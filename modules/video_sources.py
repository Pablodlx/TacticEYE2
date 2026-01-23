"""
Video Source Ingestion Layer
============================

Abstracción genérica para obtener frames desde múltiples fuentes:
- Archivos locales
- YouTube VOD/Live
- URLs HLS/RTMP
- Streams Veo

Todas las fuentes exponen un generador de frames compatible con el pipeline.
"""

import cv2
import subprocess
import json
import tempfile
from typing import Iterator, Tuple, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
import numpy as np


class SourceType(str, Enum):
    """Tipos de fuentes de video soportadas"""
    UPLOADED_FILE = "uploaded_file"
    YOUTUBE_VOD = "youtube_vod"
    YOUTUBE_LIVE = "youtube_live"
    VEO = "veo"
    HLS = "hls"
    RTMP = "rtmp"
    WEBCAM = "webcam"


@dataclass
class VideoMetadata:
    """Metadata del video/stream"""
    fps: float
    width: int
    height: int
    total_frames: Optional[int] = None  # None para streams infinitos
    duration_seconds: Optional[float] = None
    is_live: bool = False


class VideoSource:
    """
    Interfaz base para todas las fuentes de video.
    
    Contrato:
    - Implementar get_frame_generator() que retorna Iterator[np.ndarray]
    - Implementar get_metadata() que retorna VideoMetadata
    - Implementar close() para liberar recursos
    """
    
    def get_frame_generator(self) -> Iterator[np.ndarray]:
        """Retorna generador de frames (H, W, C) en formato BGR"""
        raise NotImplementedError
    
    def get_metadata(self) -> VideoMetadata:
        """Retorna metadata del video"""
        raise NotImplementedError
    
    def close(self):
        """Libera recursos"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LocalFileSource(VideoSource):
    """Fuente: archivo de video local"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {file_path}")
    
    def get_metadata(self) -> VideoMetadata:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else None
        
        return VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration_seconds=duration,
            is_live=False
        )
    
    def get_frame_generator(self) -> Iterator[np.ndarray]:
        """Genera frames uno a uno"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
    
    def close(self):
        if self.cap:
            self.cap.release()


class FFmpegStreamSource(VideoSource):
    """
    Fuente genérica usando FFmpeg para:
    - YouTube (con yt-dlp)
    - HLS
    - RTMP
    - Cualquier URL que soporte FFmpeg
    """
    
    def __init__(self, url: str, is_live: bool = False):
        self.url = url
        self.is_live = is_live
        self.process = None
        self._metadata = None
        self._probe_metadata()
    
    def _probe_metadata(self):
        """Usa ffprobe para obtener metadata"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
                '-of', 'json',
                self.url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            data = json.loads(result.stdout)
            stream = data['streams'][0]
            
            # Parse FPS (formato "30/1" o "30000/1001")
            fps_parts = stream['r_frame_rate'].split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1])
            
            width = int(stream['width'])
            height = int(stream['height'])
            
            # Para streams live, estos valores pueden no estar disponibles
            total_frames = int(stream.get('nb_frames', 0)) or None
            duration = float(stream.get('duration', 0)) or None
            
            self._metadata = VideoMetadata(
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                duration_seconds=duration,
                is_live=self.is_live
            )
        
        except Exception as e:
            # Fallback: usar valores por defecto
            print(f"Warning: No se pudo obtener metadata con ffprobe: {e}")
            self._metadata = VideoMetadata(
                fps=30.0,  # Asumimos 30 fps
                width=1920,
                height=1080,
                is_live=self.is_live
            )
    
    def get_metadata(self) -> VideoMetadata:
        return self._metadata
    
    def get_frame_generator(self) -> Iterator[np.ndarray]:
        """
        Usa FFmpeg para decodificar frames y los lee como raw video.
        
        Comando FFmpeg típico:
        ffmpeg -i <URL> -f rawvideo -pix_fmt bgr24 pipe:1
        """
        
        cmd = [
            'ffmpeg',
            '-i', self.url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',  # OpenCV usa BGR
            '-an',  # Sin audio
            'pipe:1'
        ]
        
        if self.is_live:
            # Para live: buffer pequeño, baja latencia
            cmd.insert(1, '-fflags')
            cmd.insert(2, 'nobuffer')
            cmd.insert(3, '-flags')
            cmd.insert(4, 'low_delay')
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )
        
        width = self._metadata.width
        height = self._metadata.height
        frame_size = width * height * 3  # BGR = 3 bytes por pixel
        
        while True:
            raw_frame = self.process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                break
            
            # Convertir bytes a numpy array
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            
            yield frame
    
    def close(self):
        if self.process:
            self.process.kill()
            self.process.wait()


class YouTubeSource(FFmpegStreamSource):
    """
    Fuente específica para YouTube (VOD y Live).
    
    Usa yt-dlp para resolver la URL real del stream.
    """
    
    def __init__(self, youtube_url: str, is_live: bool = False):
        # Resolver URL real con yt-dlp
        real_url = self._resolve_youtube_url(youtube_url)
        super().__init__(real_url, is_live=is_live)
        self.youtube_url = youtube_url
    
    @staticmethod
    def _resolve_youtube_url(youtube_url: str) -> str:
        """
        Usa yt-dlp para obtener la URL directa del stream.
        
        Alternativa: pytube, youtube-dl
        """
        try:
            cmd = [
                'yt-dlp',
                '-f', 'best[ext=mp4]',  # Mejor calidad MP4
                '-g',  # Get URL
                youtube_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            real_url = result.stdout.strip()
            
            if not real_url:
                raise ValueError(f"No se pudo resolver URL de YouTube: {youtube_url}")
            
            return real_url
        
        except subprocess.TimeoutExpired:
            raise ValueError("Timeout al resolver URL de YouTube")
        except Exception as e:
            raise ValueError(f"Error al resolver YouTube URL: {e}")


class HLSSource(FFmpegStreamSource):
    """Fuente para streams HLS (.m3u8)"""
    
    def __init__(self, hls_url: str):
        # HLS puede ser live o VOD, detectamos si es live
        is_live = self._detect_live(hls_url)
        super().__init__(hls_url, is_live=is_live)
    
    @staticmethod
    def _detect_live(hls_url: str) -> bool:
        """
        Detecta si un HLS es live o VOD.
        
        Heurística simple: si el playlist contiene #EXT-X-ENDLIST, es VOD.
        """
        try:
            import urllib.request
            with urllib.request.urlopen(hls_url, timeout=5) as response:
                content = response.read().decode('utf-8')
                return '#EXT-X-ENDLIST' not in content
        except:
            # Por defecto asumimos live
            return True


class RTMPSource(FFmpegStreamSource):
    """Fuente para streams RTMP"""
    
    def __init__(self, rtmp_url: str):
        super().__init__(rtmp_url, is_live=True)


class VeoSource(FFmpegStreamSource):
    """
    Fuente para Veo (plataforma de análisis de fútbol).
    
    Nota: Veo puede requerir autenticación. Este es un ejemplo genérico.
    En producción necesitarías manejar tokens/cookies.
    """
    
    def __init__(self, veo_url: str, auth_token: Optional[str] = None):
        self.auth_token = auth_token
        # Veo típicamente usa HLS o similar
        super().__init__(veo_url, is_live=False)
    
    def get_frame_generator(self) -> Iterator[np.ndarray]:
        """Override para añadir autenticación si es necesario"""
        # Aquí podrías modificar el comando FFmpeg para incluir headers
        # Por ejemplo: -headers "Authorization: Bearer <token>"
        return super().get_frame_generator()


def open_source(source_type: SourceType, source: str, **kwargs) -> VideoSource:
    """
    Factory function para crear la fuente de video apropiada.
    
    Args:
        source_type: Tipo de fuente (enum SourceType)
        source: URL o path del video
        **kwargs: Argumentos adicionales (ej: is_live, auth_token)
    
    Returns:
        VideoSource listo para generar frames
    
    Ejemplo:
        >>> with open_source(SourceType.UPLOADED_FILE, "match.mp4") as src:
        ...     for frame in src.get_frame_generator():
        ...         process(frame)
    """
    
    source_map = {
        SourceType.UPLOADED_FILE: lambda: LocalFileSource(source),
        SourceType.YOUTUBE_VOD: lambda: YouTubeSource(source, is_live=False),
        SourceType.YOUTUBE_LIVE: lambda: YouTubeSource(source, is_live=True),
        SourceType.HLS: lambda: HLSSource(source),
        SourceType.RTMP: lambda: RTMPSource(source),
        SourceType.VEO: lambda: VeoSource(source, kwargs.get('auth_token')),
    }
    
    if source_type not in source_map:
        raise ValueError(f"Tipo de fuente no soportado: {source_type}")
    
    return source_map[source_type]()


def read_frame_batches(
    stream: Iterator[np.ndarray],
    batch_size_frames: int,
    max_batches: Optional[int] = None
) -> Iterator[Tuple[int, list]]:
    """
    Lee frames del stream en micro-batches.
    
    Args:
        stream: Generador de frames
        batch_size_frames: Número de frames por batch
        max_batches: Límite de batches (útil para testing), None = infinito
    
    Yields:
        (batch_idx, frames): Tupla con índice de batch y lista de frames
    
    Ejemplo:
        Para video a 30 fps, un batch de 3 segundos = 90 frames
        
        >>> for batch_idx, frames in read_frame_batches(stream, 90):
        ...     print(f"Batch {batch_idx}: {len(frames)} frames")
    """
    
    batch_idx = 0
    current_batch = []
    
    for frame in stream:
        current_batch.append(frame)
        
        if len(current_batch) >= batch_size_frames:
            yield (batch_idx, current_batch)
            batch_idx += 1
            current_batch = []
            
            if max_batches and batch_idx >= max_batches:
                break
    
    # Último batch parcial
    if current_batch:
        yield (batch_idx, current_batch)


def calculate_batch_size(fps: float, seconds_per_batch: float = 3.0) -> int:
    """
    Calcula el tamaño de batch óptimo.
    
    Args:
        fps: Frames por segundo del video
        seconds_per_batch: Segundos de video por batch
    
    Returns:
        Número de frames por batch
    
    Recomendaciones:
        - 2-3 segundos para análisis casi tiempo real
        - 5-10 segundos para análisis offline optimizado
        - 1 segundo para streams con latencia ultra-baja
    """
    return int(fps * seconds_per_batch)


# ============================================================================
# Utilidades de testing
# ============================================================================

def test_source(source_type: SourceType, source: str, max_frames: int = 100):
    """
    Función de testing para verificar que una fuente funciona.
    
    Ejemplo:
        >>> test_source(SourceType.YOUTUBE_VOD, "https://youtube.com/watch?v=...")
    """
    print(f"\n{'='*60}")
    print(f"Testing {source_type.value}")
    print(f"Source: {source}")
    print(f"{'='*60}\n")
    
    try:
        with open_source(source_type, source) as src:
            metadata = src.get_metadata()
            
            print(f"Metadata:")
            print(f"  FPS: {metadata.fps}")
            print(f"  Resolution: {metadata.width}x{metadata.height}")
            print(f"  Total frames: {metadata.total_frames}")
            print(f"  Duration: {metadata.duration_seconds}s")
            print(f"  Is live: {metadata.is_live}")
            print()
            
            print(f"Reading first {max_frames} frames...")
            frame_count = 0
            
            for frame in src.get_frame_generator():
                frame_count += 1
                if frame_count == 1:
                    print(f"  First frame shape: {frame.shape}")
                
                if frame_count >= max_frames:
                    break
            
            print(f"✓ Successfully read {frame_count} frames")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ejemplo de uso
    print("TacticEYE - Video Source Ingestion Layer")
    print("=" * 60)
    
    # Test con archivo local (si existe)
    import os
    test_file = "sample_match.mp4"
    if os.path.exists(test_file):
        test_source(SourceType.UPLOADED_FILE, test_file, max_frames=30)
