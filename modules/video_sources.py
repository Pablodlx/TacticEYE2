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
                total_frames=None,  # No conocido para streams
                duration_seconds=None,  # No conocido para streams
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
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '5',
            '-user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            '-i', self.url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',  # OpenCV usa BGR
            '-an',  # Sin audio
            'pipe:1'
        ]
        
        if self.is_live:
            # Para live: buffer pequeño, baja latencia (ya incluidas las opciones de reconnect arriba)
            pass
        
        print(f"Iniciando FFmpeg para lectura de frames...")
        print(f"Comando FFmpeg con opciones de reconnect y user-agent")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capturar stderr para debugging
            bufsize=10**8
        )
        
        width = self._metadata.width
        height = self._metadata.height
        frame_size = width * height * 3  # BGR = 3 bytes por pixel
        
        print(f"Esperando frames de FFmpeg (frame_size={frame_size} bytes, {width}x{height})...")
        
        frame_count = 0
        while True:
            raw_frame = self.process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                # Verificar si FFmpeg tuvo algún error
                if frame_count == 0:
                    stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                    print(f"FFmpeg no produjo frames. Error stderr:\n{stderr_output[-500:]}")
                break
            
            frame_count += 1
            if frame_count == 1:
                print(f"Primer frame recibido correctamente")
            elif frame_count % 1000 == 0:
                print(f"Procesados {frame_count} frames...")
            
            # Convertir bytes a numpy array
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            
            yield frame
    
    def close(self):
        if self.process:
            self.process.kill()
            self.process.wait()


class YouTubeSource(VideoSource):
    """
    Fuente específica para YouTube (VOD y Live) con streaming directo.
    
    Usa yt-dlp para obtener la mejor URL de streaming y OpenCV para leer frames.
    """
    
    def __init__(self, youtube_url: str, is_live: bool = False):
        self.youtube_url = youtube_url
        self.is_live = is_live
        self.cap = None
        self._metadata = None
        
        # Obtener metadata y URL de streaming
        self.stream_url = self._get_stream_url()
        self._probe_metadata()
    
    def _get_stream_url(self) -> str:
        """Obtiene la mejor URL de streaming usando yt-dlp"""
        try:
            print(f"Resolviendo URL de YouTube para streaming: {self.youtube_url}")
            
            # Obtener metadata primero
            cmd_json = [
                'yt-dlp',
                '--no-playlist',
                '--dump-json',
                '--no-warnings',
                self.youtube_url
            ]
            
            result_json = subprocess.run(cmd_json, capture_output=True, text=True, timeout=30)
            
            # Obtener URL de streaming - usar formato que OpenCV/FFmpeg puedan leer directamente
            # Preferir formatos progresivos (no DASH/HLS) para mejor compatibilidad
            cmd = [
                'yt-dlp',
                '--no-playlist',
                '--no-warnings',
                '-f', 'best[protocol^=http][ext=mp4]/best[protocol^=http]/best',  # HTTP progresivo preferido
                '-g',  # Get URL
                self.youtube_url
            ]
            
            print("Obteniendo URL de streaming con yt-dlp...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Error desconocido"
                raise ValueError(f"yt-dlp error: {error_msg}")
            
            stream_url = result.stdout.strip()
            
            if not stream_url:
                raise ValueError("No se pudo obtener URL de streaming")
            
            print(f"URL de streaming obtenida: {stream_url[:100]}...")
            return stream_url
            
        except Exception as e:
            print(f"Error obteniendo URL de streaming: {e}")
            raise ValueError(f"Error al obtener URL de YouTube: {e}")
    
    def _probe_metadata(self):
        """Obtiene metadata del video"""
        try:
            # Abrir video con OpenCV
            self.cap = cv2.VideoCapture(self.stream_url)
            
            if not self.cap.isOpened():
                raise ValueError("No se pudo abrir el stream con OpenCV")
            
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Para streams, total_frames puede ser 0
            if total_frames == 0:
                total_frames = None
                duration = None
            else:
                duration = total_frames / fps if fps > 0 else None
            
            print(f"Metadata del stream: {width}x{height} @ {fps}fps")
            
            self._metadata = VideoMetadata(
                fps=fps if fps > 0 else 30.0,
                width=width,
                height=height,
                total_frames=total_frames,
                duration_seconds=duration,
                is_live=self.is_live
            )
            
        except Exception as e:
            print(f"Error obteniendo metadata con OpenCV: {e}")
            # Intentar obtener de yt-dlp
            try:
                cmd = [
                    'yt-dlp',
                    '--no-playlist',
                    '--dump-json',
                    '--no-warnings',
                    self.youtube_url
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and result.stdout:
                    metadata = json.loads(result.stdout)
                    
                    fps = float(metadata.get('fps', 30.0))
                    width = int(metadata.get('width', 1920))
                    height = int(metadata.get('height', 1080))
                    duration = float(metadata.get('duration', 0)) if not self.is_live else None
                    total_frames = int(duration * fps) if duration else None
                    
                    self._metadata = VideoMetadata(
                        fps=fps,
                        width=width,
                        height=height,
                        total_frames=total_frames,
                        duration_seconds=duration,
                        is_live=self.is_live
                    )
                    
                    # Reabrir con OpenCV
                    self.cap = cv2.VideoCapture(self.stream_url)
                else:
                    raise ValueError("No se pudo obtener metadata")
                    
            except Exception as e2:
                print(f"Error obteniendo metadata con yt-dlp: {e2}")
                raise ValueError(f"No se pudo obtener metadata del video: {e}, {e2}")
    
    def get_metadata(self) -> VideoMetadata:
        return self._metadata
    
    def get_frame_generator(self) -> Iterator[np.ndarray]:
        """Genera frames usando OpenCV"""
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.stream_url)
            
        if not self.cap.isOpened():
            raise ValueError("No se pudo abrir el video stream")
        
        print(f"Iniciando lectura de frames con OpenCV...")
        
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Stream finalizado. Total frames: {frame_count}")
                break
            
            frame_count += 1
            if frame_count == 1:
                print("Primer frame recibido correctamente!")
            elif frame_count % 1000 == 0:
                print(f"Procesados {frame_count} frames...")
            
            yield frame
    
    def close(self):
        if self.cap:
            self.cap.release()




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
