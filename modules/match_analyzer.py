"""
Match Analyzer - Loop Principal de Micro-Batching
==================================================

Orquesta el análisis completo de un partido usando micro-batching.
Soporta múltiples fuentes de video y recuperación ante fallos.
"""

import time
from typing import Optional, Callable
from dataclasses import dataclass

from modules.video_sources import (
    open_source, SourceType, read_frame_batches, calculate_batch_size
)
from modules.match_state import (
    MatchState, get_default_storage, StateStorage
)
from modules.batch_processor import (
    BatchProcessor, ChunkOutput, save_chunk_output
)


@dataclass
class AnalysisConfig:
    """Configuración del análisis"""
    # Video
    source_type: SourceType
    source: str  # URL o path del archivo
    
    # Batching
    batch_size_seconds: float = 3.0  # Segundos por batch
    batch_size_frames: Optional[int] = None  # O frames directos
    
    # Modelo
    model_path: str = "weights/best.pt"
    device: str = "cuda"
    conf_threshold: float = 0.3
    imgsz: int = 640
    
    # Tracker
    max_age: int = 30
    max_lost_time: float = 120.0
    
    # Team Classifier
    kmeans_min_tracks: int = 10
    vote_history: int = 4
    use_L_channel: bool = True
    L_weight: float = 0.5
    
    # Storage
    storage: Optional[StateStorage] = None
    output_dir: str = "outputs_streaming"
    
    # Callbacks
    on_batch_complete: Optional[Callable] = None
    on_progress: Optional[Callable] = None
    on_error: Optional[Callable] = None
    on_frame_visualized: Optional[Callable] = None  # Callback para enviar frames anotados
    
    # Límites (para testing)
    max_batches: Optional[int] = None


def run_match_analysis(
    match_id: str,
    config: AnalysisConfig,
    resume: bool = True
) -> MatchState:
    """
    Loop principal de análisis de partido con micro-batching.
    
    Args:
        match_id: Identificador único del partido
        config: Configuración del análisis
        resume: Si True, intenta reanudar desde el último estado guardado
    
    Returns:
        MatchState final del análisis
    
    Ejemplo:
        >>> config = AnalysisConfig(
        ...     source_type=SourceType.UPLOADED_FILE,
        ...     source="match.mp4",
        ...     batch_size_seconds=3.0
        ... )
        >>> state = run_match_analysis("match_001", config)
        >>> print(state.get_summary())
    """
    
    storage = config.storage or get_default_storage()
    
    # 1. CARGAR O CREAR ESTADO
    if resume and storage.exists(match_id):
        print(f"[{match_id}] Reanudando análisis desde estado guardado...")
        match_state = storage.load(match_id)
        start_batch = match_state.last_batch_idx + 1
        print(f"[{match_id}] Reanudando desde batch {start_batch}")
    else:
        print(f"[{match_id}] Iniciando nuevo análisis...")
        match_state = MatchState(
            match_id=match_id,
            source_type=config.source_type.value,
            source_url=config.source
        )
        start_batch = 0
    
    match_state.mark_running()
    
    # 2. ABRIR FUENTE DE VIDEO
    print(f"[{match_id}] Abriendo fuente: {config.source_type.value}")
    
    try:
        with open_source(config.source_type, config.source) as video_source:
            
            # Obtener metadata
            metadata = video_source.get_metadata()
            match_state.fps = metadata.fps
            match_state.metadata['video_metadata'] = {
                'fps': metadata.fps,
                'width': metadata.width,
                'height': metadata.height,
                'total_frames': metadata.total_frames,
                'duration_seconds': metadata.duration_seconds,
                'is_live': metadata.is_live
            }
            
            print(f"[{match_id}] Video: {metadata.width}x{metadata.height} @ {metadata.fps} fps")
            
            if metadata.is_live:
                print(f"[{match_id}] Modo: LIVE STREAM (análisis continuo)")
            else:
                duration_str = f"{metadata.duration_seconds:.1f}s" if metadata.duration_seconds else "duración desconocida"
                frames_str = f"{metadata.total_frames} frames" if metadata.total_frames else "frames desconocidos"
                print(f"[{match_id}] Modo: VOD ({frames_str}, {duration_str})")
            
            # Calcular batch size
            if config.batch_size_frames:
                batch_size = config.batch_size_frames
            else:
                batch_size = calculate_batch_size(metadata.fps, config.batch_size_seconds)
            
            print(f"[{match_id}] Batch size: {batch_size} frames ({config.batch_size_seconds}s)")
            
            # 3. INICIALIZAR PROCESADOR
            processor = BatchProcessor(
                model_path=config.model_path,
                device=config.device,
                conf_threshold=config.conf_threshold,
                imgsz=config.imgsz,
                max_age=config.max_age,
                max_lost_time=config.max_lost_time,
                kmeans_min_tracks=config.kmeans_min_tracks,
                vote_history=config.vote_history,
                use_L=config.use_L_channel,
                L_weight=config.L_weight
            )
            
            # 4. LOOP DE MICRO-BATCHING
            frame_generator = video_source.get_frame_generator()
            batch_generator = read_frame_batches(
                frame_generator,
                batch_size,
                max_batches=config.max_batches
            )
            
            # Skip batches ya procesados (si reanudamos)
            for _ in range(start_batch):
                try:
                    next(batch_generator)
                except StopIteration:
                    break
            
            print(f"[{match_id}] Iniciando procesamiento de batches...\n")
            
            total_processing_time = 0.0
            batches_processed = 0
            
            for batch_idx, frames in batch_generator:
                batch_start_time = time.time()
                
                # Calcular frame global inicial
                start_frame = batch_idx * batch_size
                
                print(f"[{match_id}] Batch {batch_idx}: frames {start_frame}-{start_frame + len(frames) - 1} ({len(frames)} frames)")
                
                try:
                    # PROCESAR CHUNK
                    # Crear callback para visualización
                    def send_viz(frame, frame_idx):
                        if config.on_frame_visualized:
                            config.on_frame_visualized(match_id, frame, frame_idx)
                    
                    match_state, chunk_output = processor.process_chunk(
                        match_state,
                        frames,
                        start_frame,
                        metadata.fps,
                        visualize_interval=15,
                        send_visualization=send_viz if config.on_frame_visualized else None
                    )
                    
                    # Guardar outputs
                    save_chunk_output(match_id, chunk_output, config.output_dir)
                    
                    # Guardar estado (checkpoint)
                    storage.save(match_id, match_state)
                    
                    batch_time = time.time() - batch_start_time
                    total_processing_time += batch_time
                    batches_processed += 1
                    
                    # Calcular métricas de performance
                    fps_processing = len(frames) / batch_time
                    realtime_factor = fps_processing / metadata.fps
                    
                    print(f"  ✓ Procesado en {batch_time:.2f}s ({fps_processing:.1f} fps, {realtime_factor:.2f}x realtime)")
                    print(f"  ✓ Detecciones: {chunk_output.chunk_stats['detections_count']}")
                    print(f"  ✓ Eventos: {chunk_output.chunk_stats['events_count']}")
                    print(f"  ✓ Posesión: Team {chunk_output.chunk_stats['possession_team']}")
                    
                    # Callback de progreso
                    if config.on_progress:
                        config.on_progress(
                            match_id,
                            batch_idx,
                            match_state.total_frames_processed,
                            metadata.total_frames
                        )
                    
                    # Callback de batch completo
                    if config.on_batch_complete:
                        config.on_batch_complete(match_id, batch_idx, chunk_output, match_state)
                    
                    print()
                
                except Exception as e:
                    print(f"  ✗ Error en batch {batch_idx}: {e}")
                    
                    if config.on_error:
                        config.on_error(match_id, batch_idx, e)
                    
                    # Guardar estado de error
                    match_state.mark_failed(str(e))
                    storage.save(match_id, match_state)
                    
                    raise
            
            # 5. FINALIZACIÓN
            match_state.mark_completed()
            storage.save(match_id, match_state)
            
            print(f"\n{'='*60}")
            print(f"[{match_id}] ANÁLISIS COMPLETADO")
            print(f"{'='*60}")
            print(f"Batches procesados: {batches_processed}")
            print(f"Frames totales: {match_state.total_frames_processed}")
            print(f"Tiempo total: {total_processing_time:.2f}s")
            
            if batches_processed > 0:
                avg_fps = match_state.total_frames_processed / total_processing_time
                print(f"FPS promedio: {avg_fps:.1f}")
                print(f"Factor realtime: {avg_fps / metadata.fps:.2f}x")
            
            print()
            
            # Imprimir resumen
            summary = match_state.get_summary()
            print("RESUMEN:")
            print(f"  Posesión Team 0: {summary['possession']['percent_by_team'].get(0, 0):.1f}%")
            print(f"  Posesión Team 1: {summary['possession']['percent_by_team'].get(1, 0):.1f}%")
            print(f"  Pases Team 0: {summary['passes']['by_team'].get(0, 0)}")
            print(f"  Pases Team 1: {summary['passes']['by_team'].get(1, 0)}")
            print(f"  Jugadores Team 0: {summary['teams']['team_0_players']}")
            print(f"  Jugadores Team 1: {summary['teams']['team_1_players']}")
            print()
            
            return match_state
    
    except Exception as e:
        print(f"\n[{match_id}] ERROR FATAL: {e}")
        match_state.mark_failed(str(e))
        storage.save(match_id, match_state)
        raise


def analyze_local_file(
    match_id: str,
    file_path: str,
    **kwargs
) -> MatchState:
    """
    Shortcut para analizar un archivo local.
    
    Ejemplo:
        >>> state = analyze_local_file("match_001", "video.mp4")
    """
    config = AnalysisConfig(
        source_type=SourceType.UPLOADED_FILE,
        source_url=file_path,
        **kwargs
    )
    return run_match_analysis(match_id, config)


def analyze_youtube(
    match_id: str,
    youtube_url: str,
    is_live: bool = False,
    **kwargs
) -> MatchState:
    """
    Shortcut para analizar un video de YouTube.
    
    Ejemplo:
        >>> # VOD
        >>> state = analyze_youtube("match_002", "https://youtube.com/watch?v=...")
        >>> # Live
        >>> state = analyze_youtube("match_003", "https://youtube.com/watch?v=...", is_live=True)
    """
    source_type = SourceType.YOUTUBE_LIVE if is_live else SourceType.YOUTUBE_VOD
    config = AnalysisConfig(
        source_type=source_type,
        source_url=youtube_url,
        **kwargs
    )
    return run_match_analysis(match_id, config)


def analyze_hls_stream(
    match_id: str,
    hls_url: str,
    **kwargs
) -> MatchState:
    """
    Shortcut para analizar un stream HLS.
    
    Ejemplo:
        >>> state = analyze_hls_stream("match_004", "https://example.com/stream.m3u8")
    """
    config = AnalysisConfig(
        source_type=SourceType.HLS,
        source_url=hls_url,
        **kwargs
    )
    return run_match_analysis(match_id, config)


# ============================================================================
# Ejemplo de uso
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python match_analyzer.py <match_id> <video_file>")
        print("\nEjemplo:")
        print("  python match_analyzer.py match_001 sample_match.mp4")
        sys.exit(1)
    
    match_id = sys.argv[1]
    video_file = sys.argv[2]
    
    # Callbacks de ejemplo
    def on_progress(match_id, progress):
        percent = 0
        if progress.get('total_frames'):
            percent = (progress['frames_processed'] / progress['total_frames']) * 100
        print(f"  >> Progreso: {percent:.1f}%")
    
    def on_batch_complete(match_id, output):
        print(f"  >> Batch {output.batch_idx} guardado")
    
    # Configuración
    config = AnalysisConfig(
        source_type=SourceType.UPLOADED_FILE,
        source_url=video_file,
        batch_size_seconds=3.0,
        model_path="weights/best.pt",
        on_progress=on_progress,
        on_batch_complete=on_batch_complete
    )
    
    # Ejecutar análisis
    try:
        final_state = run_match_analysis(match_id, config, resume=True)
        print(f"\n✓ Análisis completado exitosamente")
        print(f"  Estado guardado en: match_states/{match_id}.json")
        print(f"  Outputs guardados en: outputs_streaming/{match_id}/")
    
    except KeyboardInterrupt:
        print(f"\n⚠ Análisis interrumpido por el usuario")
        print(f"  Puedes reanudar ejecutando el mismo comando")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
