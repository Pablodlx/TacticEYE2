#!/usr/bin/env python3
"""
TacticEYE Streaming - Ejemplo de Uso Completo
==============================================

Demuestra todas las capacidades del sistema de micro-batching.
"""

import sys
import time
from modules.match_analyzer import (
    run_match_analysis,
    AnalysisConfig,
    analyze_local_file,
    analyze_youtube,
    analyze_hls_stream
)
from modules.video_sources import SourceType
from modules.match_state import get_default_storage


def example_1_local_file():
    """Ejemplo 1: Análisis simple de archivo local"""
    
    print("\n" + "="*60)
    print("EJEMPLO 1: Archivo Local")
    print("="*60 + "\n")
    
    state = analyze_local_file(
        match_id="example_local",
        file_path="sample_match.mp4",
        batch_size_seconds=3.0,
        max_batches=5  # Solo 5 batches para demo
    )
    
    summary = state.get_summary()
    
    print("\n📊 RESUMEN:")
    print(f"  Frames procesados: {summary['progress']['total_frames']}")
    print(f"  Posesión Team 0: {summary['possession']['percent_by_team'].get(0, 0):.1f}%")
    print(f"  Posesión Team 1: {summary['possession']['percent_by_team'].get(1, 0):.1f}%")
    print(f"  Pases Team 0: {summary['passes']['by_team'].get(0, 0)}")
    print(f"  Pases Team 1: {summary['passes']['by_team'].get(1, 0)}")


def example_2_with_callbacks():
    """Ejemplo 2: Análisis con callbacks personalizados"""
    
    print("\n" + "="*60)
    print("EJEMPLO 2: Con Callbacks")
    print("="*60 + "\n")
    
    start_time = time.time()
    batches_count = 0
    
    def on_progress(match_id, progress):
        """Callback de progreso"""
        percent = 0
        if progress.get('total_frames'):
            percent = (progress['frames_processed'] / progress['total_frames']) * 100
        
        print(f"  📈 Progreso: {percent:.1f}% "
              f"(Frame {progress['frames_processed']}/{progress.get('total_frames', '?')})")
        print(f"     FPS: {progress.get('fps_processing', 0):.1f}, "
              f"Realtime: {progress.get('realtime_factor', 0):.2f}x")
    
    def on_batch_complete(match_id, output):
        """Callback de batch completado"""
        nonlocal batches_count
        batches_count += 1
        
        print(f"  ✓ Batch {output.batch_idx} completado:")
        print(f"    - Frames: {output.start_frame}-{output.end_frame}")
        print(f"    - Detecciones: {output.chunk_stats['detections_count']}")
        print(f"    - Eventos: {len(output.events)}")
        print(f"    - Tiempo: {output.processing_time_ms:.0f}ms")
        print()
    
    def on_error(match_id, batch_idx, error):
        """Callback de error"""
        print(f"  ✗ Error en batch {batch_idx}: {error}")
    
    config = AnalysisConfig(
        source_type=SourceType.UPLOADED_FILE,
        source_url="sample_match.mp4",
        batch_size_seconds=2.0,
        on_progress=on_progress,
        on_batch_complete=on_batch_complete,
        on_error=on_error,
        max_batches=3  # Solo 3 batches para demo
    )
    
    state = run_match_analysis("example_callbacks", config)
    
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Tiempo total: {elapsed:.2f}s")
    print(f"📦 Batches procesados: {batches_count}")


def example_3_resume():
    """Ejemplo 3: Demostrar recuperación ante fallos"""
    
    print("\n" + "="*60)
    print("EJEMPLO 3: Recuperación ante Fallos")
    print("="*60 + "\n")
    
    match_id = "example_resume"
    
    # Primera ejecución (interrumpir manualmente o simular fallo)
    print("🔄 Primera ejecución (procesará 2 batches)...\n")
    
    config = AnalysisConfig(
        source_type=SourceType.UPLOADED_FILE,
        source_url="sample_match.mp4",
        batch_size_seconds=3.0,
        max_batches=2
    )
    
    try:
        state1 = run_match_analysis(match_id, config, resume=False)
        print(f"\n✓ Primera ejecución completada. Batches: {state1.last_batch_idx + 1}")
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por el usuario")
    
    # Verificar estado guardado
    storage = get_default_storage()
    saved_state = storage.load(match_id)
    
    if saved_state:
        print(f"\n💾 Estado guardado encontrado:")
        print(f"   - Match ID: {saved_state.match_id}")
        print(f"   - Último batch: {saved_state.last_batch_idx}")
        print(f"   - Frames procesados: {saved_state.total_frames_processed}")
        print(f"   - Estado: {saved_state.status}")
    
    # Segunda ejecución (reanudar)
    print(f"\n🔄 Segunda ejecución (reanudará desde batch {saved_state.last_batch_idx + 1})...\n")
    
    config.max_batches = 5  # Procesar hasta batch 5
    
    state2 = run_match_analysis(match_id, config, resume=True)
    
    print(f"\n✓ Segunda ejecución completada. Batches totales: {state2.last_batch_idx + 1}")


def example_4_query_results():
    """Ejemplo 4: Consultar resultados almacenados"""
    
    print("\n" + "="*60)
    print("EJEMPLO 4: Consultar Resultados")
    print("="*60 + "\n")
    
    from modules.batch_processor import load_match_outputs
    
    # Primero analizar (o usar análisis previo)
    match_id = "example_query"
    
    if not get_default_storage().exists(match_id):
        print("📹 Analizando video primero...\n")
        analyze_local_file(
            match_id=match_id,
            file_path="sample_match.mp4",
            batch_size_seconds=3.0,
            max_batches=3
        )
    
    # Cargar estado
    state = get_default_storage().load(match_id)
    summary = state.get_summary()
    
    print("📊 RESUMEN DEL PARTIDO:")
    print(f"   Estado: {summary['status']}")
    print(f"   Frames: {summary['progress']['total_frames']}")
    print(f"   Duración: {summary['progress']['total_seconds']:.1f}s")
    print()
    
    print("⚽ POSESIÓN:")
    print(f"   Team 0: {summary['possession']['percent_by_team'].get(0, 0):.1f}% "
          f"({summary['possession']['seconds_by_team'].get(0, 0):.1f}s)")
    print(f"   Team 1: {summary['possession']['percent_by_team'].get(1, 0):.1f}% "
          f"({summary['possession']['seconds_by_team'].get(1, 0):.1f}s)")
    print()
    
    print("🎯 PASES:")
    print(f"   Team 0: {summary['passes']['by_team'].get(0, 0)} pases")
    print(f"   Team 1: {summary['passes']['by_team'].get(1, 0)} pases")
    print()
    
    # Cargar outputs detallados
    outputs = load_match_outputs(match_id)
    
    print(f"📦 OUTPUTS ALMACENADOS:")
    print(f"   Batches: {len(outputs)}")
    
    if outputs:
        total_events = sum(len(o.events) for o in outputs)
        total_positions = sum(len(o.player_positions) for o in outputs)
        
        print(f"   Eventos totales: {total_events}")
        print(f"   Posiciones registradas: {total_positions}")
        
        # Mostrar algunos eventos
        if total_events > 0:
            print("\n   📋 Primeros eventos:")
            for output in outputs[:2]:
                for event in output.events[:3]:
                    print(f"      - Frame {event['frame']}: {event['type']} (Team {event.get('team', '?')})")


def example_5_youtube():
    """Ejemplo 5: Análisis de YouTube (requiere URL válida)"""
    
    print("\n" + "="*60)
    print("EJEMPLO 5: YouTube")
    print("="*60 + "\n")
    
    print("⚠️  Este ejemplo requiere una URL de YouTube válida.")
    print("    Modifica el código para probar con un video real.\n")
    
    # Descomentar y modificar con URL real:
    """
    state = analyze_youtube(
        match_id="youtube_match",
        youtube_url="https://youtube.com/watch?v=YOUR_VIDEO_ID",
        is_live=False,
        batch_size_seconds=3.0,
        max_batches=5
    )
    
    summary = state.get_summary()
    print(f"✓ Análisis de YouTube completado")
    print(f"  Posesión: {summary['possession']['percent_by_team']}")
    """
    
    print("💡 Para probar:")
    print("   1. Encuentra un video de fútbol en YouTube")
    print("   2. Copia la URL")
    print("   3. Descomenta el código en este ejemplo")
    print("   4. Ejecuta: python demo_streaming.py youtube")


def main():
    """Menu principal"""
    
    print("\n" + "="*60)
    print("TacticEYE Streaming - Ejemplos de Uso")
    print("="*60)
    
    examples = {
        '1': ("Archivo local simple", example_1_local_file),
        '2': ("Con callbacks", example_2_with_callbacks),
        '3': ("Recuperación ante fallos", example_3_resume),
        '4': ("Consultar resultados", example_4_query_results),
        '5': ("YouTube (requiere URL)", example_5_youtube),
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\nEjemplos disponibles:")
        for key, (name, _) in examples.items():
            print(f"  {key}. {name}")
        
        print("\nUso:")
        print("  python demo_streaming.py [1-5]")
        print("  o ejecuta sin argumentos para ver este menú")
        print()
        
        choice = input("Selecciona un ejemplo (1-5): ")
    
    if choice in examples:
        name, func = examples[choice]
        try:
            func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrumpido por el usuario")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n❌ Ejemplo inválido: {choice}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ Ejemplo completado")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
