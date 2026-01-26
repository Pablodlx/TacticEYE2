#!/usr/bin/env python3
"""
quick_test_spatial.py - Prueba rápida del sistema de calibración espacial

Ejecuta un análisis corto (primeros 300 frames) para verificar que todo funciona.
"""

import sys
from pathlib import Path

# Verificar que hay un video como argumento
if len(sys.argv) < 2:
    print("Uso: python quick_test_spatial.py <video.mp4>")
    print("\nEjemplo:")
    print("  python quick_test_spatial.py sample_match.mp4")
    sys.exit(1)

video_path = sys.argv[1]

if not Path(video_path).exists():
    print(f"❌ Error: Video no encontrado: {video_path}")
    sys.exit(1)

print("="*80)
print("PRUEBA RÁPIDA DE CALIBRACIÓN ESPACIAL")
print("="*80)
print(f"\nVideo: {video_path}")
print("Procesando primeros 300 frames (~10 segundos)...")
print()

# Ejecutar análisis
from modules.match_analyzer import run_match_analysis, AnalysisConfig
from modules.video_sources import SourceType

config = AnalysisConfig(
    source_type=SourceType.UPLOADED_FILE,
    source=video_path,
    batch_size_seconds=3.0,
    model_path="weights/best.pt",
    output_dir="outputs_quick_test",
    # Habilitar tracking espacial
    enable_spatial_tracking=True,
    zone_partition_type='thirds_lanes',
    enable_heatmaps=True,
    heatmap_resolution=(50, 34),
    # Limitar a 300 frames
    max_batches=4  # 4 batches × 90 frames = 360 frames ≈ 12 segundos
)

match_id = Path(video_path).stem

try:
    print("Iniciando análisis...\n")
    match_state = run_match_analysis(match_id, config, resume=False)
    
    print("\n" + "="*80)
    print("✓ PRUEBA COMPLETADA EXITOSAMENTE")
    print("="*80)
    
    # Mostrar resumen
    summary = match_state.get_summary()
    
    print(f"\nFrames procesados: {summary['progress']['total_frames']}")
    print(f"Duración: {summary['progress']['total_seconds']:.1f}s")
    
    print("\nPosesión:")
    for team_id in [0, 1]:
        pct = summary['possession']['percent_by_team'].get(team_id, 0)
        print(f"  Team {team_id}: {pct:.1f}%")
    
    print(f"\nOutputs guardados en: outputs_quick_test/{match_id}/")
    print(f"Heatmaps guardados en: outputs_quick_test/{match_id}_heatmaps.npz")
    
    print("\n✓ Sistema de calibración espacial funcionando correctamente!")
    print("\nPara análisis completo, ejecuta:")
    print(f"  python test_spatial_tracking.py {video_path}")
    
except KeyboardInterrupt:
    print("\n\n⚠ Prueba interrumpida")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
