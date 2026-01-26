#!/usr/bin/env python3
"""
test_spatial_tracking.py - Script de prueba para calibración automática y tracking espacial

Ejecuta el pipeline completo con el nuevo sistema de calibración y muestra
estadísticas espaciales de posesión.

Uso:
    python test_spatial_tracking.py sample_match.mp4
    python test_spatial_tracking.py sample_match.mp4 --zones thirds_lanes
    python test_spatial_tracking.py sample_match.mp4 --no-heatmaps
"""

import argparse
import sys
import cv2
import numpy as np
import json
from pathlib import Path

# Imports del pipeline
from modules.match_analyzer import run_match_analysis, AnalysisConfig
from modules.video_sources import SourceType
from modules.batch_processor import export_spatial_heatmaps


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test de calibración automática y tracking espacial'
    )
    
    parser.add_argument(
        'video',
        type=str,
        help='Ruta al video de fútbol'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='weights/best.pt',
        help='Ruta al modelo YOLO (default: weights/best.pt)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs_spatial_test',
        help='Directorio de salida (default: outputs_spatial_test)'
    )
    
    parser.add_argument(
        '--batch-seconds',
        type=float,
        default=3.0,
        help='Tamaño de batch en segundos (default: 3.0)'
    )
    
    parser.add_argument(
        '--zones',
        type=str,
        choices=['grid', 'thirds', 'thirds_lanes'],
        default='thirds_lanes',
        help='Tipo de partición de zonas (default: thirds_lanes)'
    )
    
    parser.add_argument(
        '--zone-nx',
        type=int,
        default=6,
        help='Divisiones en X para grid (default: 6)'
    )
    
    parser.add_argument(
        '--zone-ny',
        type=int,
        default=4,
        help='Divisiones en Y para grid (default: 4)'
    )
    
    parser.add_argument(
        '--no-heatmaps',
        action='store_true',
        help='Deshabilitar generación de heatmaps'
    )
    
    parser.add_argument(
        '--heatmap-resolution',
        type=str,
        default='50,34',
        help='Resolución del heatmap como "width,height" (default: 50,34)'
    )
    
    parser.add_argument(
        '--disable-spatial',
        action='store_true',
        help='Deshabilitar tracking espacial completamente'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Número máximo de frames a procesar (None = todo el video)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Mostrar visualización de calibración en tiempo real'
    )
    
    return parser.parse_args()


def print_spatial_statistics(match_state, processor):
    """Imprime estadísticas espaciales detalladas"""
    
    if not processor.enable_spatial_tracking or processor.spatial_tracker is None:
        print("\n⚠ Tracking espacial no está habilitado")
        return
    
    print("\n" + "="*80)
    print("ESTADÍSTICAS ESPACIALES")
    print("="*80)
    
    # Verificar calibración
    if processor.field_calibrator.has_valid_calibration():
        print("✓ Calibración de campo: VÁLIDA")
    else:
        print("⚠ Calibración de campo: NO DISPONIBLE")
        print("  (Las estadísticas pueden estar incompletas)")
    
    # Obtener estadísticas por zona
    zone_stats = processor.spatial_tracker.get_zone_statistics()
    
    print(f"\nTipo de partición: {zone_stats['partition_type']}")
    print(f"Número de zonas: {zone_stats['num_zones']}")
    print()
    
    # Tabla de estadísticas por zona
    print(f"{'ZONA':<20} {'TEAM 0':>15} {'TEAM 1':>15} {'TOTAL':>10}")
    print("-" * 65)
    
    for zone_info in zone_stats['zones']:
        zone_name = zone_info['zone_name']
        team_0_frames = zone_info['team_0_frames']
        team_1_frames = zone_info['team_1_frames']
        total = team_0_frames + team_1_frames
        
        if total > 0:
            team_0_pct = zone_info['team_0_percent']
            team_1_pct = zone_info['team_1_percent']
            
            print(f"{zone_name:<20} {team_0_frames:>6} ({team_0_pct:>5.1f}%) "
                  f"{team_1_frames:>6} ({team_1_pct:>5.1f}%) {total:>10}")
    
    print()
    
    # Resumen por equipo
    spatial_stats = processor.spatial_tracker.get_spatial_statistics()
    
    print("\nRESUMEN POR EQUIPO:")
    print("-" * 65)
    
    for team_id in [0, 1]:
        zone_times = spatial_stats['possession_by_zone'][team_id]
        total_time = sum(zone_times)
        
        if total_time > 0:
            fps = match_state.fps if match_state.fps > 0 else 30.0
            duration_sec = total_time / fps
            
            print(f"\nTeam {team_id}:")
            print(f"  Frames totales con posición: {total_time}")
            print(f"  Duración: {duration_sec:.1f} segundos")
            
            # Top 3 zonas
            zone_list = list(enumerate(zone_times))
            zone_list.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  Top 3 zonas:")
            for zone_id, frames in zone_list[:3]:
                if frames > 0:
                    zone_name = processor.spatial_tracker.zone_model.get_zone_name(zone_id)
                    pct = (frames / total_time) * 100
                    print(f"    - {zone_name}: {frames} frames ({pct:.1f}%)")
    
    print()


def visualize_heatmaps(heatmap_path: str):
    """Visualiza los heatmaps generados"""
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("\n⚠ matplotlib no disponible - saltando visualización")
        return
    
    # Cargar heatmaps
    data = np.load(heatmap_path)
    
    hm_0 = data['team_0_heatmap']
    hm_1 = data['team_1_heatmap']
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap Team 0 (verde)
    cmap_green = LinearSegmentedColormap.from_list(
        'green_heatmap', 
        ['white', 'lightgreen', 'green', 'darkgreen']
    )
    
    im1 = ax1.imshow(hm_0, cmap=cmap_green, aspect='auto', origin='lower')
    ax1.set_title('Team 0 - Possession Heatmap', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Field Length')
    ax1.set_ylabel('Field Width')
    plt.colorbar(im1, ax=ax1, label='Normalized Possession Density')
    
    # Heatmap Team 1 (rojo)
    cmap_red = LinearSegmentedColormap.from_list(
        'red_heatmap', 
        ['white', 'lightcoral', 'red', 'darkred']
    )
    
    im2 = ax2.imshow(hm_1, cmap=cmap_red, aspect='auto', origin='lower')
    ax2.set_title('Team 1 - Possession Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Field Length')
    ax2.set_ylabel('Field Width')
    plt.colorbar(im2, ax=ax2, label='Normalized Possession Density')
    
    plt.tight_layout()
    
    # Guardar figura
    output_png = heatmap_path.replace('.npz', '.png')
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"✓ Visualización guardada: {output_png}")
    
    # Mostrar
    plt.show()


def main():
    args = parse_args()
    
    # Verificar que el video existe
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video no encontrado: {args.video}")
        sys.exit(1)
    
    # Parsear resolución de heatmap
    heatmap_res = tuple(map(int, args.heatmap_resolution.split(',')))
    if len(heatmap_res) != 2:
        print(f"❌ Error: Formato de resolución inválido: {args.heatmap_resolution}")
        print("   Use formato: width,height (ej: 50,34)")
        sys.exit(1)
    
    print("="*80)
    print("TEST DE CALIBRACIÓN AUTOMÁTICA Y TRACKING ESPACIAL")
    print("="*80)
    print(f"\nVideo: {args.video}")
    print(f"Modelo: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Tracking espacial: {'DESHABILITADO' if args.disable_spatial else 'HABILITADO'}")
    
    if not args.disable_spatial:
        print(f"  - Tipo de zonas: {args.zones}")
        if args.zones == 'grid':
            print(f"  - Grid: {args.zone_nx}×{args.zone_ny}")
        print(f"  - Heatmaps: {'DESHABILITADOS' if args.no_heatmaps else 'HABILITADOS'}")
        if not args.no_heatmaps:
            print(f"  - Resolución: {heatmap_res[0]}×{heatmap_res[1]}")
    
    print()
    
    # Configurar análisis
    config = AnalysisConfig(
        source_type=SourceType.UPLOADED_FILE,
        source=str(video_path),
        batch_size_seconds=args.batch_seconds,
        model_path=args.model,
        output_dir=args.output_dir,
        # Parámetros espaciales
        enable_spatial_tracking=not args.disable_spatial,
        zone_partition_type=args.zones,
        zone_nx=args.zone_nx,
        zone_ny=args.zone_ny,
        enable_heatmaps=not args.no_heatmaps,
        heatmap_resolution=heatmap_res
    )
    
    # Ejecutar análisis
    match_id = video_path.stem
    
    try:
        print("Iniciando análisis...\n")
        match_state = run_match_analysis(match_id, config, resume=False)
        
        print("\n" + "="*80)
        print("ANÁLISIS COMPLETADO")
        print("="*80)
        
        # Obtener summary
        summary = match_state.get_summary()
        
        # Imprimir resumen básico
        print("\nRESUMEN GENERAL:")
        print(f"  Frames procesados: {summary['progress']['total_frames']}")
        print(f"  Duración: {summary['progress']['total_seconds']:.1f} segundos")
        print(f"  Batches: {summary['progress']['last_batch'] + 1}")
        
        print("\nPOSESIÓN GLOBAL:")
        for team_id in [0, 1]:
            pct = summary['possession']['percent_by_team'].get(team_id, 0)
            sec = summary['possession']['seconds_by_team'].get(team_id, 0)
            print(f"  Team {team_id}: {pct:.1f}% ({sec:.1f}s)")
        
        print("\nPASES:")
        for team_id in [0, 1]:
            passes = summary['passes']['by_team'].get(team_id, 0)
            print(f"  Team {team_id}: {passes} pases")
        
        # Imprimir estadísticas espaciales si están disponibles
        if not args.disable_spatial:
            # Necesitamos acceder al processor para obtener las estadísticas detalladas
            # Por ahora, crear uno temporal para demostración
            from modules.batch_processor import BatchProcessor
            
            processor = BatchProcessor(
                model_path=args.model,
                enable_spatial_tracking=True,
                zone_partition_type=args.zones,
                zone_nx=args.zone_nx,
                zone_ny=args.zone_ny,
                enable_heatmaps=not args.no_heatmaps,
                heatmap_resolution=heatmap_res
            )
            
            processor.initialize_modules(match_state)
            
            # Aquí normalmente cargaríamos el estado espacial guardado
            # Por ahora, solo mostramos el mensaje
            print("\n⚠ Nota: Para ver estadísticas espaciales detalladas,")
            print("  necesitamos cargar el estado del spatial_tracker guardado.")
            print("  Esto se implementará en la próxima iteración.")
            
            # Exportar heatmaps si están disponibles
            if not args.no_heatmaps:
                heatmap_path = Path(args.output_dir) / f"{match_id}_heatmaps.npz"
                
                # TODO: Cargar y exportar heatmaps desde el estado guardado
                print(f"\n✓ Heatmaps se guardarán en: {heatmap_path}")
                
                if args.visualize:
                    print("  (Visualización de heatmaps se implementará)")
        
        print("\n✓ Análisis completado exitosamente!")
        print(f"✓ Outputs guardados en: {args.output_dir}/{match_id}/")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Análisis interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
