"""
Generador de Heatmaps desde Análisis Existente
===============================================

Script para generar heatmaps de posición a partir de análisis de partidos
guardados en formato JSON/NPZ.

Uso:
    python generate_heatmaps_from_analysis.py outputs_streaming/SESSION_ID_trajectories.json

Autor: TacticEYE2
Fecha: 2026-01-29
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from modules.field_heatmap_system import (
    FIELD_POINTS,
    HeatmapAccumulator,
    process_sequence
)


def load_trajectories(json_path: str) -> Dict:
    """Carga el archivo JSON de trayectorias."""
    with open(json_path, 'r') as f:
        return json.load(f)


def convert_to_heatmap_format(trajectories: Dict) -> tuple:
    """
    Convierte el formato de trayectorias al formato esperado por el sistema de heatmaps.
    
    Args:
        trajectories: Diccionario con estructura:
            {
                "metadata": {...},
                "trajectories": {
                    "track_id": {
                        "team_id": int,
                        "frames": [frame_idx, ...],
                        "positions": [[x, y], ...],
                        ...
                    }
                }
            }
    
    Returns:
        (frames_keypoints, frames_players): Formato para process_sequence
    """
    metadata = trajectories.get('metadata', {})
    total_frames = metadata.get('total_frames', 0)
    tracks = trajectories.get('trajectories', {})
    
    # Inicializar listas por frame
    frames_keypoints = [[] for _ in range(total_frames)]
    frames_players = [[] for _ in range(total_frames)]
    
    # Llenar detecciones de jugadores por frame
    for track_id, track_data in tracks.items():
        team_id = track_data.get('team_id', -1)
        
        # Saltar árbitros y objetos sin equipo
        if team_id == -1:
            continue
        
        frames = track_data.get('frames', [])
        positions = track_data.get('positions', [])
        
        for frame_idx, (x, y) in zip(frames, positions):
            if frame_idx < total_frames:
                frames_players[frame_idx].append({
                    'team_id': team_id,
                    'xy': (x, y),
                    'conf': 0.9  # Asumimos alta confianza para tracks confirmados
                })
    
    print(f"✓ Convertido: {total_frames} frames, {len(tracks)} tracks")
    
    return frames_keypoints, frames_players


def generate_heatmaps_from_json(
    json_path: str,
    output_dir: str = None,
    grid_size: tuple = (105, 68)
):
    """
    Genera heatmaps a partir de un archivo JSON de trayectorias.
    
    Args:
        json_path: Ruta al archivo JSON de trayectorias
        output_dir: Directorio de salida (si None, usa el mismo directorio del JSON)
        grid_size: Tamaño de la cuadrícula (nx, ny)
    """
    print("=" * 70)
    print(f"GENERANDO HEATMAPS DESDE: {json_path}")
    print("=" * 70)
    
    # Cargar datos
    print("\n1. Cargando trayectorias...")
    trajectories = load_trajectories(json_path)
    
    # Convertir formato
    print("\n2. Convirtiendo formato...")
    frames_keypoints, frames_players = convert_to_heatmap_format(trajectories)
    
    # Crear acumulador
    print(f"\n3. Creando acumulador de heatmaps (grid: {grid_size})...")
    accumulator = HeatmapAccumulator(nx=grid_size[0], ny=grid_size[1])
    
    # NOTA: Este script asume que YA tienes homografías calculadas
    # Si no tienes keypoints detectados en cada frame, necesitarás:
    # 1. Ejecutar el análisis con detección de keypoints activada, O
    # 2. Usar una homografía fija/promedio, O
    # 3. Implementar interpolación de homografías
    
    print("\n⚠ ADVERTENCIA: Este script requiere keypoints detectados por frame")
    print("   Si tus trayectorias no incluyen keypoints, considera:")
    print("   - Reejecutar el análisis con --calibrate --show-keypoints")
    print("   - Usar el sistema de heatmaps directamente en batch_processor.py")
    
    # Determinar directorio de salida
    if output_dir is None:
        output_dir = Path(json_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Nombre base del archivo
    session_id = Path(json_path).stem.replace('_trajectories', '')
    
    print(f"\n✓ Configuración completada")
    print(f"  Directorio de salida: {output_dir}")
    print(f"  Session ID: {session_id}")
    
    # TODO: Aquí necesitarías las homografías por frame
    # Por ahora, el script es un template
    
    print("\n" + "=" * 70)
    print("NOTA: Para implementación completa, integrar con batch_processor.py")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Genera heatmaps de posición desde análisis existente"
    )
    parser.add_argument(
        'trajectories_json',
        help='Ruta al archivo JSON de trayectorias (_trajectories.json)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Directorio de salida (default: mismo que el JSON)',
        default=None
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        nargs=2,
        default=[105, 68],
        metavar=('NX', 'NY'),
        help='Tamaño de la cuadrícula (default: 105 68)'
    )
    
    args = parser.parse_args()
    
    # Validar archivo de entrada
    if not Path(args.trajectories_json).exists():
        print(f"❌ Error: Archivo no encontrado: {args.trajectories_json}")
        return 1
    
    # Generar heatmaps
    generate_heatmaps_from_json(
        args.trajectories_json,
        output_dir=args.output_dir,
        grid_size=tuple(args.grid_size)
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
