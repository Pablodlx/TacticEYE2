#!/usr/bin/env python3
"""
Script de diagnóstico para heatmaps
Verifica el contenido de archivos .npz y muestra estadísticas
"""

import numpy as np
import sys
from pathlib import Path

def diagnose_heatmap(file_path):
    """Diagnostica un archivo de heatmap .npz"""
    
    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO DE HEATMAP: {file_path}")
    print('='*70)
    
    if not Path(file_path).exists():
        print(f"❌ Archivo no encontrado: {file_path}")
        return
    
    # Cargar archivo
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ Error cargando archivo: {e}")
        return
    
    # Listar claves
    print(f"\n📋 Claves disponibles ({len(data.keys())}):")
    for key in sorted(data.keys()):
        print(f"  - {key}")
    
    # Analizar heatmaps de equipos
    print(f"\n🔥 HEATMAPS DE EQUIPOS:")
    print("-" * 70)
    
    for team_id in [0, 1]:
        key = f'team_{team_id}_heatmap'
        
        if key in data:
            heatmap = data[key]
            
            print(f"\nTeam {team_id}:")
            print(f"  ✓ Clave encontrada: {key}")
            print(f"  - Shape: {heatmap.shape}")
            print(f"  - Dtype: {heatmap.dtype}")
            print(f"  - Min: {heatmap.min():.4f}")
            print(f"  - Max: {heatmap.max():.4f}")
            print(f"  - Mean: {heatmap.mean():.4f}")
            print(f"  - Sum: {heatmap.sum():.2f}")
            print(f"  - Non-zero: {np.count_nonzero(heatmap)} / {heatmap.size} píxeles")
            
            if heatmap.sum() == 0:
                print(f"  ⚠️  HEATMAP VACÍO (sum=0)")
                print(f"       Posible causa: Team {team_id} no tiene jugadores clasificados")
            elif heatmap.sum() < 10:
                print(f"  ⚠️  HEATMAP MUY DÉBIL (sum<10)")
                print(f"       Posible causa: Pocos frames con jugadores del Team {team_id}")
            else:
                print(f"  ✓ Heatmap con datos válidos")
        else:
            print(f"\nTeam {team_id}:")
            print(f"  ❌ Clave NO encontrada: {key}")
    
    # Analizar estadísticas de zonas si existen
    if 'possession_by_zone_team_0' in data or 'possession_by_zone_team_1' in data:
        print(f"\n📊 ESTADÍSTICAS DE ZONAS:")
        print("-" * 70)
        
        for team_id in [0, 1]:
            poss_key = f'possession_by_zone_team_{team_id}'
            zone_key = f'zone_percentages_team_{team_id}'
            
            if poss_key in data:
                poss = data[poss_key]
                print(f"\nTeam {team_id} - Posesión por zona:")
                print(f"  Shape: {poss.shape}")
                print(f"  Total frames: {poss.sum():.0f}")
                
                if zone_key in data:
                    percentages = data[zone_key]
                    print(f"  Porcentajes: {percentages}")
    
    # Metadata
    if 'metadata' in data:
        print(f"\n⚙️  METADATA:")
        print("-" * 70)
        metadata = data['metadata'].item()
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python diagnose_heatmap.py <archivo.npz>")
        print("\nEjemplo:")
        print("  python diagnose_heatmap.py outputs_streaming/session_heatmaps.npz")
        sys.exit(1)
    
    file_path = sys.argv[1]
    diagnose_heatmap(file_path)
