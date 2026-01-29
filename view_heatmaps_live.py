#!/usr/bin/env python3
"""
Visualizador de heatmaps - Muestra los heatmaps generados por el análisis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def visualize_heatmaps(npz_path: str):
    """
    Visualiza los heatmaps de un archivo .npz
    
    Args:
        npz_path: Ruta al archivo .npz con los heatmaps
    """
    # Cargar datos
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"\n{'='*70}")
    print(f"VISUALIZANDO HEATMAPS: {Path(npz_path).name}")
    print(f"{'='*70}\n")
    
    # Mostrar keys disponibles
    print("📊 Datos disponibles:")
    for key in data.keys():
        arr = data[key]
        if isinstance(arr, np.ndarray):
            arr_sum = float(arr.sum()) if arr.dtype != object else 'N/A'
            print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}, sum={arr_sum}")
        else:
            print(f"  - {key}: {type(arr).__name__}")
    
    # Extraer heatmaps
    if 'team_0_heatmap' in data and 'team_1_heatmap' in data:
        heatmap_0 = data['team_0_heatmap']
        heatmap_1 = data['team_1_heatmap']
        
        # Crear visualización
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Team 0
        im0 = axes[0].imshow(heatmap_0, cmap='hot', interpolation='bilinear', origin='lower')
        axes[0].set_title(f'Team 0 Heatmap (sum={heatmap_0.sum():.1f})', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X (campo)')
        axes[0].set_ylabel('Y (campo)')
        plt.colorbar(im0, ax=axes[0], label='Intensidad')
        
        # Team 1
        im1 = axes[1].imshow(heatmap_1, cmap='hot', interpolation='bilinear', origin='lower')
        axes[1].set_title(f'Team 1 Heatmap (sum={heatmap_1.sum():.1f})', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X (campo)')
        axes[1].set_ylabel('Y (campo)')
        plt.colorbar(im1, ax=axes[1], label='Intensidad')
        
        plt.tight_layout()
        
        # Guardar imagen
        output_img = Path(npz_path).with_suffix('.png')
        plt.savefig(output_img, dpi=150, bbox_inches='tight')
        print(f"\n✓ Imagen guardada: {output_img}")
        
        # Mostrar
        plt.show()
        
        # Estadísticas
        print(f"\n📈 ESTADÍSTICAS:")
        print(f"  Team 0: min={heatmap_0.min():.2f}, max={heatmap_0.max():.2f}, mean={heatmap_0.mean():.2f}")
        print(f"  Team 1: min={heatmap_1.min():.2f}, max={heatmap_1.max():.2f}, mean={heatmap_1.mean():.2f}")
        
        # Verificar si hay datos
        if heatmap_0.sum() == 0 and heatmap_1.sum() == 0:
            print(f"\n⚠️  ADVERTENCIA: Ambos heatmaps están vacíos (sum=0)")
            print(f"   Esto puede indicar que la calibración no está funcionando correctamente.")
        elif heatmap_0.sum() > 0 or heatmap_1.sum() > 0:
            print(f"\n✓ Heatmaps contienen datos!")
    else:
        print("⚠️  No se encontraron heatmaps de equipos en el archivo")
        print(f"   Keys disponibles: {list(data.keys())}")


if __name__ == "__main__":
    # Buscar el archivo más reciente
    outputs_dir = Path("outputs_streaming")
    
    if len(sys.argv) > 1:
        # Usar archivo especificado
        npz_file = sys.argv[1]
    else:
        # Buscar el más reciente
        npz_files = sorted(outputs_dir.glob("*_heatmaps.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not npz_files:
            print("❌ No se encontraron archivos de heatmaps en outputs_streaming/")
            sys.exit(1)
        
        npz_file = npz_files[0]
        print(f"📁 Usando archivo más reciente: {npz_file.name}")
    
    visualize_heatmaps(str(npz_file))
