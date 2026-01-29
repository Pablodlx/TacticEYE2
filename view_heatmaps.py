#!/usr/bin/env python3
"""
Script para visualizar archivos .npz de heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def view_npz(filepath):
    """Muestra el contenido de un archivo .npz"""
    
    if not Path(filepath).exists():
        print(f"❌ Archivo no encontrado: {filepath}")
        return
    
    # Cargar archivo
    print(f"📂 Cargando: {filepath}")
    print()
    
    data = np.load(filepath, allow_pickle=True)
    
    # Mostrar todas las claves
    print("🔑 Claves disponibles:")
    for key in data.files:
        try:
            array = data[key]
            print(f"  - {key}: shape={array.shape}, dtype={array.dtype}")
            if array.size < 20:  # Si es pequeño, mostrar valores
                print(f"    valores: {array}")
            else:  # Si es grande, mostrar estadísticas
                print(f"    min={array.min():.2f}, max={array.max():.2f}, mean={array.mean():.2f}, sum={array.sum():.0f}")
        except Exception as e:
            print(f"    (metadata: {array})")
    print()
    
    # Visualizar heatmaps si existen
    heatmap_keys = [k for k in data.files if 'heatmap' in k.lower()]
    
    if heatmap_keys:
        print("🎨 Generando visualización de heatmaps...")
        
        num_heatmaps = len(heatmap_keys)
        fig, axes = plt.subplots(1, num_heatmaps, figsize=(6*num_heatmaps, 5))
        
        if num_heatmaps == 1:
            axes = [axes]
        
        for idx, key in enumerate(heatmap_keys):
            heatmap = data[key]
            
            # Determinar colormap
            if 'team_0' in key or '0' in key:
                cmap = 'Greens'
                title = 'Team 0 Heatmap'
            elif 'team_1' in key or '1' in key:
                cmap = 'Reds'
                title = 'Team 1 Heatmap'
            else:
                cmap = 'viridis'
                title = key
            
            # Plotear
            im = axes[idx].imshow(heatmap, cmap=cmap, aspect='auto')
            axes[idx].set_title(title)
            axes[idx].set_xlabel('X (campo)')
            axes[idx].set_ylabel('Y (campo)')
            plt.colorbar(im, ax=axes[idx], label='Intensidad')
        
        plt.tight_layout()
        
        # Guardar imagen
        output_path = Path(filepath).with_suffix('.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualización guardada en: {output_path}")
        
        # Mostrar
        plt.show()
    else:
        print("ℹ️  No se encontraron heatmaps en el archivo")
    
    data.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Buscar el más reciente
        import glob
        files = glob.glob('outputs_streaming/*_heatmaps.npz')
        if files:
            latest = max(files, key=lambda x: Path(x).stat().st_mtime)
            print(f"📌 Usando el archivo más reciente: {latest}\n")
            view_npz(latest)
        else:
            print("Uso: python view_heatmaps.py <archivo.npz>")
            print("\nO simplemente: python view_heatmaps.py")
            print("  (usará el archivo más reciente en outputs_streaming/)")
    else:
        view_npz(sys.argv[1])
