"""
Heatmap Visualizer - Visualizador de mapas de calor
===================================================
Script para visualizar heatmaps exportados
"""

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import subprocess


def load_heatmap(npz_path: str):
    """Carga heatmaps desde archivo .npz"""
    data = np.load(npz_path)
    
    heatmaps = {}
    field_size = tuple(data['field_size'])
    
    for key in data.keys():
        if key != 'field_size':
            heatmaps[key] = data[key]
    
    return heatmaps, field_size


def visualize_heatmap(heatmap_grid: np.ndarray,
                     field_size: tuple,
                     title: str = "Heatmap",
                     colormap: str = 'hot'):
    """Visualiza un heatmap individual"""
    # Resize a tamaño de campo
    heatmap = cv2.resize(heatmap_grid, field_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalizar
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Dibujar campo base
    ax.set_xlim(0, field_size[0])
    ax.set_ylim(0, field_size[1])
    ax.set_aspect('equal')
    ax.set_facecolor('#2E7D32')  # Verde césped
    
    # Superponer heatmap
    im = ax.imshow(
        heatmap,
        cmap=colormap,
        alpha=0.7,
        extent=[0, field_size[0], 0, field_size[1]],
        origin='upper',
        interpolation='bilinear'
    )
    
    # Dibujar líneas del campo (simplificado)
    line_color = 'white'
    line_width = 2
    
    # Perímetro
    ax.plot([0, field_size[0], field_size[0], 0, 0],
           [0, 0, field_size[1], field_size[1], 0],
           color=line_color, linewidth=line_width)
    
    # Línea central
    mid_x = field_size[0] / 2
    ax.plot([mid_x, mid_x], [0, field_size[1]],
           color=line_color, linewidth=line_width)
    
    # Círculo central
    circle = plt.Circle((mid_x, field_size[1]/2), 
                       91.5,  # 9.15m * 10px/m
                       fill=False, 
                       color=line_color, 
                       linewidth=line_width)
    ax.add_patch(circle)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Densidad de Presencia', rotation=270, labelpad=20)
    
    # Título
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Labels
    ax.set_xlabel('Longitud del campo (m)', fontsize=12)
    ax.set_ylabel('Ancho del campo (m)', fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return fig


def visualize_all_heatmaps(npz_path: str, output_dir: str = None, show_display: bool = True):
    """Visualiza todos los heatmaps de un archivo"""
    print(f"📊 Cargando heatmaps desde: {npz_path}")
    
    heatmaps, field_size = load_heatmap(npz_path)
    
    print(f"✅ Cargados {len(heatmaps)} heatmaps")
    print(f"   Tamaño del campo: {field_size}\n")
    
    # Mapeo de nombres y colores
    heatmap_config = {
        'team_0': ('Equipo Local', 'YlOrRd'),
        'team_1': ('Equipo Visitante', 'Blues'),
        'team_2': ('Árbitros', 'Greens'),
        'ball': ('Balón', 'hot')
    }
    
    # Crear directorio de salida
    output_path = Path(output_dir) if output_dir else Path('heatmap_images')
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Visualizar cada heatmap
    for key, grid in heatmaps.items():
        if key in heatmap_config:
            title, colormap = heatmap_config[key]
            
            print(f"🔥 Generando heatmap: {title}")
            
            fig = visualize_heatmap(
                grid,
                field_size,
                title=f"Mapa de Calor - {title}",
                colormap=colormap
            )
            
            # Guardar imagen
            output_file = output_path / f"heatmap_{key}.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   ✓ Guardado: {output_file}")
            saved_files.append(output_file)
            
            plt.close(fig)
    
    # Crear heatmap combinado
    print("\n🎨 Generando heatmap combinado...")
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Dibujar campo
    ax.set_xlim(0, field_size[0])
    ax.set_ylim(0, field_size[1])
    ax.set_aspect('equal')
    ax.set_facecolor('#2E7D32')
    
    # Combinar team_0 y team_1
    if 'team_0' in heatmaps and 'team_1' in heatmaps:
        hm0 = cv2.resize(heatmaps['team_0'], field_size, interpolation=cv2.INTER_LINEAR)
        hm1 = cv2.resize(heatmaps['team_1'], field_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalizar
        if hm0.max() > 0:
            hm0 = hm0 / hm0.max()
        if hm1.max() > 0:
            hm1 = hm1 / hm1.max()
        
        # Crear imagen RGB
        combined = np.zeros((field_size[1], field_size[0], 3))
        combined[:, :, 0] = hm1  # Azul para team_1
        combined[:, :, 2] = hm0  # Rojo para team_0
        
        ax.imshow(
            combined,
            alpha=0.6,
            extent=[0, field_size[0], 0, field_size[1]],
            origin='upper'
        )
    
    # Líneas del campo
    line_color = 'white'
    line_width = 2
    
    ax.plot([0, field_size[0], field_size[0], 0, 0],
           [0, 0, field_size[1], field_size[1], 0],
           color=line_color, linewidth=line_width)
    
    mid_x = field_size[0] / 2
    ax.plot([mid_x, mid_x], [0, field_size[1]],
           color=line_color, linewidth=line_width)
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Equipo Local'),
        Patch(facecolor='blue', alpha=0.6, label='Equipo Visitante')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    ax.set_title('Mapa de Calor Combinado - Ambos Equipos', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitud del campo (m)', fontsize=12)
    ax.set_ylabel('Ancho del campo (m)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_file = output_path / "heatmap_combined.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_file}")
    saved_files.append(output_file)
    
    plt.close(fig)
    
    print("\n✅ Visualización completada")
    print(f"📁 Imágenes guardadas en: {output_path}")
    print(f"   Total archivos: {len(saved_files)}\n")
    
    # Intentar abrir las imágenes con visor del sistema
    try:
        print("🖼️  Abriendo imágenes...")
        for img_file in saved_files:
            if os.name == 'posix':  # Linux/Mac
                subprocess.Popen(['xdg-open', str(img_file)], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            elif os.name == 'nt':  # Windows
                os.startfile(str(img_file))
        print("✓ Imágenes abiertas con visor del sistema")
    except Exception as e:
        print(f"ℹ️  No se pudieron abrir automáticamente. Usa:")
        print(f"   xdg-open {output_path}")
        print(f"   o abre manualmente: {output_path}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Visualizador de Heatmaps de TacticEYE2'
    )
    parser.add_argument(
        'heatmap_file',
        type=str,
        help='Ruta al archivo .npz de heatmaps'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Directorio para guardar imágenes (opcional)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Solo guardar imágenes, no mostrar ventanas'
    )
    
    args = parser.parse_args()
    
    # Verificar archivo
    if not Path(args.heatmap_file).exists():
        print(f"❌ Error: Archivo no encontrado: {args.heatmap_file}")
        return
    
    # Si no se especifica output, crear carpeta automática
    if args.output is None:
        heatmap_path = Path(args.heatmap_file)
        args.output = heatmap_path.parent / "heatmap_images"
        print(f"📁 Guardando imágenes en: {args.output}")
    
    # Configurar backend matplotlib
    matplotlib.use('Agg')  # Backend sin display para evitar errores de X
    
    # Visualizar
    visualize_all_heatmaps(args.heatmap_file, args.output, show_display=not args.no_display)


if __name__ == '__main__':
    main()
