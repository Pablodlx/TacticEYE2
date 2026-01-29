#!/usr/bin/env python3
"""
Visualizador de Mapas de Calor basados en Keypoints
Procesa los datos JSON generados por pruebatrackequipo.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
import argparse
from pathlib import Path


def draw_football_field(ax, length=105.0, width=68.0):
    """Dibuja un campo de fútbol con dimensiones reales"""
    
    # Campo principal
    ax.add_patch(Rectangle((0, 0), length, width, fill=False, edgecolor='white', linewidth=2))
    
    # Línea media
    ax.plot([length/2, length/2], [0, width], 'white', linewidth=2)
    
    # Círculo central
    center_circle = Circle((length/2, width/2), 9.15, fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(center_circle)
    
    # Áreas grandes (penalty areas)
    # Izquierda
    ax.add_patch(Rectangle((0, width/2 - 20.16), 16.5, 40.32, fill=False, edgecolor='white', linewidth=2))
    # Derecha
    ax.add_patch(Rectangle((length - 16.5, width/2 - 20.16), 16.5, 40.32, fill=False, edgecolor='white', linewidth=2))
    
    # Áreas pequeñas (goal areas)
    # Izquierda
    ax.add_patch(Rectangle((0, width/2 - 9.16), 5.5, 18.32, fill=False, edgecolor='white', linewidth=2))
    # Derecha
    ax.add_patch(Rectangle((length - 5.5, width/2 - 9.16), 5.5, 18.32, fill=False, edgecolor='white', linewidth=2))
    
    # Arcos de penalty
    # Izquierda
    left_arc = Arc((11, width/2), 18.3, 18.3, angle=0, theta1=308, theta2=52, 
                   edgecolor='white', linewidth=2)
    ax.add_patch(left_arc)
    # Derecha
    right_arc = Arc((length - 11, width/2), 18.3, 18.3, angle=0, theta1=128, theta2=232,
                    edgecolor='white', linewidth=2)
    ax.add_patch(right_arc)
    
    # Puntos de penalty
    ax.plot(11, width/2, 'wo', markersize=4)
    ax.plot(length - 11, width/2, 'wo', markersize=4)
    
    # Centro del campo
    ax.plot(length/2, width/2, 'wo', markersize=4)
    
    ax.set_xlim(-5, length + 5)
    ax.set_ylim(-5, width + 5)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a5f1a')
    ax.axis('off')


def generate_heatmap(positions, field_length=105.0, field_width=68.0, bins=50):
    """Genera un heatmap 2D desde las posiciones"""
    
    if not positions:
        return None, None, None
    
    # Extraer coordenadas
    xs = [p['position'][0] for p in positions]
    ys = [p['position'][1] for p in positions]
    
    # Crear heatmap 2D
    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=bins, 
                                              range=[[0, field_length], [0, field_width]])
    
    # Suavizar con filtro gaussiano
    from scipy.ndimage import gaussian_filter
    heatmap_smooth = gaussian_filter(heatmap.T, sigma=2)
    
    return heatmap_smooth, xedges, yedges


def visualize_heatmap(data_file, output_file=None, show=True):
    """Visualiza los mapas de calor de ambos equipos"""
    
    # Cargar datos
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    team_0 = data['team_0']
    team_1 = data['team_1']
    
    print(f"Procesando: {data_file}")
    print(f"  Team 0: {summary['team_0_positions']} posiciones")
    print(f"  Team 1: {summary['team_1_positions']} posiciones")
    print(f"  Frames totales: {summary['total_frames']}")
    print(f"  Calibración: {'Exitosa' if summary['calibration_success'] else 'Pendiente'}")
    
    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor('#2d2d2d')
    
    # Team 0
    ax0 = axes[0]
    draw_football_field(ax0)
    
    if team_0:
        heatmap0, xedges0, yedges0 = generate_heatmap(team_0)
        if heatmap0 is not None:
            extent0 = [xedges0[0], xedges0[-1], yedges0[0], yedges0[-1]]
            im0 = ax0.imshow(heatmap0, extent=extent0, origin='lower', 
                            cmap='hot', alpha=0.6, interpolation='bilinear')
            plt.colorbar(im0, ax=ax0, label='Densidad', fraction=0.046)
    
    ax0.set_title(f'Team 0 - Heatmap ({len(team_0)} posiciones)', 
                  color='white', fontsize=16, fontweight='bold')
    
    # Team 1
    ax1 = axes[1]
    draw_football_field(ax1)
    
    if team_1:
        heatmap1, xedges1, yedges1 = generate_heatmap(team_1)
        if heatmap1 is not None:
            extent1 = [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]]
            im1 = ax1.imshow(heatmap1, extent=extent1, origin='lower',
                            cmap='hot', alpha=0.6, interpolation='bilinear')
            plt.colorbar(im1, ax=ax1, label='Densidad', fraction=0.046)
    
    ax1.set_title(f'Team 1 - Heatmap ({len(team_1)} posiciones)',
                  color='white', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar si se especifica
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#2d2d2d')
        print(f"\n✓ Heatmap guardado en: {output_file}")
    
    # Mostrar
    if show:
        plt.show()
    
    plt.close()


def analyze_keypoint_relations(data_file):
    """Analiza las relaciones con keypoints"""
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    team_0 = data['team_0']
    team_1 = data['team_1']
    
    print("\n" + "="*70)
    print("ANÁLISIS DE RELACIONES CON KEYPOINTS (CON JERARQUÍA)")
    print("="*70)
    
    for team_id, team_data in [(0, team_0), (1, team_1)]:
        if not team_data:
            continue
        
        print(f"\nTeam {team_id}:")
        
        # Contar keypoints más cercanos y promediar scores
        kp_counts = {}
        kp_scores = {}
        kp_priorities = {}
        distances = []
        
        for pos in team_data:
            kp = pos.get('nearest_kp')
            dist = pos.get('kp_distance')
            score = pos.get('kp_score', 0)
            priority = pos.get('kp_priority', 50)
            
            if kp:
                kp_counts[kp] = kp_counts.get(kp, 0) + 1
                if kp not in kp_scores:
                    kp_scores[kp] = []
                kp_scores[kp].append(score)
                kp_priorities[kp] = priority
            if dist is not None:
                distances.append(dist)
        
        # Top 5 keypoints más frecuentes con su prioridad
        print("  Keypoints más usados (con prioridad jerárquica):")
        sorted_kps = sorted(kp_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for kp, count in sorted_kps:
            pct = (count / len(team_data)) * 100
            avg_score = np.mean(kp_scores[kp]) if kp in kp_scores else 0
            priority = kp_priorities.get(kp, 50)
            
            # Clasificar prioridad
            if priority >= 90:
                priority_label = "ALTA"
            elif priority >= 70:
                priority_label = "Media-Alta"
            elif priority >= 50:
                priority_label = "Media"
            else:
                priority_label = "Baja"
            
            print(f"    - {kp:40s}: {count:4d} veces ({pct:5.1f}%) | Prioridad: {priority_label:12s} (score: {avg_score:.1f})")
        
        # Estadísticas de distancia
        if distances:
            print(f"\n  Distancias a keypoints:")
            print(f"    - Promedio: {np.mean(distances):.1f}px")
            print(f"    - Mediana: {np.median(distances):.1f}px")
            print(f"    - Min: {np.min(distances):.1f}px")
            print(f"    - Max: {np.max(distances):.1f}px")
        
        # Distribución por prioridad
        priority_dist = {'ALTA': 0, 'Media-Alta': 0, 'Media': 0, 'Baja': 0}
        for kp, count in kp_counts.items():
            priority = kp_priorities.get(kp, 50)
            if priority >= 90:
                priority_dist['ALTA'] += count
            elif priority >= 70:
                priority_dist['Media-Alta'] += count
            elif priority >= 50:
                priority_dist['Media'] += count
            else:
                priority_dist['Baja'] += count
        
        total = sum(priority_dist.values())
        if total > 0:
            print(f"\n  Distribución por nivel de prioridad:")
            for level, count in priority_dist.items():
                pct = (count / total) * 100
                print(f"    - {level:12s}: {count:4d} referencias ({pct:5.1f}%)")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualizar mapas de calor desde datos de keypoints')
    parser.add_argument('data_file', help='Archivo JSON con datos de heatmap')
    parser.add_argument('--output', '-o', help='Archivo de salida para imagen (PNG)')
    parser.add_argument('--no-show', action='store_true', help='No mostrar ventana interactiva')
    parser.add_argument('--analyze', '-a', action='store_true', help='Análisis detallado de keypoints')
    
    args = parser.parse_args()
    
    # Verificar que existe el archivo
    if not Path(args.data_file).exists():
        print(f"Error: No se encuentra {args.data_file}")
        return
    
    # Visualizar heatmap
    visualize_heatmap(args.data_file, args.output, not args.no_show)
    
    # Análisis detallado si se solicita
    if args.analyze:
        analyze_keypoint_relations(args.data_file)


if __name__ == '__main__':
    main()
