"""
Test Interactivo del Sistema de Heatmaps
=========================================

Script de prueba para el sistema de heatmaps con visualización interactiva.
"""

import numpy as np
import matplotlib.pyplot as plt
from modules.field_heatmap_system import (
    FIELD_POINTS,
    HeatmapAccumulator,
    estimate_homography_with_flip_resolution,
    project_points
)


def generate_realistic_test_data(num_frames=30):
    """
    Genera datos de prueba más realistas simulando un partido.
    """
    np.random.seed(42)
    
    frames_keypoints = []
    frames_players = []
    
    # Dimensiones de imagen simulada
    IMG_W, IMG_H = 1920, 1080
    
    for frame_idx in range(num_frames):
        # Simular movimiento de cámara (pan/zoom)
        pan_offset = 200 * np.sin(frame_idx / 10)
        zoom_factor = 1.0 + 0.2 * np.sin(frame_idx / 15)
        
        # Simular flip aleatorio cada 10 frames
        is_flipped = (frame_idx // 10) % 2 == 1
        
        # Generar keypoints con ruido
        kps = []
        
        # Línea central siempre visible
        mid_x = IMG_W / 2 + pan_offset
        kps.append({
            "cls_name": "midline_top_intersection",
            "xy": (mid_x + np.random.randn() * 5, 100 + np.random.randn() * 10),
            "conf": 0.90 + np.random.rand() * 0.1
        })
        kps.append({
            "cls_name": "midline_bottom_intersection",
            "xy": (mid_x + np.random.randn() * 5, IMG_H - 100 + np.random.randn() * 10),
            "conf": 0.88 + np.random.rand() * 0.1
        })
        
        # Círculo central
        if np.random.rand() > 0.3:
            kps.append({
                "cls_name": "halfcircle_top",
                "xy": (mid_x + np.random.randn() * 5, IMG_H/2 - 150 + np.random.randn() * 10),
                "conf": 0.85 + np.random.rand() * 0.1
            })
        if np.random.rand() > 0.3:
            kps.append({
                "cls_name": "halfcircle_bottom",
                "xy": (mid_x + np.random.randn() * 5, IMG_H/2 + 150 + np.random.randn() * 10),
                "conf": 0.83 + np.random.rand() * 0.1
            })
        
        # Área grande (visible según zoom)
        if zoom_factor < 1.15:
            box_x = 300 if not is_flipped else IMG_W - 300
            box_x += pan_offset
            kps.append({
                "cls_name": "bigarea_top_inner",
                "xy": (box_x + np.random.randn() * 5, IMG_H/2 - 200 + np.random.randn() * 10),
                "conf": 0.82 + np.random.rand() * 0.1
            })
            kps.append({
                "cls_name": "bigarea_bottom_inner",
                "xy": (box_x + np.random.randn() * 5, IMG_H/2 + 200 + np.random.randn() * 10),
                "conf": 0.80 + np.random.rand() * 0.1
            })
        
        frames_keypoints.append(kps)
        
        # Generar jugadores (22 jugadores: 11 por equipo)
        players = []
        
        for team_id in [0, 1]:
            num_visible = np.random.randint(4, 7)  # 4-6 jugadores visibles por equipo
            
            for _ in range(num_visible):
                # Distribución espacial por equipo
                if team_id == 0:
                    # Equipo local: más en la mitad izquierda
                    x = np.random.rand() * IMG_W * 0.6 + pan_offset
                else:
                    # Equipo visitante: más en la mitad derecha
                    x = np.random.rand() * IMG_W * 0.6 + IMG_W * 0.4 + pan_offset
                
                y = np.random.rand() * (IMG_H - 200) + 100
                
                players.append({
                    "team_id": team_id,
                    "xy": (x, y),
                    "conf": 0.85 + np.random.rand() * 0.15
                })
        
        frames_players.append(players)
    
    return frames_keypoints, frames_players


def visualize_field_with_heatmaps(heatmap_team0, heatmap_team1, save_path='test_heatmaps.png'):
    """
    Visualiza los heatmaps superpuestos sobre un campo de fútbol.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Campo simplificado
    field_color = '#2d5f2e'
    line_color = 'white'
    
    # Heatmap Team 0
    ax = axes[0, 0]
    ax.set_facecolor(field_color)
    im0 = ax.imshow(heatmap_team0, cmap='Reds', origin='lower', aspect='auto', alpha=0.7, extent=[0, 105, 0, 68])
    ax.set_title('Heatmap Team 0 (Local)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitud (m)')
    ax.set_ylabel('Ancho (m)')
    
    # Dibujar líneas de campo
    ax.axvline(52.5, color=line_color, linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(34, color=line_color, linewidth=1, linestyle='--', alpha=0.5)
    plt.colorbar(im0, ax=ax, label='Densidad')
    
    # Heatmap Team 1
    ax = axes[0, 1]
    ax.set_facecolor(field_color)
    im1 = ax.imshow(heatmap_team1, cmap='Blues', origin='lower', aspect='auto', alpha=0.7, extent=[0, 105, 0, 68])
    ax.set_title('Heatmap Team 1 (Visitante)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitud (m)')
    ax.set_ylabel('Ancho (m)')
    
    # Dibujar líneas de campo
    ax.axvline(52.5, color=line_color, linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(34, color=line_color, linewidth=1, linestyle='--', alpha=0.5)
    plt.colorbar(im1, ax=ax, label='Densidad')
    
    # Heatmap combinado
    ax = axes[1, 0]
    ax.set_facecolor(field_color)
    # Combinar ambos heatmaps con colores diferentes
    combined = np.zeros((*heatmap_team0.shape, 3))
    combined[:, :, 0] = heatmap_team0  # Rojo para team 0
    combined[:, :, 2] = heatmap_team1  # Azul para team 1
    ax.imshow(combined, origin='lower', aspect='auto', alpha=0.7, extent=[0, 105, 0, 68])
    ax.set_title('Heatmap Combinado', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitud (m)')
    ax.set_ylabel('Ancho (m)')
    
    # Dibujar líneas de campo
    ax.axvline(52.5, color=line_color, linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(34, color=line_color, linewidth=1, linestyle='--', alpha=0.5)
    
    # Diferencia de presencia (Team 0 - Team 1)
    ax = axes[1, 1]
    ax.set_facecolor(field_color)
    diff = heatmap_team0 - heatmap_team1
    im_diff = ax.imshow(diff, cmap='RdBu_r', origin='lower', aspect='auto', alpha=0.7, extent=[0, 105, 0, 68], vmin=-1, vmax=1)
    ax.set_title('Diferencia de Presencia (Rojo=Team 0, Azul=Team 1)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitud (m)')
    ax.set_ylabel('Ancho (m)')
    
    # Dibujar líneas de campo
    ax.axvline(52.5, color='yellow', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(34, color='yellow', linewidth=1, linestyle='--', alpha=0.5)
    plt.colorbar(im_diff, ax=ax, label='Diferencia')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualización guardada en: {save_path}")
    
    return fig


def main():
    print("=" * 70)
    print("TEST INTERACTIVO: Sistema de Heatmaps")
    print("=" * 70)
    
    # Generar datos de prueba
    print("\n1. Generando datos de prueba (30 frames)...")
    frames_keypoints, frames_players = generate_realistic_test_data(num_frames=30)
    
    print(f"   ✓ {len(frames_keypoints)} frames generados")
    print(f"   ✓ Promedio de keypoints por frame: {np.mean([len(kps) for kps in frames_keypoints]):.1f}")
    print(f"   ✓ Promedio de jugadores por frame: {np.mean([len(pls) for pls in frames_players]):.1f}")
    
    # Crear acumulador
    print("\n2. Creando acumulador (grid: 42x28 ~= 2.5m por celda)...")
    accumulator = HeatmapAccumulator(nx=42, ny=28)
    
    # Procesar frame por frame con detalles
    print("\n3. Procesando frames...")
    num_successful = 0
    num_flipped = 0
    
    for frame_idx, (kps, players) in enumerate(zip(frames_keypoints, frames_players)):
        # Estimar homografía con resolución de flip
        H, is_flipped = estimate_homography_with_flip_resolution(kps, FIELD_POINTS, min_points=3)
        
        if H is not None:
            accumulator.add_frame(H, players)
            num_successful += 1
            if is_flipped:
                num_flipped += 1
            
            if frame_idx % 10 == 0:
                print(f"   Frame {frame_idx:2d}: ✓ H estimada (flip={is_flipped}), {len(players)} jugadores")
        else:
            if frame_idx % 10 == 0:
                print(f"   Frame {frame_idx:2d}: ✗ Sin homografía ({len(kps)} keypoints)")
    
    print(f"\n   ✓ Frames exitosos: {num_successful}/{len(frames_keypoints)} ({100*num_successful/len(frames_keypoints):.1f}%)")
    print(f"   ✓ Frames con flip: {num_flipped}/{num_successful}")
    
    # Obtener heatmaps
    print("\n4. Generando heatmaps...")
    heatmap_team0 = accumulator.get_heatmap(0, normalize='max')
    heatmap_team1 = accumulator.get_heatmap(1, normalize='max')
    
    print(f"   ✓ Team 0: max={heatmap_team0.max():.2f}, sum={heatmap_team0.sum():.1f}")
    print(f"   ✓ Team 1: max={heatmap_team1.max():.2f}, sum={heatmap_team1.sum():.1f}")
    
    # Visualizar
    print("\n5. Generando visualización...")
    visualize_field_with_heatmaps(heatmap_team0, heatmap_team1)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETADO EXITOSAMENTE")
    print("=" * 70)


if __name__ == '__main__':
    main()
