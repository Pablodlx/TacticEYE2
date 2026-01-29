"""
Sistema de Mapas de Calor para Análisis de Fútbol
==================================================

Sistema completo para generar heatmaps de posición de jugadores proyectados
a coordenadas de campo, resolviendo la ambigüedad de flip horizontal cuando
los keypoints detectados no distinguen izquierda/derecha.

Autor: TacticEYE2
Fecha: 2026-01-29
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# 1. DEFINICIÓN DEL CAMPO TEÓRICO
# ============================================================================

# Dimensiones estándar FIFA (metros)
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

# Dimensiones reglamentarias
BIG_BOX_LENGTH = 16.5  # Distancia del área grande desde línea de gol
BIG_BOX_WIDTH = 40.3   # Ancho del área grande
SMALL_BOX_LENGTH = 5.5
SMALL_BOX_WIDTH = 18.3
PENALTY_SPOT_DIST = 11.0
CENTER_CIRCLE_RADIUS = 9.15

# Coordenadas teóricas de keypoints "neutros" (sin distinguir izq/der)
# IMPORTANTE: Solo usar keypoints INEQUÍVOCOS para estimación inicial
FIELD_POINTS = {
    # ========================================================================
    # KEYPOINTS INEQUÍVOCOS (no necesitan flip detection)
    # ========================================================================
    # Intersecciones de línea central con bordes (siempre en X=52.5m)
    'midline_top_intersection': (FIELD_LENGTH / 2, FIELD_WIDTH),
    'midline_bottom_intersection': (FIELD_LENGTH / 2, 0),
    
    # Círculo central (puntos cardinales, siempre en X=52.5m)
    'halfcircle_top': (FIELD_LENGTH / 2, FIELD_WIDTH / 2 + CENTER_CIRCLE_RADIUS),
    'halfcircle_bottom': (FIELD_LENGTH / 2, FIELD_WIDTH / 2 - CENTER_CIRCLE_RADIUS),
    'center': (FIELD_LENGTH / 2, FIELD_WIDTH / 2),
    
    # ========================================================================
    # KEYPOINTS AMBIGUOS (requieren flip detection para saber si izq/der)
    # Se definen para UNA portería (la que se asume según orientación)
    # ========================================================================
    # Área grande - ASUMIENDO portería en X cercano a 0
    'bigarea_top_inner': (BIG_BOX_LENGTH, (FIELD_WIDTH + BIG_BOX_WIDTH) / 2),
    'bigarea_bottom_inner': (BIG_BOX_LENGTH, (FIELD_WIDTH - BIG_BOX_WIDTH) / 2),
    'bigarea_top_outter': (0, (FIELD_WIDTH + BIG_BOX_WIDTH) / 2),
    'bigarea_bottom_outter': (0, (FIELD_WIDTH - BIG_BOX_WIDTH) / 2),
    
    # Área pequeña - ASUMIENDO portería en X cercano a 0
    'smallarea_top_inner': (SMALL_BOX_LENGTH, (FIELD_WIDTH + SMALL_BOX_WIDTH) / 2),
    'smallarea_bottom_inner': (SMALL_BOX_LENGTH, (FIELD_WIDTH - SMALL_BOX_WIDTH) / 2),
    'smallarea_top_outter': (0, (FIELD_WIDTH + SMALL_BOX_WIDTH) / 2),
    'smallarea_bottom_outter': (0, (FIELD_WIDTH - SMALL_BOX_WIDTH) / 2),
    
    # Intersecciones arco-área (arcos de penalti) - ASUMIENDO portería en X cercano a 0
    'top_arc_area_intersection': (PENALTY_SPOT_DIST, FIELD_WIDTH / 2 + CENTER_CIRCLE_RADIUS),
    'bottom_arc_area_intersection': (PENALTY_SPOT_DIST, FIELD_WIDTH / 2 - CENTER_CIRCLE_RADIUS),
    
    # Esquinas - ambiguas, se pueden mapear a cualquiera
    'corner': (0, 0),
}


# Keypoints que NO necesitan flip (siempre en X=52.5m, línea central)
UNAMBIGUOUS_KEYPOINTS = {
    'midline_top_intersection',
    'midline_bottom_intersection', 
    'halfcircle_top',
    'halfcircle_bottom',
    'center'
}


# ============================================================================
# 2. ESTIMACIÓN DE HOMOGRAFÍA
# ============================================================================

def estimate_homography(
    frame_keypoints: List[Dict],
    field_points: Dict[str, Tuple[float, float]],
    min_points: int = 4,
    conf_threshold: float = 0.4
) -> Optional[np.ndarray]:
    """
    Estima la homografía imagen -> campo para un frame.
    
    Args:
        frame_keypoints: Lista de detecciones de keypoints:
            [{"cls_name": str, "xy": (x, y), "conf": float}, ...]
        field_points: Diccionario {cls_name: (X, Y)} en coordenadas de campo
        min_points: Número mínimo de puntos para estimar homografía
        conf_threshold: Umbral de confianza para filtrar keypoints
    
    Returns:
        H: Matriz 3x3 de homografía (imagen -> campo) o None si no hay puntos suficientes
    """
    # Filtrar y emparejar puntos
    img_pts = []
    field_pts = []
    
    for kp in frame_keypoints:
        cls_name = kp['cls_name']
        conf = kp['conf']
        xy = kp['xy']
        
        # Validar confianza y que exista en el modelo de campo
        if conf >= conf_threshold and cls_name in field_points:
            img_pts.append(xy)
            field_pts.append(field_points[cls_name])
    
    # Verificar puntos suficientes
    if len(img_pts) < min_points:
        return None
    
    # Convertir a arrays numpy
    img_pts = np.array(img_pts, dtype=np.float32)
    field_pts = np.array(field_pts, dtype=np.float32)
    
    # Verificar que tenemos exactamente 4 o más puntos para RANSAC
    if len(img_pts) < 4:
        return None
    
    # Estimar homografía con RANSAC
    H, mask = cv2.findHomography(
        img_pts,
        field_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        confidence=0.99
    )
    
    if H is None:
        return None
    
    # Verificar calidad: al menos 50% de inliers
    num_inliers = np.sum(mask) if mask is not None else 0
    if num_inliers < min_points or num_inliers / len(img_pts) < 0.5:
        return None
    
    return H


# ============================================================================
# 3. RESOLUCIÓN DE FLIP HORIZONTAL
# ============================================================================

def flip_field_points(field_points: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """
    Aplica transformación de flip horizontal a los puntos del campo.
    
    Transformación: (X, Y) -> (L - X, Y)
    
    Args:
        field_points: Diccionario original {cls_name: (X, Y)}
    
    Returns:
        Diccionario con puntos flipped
    """
    flipped = {}
    for cls_name, (X, Y) in field_points.items():
        flipped[cls_name] = (FIELD_LENGTH - X, Y)
    return flipped


def homography_geom_error(
    H: np.ndarray,
    frame_keypoints: List[Dict],
    field_points: Dict[str, Tuple[float, float]],
    conf_threshold: float = 0.4
) -> float:
    """
    Calcula el error geométrico de una homografía.
    
    Proyecta keypoints de imagen al campo y compara distancias relativas
    con las distancias teóricas en el modelo de campo.
    
    Args:
        H: Homografía imagen -> campo
        frame_keypoints: Keypoints detectados en imagen
        field_points: Modelo teórico de campo
        conf_threshold: Umbral de confianza
    
    Returns:
        Error escalar (menor es mejor). Inf si no hay puntos válidos.
    """
    # Filtrar keypoints válidos
    valid_kps = []
    for kp in frame_keypoints:
        if kp['conf'] >= conf_threshold and kp['cls_name'] in field_points:
            valid_kps.append(kp)
    
    if len(valid_kps) < 3:
        return float('inf')
    
    # Proyectar puntos de imagen al campo con H
    img_pts = np.array([kp['xy'] for kp in valid_kps], dtype=np.float32)
    projected = project_points(H, img_pts)
    
    # Comparar distancias relativas
    errors = []
    for i in range(len(valid_kps)):
        for j in range(i + 1, len(valid_kps)):
            # Distancia proyectada
            dist_proj = np.linalg.norm(projected[i] - projected[j])
            
            # Distancia teórica
            cls_i = valid_kps[i]['cls_name']
            cls_j = valid_kps[j]['cls_name']
            pt_i = np.array(field_points[cls_i])
            pt_j = np.array(field_points[cls_j])
            dist_theory = np.linalg.norm(pt_i - pt_j)
            
            # Error relativo
            if dist_theory > 0:
                rel_error = abs(dist_proj - dist_theory) / dist_theory
                errors.append(rel_error)
    
    return np.mean(errors) if errors else float('inf')


def estimate_homography_with_flip_resolution(
    frame_keypoints: List[Dict],
    field_points: Dict[str, Tuple[float, float]],
    min_points: int = 4,
    conf_threshold: float = 0.4
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Estima homografía resolviendo la ambigüedad de flip horizontal.
    
    Estrategia MEJORADA (2 pasos):
    
    PASO 1: Determinar orientación usando SOLO keypoints en píxeles (sin homografía)
       - Compara posiciones X en píxeles de bigarea vs smallarea
       - bigarea_inner MÁS a la derecha que smallarea_inner → NO flip
       - smallarea_inner MÁS a la derecha que bigarea_inner → SÍ flip
    
    PASO 2: Estimar homografía con orientación correcta
       - Si flip: usar flip_field_points()
       - Si no flip: usar field_points original
       - PRIORIZAR keypoints inequívocos (línea central) si hay pocos puntos
    
    Args:
        frame_keypoints: Keypoints detectados en imagen
        field_points: Modelo teórico de campo
        min_points: Puntos mínimos para estimar H
        conf_threshold: Umbral de confianza
    
    Returns:
        (H_best, is_flipped):
            - H_best: Mejor homografía
            - is_flipped: True si campo está horizontalmente invertido
    """
    # Convertir a diccionario para búsqueda rápida
    kp_dict = {kp['cls_name']: kp['xy'] for kp in frame_keypoints if kp.get('conf', 1.0) >= conf_threshold}
    
    # ========================================================================
    # PASO 1: DETERMINAR ORIENTACIÓN (heurística en píxeles, SIN homografía)
    # ========================================================================
    orientation_votes = []
    
    # Estrategia: Comparar X en PÍXELES de pares conocidos
    # En campo real: smallarea (5.5m) < bigarea (16.5m) < arc (11m en centro) < midline (52.5m)
    
    # Vote 1: smallarea vs bigarea (inner)
    for suffix in ['top_inner', 'bottom_inner']:
        small_name = f'smallarea_{suffix}'
        big_name = f'bigarea_{suffix}'
        if small_name in kp_dict and big_name in kp_dict:
            small_x_px = kp_dict[small_name][0]
            big_x_px = kp_dict[big_name][0]
            
            # En orientación NORMAL: smallarea (5.5m) debe estar a la IZQUIERDA de bigarea (16.5m)
            # Si en píxeles: small_x < big_x → NO flip
            # Si en píxeles: small_x > big_x → SÍ flip (campo invertido)
            needs_flip = small_x_px > big_x_px
            orientation_votes.append(needs_flip)
    
    # Vote 2: smallarea vs bigarea (outter) - ambos en X=0 pero con diferentes Y
    # Este voto es menos útil, skip
    
    # Vote 3: bigarea_inner vs midline (si ambos visibles)
    for suffix in ['top', 'bottom']:
        big_name = f'bigarea_{suffix}_inner'
        mid_name = f'midline_{suffix}_intersection'
        if big_name in kp_dict and mid_name in kp_dict:
            big_x_px = kp_dict[big_name][0]
            mid_x_px = kp_dict[mid_name][0]
            
            # En orientación NORMAL: bigarea (16.5m) debe estar a la IZQUIERDA de midline (52.5m)
            # Si en píxeles: big_x < mid_x → NO flip
            # Si en píxeles: big_x > mid_x → SÍ flip
            needs_flip = big_x_px > mid_x_px
            orientation_votes.append(needs_flip)
    
    # Vote 4: arc_area vs midline
    for suffix in ['top', 'bottom']:
        arc_name = f'{suffix}_arc_area_intersection'
        mid_name = f'midline_{suffix}_intersection'
        if arc_name in kp_dict and mid_name in kp_dict:
            arc_x_px = kp_dict[arc_name][0]
            mid_x_px = kp_dict[mid_name][0]
            
            # En orientación NORMAL: arc (11m) debe estar a la IZQUIERDA de midline (52.5m)
            needs_flip = arc_x_px > mid_x_px
            orientation_votes.append(needs_flip)
    
    # ========================================================================
    # DECISIÓN DE ORIENTACIÓN POR CONSENSO
    # ========================================================================
    use_flip = False
    if len(orientation_votes) >= 1:
        # Consenso mayoritario
        flip_count = sum(orientation_votes)
        use_flip = flip_count > len(orientation_votes) / 2
    
    # ========================================================================
    # PASO 2: ESTIMAR HOMOGRAFÍA CON ORIENTACIÓN CORRECTA
    # ========================================================================
    field_points_to_use = flip_field_points(field_points) if use_flip else field_points
    
    # Intentar estimar con todos los keypoints
    H = estimate_homography(frame_keypoints, field_points_to_use, min_points, conf_threshold)
    
    # Si falla, intentar SOLO con keypoints inequívocos (línea central)
    if H is None:
        unambiguous_kps = [kp for kp in frame_keypoints 
                          if kp['cls_name'] in UNAMBIGUOUS_KEYPOINTS 
                          and kp.get('conf', 1.0) >= conf_threshold]
        
        if len(unambiguous_kps) >= min_points:
            H = estimate_homography(unambiguous_kps, field_points_to_use, min_points, conf_threshold)
    
    return H, use_flip


# ============================================================================
# 4. PROYECCIÓN POR TRIANGULACIÓN (Método Geométrico Directo)
# ============================================================================

def project_points_by_triangulation(
    player_positions_px: List[Tuple[float, float]],
    frame_keypoints: List[Dict],
    field_points: Dict[str, Tuple[float, float]],
    is_flipped: bool,
    min_references: int = 2,
    max_references: int = 4,
    conf_threshold: float = 0.3
) -> np.ndarray:
    """
    Proyecta posiciones de jugadores usando triangulación geométrica.
    
    En lugar de usar homografía completa (que puede tener distorsiones),
    usa los keypoints cercanos como referencias y calcula la posición
    mediante interpolación ponderada por distancia.
    
    Algoritmo:
    1. Para cada jugador, encuentra los K keypoints más cercanos (en píxeles)
    2. Calcula pesos inversamente proporcionales a la distancia
    3. Interpola la posición en el campo usando esos pesos
    
    Args:
        player_positions_px: Lista de (x, y) en píxeles
        frame_keypoints: Keypoints detectados [{"cls_name": str, "xy": (x,y), "conf": float}]
        field_points: Coordenadas teóricas en el campo
        is_flipped: Si True, aplicar flip a coordenadas de campo
        min_references: Mínimo de keypoints para proyectar
        max_references: Máximo de keypoints a usar por jugador
        conf_threshold: Umbral de confianza
    
    Returns:
        Array Nx2 con posiciones (X, Y) en metros en el campo
    """
    # Aplicar flip si necesario
    field_coords = flip_field_points(field_points) if is_flipped else field_points
    
    # Filtrar keypoints por confianza y que existan en el modelo
    valid_kps = []
    for kp in frame_keypoints:
        if kp.get('conf', 1.0) >= conf_threshold and kp['cls_name'] in field_coords:
            valid_kps.append({
                'name': kp['cls_name'],
                'px': np.array(kp['xy']),
                'field': np.array(field_coords[kp['cls_name']])
            })
    
    if len(valid_kps) < min_references:
        # No hay suficientes referencias, retornar posiciones inválidas
        return np.full((len(player_positions_px), 2), np.nan)
    
    # Proyectar cada jugador
    projected = []
    for player_px in player_positions_px:
        player_px = np.array(player_px)
        
        # Calcular distancias a todos los keypoints (en píxeles)
        distances = []
        for kp in valid_kps:
            dist = np.linalg.norm(player_px - kp['px'])
            distances.append(dist)
        
        # Ordenar por distancia y tomar los K más cercanos
        sorted_indices = np.argsort(distances)[:max_references]
        
        # Calcular pesos (inverso de la distancia, evitar división por 0)
        weights = []
        nearby_kps = []
        for idx in sorted_indices:
            dist = distances[idx]
            if dist < 1.0:  # Si está muy cerca de un keypoint
                dist = 1.0
            weight = 1.0 / (dist ** 2)  # Peso cuadrático (más peso a los cercanos)
            weights.append(weight)
            nearby_kps.append(valid_kps[idx])
        
        if len(weights) < min_references:
            projected.append([np.nan, np.nan])
            continue
        
        # Normalizar pesos
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Interpolación ponderada en píxeles
        # 1. Calcular vector desde el keypoint más cercano al jugador (en píxeles)
        closest_kp = nearby_kps[0]
        delta_px = player_px - closest_kp['px']
        
        # 2. Estimar escala píxel->metro local (usando pares de keypoints cercanos)
        px_per_meter = []
        for i in range(len(nearby_kps)):
            for j in range(i + 1, min(i + 3, len(nearby_kps))):  # Solo pares cercanos
                kp_i = nearby_kps[i]
                kp_j = nearby_kps[j]
                
                dist_px = np.linalg.norm(kp_i['px'] - kp_j['px'])
                dist_field = np.linalg.norm(kp_i['field'] - kp_j['field'])
                
                if dist_field > 2.0:  # Evitar keypoints demasiado cercanos
                    scale = dist_px / dist_field
                    px_per_meter.append(scale)
        
        # Usar mediana para robustez
        if len(px_per_meter) > 0:
            scale = np.median(px_per_meter)
        else:
            scale = 10.0  # Fallback conservador
        
        # 3. Convertir delta en píxeles a delta en metros
        if scale > 0:
            delta_meters = delta_px / scale
            # INVERTIR componente Y: en imagen Y crece hacia abajo, en campo hacia arriba
            delta_meters[1] = -delta_meters[1]
        else:
            delta_meters = np.array([0.0, 0.0])
        
        # 4. Posición final = posición del keypoint más cercano + delta en metros
        position_field = closest_kp['field'] + delta_meters
        
        # 5. Clipping: asegurar que esté dentro del campo
        position_field[0] = np.clip(position_field[0], 0, FIELD_LENGTH)
        position_field[1] = np.clip(position_field[1], 0, FIELD_WIDTH)
        
        projected.append(position_field)
    
    return np.array(projected)


# ============================================================================
# 5. PROYECCIÓN POR HOMOGRAFÍA (Método Clásico)
# ============================================================================

def project_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Proyecta puntos de imagen a coordenadas de campo usando homografía.
    
    Args:
        H: Homografía 3x3 (imagen -> campo)
        points: Array Nx2 de puntos (x, y) en píxeles
    
    Returns:
        Array Nx2 de puntos (X, Y) en coordenadas de campo
    """
    points = np.atleast_2d(points)
    
    # Convertir a coordenadas homogéneas
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # Aplicar homografía
    projected_h = (H @ points_h.T).T
    
    # Normalizar (dividir por tercera coordenada)
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    
    return projected


# ============================================================================
# 5. ACUMULADOR DE MAPAS DE CALOR
# ============================================================================

class HeatmapAccumulator:
    """
    Acumula posiciones de jugadores en una cuadrícula sobre el campo.
    """
    
    def __init__(
        self,
        field_length: float = FIELD_LENGTH,
        field_width: float = FIELD_WIDTH,
        nx: int = 105,
        ny: int = 68
    ):
        """
        Args:
            field_length: Longitud del campo (metros)
            field_width: Ancho del campo (metros)
            nx: Número de celdas en eje X
            ny: Número de celdas en eje Y
        """
        self.field_length = field_length
        self.field_width = field_width
        self.nx = nx
        self.ny = ny
        
        # Matrices de conteo por equipo
        self.counts_team0 = np.zeros((ny, nx), dtype=np.float32)
        self.counts_team1 = np.zeros((ny, nx), dtype=np.float32)
        
        # Contador de frames procesados
        self.num_frames = 0
    
    def add_frame(
        self,
        H: np.ndarray,
        player_dets: List[Dict]
    ):
        """
        Añade detecciones de jugadores de un frame al heatmap.
        
        Args:
            H: Homografía imagen -> campo para este frame
            player_dets: Lista de detecciones:
                [{"team_id": int, "xy": (x, y), "conf": float}, ...]
        """
        if not player_dets:
            return
        
        # Extraer puntos y team_ids
        points = []
        team_ids = []
        for det in player_dets:
            points.append(det['xy'])
            team_ids.append(det['team_id'])
        
        points = np.array(points, dtype=np.float32)
        
        # Proyectar al campo
        field_coords = project_points(H, points)
        
        # Acumular en celdas
        for (X, Y), team_id in zip(field_coords, team_ids):
            # Convertir coordenadas continuas a índices de celda
            i = int((Y / self.field_width) * self.ny)
            j = int((X / self.field_length) * self.nx)
            
            # Validar límites
            if 0 <= i < self.ny and 0 <= j < self.nx:
                if team_id == 0:
                    self.counts_team0[i, j] += 1
                elif team_id == 1:
                    self.counts_team1[i, j] += 1
        
        self.num_frames += 1
    
    def get_heatmap(self, team_id: int, normalize: str = 'max') -> np.ndarray:
        """
        Obtiene el heatmap de un equipo.
        
        Args:
            team_id: ID del equipo (0 o 1)
            normalize: Método de normalización:
                - 'max': Dividir por valor máximo (rango [0, 1])
                - 'sum': Dividir por suma total (densidad de probabilidad)
                - 'frames': Dividir por número de frames
                - None: Sin normalizar
        
        Returns:
            Matriz ny x nx con el heatmap
        """
        counts = self.counts_team0 if team_id == 0 else self.counts_team1
        
        if normalize == 'max':
            max_val = counts.max()
            return counts / max_val if max_val > 0 else counts
        elif normalize == 'sum':
            total = counts.sum()
            return counts / total if total > 0 else counts
        elif normalize == 'frames':
            return counts / self.num_frames if self.num_frames > 0 else counts
        else:
            return counts.copy()
    
    def reset(self):
        """Reinicia los contadores."""
        self.counts_team0.fill(0)
        self.counts_team1.fill(0)
        self.num_frames = 0


# ============================================================================
# 6. PIPELINE COMPLETO
# ============================================================================

def process_sequence(
    frames_keypoints: List[List[Dict]],
    frames_players: List[List[Dict]],
    field_points: Dict[str, Tuple[float, float]],
    accumulator: HeatmapAccumulator,
    verbose: bool = True
) -> Dict:
    """
    Procesa una secuencia de frames para generar heatmaps.
    
    Args:
        frames_keypoints: Lista de detecciones de keypoints por frame
        frames_players: Lista de detecciones de jugadores por frame
        field_points: Modelo teórico del campo
        accumulator: Acumulador de heatmaps
        verbose: Si True, imprime información de progreso
    
    Returns:
        Diccionario con estadísticas del procesamiento
    """
    num_frames = len(frames_keypoints)
    num_successful = 0
    num_flipped = 0
    
    for frame_idx in range(num_frames):
        kps = frames_keypoints[frame_idx]
        players = frames_players[frame_idx]
        
        # Estimar homografía con resolución de flip
        H, is_flipped = estimate_homography_with_flip_resolution(kps, field_points)
        
        if H is not None:
            # Añadir frame al acumulador
            accumulator.add_frame(H, players)
            num_successful += 1
            if is_flipped:
                num_flipped += 1
        elif verbose and frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}: No se pudo estimar homografía (keypoints insuficientes)")
    
    stats = {
        'total_frames': num_frames,
        'successful_frames': num_successful,
        'flipped_frames': num_flipped,
        'success_rate': num_successful / num_frames if num_frames > 0 else 0
    }
    
    if verbose:
        print(f"\n✓ Procesamiento completado:")
        print(f"  Frames procesados: {num_successful}/{num_frames} ({stats['success_rate']:.1%})")
        print(f"  Frames con flip: {num_flipped}/{num_successful}")
    
    return stats


# ============================================================================
# 7. EJEMPLO DE USO
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("EJEMPLO DE USO: Sistema de Mapas de Calor")
    print("=" * 70)
    
    # Datos sintéticos de ejemplo (3 frames)
    frames_keypoints = [
        # Frame 0: Vista desde la izquierda
        [
            {"cls_name": "midline_top_intersection", "xy": (960, 100), "conf": 0.95},
            {"cls_name": "midline_bottom_intersection", "xy": (960, 980), "conf": 0.93},
            {"cls_name": "bigarea_top_inner", "xy": (400, 300), "conf": 0.88},
            {"cls_name": "bigarea_bottom_inner", "xy": (400, 700), "conf": 0.87},
        ],
        # Frame 1: Vista desde la derecha (flipped)
        [
            {"cls_name": "midline_top_intersection", "xy": (960, 100), "conf": 0.94},
            {"cls_name": "midline_bottom_intersection", "xy": (960, 980), "conf": 0.92},
            {"cls_name": "bigarea_top_inner", "xy": (1520, 300), "conf": 0.89},
            {"cls_name": "bigarea_bottom_inner", "xy": (1520, 700), "conf": 0.86},
        ],
        # Frame 2: Vista central
        [
            {"cls_name": "midline_top_intersection", "xy": (960, 150), "conf": 0.96},
            {"cls_name": "midline_bottom_intersection", "xy": (960, 930), "conf": 0.94},
            {"cls_name": "halfcircle_top", "xy": (960, 400), "conf": 0.91},
            {"cls_name": "halfcircle_bottom", "xy": (960, 680), "conf": 0.90},
        ],
    ]
    
    frames_players = [
        # Frame 0: Jugadores distribuidos
        [
            {"team_id": 0, "xy": (300, 400), "conf": 0.95},
            {"team_id": 0, "xy": (500, 500), "conf": 0.93},
            {"team_id": 1, "xy": (1200, 400), "conf": 0.94},
            {"team_id": 1, "xy": (1400, 600), "conf": 0.92},
        ],
        # Frame 1
        [
            {"team_id": 0, "xy": (1600, 400), "conf": 0.94},
            {"team_id": 0, "xy": (1400, 500), "conf": 0.92},
            {"team_id": 1, "xy": (700, 400), "conf": 0.93},
            {"team_id": 1, "xy": (500, 600), "conf": 0.91},
        ],
        # Frame 2
        [
            {"team_id": 0, "xy": (800, 300), "conf": 0.95},
            {"team_id": 0, "xy": (900, 700), "conf": 0.94},
            {"team_id": 1, "xy": (1100, 400), "conf": 0.93},
            {"team_id": 1, "xy": (1200, 600), "conf": 0.92},
        ],
    ]
    
    # Crear acumulador
    print("\n1. Creando acumulador de heatmaps...")
    accumulator = HeatmapAccumulator(nx=21, ny=14)  # Cuadrícula más gruesa para ejemplo
    
    # Procesar secuencia
    print("\n2. Procesando secuencia...")
    stats = process_sequence(
        frames_keypoints,
        frames_players,
        FIELD_POINTS,
        accumulator,
        verbose=True
    )
    
    # Obtener heatmaps
    print("\n3. Generando heatmaps...")
    heatmap_team0 = accumulator.get_heatmap(0, normalize='max')
    heatmap_team1 = accumulator.get_heatmap(1, normalize='max')
    
    print(f"\n✓ Heatmap Team 0: shape={heatmap_team0.shape}, max={heatmap_team0.max():.2f}")
    print(f"✓ Heatmap Team 1: shape={heatmap_team1.shape}, max={heatmap_team1.max():.2f}")
    
    # Visualizar (ASCII art simplificado)
    print("\n4. Heatmap Team 0 (visualización ASCII):")
    print("   " + "=" * heatmap_team0.shape[1])
    for row in heatmap_team0:
        print("   " + "".join("█" if v > 0.5 else "▓" if v > 0.3 else "░" if v > 0.1 else " " for v in row))
    print("   " + "=" * heatmap_team0.shape[1])
    
    print("\n5. Heatmap Team 1 (visualización ASCII):")
    print("   " + "=" * heatmap_team1.shape[1])
    for row in heatmap_team1:
        print("   " + "".join("█" if v > 0.5 else "▓" if v > 0.3 else "░" if v > 0.1 else " " for v in row))
    print("   " + "=" * heatmap_team1.shape[1])
    
    # Intentar visualización con matplotlib si está disponible
    try:
        import matplotlib.pyplot as plt
        
        print("\n6. Generando visualización con matplotlib...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Team 0
        im0 = axes[0].imshow(heatmap_team0, cmap='Reds', origin='lower', aspect='auto')
        axes[0].set_title('Heatmap Team 0 (Local)')
        axes[0].set_xlabel('Eje X (longitud campo)')
        axes[0].set_ylabel('Eje Y (ancho campo)')
        plt.colorbar(im0, ax=axes[0])
        
        # Team 1
        im1 = axes[1].imshow(heatmap_team1, cmap='Blues', origin='lower', aspect='auto')
        axes[1].set_title('Heatmap Team 1 (Visitante)')
        axes[1].set_xlabel('Eje X (longitud campo)')
        axes[1].set_ylabel('Eje Y (ancho campo)')
        plt.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('heatmap_example.png', dpi=150)
        print("✓ Visualización guardada en: heatmap_example.png")
        
    except ImportError:
        print("\n  (matplotlib no disponible, saltando visualización gráfica)")
    
    print("\n" + "=" * 70)
    print("EJEMPLO COMPLETADO")
    print("=" * 70)
