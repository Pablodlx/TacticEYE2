"""
SpatialPossessionTracker - Tracking de posesión con información espacial

Extiende PossessionTracker para incluir análisis espacial:
- Reproyección de posiciones a coordenadas de campo
- Acumulación de posesión por zonas
- Heatmaps y mapas de calor por equipo
- Integración con FieldCalibrator
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict

from .possession_tracker import PossessionTracker
from .field_calibration import FieldCalibrator
from .field_model import FieldModel, ZoneModel


class SpatialPossessionTracker(PossessionTracker):
    """
    Tracker de posesión extendido con información espacial.
    
    Características adicionales sobre PossessionTracker base:
    - Mantiene posición en campo del poseedor del balón
    - Acumula tiempo de posesión por zona del campo
    - Genera heatmaps de posesión por equipo
    - Maneja casos donde no hay calibración disponible
    
    Ejemplo de uso:
        >>> calibrator = FieldCalibrator()
        >>> zone_model = ZoneModel(partition_type='thirds_lanes', nx=6, ny=4)
        >>> tracker = SpatialPossessionTracker(calibrator, zone_model)
        >>> 
        >>> # En cada frame:
        >>> tracker.update(ball_pos, players, frame)
        >>> stats = tracker.get_spatial_statistics()
    """
    
    def __init__(self,
                 calibrator: FieldCalibrator,
                 zone_model: Optional[ZoneModel] = None,
                 enable_heatmaps: bool = True,
                 heatmap_resolution: Tuple[int, int] = (50, 34),
                 **base_kwargs):
        """
        Args:
            calibrator: FieldCalibrator para reproyección
            zone_model: Modelo de zonas (usa default si es None)
            enable_heatmaps: Activar generación de heatmaps
            heatmap_resolution: Resolución del heatmap (width, height)
            **base_kwargs: Argumentos para PossessionTracker base
        """
        # Inicializar clase base
        super().__init__(**base_kwargs)
        
        self.calibrator = calibrator
        self.field_model = calibrator.field_model
        
        # Modelo de zonas
        if zone_model is None:
            self.zone_model = ZoneModel(
                self.field_model,
                partition_type='grid',
                nx=6,
                ny=4
            )
        else:
            self.zone_model = zone_model
        
        # Configuración de heatmaps
        self.enable_heatmaps = enable_heatmaps
        self.heatmap_resolution = heatmap_resolution
        
        # Estadísticas espaciales
        self._init_spatial_stats()
    
    def _init_spatial_stats(self):
        """Inicializa estructuras para estadísticas espaciales"""
        num_zones = self.zone_model.num_zones
        
        # Tiempo por equipo y zona (en frames)
        self.time_by_team_and_zone = {
            0: np.zeros(num_zones, dtype=np.int32),
            1: np.zeros(num_zones, dtype=np.int32),
            -1: np.zeros(num_zones, dtype=np.int32)  # Sin equipo / contested
        }
        
        # Heatmaps acumulativos (si están activados)
        if self.enable_heatmaps:
            h, w = self.heatmap_resolution
            self.heatmaps = {
                0: np.zeros((h, w), dtype=np.float32),
                1: np.zeros((h, w), dtype=np.float32)
            }
        else:
            self.heatmaps = None
        
        # Última posición conocida por equipo (para fallback)
        self.last_field_pos = {
            0: None,
            1: None
        }
        
        # Última zona conocida por equipo
        self.last_zone = {
            0: -1,
            1: -1
        }
        
        # Contadores de frames sin posición válida
        self.frames_without_position = {
            0: 0,
            1: 0
        }
    
    def update(self, 
               ball_pos: Optional[Tuple[float, float]],
               players: List,
               frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Actualiza estado de posesión con información espacial.
        
        Args:
            ball_pos: Posición del balón en imagen (x, y) o None
            players: Lista de jugadores (mismo formato que PossessionTracker base)
            frame: Frame actual (para actualizar calibración)
            
        Returns:
            Estado actualizado con info espacial adicional
        """
        # Actualizar calibración si hay frame disponible
        if frame is not None:
            self.calibrator.estimate_homography(frame)
        
        # Actualizar posesión base (no retorna nada, actualiza self.*)
        super().update(ball_pos, players)
        
        # Construir estado base desde atributos del tracker
        base_state = {
            'current_state': self.current_state,
            'current_team': self.current_team,
            'current_player': self.current_player,
            'frames_since_change': self.frames_since_change
        }
        
        # Añadir información espacial
        spatial_info = self._update_spatial(base_state, players)
        
        # Combinar estados
        full_state = {**base_state, **spatial_info}
        
        return full_state
    
    def _update_spatial(self, 
                       base_state: Dict[str, Any],
                       players: List) -> Dict[str, Any]:
        """
        Actualiza estadísticas espaciales.
        
        Args:
            base_state: Estado de posesión base
            players: Lista de jugadores
            
        Returns:
            Diccionario con info espacial adicional
        """
        spatial_info = {
            'field_position': None,
            'zone_id': -1,
            'calibration_valid': self.calibrator.has_valid_calibration()
        }
        
        # Obtener equipo y jugador en posesión desde base_state
        team_id = base_state.get('current_team')
        player_id = base_state.get('current_player')
        
        # Si no hay posesión definida, no hay nada que actualizar
        if team_id is None or player_id is None:
            return spatial_info
        
        # Intentar obtener posición en campo
        field_pos = None
        
        if self.calibrator.has_valid_calibration():
            # Encontrar el jugador con posesión
            player_pos_img = None
            
            for p in players:
                pid = p.get('track_id') if isinstance(p, dict) else p[0]
                if pid == player_id:
                    # Extraer posición del jugador
                    if isinstance(p, dict):
                        if 'pos' in p:
                            player_pos_img = p['pos']
                        elif 'bbox' in p:
                            # Usar base del bbox (pies del jugador)
                            x1, y1, x2, y2 = p['bbox']
                            player_pos_img = ((x1 + x2) / 2, y2)  # Centro-base
                    else:
                        player_pos_img = p[1]
                    break
            
            # Reproyectar a coordenadas de campo
            if player_pos_img is not None:
                field_pos = self.calibrator.image_to_field(*player_pos_img)
                
                if field_pos is not None:
                    spatial_info['field_position'] = field_pos
                    
                    # Determinar zona
                    zone_id = self.zone_model.zone_from_xy(
                        field_pos[0], 
                        field_pos[1],
                        team_id
                    )
                    spatial_info['zone_id'] = zone_id
                    
                    # Actualizar última posición conocida
                    if team_id is not None:
                        self.last_field_pos[team_id] = field_pos
                        self.last_zone[team_id] = zone_id
                        self.frames_without_position[team_id] = 0
        
        # Fallback: usar última posición conocida si no tenemos actual
        if field_pos is None and team_id is not None:
            self.frames_without_position[team_id] += 1
            
            # Solo usar fallback si no ha pasado mucho tiempo
            if self.frames_without_position[team_id] < 30:  # ~1 segundo a 30fps
                field_pos = self.last_field_pos.get(team_id)
                zone_id = self.last_zone.get(team_id, -1)
                
                spatial_info['field_position'] = field_pos
                spatial_info['zone_id'] = zone_id
                spatial_info['position_fallback'] = True
            else:
                spatial_info['position_fallback'] = False
        else:
            spatial_info['position_fallback'] = False
        
        # Acumular tiempo en zona
        zone_id = spatial_info['zone_id']
        if zone_id >= 0 and team_id is not None:
            self.time_by_team_and_zone[team_id][zone_id] += 1
        
        # Actualizar heatmap
        if self.enable_heatmaps and field_pos is not None and team_id is not None:
            self._update_heatmap(field_pos, team_id)
        
        return spatial_info
    
    def _update_heatmap(self, 
                       field_pos: Tuple[float, float],
                       team_id: int):
        """
        Actualiza heatmap con nueva posición.
        
        Args:
            field_pos: Posición en coordenadas de campo (x, y)
            team_id: ID del equipo (0 o 1)
        """
        if self.heatmaps is None:
            return
        
        # Normalizar coordenadas a [0, 1]
        x_norm, y_norm = self.field_model.normalize_coordinates(*field_pos)
        
        # Convertir a índices de heatmap
        h, w = self.heatmap_resolution
        ix = int(x_norm * w)
        iy = int(y_norm * h)
        
        # Verificar límites
        if 0 <= ix < w and 0 <= iy < h:
            # Incrementar con kernel gaussiano pequeño para suavizar
            self.heatmaps[team_id][iy, ix] += 1.0
            
            # Opcional: difusión a celdas vecinas
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = iy + dy, ix + dx
                    if 0 <= nx < w and 0 <= ny < h and (dx != 0 or dy != 0):
                        self.heatmaps[team_id][ny, nx] += 0.3
    
    def get_spatial_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas espaciales completas.
        
        Returns:
            Diccionario con:
            - possession_by_zone: Tiempo por zona y equipo
            - heatmaps: Heatmaps normalizados (si están activados)
            - zone_percentages: Porcentaje de tiempo por zona
        """
        stats = {
            'possession_by_zone': {},
            'zone_percentages': {},
            'heatmaps': None
        }
        
        # Calcular porcentajes por zona
        total_time = sum(
            self.time_by_team_and_zone[tid].sum() 
            for tid in [0, 1]
        )
        
        for team_id in [0, 1]:
            zone_times = self.time_by_team_and_zone[team_id]
            stats['possession_by_zone'][team_id] = zone_times.tolist()
            
            if total_time > 0:
                zone_pcts = (zone_times / total_time) * 100
                stats['zone_percentages'][team_id] = zone_pcts.tolist()
            else:
                stats['zone_percentages'][team_id] = [0.0] * len(zone_times)
        
        # Normalizar heatmaps
        if self.enable_heatmaps and self.heatmaps is not None:
            stats['heatmaps'] = {}
            for team_id in [0, 1]:
                hm = self.heatmaps[team_id].copy()
                max_val = hm.max()
                if max_val > 0:
                    hm = hm / max_val  # Normalizar a [0, 1]
                stats['heatmaps'][team_id] = hm
        
        return stats
    
    def get_zone_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas por zona en formato legible.
        
        Returns:
            Diccionario con nombre de zona y tiempos por equipo
        """
        zone_stats = []
        
        for zone_id in range(self.zone_model.num_zones):
            zone_name = self.zone_model.get_zone_name(zone_id)
            
            zone_info = {
                'zone_id': zone_id,
                'zone_name': zone_name,
                'team_0_frames': int(self.time_by_team_and_zone[0][zone_id]),
                'team_1_frames': int(self.time_by_team_and_zone[1][zone_id]),
            }
            
            # Calcular porcentajes
            total = zone_info['team_0_frames'] + zone_info['team_1_frames']
            if total > 0:
                zone_info['team_0_percent'] = (zone_info['team_0_frames'] / total) * 100
                zone_info['team_1_percent'] = (zone_info['team_1_frames'] / total) * 100
            else:
                zone_info['team_0_percent'] = 0.0
                zone_info['team_1_percent'] = 0.0
            
            zone_stats.append(zone_info)
        
        return {
            'zones': zone_stats,
            'partition_type': self.zone_model.partition_type,
            'num_zones': self.zone_model.num_zones
        }
    
    def reset_spatial_stats(self):
        """Resetea estadísticas espaciales (mantiene calibración)"""
        self._init_spatial_stats()
    
    def reset_all(self):
        """Resetea todo (incluyendo calibración)"""
        self.reset()  # Reset del PossessionTracker base
        self.reset_spatial_stats()
        self.calibrator.reset()
    
    def export_heatmap(self, team_id: int, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Exporta heatmap para visualización.
        
        Args:
            team_id: 0 o 1
            normalize: Si True, normaliza a [0, 1]
            
        Returns:
            Array 2D con el heatmap, o None si no está disponible
        """
        if not self.enable_heatmaps or self.heatmaps is None:
            return None
        
        hm = self.heatmaps[team_id].copy()
        
        if normalize:
            max_val = hm.max()
            if max_val > 0:
                hm = hm / max_val
        
        return hm
    
    def visualize_zones_on_frame(self, 
                                 frame: np.ndarray,
                                 show_possession: bool = True) -> np.ndarray:
        """
        Dibuja zonas y estadísticas de posesión sobre el frame.
        
        Args:
            frame: Frame original
            show_possession: Mostrar % de posesión por zona
            
        Returns:
            Frame con visualización
        """
        if not self.calibrator.has_valid_calibration():
            return frame
        
        vis = frame.copy()
        
        # Dibujar límites de zonas proyectados en imagen
        for zone_id in range(self.zone_model.num_zones):
            bounds = self.zone_model.get_zone_bounds(zone_id)
            if bounds is None:
                continue
            
            x_min, y_min, x_max, y_max = bounds
            
            # Proyectar esquinas de zona a imagen
            corners_field = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ], dtype=np.float32)
            
            corners_img = self.calibrator.image_to_field_batch(corners_field)
            
            if corners_img is not None:
                # Dibujar polígono
                pts = corners_img.astype(np.int32)
                cv2.polylines(vis, [pts], True, (255, 255, 0), 2)
                
                if show_possession:
                    # Mostrar estadísticas
                    t0 = self.time_by_team_and_zone[0][zone_id]
                    t1 = self.time_by_team_and_zone[1][zone_id]
                    total = t0 + t1
                    
                    if total > 0:
                        pct_0 = (t0 / total) * 100
                        text = f"{pct_0:.0f}%"
                        
                        # Posición del texto (centro de la zona)
                        center = pts.mean(axis=0).astype(int)
                        cv2.putText(vis, text, tuple(center),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
