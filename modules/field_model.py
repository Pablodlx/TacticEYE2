"""
FieldModel - Modelo geométrico del campo de fútbol estándar

Define las dimensiones y elementos clave del campo según normativa FIFA.
Proporciona correspondencias entre elementos detectables y posiciones reales.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FieldDimensions:
    """Dimensiones estándar del campo de fútbol (en metros)"""
    length: float = 105.0  # Largo del campo (FIFA estándar)
    width: float = 68.0    # Ancho del campo
    
    # Áreas
    penalty_area_length: float = 16.5  # Profundidad del área grande
    penalty_area_width: float = 40.3   # Ancho del área grande
    goal_area_length: float = 5.5      # Profundidad del área pequeña
    goal_area_width: float = 18.3      # Ancho del área pequeña
    
    # Otros elementos
    penalty_spot_distance: float = 11.0  # Distancia del penalty a la línea de gol
    center_circle_radius: float = 9.15   # Radio del círculo central
    corner_arc_radius: float = 1.0       # Radio del cuarto de círculo en esquinas


class FieldModel:
    """
    Modelo del campo de fútbol con puntos y líneas de referencia.
    
    Sistema de coordenadas del campo:
    - Origen (0,0) en el centro del campo
    - Eje X: a lo largo del campo (-length/2 a +length/2)
    - Eje Y: a lo ancho del campo (-width/2 a +width/2)
    """
    
    def __init__(self, dimensions: Optional[FieldDimensions] = None):
        self.dims = dimensions or FieldDimensions()
        self._build_keypoints()
        self._build_lines()
        
    def _build_keypoints(self):
        """Construye puntos clave del campo en coordenadas reales"""
        d = self.dims
        half_l = d.length / 2
        half_w = d.width / 2
        
        self.keypoints = {
            # Esquinas del campo
            'corner_tl': (-half_l, half_w),
            'corner_tr': (half_l, half_w),
            'corner_bl': (-half_l, -half_w),
            'corner_br': (half_l, -half_w),
            
            # Centro del campo
            'center': (0.0, 0.0),
            
            # Puntos de penalti
            'penalty_left': (-half_l + d.penalty_spot_distance, 0.0),
            'penalty_right': (half_l - d.penalty_spot_distance, 0.0),
            
            # Esquinas del área grande (izquierda)
            'penalty_area_left_tl': (-half_l, d.penalty_area_width/2),
            'penalty_area_left_tr': (-half_l + d.penalty_area_length, d.penalty_area_width/2),
            'penalty_area_left_bl': (-half_l, -d.penalty_area_width/2),
            'penalty_area_left_br': (-half_l + d.penalty_area_length, -d.penalty_area_width/2),
            
            # Esquinas del área grande (derecha)
            'penalty_area_right_tl': (half_l, d.penalty_area_width/2),
            'penalty_area_right_tr': (half_l - d.penalty_area_length, d.penalty_area_width/2),
            'penalty_area_right_bl': (half_l, -d.penalty_area_width/2),
            'penalty_area_right_br': (half_l - d.penalty_area_length, -d.penalty_area_width/2),
            
            # Esquinas del área pequeña (izquierda)
            'goal_area_left_tl': (-half_l, d.goal_area_width/2),
            'goal_area_left_tr': (-half_l + d.goal_area_length, d.goal_area_width/2),
            'goal_area_left_bl': (-half_l, -d.goal_area_width/2),
            'goal_area_left_br': (-half_l + d.goal_area_length, -d.goal_area_width/2),
            
            # Esquinas del área pequeña (derecha)
            'goal_area_right_tl': (half_l, d.goal_area_width/2),
            'goal_area_right_tr': (half_l - d.goal_area_length, d.goal_area_width/2),
            'goal_area_right_bl': (half_l, -d.goal_area_width/2),
            'goal_area_right_br': (half_l - d.goal_area_length, -d.goal_area_width/2),
        }
    
    def _build_lines(self):
        """Construye líneas principales del campo"""
        d = self.dims
        half_l = d.length / 2
        half_w = d.width / 2
        
        self.lines = {
            # Líneas del perímetro
            'sideline_top': [(-half_l, half_w), (half_l, half_w)],
            'sideline_bottom': [(-half_l, -half_w), (half_l, -half_w)],
            'goal_line_left': [(-half_l, -half_w), (-half_l, half_w)],
            'goal_line_right': [(half_l, -half_w), (half_l, half_w)],
            
            # Línea central
            'halfway_line': [(0, -half_w), (0, half_w)],
            
            # Área grande izquierda
            'penalty_area_left_top': [
                (-half_l, d.penalty_area_width/2),
                (-half_l + d.penalty_area_length, d.penalty_area_width/2)
            ],
            'penalty_area_left_bottom': [
                (-half_l, -d.penalty_area_width/2),
                (-half_l + d.penalty_area_length, -d.penalty_area_width/2)
            ],
            'penalty_area_left_front': [
                (-half_l + d.penalty_area_length, -d.penalty_area_width/2),
                (-half_l + d.penalty_area_length, d.penalty_area_width/2)
            ],
            
            # Área grande derecha
            'penalty_area_right_top': [
                (half_l, d.penalty_area_width/2),
                (half_l - d.penalty_area_length, d.penalty_area_width/2)
            ],
            'penalty_area_right_bottom': [
                (half_l, -d.penalty_area_width/2),
                (half_l - d.penalty_area_length, -d.penalty_area_width/2)
            ],
            'penalty_area_right_front': [
                (half_l - d.penalty_area_length, -d.penalty_area_width/2),
                (half_l - d.penalty_area_length, d.penalty_area_width/2)
            ],
        }
    
    def get_all_keypoints(self) -> Dict[str, Tuple[float, float]]:
        """Retorna todos los puntos clave del campo"""
        return self.keypoints.copy()
    
    def get_all_lines(self) -> Dict[str, List[Tuple[float, float]]]:
        """Retorna todas las líneas del campo"""
        return self.lines.copy()
    
    def normalize_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """
        Normaliza coordenadas de campo a rango [0, 1]
        
        Args:
            x, y: Coordenadas en metros (centradas en 0,0)
            
        Returns:
            (x_norm, y_norm) en rango [0, 1]
        """
        x_norm = (x + self.dims.length/2) / self.dims.length
        y_norm = (y + self.dims.width/2) / self.dims.width
        return x_norm, y_norm
    
    def denormalize_coordinates(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        """
        Desnormaliza coordenadas de [0,1] a coordenadas de campo en metros
        """
        x = x_norm * self.dims.length - self.dims.length/2
        y = y_norm * self.dims.width - self.dims.width/2
        return x, y


class ZoneModel:
    """
    Modelo de zonas del campo para análisis espacial de posesión.
    
    Soporta diferentes tipos de partición:
    - 'grid': Rejilla regular (nx × ny)
    - 'thirds': Tercios defensivo/medio/ofensivo
    - 'thirds_lanes': Tercios × Carriles (lateral izq/central/lateral der)
    """
    
    def __init__(self, 
                 field_model: FieldModel,
                 partition_type: str = 'grid',
                 nx: int = 6,
                 ny: int = 4):
        """
        Args:
            field_model: Modelo del campo
            partition_type: 'grid', 'thirds', 'thirds_lanes'
            nx: Divisiones en X (si partition_type='grid')
            ny: Divisiones en Y (si partition_type='grid')
        """
        self.field_model = field_model
        self.partition_type = partition_type
        self.nx = nx
        self.ny = ny
        
        self._build_zones()
    
    def _build_zones(self):
        """Construye las zonas según el tipo de partición"""
        dims = self.field_model.dims
        
        if self.partition_type == 'grid':
            self.num_zones = self.nx * self.ny
            self.zone_names = [f"zone_{i}" for i in range(self.num_zones)]
            
        elif self.partition_type == 'thirds':
            self.num_zones = 3
            self.zone_names = ['defensive', 'middle', 'offensive']
            
        elif self.partition_type == 'thirds_lanes':
            self.num_zones = 9  # 3 tercios × 3 carriles
            self.zone_names = [
                'def_left', 'def_center', 'def_right',
                'mid_left', 'mid_center', 'mid_right',
                'off_left', 'off_center', 'off_right'
            ]
    
    def zone_from_xy(self, x: float, y: float, team_id: int = 0) -> int:
        """
        Determina la zona a partir de coordenadas de campo.
        
        Args:
            x, y: Coordenadas en metros (sistema del campo)
            team_id: 0 o 1 (para orientar tercios si es necesario)
            
        Returns:
            ID de zona (0 a num_zones-1), o -1 si está fuera del campo
        """
        dims = self.field_model.dims
        half_l = dims.length / 2
        half_w = dims.width / 2
        
        # Verificar que esté dentro del campo
        if not (-half_l <= x <= half_l and -half_w <= y <= half_w):
            return -1
        
        if self.partition_type == 'grid':
            # Normalizar a [0, 1]
            x_norm = (x + half_l) / dims.length
            y_norm = (y + half_w) / dims.width
            
            # Calcular índices de rejilla
            ix = min(int(x_norm * self.nx), self.nx - 1)
            iy = min(int(y_norm * self.ny), self.ny - 1)
            
            return iy * self.nx + ix
        
        elif self.partition_type == 'thirds':
            # Ajustar orientación según equipo (team 0 ataca a la derecha)
            x_oriented = x if team_id == 0 else -x
            
            if x_oriented < -dims.length / 6:
                return 0  # defensive
            elif x_oriented < dims.length / 6:
                return 1  # middle
            else:
                return 2  # offensive
        
        elif self.partition_type == 'thirds_lanes':
            # Tercios en X
            x_oriented = x if team_id == 0 else -x
            if x_oriented < -dims.length / 6:
                third = 0  # defensive
            elif x_oriented < dims.length / 6:
                third = 1  # middle
            else:
                third = 2  # offensive
            
            # Carriles en Y
            if y < -dims.width / 6:
                lane = 0  # left
            elif y < dims.width / 6:
                lane = 1  # center
            else:
                lane = 2  # right
            
            return third * 3 + lane
        
        return -1
    
    def get_zone_name(self, zone_id: int) -> str:
        """Retorna el nombre de la zona"""
        if 0 <= zone_id < self.num_zones:
            return self.zone_names[zone_id]
        return "unknown"
    
    def get_zone_bounds(self, zone_id: int) -> Optional[Tuple[float, float, float, float]]:
        """
        Retorna los límites de una zona (x_min, y_min, x_max, y_max)
        """
        dims = self.field_model.dims
        half_l = dims.length / 2
        half_w = dims.width / 2
        
        if self.partition_type == 'grid':
            if not (0 <= zone_id < self.num_zones):
                return None
            
            iy = zone_id // self.nx
            ix = zone_id % self.nx
            
            x_min = -half_l + (ix * dims.length / self.nx)
            x_max = -half_l + ((ix + 1) * dims.length / self.nx)
            y_min = -half_w + (iy * dims.width / self.ny)
            y_max = -half_w + ((iy + 1) * dims.width / self.ny)
            
            return (x_min, y_min, x_max, y_max)
        
        # TODO: Implementar para otros tipos de partición
        return None
