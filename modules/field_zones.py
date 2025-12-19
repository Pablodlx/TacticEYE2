"""
Field Zones - Sistema de Zonificación Táctica del Campo
========================================================

Este módulo divide el campo de fútbol en zonas tácticas para análisis.
Soporta diferentes configuraciones de grid (3x5, 4x6, etc.) y proporciona
información semántica sobre cada zona (defensa, medio, ataque, flancos, etc.).

Diseñado para análisis táctico, no precisión milimétrica.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ZoneType(Enum):
    """Tipos de zonas tácticas"""
    DEFENSIVE = "defensive"
    MIDFIELD = "midfield"
    ATTACKING = "attacking"
    WING = "wing"
    CENTRAL = "central"
    PENALTY_AREA = "penalty_area"
    GOAL_AREA = "goal_area"
    CENTER_CIRCLE = "center_circle"


@dataclass
class FieldZone:
    """Representa una zona táctica del campo"""
    zone_id: int
    name: str
    bounds: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    center: Tuple[float, float]  # (x, y) en metros
    zone_type: ZoneType
    tactical_info: Dict[str, any]


class FieldZoneManager:
    """
    Gestiona la zonificación táctica del campo.
    
    Divide el campo en un grid configurable y proporciona información
    semántica sobre cada zona para análisis táctico.
    """
    
    def __init__(
        self,
        field_length: float = 105.0,
        field_width: float = 68.0,
        grid_cols: int = 6,
        grid_rows: int = 3,
        penalty_area_length: float = 16.5,
        penalty_area_width: float = 40.32,
        goal_area_length: float = 5.5,
        goal_area_width: float = 18.32,
        center_circle_radius: float = 9.15
    ):
        """
        Args:
            field_length: Longitud del campo en metros
            field_width: Ancho del campo en metros
            grid_cols: Número de columnas en el grid (dirección X)
            grid_rows: Número de filas en el grid (dirección Y)
            penalty_area_length: Longitud del área de penalti
            penalty_area_width: Ancho del área de penalti
            goal_area_length: Longitud del área de portería
            goal_area_width: Ancho del área de portería
            center_circle_radius: Radio del círculo central
        """
        self.field_length = field_length
        self.field_width = field_width
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.penalty_area_length = penalty_area_length
        self.penalty_area_width = penalty_area_width
        self.goal_area_length = goal_area_length
        self.goal_area_width = goal_area_width
        self.center_circle_radius = center_circle_radius
        
        # Generar zonas
        self.zones = self._generate_zones()
        
    def _generate_zones(self) -> List[FieldZone]:
        """
        Genera todas las zonas del campo basadas en el grid configurado.
        
        Returns:
            Lista de FieldZone ordenadas por zone_id
        """
        zones = []
        
        # Dimensiones de cada celda del grid
        col_width = self.field_length / self.grid_cols
        row_height = self.field_width / self.grid_rows
        
        zone_id = 1
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Calcular bounds de la zona
                x_min = col * col_width
                x_max = (col + 1) * col_width
                y_min = row * row_height
                y_max = (row + 1) * row_height
                
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                # Determinar tipo de zona
                zone_type, tactical_info = self._classify_zone(
                    center_x, center_y, x_min, y_min, x_max, y_max
                )
                
                # Generar nombre descriptivo
                name = self._generate_zone_name(row, col, zone_type)
                
                zone = FieldZone(
                    zone_id=zone_id,
                    name=name,
                    bounds=(x_min, y_min, x_max, y_max),
                    center=(center_x, center_y),
                    zone_type=zone_type,
                    tactical_info=tactical_info
                )
                
                zones.append(zone)
                zone_id += 1
        
        return zones
    
    def _classify_zone(
        self,
        center_x: float,
        center_y: float,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float
    ) -> Tuple[ZoneType, Dict[str, any]]:
        """
        Clasifica una zona según su posición y características tácticas.
        
        Args:
            center_x, center_y: Centro de la zona
            x_min, y_min, x_max, y_max: Bounds de la zona
            
        Returns:
            (tipo_zona, info_táctica)
        """
        tactical_info = {}
        
        # === CLASIFICACIÓN POR PROFUNDIDAD (X) ===
        # Defensiva: X < 35m (tercio defensivo)
        # Medio: 35m <= X < 70m (tercio medio)
        # Ataque: X >= 70m (tercio ofensivo)
        
        if center_x < 35.0:
            depth_type = ZoneType.DEFENSIVE
            tactical_info['depth'] = 'defensive'
            tactical_info['third'] = 'defensive_third'
        elif center_x < 70.0:
            depth_type = ZoneType.MIDFIELD
            tactical_info['depth'] = 'midfield'
            tactical_info['third'] = 'middle_third'
        else:
            depth_type = ZoneType.ATTACKING
            tactical_info['depth'] = 'attacking'
            tactical_info['third'] = 'attacking_third'
        
        # === CLASIFICACIÓN POR ANCHO (Y) ===
        # Wing: Y < 17m o Y > 51m (tercios laterales)
        # Central: 17m <= Y <= 51m (tercio central)
        
        wing_margin = self.field_width / 3  # ~22.67m
        
        if center_y < wing_margin or center_y > (self.field_width - wing_margin):
            width_type = ZoneType.WING
            tactical_info['width'] = 'wing'
            if center_y < wing_margin:
                tactical_info['side'] = 'left'
            else:
                tactical_info['side'] = 'right'
        else:
            width_type = ZoneType.CENTRAL
            tactical_info['width'] = 'central'
            tactical_info['side'] = 'center'
        
        # === CLASIFICACIÓN POR ÁREAS ESPECIALES ===
        # Área de penalti
        penalty_y_min = (self.field_width - self.penalty_area_width) / 2
        penalty_y_max = penalty_y_min + self.penalty_area_width
        
        if (x_min < self.penalty_area_length and 
            penalty_y_min <= center_y <= penalty_y_max):
            tactical_info['special_area'] = 'penalty_area'
            tactical_info['goal_side'] = 'left'
            return ZoneType.PENALTY_AREA, tactical_info
        
        if (x_max > (self.field_length - self.penalty_area_length) and
            penalty_y_min <= center_y <= penalty_y_max):
            tactical_info['special_area'] = 'penalty_area'
            tactical_info['goal_side'] = 'right'
            return ZoneType.PENALTY_AREA, tactical_info
        
        # Área de portería
        goal_y_min = (self.field_width - self.goal_area_width) / 2
        goal_y_max = goal_y_min + self.goal_area_width
        
        if (x_min < self.goal_area_length and
            goal_y_min <= center_y <= goal_y_max):
            tactical_info['special_area'] = 'goal_area'
            tactical_info['goal_side'] = 'left'
            return ZoneType.GOAL_AREA, tactical_info
        
        if (x_max > (self.field_length - self.goal_area_length) and
            goal_y_min <= center_y <= goal_y_max):
            tactical_info['special_area'] = 'goal_area'
            tactical_info['goal_side'] = 'right'
            return ZoneType.GOAL_AREA, tactical_info
        
        # Círculo central
        center_x_field = self.field_length / 2
        center_y_field = self.field_width / 2
        dist_from_center = np.sqrt(
            (center_x - center_x_field)**2 + (center_y - center_y_field)**2
        )
        
        if dist_from_center <= self.center_circle_radius:
            tactical_info['special_area'] = 'center_circle'
            return ZoneType.CENTER_CIRCLE, tactical_info
        
        # Zona normal: combinar profundidad y ancho
        if width_type == ZoneType.WING:
            return width_type, tactical_info
        else:
            return depth_type, tactical_info
    
    def _generate_zone_name(
        self,
        row: int,
        col: int,
        zone_type: ZoneType
    ) -> str:
        """
        Genera nombre descriptivo para una zona.
        
        Convención:
        - Filas: 0=Abajo (cerca), 1=Centro, 2=Arriba (lejos)
        - Columnas: 0=Izquierda, ..., N-1=Derecha
        """
        row_names = ["Bottom", "Center", "Top"]
        col_names = ["Left", "Mid-Left", "Mid", "Mid-Right", "Right"]
        
        row_name = row_names[row] if row < len(row_names) else f"Row{row}"
        
        if self.grid_cols == 6:
            if col == 0:
                col_name = "Left"
            elif col == 1:
                col_name = "Mid-Left"
            elif col == 2:
                col_name = "Center-Left"
            elif col == 3:
                col_name = "Center-Right"
            elif col == 4:
                col_name = "Mid-Right"
            else:
                col_name = "Right"
        elif self.grid_cols == 5:
            col_name = col_names[col] if col < len(col_names) else f"Col{col}"
        else:
            col_name = f"Col{col}"
        
        # Añadir información táctica
        if zone_type == ZoneType.PENALTY_AREA:
            return f"{row_name} {col_name} (Penalty Area)"
        elif zone_type == ZoneType.GOAL_AREA:
            return f"{row_name} {col_name} (Goal Area)"
        elif zone_type == ZoneType.CENTER_CIRCLE:
            return f"{row_name} {col_name} (Center Circle)"
        else:
            return f"{row_name} {col_name}"
    
    def get_zone(self, position: np.ndarray) -> Optional[FieldZone]:
        """
        Obtiene la zona que contiene una posición dada.
        
        Args:
            position: [x, y] en metros (coordenadas del campo)
            
        Returns:
            FieldZone que contiene la posición, o None si está fuera del campo
        """
        x, y = position[0], position[1]
        
        # Verificar bounds del campo
        if x < 0 or x > self.field_length or y < 0 or y > self.field_width:
            return None
        
        # Encontrar zona correspondiente
        col_width = self.field_length / self.grid_cols
        row_height = self.field_width / self.grid_rows
        
        col = int(x / col_width)
        row = int(y / row_height)
        
        # Asegurar índices válidos
        col = min(col, self.grid_cols - 1)
        row = min(row, self.grid_rows - 1)
        
        zone_id = row * self.grid_cols + col + 1
        
        return self.zones[zone_id - 1]
    
    def get_zone_by_id(self, zone_id: int) -> Optional[FieldZone]:
        """Obtiene una zona por su ID"""
        if 1 <= zone_id <= len(self.zones):
            return self.zones[zone_id - 1]
        return None
    
    def get_zones_by_type(self, zone_type: ZoneType) -> List[FieldZone]:
        """Obtiene todas las zonas de un tipo específico"""
        return [zone for zone in self.zones if zone.zone_type == zone_type]
    
    def get_zones_in_region(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float
    ) -> List[FieldZone]:
        """
        Obtiene todas las zonas que intersectan con una región rectangular.
        
        Args:
            x_min, y_min, x_max, y_max: Bounds de la región
            
        Returns:
            Lista de zonas que intersectan
        """
        intersecting_zones = []
        
        for zone in self.zones:
            z_x_min, z_y_min, z_x_max, z_y_max = zone.bounds
            
            # Verificar intersección
            if not (x_max < z_x_min or x_min > z_x_max or
                   y_max < z_y_min or y_min > z_y_max):
                intersecting_zones.append(zone)
        
        return intersecting_zones
    
    def get_zone_statistics(self) -> Dict[str, any]:
        """
        Retorna estadísticas sobre la zonificación.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'total_zones': len(self.zones),
            'grid_size': f"{self.grid_cols}x{self.grid_rows}",
            'zones_by_type': {}
        }
        
        for zone_type in ZoneType:
            zones_of_type = self.get_zones_by_type(zone_type)
            stats['zones_by_type'][zone_type.value] = len(zones_of_type)
        
        return stats
    
    def visualize_zones(self, image_shape: Tuple[int, int], homography: np.ndarray) -> np.ndarray:
        """
        Visualiza las zonas proyectadas sobre una imagen usando la homografía.
        
        Args:
            image_shape: (height, width) de la imagen
            homography: Matriz de homografía campo -> imagen
            
        Returns:
            Imagen con overlay de zonas
        """
        import cv2
        
        h, w = image_shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Colores por tipo de zona
        color_map = {
            ZoneType.DEFENSIVE: (0, 100, 200),      # Azul oscuro
            ZoneType.MIDFIELD: (0, 200, 200),        # Cian
            ZoneType.ATTACKING: (0, 200, 100),       # Verde
            ZoneType.WING: (200, 100, 0),            # Naranja
            ZoneType.PENALTY_AREA: (0, 0, 255),      # Rojo
            ZoneType.GOAL_AREA: (255, 0, 255),       # Magenta
            ZoneType.CENTER_CIRCLE: (255, 255, 0),   # Amarillo
        }
        
        # Proyectar cada zona
        for zone in self.zones:
            x_min, y_min, x_max, y_max = zone.bounds
            
            # Esquinas de la zona en coordenadas del campo
            corners_field = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ], dtype=np.float32)
            
            # Transformar a imagen
            corners_image = cv2.perspectiveTransform(
                corners_field.reshape(-1, 1, 2),
                homography
            ).reshape(-1, 2).astype(np.int32)
            
            # Obtener color
            color = color_map.get(zone.zone_type, (128, 128, 128))
            
            # Dibujar polígono semitransparente
            overlay = vis.copy()
            cv2.fillPoly(overlay, [corners_image], color)
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            
            # Dibujar borde
            cv2.polylines(vis, [corners_image], True, color, 2)
            
            # Etiqueta con ID
            center_image = corners_image.mean(axis=0).astype(int)
            cv2.putText(vis, str(zone.zone_id),
                       tuple(center_image),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis

