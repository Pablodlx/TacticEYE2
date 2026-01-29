"""
Field Model - Modelo teórico del campo de fútbol
=================================================

Define las coordenadas de referencia de los keypoints del campo en el sistema
de coordenadas del terreno de juego (metros o normalizado).
"""

import numpy as np
import cv2
from typing import Dict, Tuple


class FieldModel:
    """
    Modelo del campo de fútbol con coordenadas teóricas de keypoints.
    
    Sistema de coordenadas:
    - Origen (0,0) en esquina inferior izquierda
    - X: eje longitudinal (0-105m para campo estándar)
    - Y: eje lateral (0-68m para campo estándar)
    
    Keypoints principales:
    - Esquinas del campo (4)
    - Esquinas del área grande (4)
    - Esquinas del área pequeña (4)
    - Puntos de penalti (2)
    - Círculo central (4 puntos cardinales)
    - Medio campo (línea central)
    """
    
    def __init__(self, 
                 field_length: float = 105.0,
                 field_width: float = 68.0,
                 use_normalized: bool = False):
        """
        Args:
            field_length: Longitud del campo en metros (105m estándar FIFA)
            field_width: Ancho del campo en metros (68m estándar FIFA)
            use_normalized: Si True, coordenadas en [0,1], si False en metros
        """
        self.field_length = field_length
        self.field_width = field_width
        self.use_normalized = use_normalized
        
        # Dimensiones reglamentarias FIFA (en metros desde borde)
        self.big_area_length = 16.5  # Área grande
        self.big_area_width = 40.3
        self.small_area_length = 5.5  # Área pequeña
        self.small_area_width = 18.3
        self.penalty_spot_distance = 11.0  # Punto de penalti desde línea de gol
        self.center_circle_radius = 9.15
        
        # Generar coordenadas de keypoints
        self.keypoints_field = self._generate_keypoints()
    
    def _generate_keypoints(self) -> Dict[str, Tuple[float, float]]:
        """
        Genera el diccionario de keypoints con sus coordenadas de campo.
        
        Nombres compatibles con lo que probablemente devuelva el modelo de Roboflow:
        - corner_*: esquinas del campo
        - big_area_*: área grande
        - small_area_*: área pequeña
        - penalty_*: puntos de penalti
        - circle_*: puntos del círculo central
        - midfield_*: puntos de la línea central
        
        Returns:
            Dict {keypoint_name: (X_field, Y_field)}
        """
        L = self.field_length
        W = self.field_width
        
        keypoints = {}
        
        # ========== ESQUINAS DEL CAMPO ==========
        # Esquinas (4 puntos)
        keypoints['corner_bottom_left'] = (0, 0)
        keypoints['corner_bottom_right'] = (L, 0)
        keypoints['corner_top_left'] = (0, W)
        keypoints['corner_top_right'] = (L, W)
        
        # Aliases comunes
        keypoints['corner_1'] = (0, 0)
        keypoints['corner_2'] = (L, 0)
        keypoints['corner_3'] = (L, W)
        keypoints['corner_4'] = (0, W)
        
        # ========== ÁREAS ==========
        # Área grande izquierda
        keypoints['big_area_left_bottom'] = (self.big_area_length, (W - self.big_area_width) / 2)
        keypoints['big_area_left_top'] = (self.big_area_length, (W + self.big_area_width) / 2)
        
        # Área grande derecha
        keypoints['big_area_right_bottom'] = (L - self.big_area_length, (W - self.big_area_width) / 2)
        keypoints['big_area_right_top'] = (L - self.big_area_length, (W + self.big_area_width) / 2)
        
        # Esquinas del área grande (más detalle)
        keypoints['big_area_left_corner_bottom'] = (0, (W - self.big_area_width) / 2)
        keypoints['big_area_left_corner_top'] = (0, (W + self.big_area_width) / 2)
        keypoints['big_area_right_corner_bottom'] = (L, (W - self.big_area_width) / 2)
        keypoints['big_area_right_corner_top'] = (L, (W + self.big_area_width) / 2)
        
        # Área pequeña izquierda
        keypoints['small_area_left_bottom'] = (self.small_area_length, (W - self.small_area_width) / 2)
        keypoints['small_area_left_top'] = (self.small_area_length, (W + self.small_area_width) / 2)
        
        # Área pequeña derecha
        keypoints['small_area_right_bottom'] = (L - self.small_area_length, (W - self.small_area_width) / 2)
        keypoints['small_area_right_top'] = (L - self.small_area_length, (W + self.small_area_width) / 2)
        
        # ========== ARCOS DE PENALTI (intersección con área) ==========
        # Distancia del arco al punto de penalti: 9.15m (radio del arco)
        # Estos son puntos donde el arco intersecta con la línea del área grande
        penalty_arc_radius = 9.15
        penalty_arc_offset_y = 9.15  # Distancia vertical desde penalti a intersección
        
        # Izquierda
        keypoints['left_penalty_arc_top'] = (self.penalty_spot_distance, W / 2 + penalty_arc_offset_y)
        keypoints['left_penalty_arc_bottom'] = (self.penalty_spot_distance, W / 2 - penalty_arc_offset_y)
        
        # Derecha
        keypoints['right_penalty_arc_top'] = (L - self.penalty_spot_distance, W / 2 + penalty_arc_offset_y)
        keypoints['right_penalty_arc_bottom'] = (L - self.penalty_spot_distance, W / 2 - penalty_arc_offset_y)
        
        # Aliases para modelo field_kp_merged_fast (genéricos - se mapean a promedio)
        # Nota: estos son aproximaciones ya que el modelo no distingue izq/der
        keypoints['top_arc_area_intersection'] = (L / 2, W / 2 + penalty_arc_offset_y)
        keypoints['bottom_arc_area_intersection'] = (L / 2, W / 2 - penalty_arc_offset_y)
        keypoints['bigarea_top_inner'] = (L / 2, (W + self.big_area_width) / 2)
        keypoints['bigarea_bottom_inner'] = (L / 2, (W - self.big_area_width) / 2)
        keypoints['bigarea_top_outter'] = (0, (W + self.big_area_width) / 2)  # Línea de gol
        keypoints['bigarea_bottom_outter'] = (0, (W - self.big_area_width) / 2)  # Línea de gol
        keypoints['smallarea_top_inner'] = (L / 2, (W + self.small_area_width) / 2)
        keypoints['smallarea_bottom_inner'] = (L / 2, (W - self.small_area_width) / 2)
        keypoints['smallarea_top_outter'] = (0, (W + self.small_area_width) / 2)  # Línea de gol
        keypoints['smallarea_bottom_outter'] = (0, (W - self.small_area_width) / 2)  # Línea de gol
        
        # ========== PUNTOS DE PENALTI ==========
        keypoints['penalty_left'] = (self.penalty_spot_distance, W / 2)
        keypoints['penalty_right'] = (L - self.penalty_spot_distance, W / 2)
        
        # ========== LÍNEA CENTRAL Y CÍRCULO CENTRAL ==========
        # Punto central del campo
        keypoints['center'] = (L / 2, W / 2)
        keypoints['center_circle_center'] = (L / 2, W / 2)
        
        # Puntos cardinales del círculo central
        R = self.center_circle_radius
        keypoints['circle_top'] = (L / 2, W / 2 + R)
        keypoints['circle_bottom'] = (L / 2, W / 2 - R)
        keypoints['circle_left'] = (L / 2 - R, W / 2)
        keypoints['circle_right'] = (L / 2 + R, W / 2)
        
        # Aliases para modelo field_kp_merged_fast
        keypoints['halfcircle_top'] = (L / 2, W / 2 + R)
        keypoints['halfcircle_bottom'] = (L / 2, W / 2 - R)
        
        # Intersecciones de línea central con bordes
        keypoints['midfield_top'] = (L / 2, W)
        keypoints['midfield_bottom'] = (L / 2, 0)
        keypoints['midline_top_intersection'] = (L / 2, W)
        keypoints['midline_bottom_intersection'] = (L / 2, 0)
        
        # Normalizar si se requiere
        if self.use_normalized:
            keypoints = {
                name: (x / L, y / W)
                for name, (x, y) in keypoints.items()
            }
        
        return keypoints
    
    def get_keypoint_coords(self, name: str) -> Tuple[float, float]:
        """
        Obtiene las coordenadas de un keypoint por nombre.
        
        Args:
            name: Nombre del keypoint
            
        Returns:
            (X, Y) en coordenadas de campo, o None si no existe
        """
        return self.keypoints_field.get(name)
    
    def normalize_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """
        Normaliza coordenadas de campo a [0,1].
        
        Args:
            x: Coordenada X en metros (o ya normalizada si use_normalized=True)
            y: Coordenada Y en metros (o ya normalizada si use_normalized=True)
            
        Returns:
            (x_norm, y_norm) en rango [0,1]
        """
        if self.use_normalized:
            # Ya están normalizadas
            return (x, y)
        else:
            # Normalizar diviendo por dimensiones del campo
            x_norm = x / self.field_length
            y_norm = y / self.field_width
            return (x_norm, y_norm)
    
    def match_keypoint_names(self, detected_names: list) -> Dict[str, str]:
        """
        Intenta hacer matching entre nombres detectados por el modelo
        y nombres del FieldModel.
        
        Útil porque el modelo de Roboflow puede usar nombres ligeramente diferentes.
        
        Args:
            detected_names: Lista de nombres detectados por el modelo
            
        Returns:
            Dict {detected_name: field_model_name} con los matchings encontrados
        """
        matches = {}
        
        # Mapeo directo para modelos que usan numeración
        # Basado en el esquema común de anotación de campos de fútbol
        numeric_mapping = {
            '1': 'corner_1',           # Esquina inferior izquierda
            '2': 'corner_2',           # Esquina inferior derecha
            '3': 'corner_3',           # Esquina superior derecha
            '4': 'corner_4',           # Esquina superior izquierda
            '5': 'big_area_left_bottom',
            '6': 'big_area_left_top',
            '7': 'big_area_right_bottom',
            '8': 'big_area_right_top',
            '9': 'small_area_left_bottom',
            '10': 'small_area_left_top',
            '11': 'small_area_right_bottom',
            '12': 'small_area_right_top',
            '13': 'penalty_left',
            '14': 'penalty_right',
            '15': 'center',
            '16': 'circle_top',
            '17': 'circle_bottom',
            '18': 'circle_left',
            '19': 'circle_right',
            '20': 'midfield_top',
            '21': 'midfield_bottom',
            '22': 'big_area_left_corner_bottom',
            '23': 'big_area_left_corner_top',
            '24': 'big_area_right_corner_bottom',
            '25': 'big_area_right_corner_top',
        }
        
        for detected in detected_names:
            # 1. Intentar mapeo numérico directo
            if detected in numeric_mapping:
                matches[detected] = numeric_mapping[detected]
                continue
            
            # 2. Normalizar nombre (lowercase, sin espacios)
            detected_norm = detected.lower().replace(' ', '_').replace('-', '_')
            
            # 3. Buscar match exacto
            if detected_norm in self.keypoints_field:
                matches[detected] = detected_norm
                continue
            
            # 4. Buscar match por similitud (estrategias de fuzzy matching)
            for field_name in self.keypoints_field.keys():
                # Match parcial
                if detected_norm in field_name or field_name in detected_norm:
                    matches[detected] = field_name
                    break
                
                # Match por tokens clave
                detected_tokens = set(detected_norm.split('_'))
                field_tokens = set(field_name.split('_'))
                
                # Si comparten >50% de tokens, es probable que sean el mismo
                common = detected_tokens & field_tokens
                if len(common) >= max(1, len(detected_tokens) // 2):
                    matches[detected] = field_name
                    break
        
        return matches
    
    def get_critical_keypoints(self) -> list:
        """
        Retorna los nombres de keypoints más críticos para una buena homografía.
        
        Prioridad:
        1. Esquinas del campo (4 puntos no colineales → homografía completa)
        2. Intersecciones de área grande (geometría conocida)
        3. Círculo central (útil para verificación)
        
        Returns:
            Lista de nombres de keypoints ordenados por importancia
        """
        critical = [
            # Nivel 1: Esquinas del campo (CRÍTICO)
            'corner_bottom_left', 'corner_bottom_right', 
            'corner_top_left', 'corner_top_right',
            'corner_1', 'corner_2', 'corner_3', 'corner_4',
            
            # Nivel 2: Área grande (MUY ÚTIL)
            'big_area_left_bottom', 'big_area_left_top',
            'big_area_right_bottom', 'big_area_right_top',
            
            # Nivel 3: Puntos de penalti y círculo central (ÚTIL)
            'penalty_left', 'penalty_right',
            'center', 'circle_top', 'circle_bottom',
            
            # Nivel 4: Área pequeña (COMPLEMENTARIO)
            'small_area_left_bottom', 'small_area_left_top',
            'small_area_right_bottom', 'small_area_right_top',
        ]
        
        return critical
    
    def visualize(self, width: int = 800) -> np.ndarray:
        """
        Genera una imagen visualizando el campo y sus keypoints.
        
        Args:
            width: Ancho de la imagen en píxeles
            
        Returns:
            Imagen BGR del campo con keypoints
        """
        # Calcular dimensiones manteniendo aspect ratio
        aspect = self.field_width / self.field_length
        height = int(width * aspect)
        
        # Crear canvas
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Dibujar campo (verde)
        cv2.rectangle(img, (0, 0), (width-1, height-1), (34, 139, 34), -1)
        
        # Función helper para convertir coordenadas de campo a imagen
        def field_to_img(x, y):
            if self.use_normalized:
                ix = int(x * width)
                iy = int(y * height)
            else:
                ix = int((x / self.field_length) * width)
                iy = int((y / self.field_width) * height)
            return (ix, iy)
        
        # Dibujar líneas del campo
        # Borde
        cv2.rectangle(img, (0, 0), (width-1, height-1), (255, 255, 255), 2)
        
        # Línea central
        cv2.line(img, 
                field_to_img(self.field_length/2, 0),
                field_to_img(self.field_length/2, self.field_width),
                (255, 255, 255), 2)
        
        # Círculo central
        center_img = field_to_img(self.field_length/2, self.field_width/2)
        radius_img = int((self.center_circle_radius / self.field_length) * width)
        cv2.circle(img, center_img, radius_img, (255, 255, 255), 2)
        
        # Dibujar keypoints
        for name, (x, y) in self.keypoints_field.items():
            pt = field_to_img(x, y)
            cv2.circle(img, pt, 4, (0, 0, 255), -1)
        
        return img


if __name__ == '__main__':
    # Ejemplo de uso
    field_model = FieldModel()
    
    print("Field Model - Keypoints disponibles:")
    print(f"Total: {len(field_model.keypoints_field)} keypoints\n")
    
    # Mostrar keypoints críticos
    critical = field_model.get_critical_keypoints()
    print("Keypoints críticos para homografía:")
    for i, name in enumerate(critical[:10], 1):
        coords = field_model.get_keypoint_coords(name)
        if coords:
            print(f"  {i}. {name}: {coords}")
    
    # Visualizar campo
    print("\nGenerando visualización...")
    img = field_model.visualize(width=1000)
    cv2.imwrite('field_model.jpg', img)
    print("✓ Campo guardado en: field_model.jpg")
