"""
FieldLineDetector - Detección automática de líneas del campo

Combina técnicas clásicas de visión por computador para detectar
líneas blancas del campo sin anotación manual.

Pipeline:
1. Preprocesamiento: filtrado de color blanco + máscaras
2. Detección de líneas: LSD (Line Segment Detector) o Hough
3. Filtrado y agrupación: clustering de líneas paralelas/perpendiculares
4. Clasificación: matching contra modelo de campo
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class LineSegment:
    """Segmento de línea detectado en la imagen"""
    x1: float
    y1: float
    x2: float
    y2: float
    angle: float  # Ángulo en radianes
    length: float
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def direction(self) -> Tuple[float, float]:
        """Vector director normalizado"""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 0:
            return (dx / norm, dy / norm)
        return (0, 0)


class FieldLineDetector:
    """
    Detector automático de líneas del campo de fútbol.
    
    Estrategia:
    - Usa detección de color blanco + morfología para encontrar líneas
    - Aplica LSD (Line Segment Detector) de OpenCV
    - Agrupa líneas por orientación (horizontales, verticales, diagonales)
    - Filtra por longitud y posición esperada
    """
    
    def __init__(self,
                 white_threshold_low: int = 130,
                 white_threshold_high: int = 255,
                 min_line_length: int = 35,
                 max_line_gap: int = 10,
                 angle_tolerance: float = 10.0):
        """
        Args:
            white_threshold_low: Umbral inferior para detección de blanco
            white_threshold_high: Umbral superior
            min_line_length: Longitud mínima de línea a detectar (píxeles)
            max_line_gap: Gap máximo para unir segmentos
            angle_tolerance: Tolerancia en grados para agrupar líneas paralelas
        """
        self.white_threshold_low = white_threshold_low
        self.white_threshold_high = white_threshold_high
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.angle_tolerance = np.deg2rad(angle_tolerance)
        
        # Crear detector LSD
        self.lsd = cv2.createLineSegmentDetector(0)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa imagen para resaltar líneas blancas del campo.
        
        Args:
            image: Frame RGB/BGR
            
        Returns:
            Máscara binaria con líneas blancas
        """
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Filtro bilateral para reducir ruido manteniendo bordes
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Umbralización para detectar líneas blancas
        _, white_mask = cv2.threshold(
            denoised, 
            self.white_threshold_low, 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Morfología para limpiar y conectar líneas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        return white_mask
    
    def detect_lines(self, image: np.ndarray) -> List[LineSegment]:
        """
        Detecta segmentos de línea en la imagen.
        
        Args:
            image: Frame RGB/BGR
            
        Returns:
            Lista de LineSegment detectados
        """
        # Preprocesar
        mask = self.preprocess_image(image)
        
        # Detectar líneas con LSD
        lines = self.lsd.detect(mask)[0]
        
        if lines is None:
            return []
        
        # Convertir a LineSegment
        segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calcular propiedades
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            
            # Filtrar por longitud mínima
            if length >= self.min_line_length:
                segments.append(LineSegment(x1, y1, x2, y2, angle, length))
        
        return segments
    
    def cluster_lines_by_orientation(self, 
                                     segments: List[LineSegment]) -> Dict[str, List[LineSegment]]:
        """
        Agrupa líneas por orientación (horizontal, vertical, diagonal).
        
        Args:
            segments: Lista de segmentos detectados
            
        Returns:
            Diccionario con keys: 'horizontal', 'vertical', 'diagonal_pos', 'diagonal_neg'
        """
        clusters = {
            'horizontal': [],
            'vertical': [],
            'diagonal_pos': [],
            'diagonal_neg': []
        }
        
        for seg in segments:
            # Normalizar ángulo a [-π/2, π/2]
            angle = seg.angle
            if angle > np.pi / 2:
                angle -= np.pi
            elif angle < -np.pi / 2:
                angle += np.pi
            
            # Clasificar por ángulo
            if abs(angle) < self.angle_tolerance:
                clusters['horizontal'].append(seg)
            elif abs(abs(angle) - np.pi/2) < self.angle_tolerance:
                clusters['vertical'].append(seg)
            elif angle > 0:
                clusters['diagonal_pos'].append(seg)
            else:
                clusters['diagonal_neg'].append(seg)
        
        return clusters
    
    def merge_collinear_segments(self, segments: List[LineSegment]) -> List[LineSegment]:
        """
        Fusiona segmentos colineales cercanos en líneas más largas.
        
        Args:
            segments: Lista de segmentos (debe ser de orientación similar)
            
        Returns:
            Lista de segmentos fusionados
        """
        if len(segments) == 0:
            return []
        
        # Ordenar por punto medio en eje principal
        if len(segments[0].direction) == 2:
            # Determinar eje principal (x para verticales, y para horizontales)
            avg_angle = np.mean([s.angle for s in segments])
            use_y = abs(avg_angle) < np.pi / 4
            
            segments_sorted = sorted(
                segments, 
                key=lambda s: s.midpoint[1] if use_y else s.midpoint[0]
            )
        else:
            segments_sorted = segments
        
        merged = []
        current = segments_sorted[0]
        
        for next_seg in segments_sorted[1:]:
            # Calcular distancia entre segmentos
            dist = np.sqrt(
                (current.midpoint[0] - next_seg.midpoint[0])**2 + 
                (current.midpoint[1] - next_seg.midpoint[1])**2
            )
            
            # Si están cerca, fusionar
            if dist < self.max_line_gap * 2:
                # Extender el segmento actual
                x_coords = [current.x1, current.x2, next_seg.x1, next_seg.x2]
                y_coords = [current.y1, current.y2, next_seg.y1, next_seg.y2]
                
                x1 = min(x_coords)
                x2 = max(x_coords)
                y1_at_x1 = current.y1 if abs(current.x1 - x1) < 1 else next_seg.y1
                y2_at_x2 = current.y2 if abs(current.x2 - x2) < 1 else next_seg.y2
                
                dx = x2 - x1
                dy = y2_at_x2 - y1_at_x1
                length = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)
                
                current = LineSegment(x1, y1_at_x1, x2, y2_at_x2, angle, length)
            else:
                merged.append(current)
                current = next_seg
        
        merged.append(current)
        return merged
    
    def detect_and_classify(self, image: np.ndarray) -> Dict[str, List[LineSegment]]:
        """
        Pipeline completo: detectar y clasificar líneas del campo.
        
        Args:
            image: Frame RGB/BGR
            
        Returns:
            Diccionario con líneas agrupadas por tipo
        """
        # Detectar líneas
        segments = self.detect_lines(image)
        
        if len(segments) == 0:
            return {
                'horizontal': [],
                'vertical': [],
                'diagonal_pos': [],
                'diagonal_neg': []
            }
        
        # Agrupar por orientación
        clusters = self.cluster_lines_by_orientation(segments)
        
        # Fusionar segmentos colineales en cada grupo
        for key in clusters:
            clusters[key] = self.merge_collinear_segments(clusters[key])
        
        return clusters
    
    def visualize_lines(self, 
                       image: np.ndarray, 
                       clusters: Dict[str, List[LineSegment]]) -> np.ndarray:
        """
        Visualiza las líneas detectadas sobre la imagen.
        
        Args:
            image: Imagen original
            clusters: Líneas agrupadas
            
        Returns:
            Imagen con líneas dibujadas
        """
        vis = image.copy()
        
        colors = {
            'horizontal': (0, 255, 0),      # Verde
            'vertical': (255, 0, 0),        # Azul
            'diagonal_pos': (0, 255, 255),  # Amarillo
            'diagonal_neg': (255, 0, 255)   # Magenta
        }
        
        for line_type, segments in clusters.items():
            color = colors.get(line_type, (128, 128, 128))
            for seg in segments:
                pt1 = (int(seg.x1), int(seg.y1))
                pt2 = (int(seg.x2), int(seg.y2))
                cv2.line(vis, pt1, pt2, color, 2)
        
        return vis


class KeypointMatcher:
    """
    Matcher para encontrar correspondencias entre líneas detectadas
    y el modelo del campo.
    
    Estrategia:
    - Busca intersecciones de líneas (candidatos a esquinas/puntos clave)
    - Compara geometría local con modelo de campo
    - Usa RANSAC-like approach para encontrar mejor conjunto consistente
    """
    
    def __init__(self, field_model):
        self.field_model = field_model
    
    def find_line_intersections(self, 
                                lines_h: List[LineSegment], 
                                lines_v: List[LineSegment]) -> List[Tuple[float, float]]:
        """
        Encuentra intersecciones entre líneas horizontales y verticales.
        
        Returns:
            Lista de puntos (x, y) de intersección
        """
        intersections = []
        
        for h_line in lines_h:
            for v_line in lines_v:
                # Calcular intersección
                x1, y1, x2, y2 = h_line.x1, h_line.y1, h_line.x2, h_line.y2
                x3, y3, x4, y4 = v_line.x1, v_line.y1, v_line.x2, v_line.y2
                
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                
                if abs(denom) < 1e-10:
                    continue
                
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                
                # Verificar que la intersección está dentro de ambos segmentos
                if 0 <= t <= 1 and 0 <= u <= 1:
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    intersections.append((x, y))
        
        return intersections
    
    def match_to_field_model(self, 
                            detected_points: List[Tuple[float, float]],
                            image_shape: Tuple[int, int]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Encuentra correspondencias entre puntos detectados y puntos del modelo.
        
        Args:
            detected_points: Puntos detectados en imagen
            image_shape: (height, width) de la imagen
            
        Returns:
            Lista de tuplas: (punto_imagen, punto_campo)
        """
        # Obtener keypoints del modelo
        field_keypoints = self.field_model.get_all_keypoints()
        
        # TODO: Implementar matching robusto
        # Por ahora, placeholder que requiere implementación específica
        # basada en geometría local y patrones del campo
        
        correspondences = []
        
        # Esta es una versión simplificada - en producción necesitaría
        # análisis geométrico más sofisticado
        
        return correspondences
