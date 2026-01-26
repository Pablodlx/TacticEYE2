"""
FieldCalibrator - Calibración automática del campo y estimación de homografía

Combina detección de líneas con optimización para estimar la transformación
imagen → campo sin intervención manual.

Pipeline:
1. Detectar líneas y keypoints candidatos en imagen
2. Usar modelo pre-entrenado (opcional) o heurísticas geométricas
3. Encontrar correspondencias imagen ↔ campo
4. Estimar homografía con RANSAC
5. Refinar y estabilizar temporalmente
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from collections import deque

from .field_model import FieldModel
from .field_line_detector import FieldLineDetector, KeypointMatcher


@dataclass
class HomographyEstimate:
    """Resultado de estimación de homografía"""
    H: np.ndarray  # Matriz 3x3
    confidence: float  # 0-1
    num_inliers: int
    reprojection_error: float
    frame_idx: int


class HomographyFilter:
    """
    Filtro temporal para suavizar homografías entre frames.
    
    Usa un buffer de homografías recientes y promedia parámetros
    para evitar jitter cuando la cámara está estática.
    """
    
    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.6):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.history = deque(maxlen=window_size)
        self.last_good_H = None
    
    def update(self, estimate: Optional[HomographyEstimate]) -> Optional[np.ndarray]:
        """
        Actualiza el filtro con nueva estimación.
        
        Args:
            estimate: Nueva estimación de homografía (o None si falló)
            
        Returns:
            Homografía filtrada, o última buena si la actual no es confiable
        """
        if estimate is not None and estimate.confidence >= self.confidence_threshold:
            self.history.append(estimate)
            self.last_good_H = estimate.H
        
        # Si no hay suficientes datos, usar la última buena
        if len(self.history) < 2:
            return self.last_good_H
        
        # Promediar homografías recientes (ponderado por confianza)
        weights = np.array([e.confidence for e in self.history])
        weights /= weights.sum()
        
        H_avg = np.zeros((3, 3))
        for w, estimate in zip(weights, self.history):
            H_avg += w * estimate.H
        
        # Normalizar para que H[2,2] = 1
        H_avg /= H_avg[2, 2]
        
        return H_avg
    
    def get_current(self) -> Optional[np.ndarray]:
        """Retorna la homografía actual filtrada"""
        if len(self.history) > 0:
            return self.update(None)
        return self.last_good_H


class FieldCalibrator:
    """
    Calibrador automático del campo de fútbol.
    
    Funcionalidades:
    - Estima homografía imagen → campo automáticamente
    - Mantiene calibración temporal con filtrado
    - Maneja cambios de cámara y zoom
    - Provee API simple para reproyección de puntos
    
    Ejemplo de uso:
        >>> calibrator = FieldCalibrator()
        >>> H = calibrator.estimate_homography(frame)
        >>> x_field, y_field = calibrator.image_to_field(x_img, y_img)
    """
    
    def __init__(self, 
                 field_model: Optional[FieldModel] = None,
                 use_temporal_filter: bool = True,
                 min_confidence: float = 0.5):
        """
        Args:
            field_model: Modelo del campo (usa estándar si es None)
            use_temporal_filter: Activar filtrado temporal
            min_confidence: Confianza mínima para aceptar estimación
        """
        self.field_model = field_model or FieldModel()
        self.line_detector = FieldLineDetector()
        self.keypoint_matcher = KeypointMatcher(self.field_model)
        
        self.use_temporal_filter = use_temporal_filter
        self.min_confidence = min_confidence
        
        if use_temporal_filter:
            self.h_filter = HomographyFilter(
                window_size=5,
                confidence_threshold=min_confidence
            )
        else:
            self.h_filter = None
        
        self.current_H = None
        self.frame_count = 0
    
    def estimate_homography(self, 
                           frame: np.ndarray,
                           force_recompute: bool = False) -> Optional[np.ndarray]:
        """
        Estima homografía para el frame dado.
        
        Args:
            frame: Frame RGB/BGR de la transmisión
            force_recompute: Forzar recálculo aunque haya homografía válida
            
        Returns:
            Matriz H 3x3 (imagen → campo), o None si falló
        """
        self.frame_count += 1
        
        # Si ya tenemos homografía y no forzamos recálculo, usar la actual
        if self.current_H is not None and not force_recompute:
            # Solo recalcular cada N frames para eficiencia
            if self.frame_count % 30 != 0:
                return self.current_H
        
        # 1. Detectar líneas del campo
        line_clusters = self.line_detector.detect_and_classify(frame)
        
        # Verificar que tengamos suficientes líneas
        total_lines = sum(len(v) for v in line_clusters.values())
        if total_lines < 4:
            # No suficientes líneas detectadas
            if self.h_filter:
                return self.h_filter.get_current()
            return self.current_H
        
        # 2. Encontrar puntos de intersección (candidatos a keypoints)
        intersections = self.keypoint_matcher.find_line_intersections(
            line_clusters['horizontal'],
            line_clusters['vertical']
        )
        
        if len(intersections) < 4:
            # Necesitamos al menos 4 puntos para homografía
            if self.h_filter:
                return self.h_filter.get_current()
            return self.current_H
        
        # 3. Matchear con modelo de campo
        correspondences = self.keypoint_matcher.match_to_field_model(
            intersections,
            frame.shape[:2]
        )
        
        if len(correspondences) < 4:
            # No pudimos matchear suficientes puntos
            # Usar estrategia de fallback (heurística simple)
            correspondences = self._fallback_matching(
                line_clusters, 
                intersections,
                frame.shape
            )
        
        if len(correspondences) < 4:
            if self.h_filter:
                return self.h_filter.get_current()
            return self.current_H
        
        # 4. Estimar homografía con RANSAC
        estimate = self._estimate_homography_ransac(correspondences)
        
        if estimate is None:
            if self.h_filter:
                return self.h_filter.get_current()
            return self.current_H
        
        # 5. Filtrar temporalmente si está activado
        if self.h_filter:
            self.current_H = self.h_filter.update(estimate)
        else:
            if estimate.confidence >= self.min_confidence:
                self.current_H = estimate.H
        
        return self.current_H
    
    def _fallback_matching(self, 
                          line_clusters: Dict,
                          intersections: List[Tuple[float, float]],
                          image_shape: Tuple[int, int]) -> List[Tuple]:
        """
        Estrategia de fallback cuando el matching automático falla.
        
        Usa heurísticas simples basadas en posiciones típicas de broadcast:
        - Línea de banda superior/inferior
        - Líneas de área
        - Intersecciones típicas
        """
        correspondences = []
        
        h, w = image_shape[:2]
        
        # Heurística: si vemos el campo desde arriba en broadcast típico,
        # las líneas horizontales superiores/inferiores suelen ser bandas
        # y las verticales cercanas a los bordes son líneas de gol/área
        
        horizontal = line_clusters.get('horizontal', [])
        vertical = line_clusters.get('vertical', [])
        
        # Ordenar líneas horizontales por posición Y
        horizontal_sorted = sorted(horizontal, key=lambda l: l.midpoint[1])
        
        if len(horizontal_sorted) >= 2:
            # Línea superior → banda superior (asumiendo vista estándar)
            top_line = horizontal_sorted[0]
            # Línea inferior → banda inferior
            bottom_line = horizontal_sorted[-1]
            
            # Mapear a coordenadas de campo
            field_width = self.field_model.dims.width
            field_length = self.field_model.dims.length
            
            # Puntos de ejemplo (simplificado - en producción sería más sofisticado)
            # Aquí asumiríamos vista de medio campo o similar
            
            # Esta es una implementación placeholder
            # En producción, usaríamos ML o geometría proyectiva más avanzada
        
        return correspondences
    
    def _estimate_homography_ransac(self, 
                                   correspondences: List[Tuple]) -> Optional[HomographyEstimate]:
        """
        Estima homografía usando RANSAC.
        
        Args:
            correspondences: Lista de (punto_imagen, punto_campo)
            
        Returns:
            HomographyEstimate o None si falló
        """
        if len(correspondences) < 4:
            return None
        
        # Separar puntos de imagen y campo
        img_pts = np.array([c[0] for c in correspondences], dtype=np.float32)
        field_pts = np.array([c[1] for c in correspondences], dtype=np.float32)
        
        # Estimar con RANSAC
        H, mask = cv2.findHomography(
            img_pts, 
            field_pts, 
            cv2.RANSAC,
            ransacReprojThreshold=5.0
        )
        
        if H is None:
            return None
        
        # Calcular métricas de confianza
        inliers = np.sum(mask)
        total = len(mask)
        confidence = inliers / total if total > 0 else 0.0
        
        # Error de reproyección
        img_pts_reproj = cv2.perspectiveTransform(
            field_pts.reshape(-1, 1, 2), 
            np.linalg.inv(H)
        ).reshape(-1, 2)
        
        errors = np.linalg.norm(img_pts - img_pts_reproj, axis=1)
        avg_error = np.mean(errors[mask.ravel() == 1])
        
        return HomographyEstimate(
            H=H,
            confidence=confidence,
            num_inliers=inliers,
            reprojection_error=avg_error,
            frame_idx=self.frame_count
        )
    
    def image_to_field(self, x_img: float, y_img: float) -> Optional[Tuple[float, float]]:
        """
        Transforma punto de imagen a coordenadas de campo.
        
        Args:
            x_img, y_img: Coordenadas en píxeles de imagen
            
        Returns:
            (x_field, y_field) en metros (sistema del campo), o None si no hay homografía
        """
        if self.current_H is None:
            return None
        
        # Aplicar homografía
        pt_img = np.array([[[x_img, y_img]]], dtype=np.float32)
        pt_field = cv2.perspectiveTransform(pt_img, self.current_H)
        
        x_field = pt_field[0, 0, 0]
        y_field = pt_field[0, 0, 1]
        
        return (x_field, y_field)
    
    def image_to_field_batch(self, 
                            points_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Transforma múltiples puntos de imagen a campo.
        
        Args:
            points_img: Array Nx2 de puntos (x, y) en imagen
            
        Returns:
            Array Nx2 de puntos en campo, o None
        """
        if self.current_H is None:
            return None
        
        points_img = points_img.reshape(-1, 1, 2).astype(np.float32)
        points_field = cv2.perspectiveTransform(points_img, self.current_H)
        
        return points_field.reshape(-1, 2)
    
    def field_to_image(self, x_field: float, y_field: float) -> Optional[Tuple[float, float]]:
        """
        Transforma punto de campo a imagen (inversa).
        
        Args:
            x_field, y_field: Coordenadas en metros
            
        Returns:
            (x_img, y_img) en píxeles, o None
        """
        if self.current_H is None:
            return None
        
        H_inv = np.linalg.inv(self.current_H)
        
        pt_field = np.array([[[x_field, y_field]]], dtype=np.float32)
        pt_img = cv2.perspectiveTransform(pt_field, H_inv)
        
        return (pt_img[0, 0, 0], pt_img[0, 0, 1])
    
    def has_valid_calibration(self) -> bool:
        """Retorna True si hay calibración válida disponible"""
        return self.current_H is not None
    
    def reset(self):
        """Resetea el calibrador (útil al cambiar de video/cámara)"""
        self.current_H = None
        self.frame_count = 0
        if self.h_filter:
            self.h_filter = HomographyFilter(
                window_size=5,
                confidence_threshold=self.min_confidence
            )
    
    def visualize_calibration(self, 
                             frame: np.ndarray,
                             draw_grid: bool = True) -> np.ndarray:
        """
        Visualiza la calibración actual sobre el frame.
        
        Args:
            frame: Frame original
            draw_grid: Si True, dibuja rejilla del campo proyectada
            
        Returns:
            Frame con visualización
        """
        if self.current_H is None:
            return frame
        
        vis = frame.copy()
        
        if draw_grid:
            # Dibujar líneas principales del campo proyectadas en imagen
            field_lines = self.field_model.get_all_lines()
            H_inv = np.linalg.inv(self.current_H)
            
            for line_name, line_pts in field_lines.items():
                if len(line_pts) == 2:
                    # Proyectar puntos de campo a imagen
                    pt1_field = np.array([[line_pts[0]]], dtype=np.float32)
                    pt2_field = np.array([[line_pts[1]]], dtype=np.float32)
                    
                    pt1_img = cv2.perspectiveTransform(pt1_field, H_inv)
                    pt2_img = cv2.perspectiveTransform(pt2_field, H_inv)
                    
                    x1, y1 = int(pt1_img[0, 0, 0]), int(pt1_img[0, 0, 1])
                    x2, y2 = int(pt2_img[0, 0, 0]), int(pt2_img[0, 0, 1])
                    
                    # Verificar que estén dentro de la imagen
                    h, w = frame.shape[:2]
                    if (0 <= x1 < w and 0 <= y1 < h) or (0 <= x2 < w and 0 <= y2 < h):
                        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return vis
