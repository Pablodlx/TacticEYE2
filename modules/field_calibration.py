"""
Field Calibration - Calibración automática del campo de fútbol
==============================================================
Detecta líneas, calcula homografía 2D→3D (píxeles → metros reales 105×68m)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class FieldDimensions:
    """Dimensiones estándar de campo FIFA"""
    length: float = 105.0  # metros
    width: float = 68.0    # metros
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.32
    goal_area_length: float = 5.5
    goal_area_width: float = 18.32
    center_circle_radius: float = 9.15


class FieldCalibration:
    """
    Sistema de calibración automática del campo
    Detecta líneas y calcula homografía para mapeo píxeles → metros reales
    """
    
    def __init__(self, field_dims: FieldDimensions = None):
        self.field_dims = field_dims or FieldDimensions()
        self.homography_matrix = None
        self.field_mask = None
        self.detected_lines = []
        self.field_corners_2d = None  # Esquinas en imagen
        self.field_corners_3d = None  # Esquinas en campo real
        
        # Sistema multi-frame
        self.homography_candidates = []  # Lista de (homography, quality_score)
        self.calibration_frames = []  # Frames donde se calibró
        self.best_homography = None
        self.last_calibration_frame = -1
        self.min_calibration_interval = 500  # Frames entre calibraciones
        
    def detect_field_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Detecta líneas del campo usando edge detection + Hough Transform
        
        Args:
            image: Frame BGR
            
        Returns:
            Imagen con líneas detectadas
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Crear máscara de campo verde (en HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Rango para césped verde (ajustar según iluminación)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morfología para limpiar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        self.field_mask = green_mask
        
        # Aplicar máscara a imagen gris
        masked_gray = cv2.bitwise_and(gray, gray, mask=green_mask)
        
        # Edge detection
        edges = cv2.Canny(masked_gray, 50, 150, apertureSize=3)
        
        # Hough Line Transform (detectar líneas)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=20
        )
        
        self.detected_lines = lines if lines is not None else []
        
        # Visualización
        line_image = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return line_image
    
    def find_field_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Encuentra las 4 esquinas del campo visible
        
        Returns:
            Array 4x2 con esquinas [top-left, top-right, bottom-right, bottom-left]
        """
        if self.field_mask is None:
            self.detect_field_lines(image)
        
        # Encontrar contorno del campo
        contours, _ = cv2.findContours(
            self.field_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Contorno más grande = campo
        field_contour = max(contours, key=cv2.contourArea)
        
        # Aproximar a polígono
        epsilon = 0.02 * cv2.arcLength(field_contour, True)
        approx = cv2.approxPolyDP(field_contour, epsilon, True)
        
        # Si tenemos 4 esquinas, perfecto
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            # Usar bounding box orientado
            rect = cv2.minAreaRect(field_contour)
            corners = cv2.boxPoints(rect).astype(np.float32)
        
        # Ordenar esquinas: TL, TR, BR, BL
        corners = self._order_corners(corners)
        self.field_corners_2d = corners
        
        return corners
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Ordena esquinas en sentido horario desde top-left"""
        # Centroide
        center = corners.mean(axis=0)
        
        # Ángulos desde centro
        angles = np.arctan2(corners[:, 1] - center[1], 
                           corners[:, 0] - center[0])
        
        # Ordenar por ángulo
        sorted_idx = np.argsort(angles)
        
        # Rotar para que top-left sea primero
        # Top-left tiene menor suma de coordenadas
        sums = corners[sorted_idx].sum(axis=1)
        tl_idx = np.argmin(sums)
        sorted_corners = np.roll(corners[sorted_idx], -tl_idx, axis=0)
        
        return sorted_corners
    
    def _compute_homography_quality(self, homography: np.ndarray, 
                                    src_points: np.ndarray) -> float:
        """
        Calcula puntuación de calidad de homografía (0-1)
        Basado en: condición de matriz, simetría, y distorsión
        """
        if homography is None:
            return 0.0
        
        try:
            # 1. Condición de matriz (no debe estar mal condicionada)
            cond = np.linalg.cond(homography)
            cond_score = 1.0 / (1.0 + np.log10(max(cond, 1.0)))
            
            # 2. Determinante (no debe ser cercano a 0)
            det = abs(np.linalg.det(homography))
            det_score = min(1.0, det / 0.1)
            
            # 3. Verificar que esquinas transformadas sean razonables
            dst_test = cv2.perspectiveTransform(src_points.reshape(1, -1, 2), homography)
            dst_test = dst_test.reshape(-1, 2)
            
            # Las esquinas deben formar un rectángulo razonable
            # Verificar que no haya inversiones o distorsiones extremas
            areas = []
            for i in range(4):
                p1 = dst_test[i]
                p2 = dst_test[(i + 1) % 4]
                p3 = dst_test[(i + 2) % 4]
                # Área del triángulo
                area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                          (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
            
            area_variance = np.std(areas) / (np.mean(areas) + 1e-6)
            area_score = 1.0 / (1.0 + area_variance)
            
            # Score combinado
            quality = (cond_score * 0.3 + det_score * 0.3 + area_score * 0.4)
            return float(quality)
            
        except:
            return 0.0
    
    def compute_homography(self, image: np.ndarray, 
                          manual_corners: Optional[np.ndarray] = None,
                          frame_number: int = 0) -> bool:
        """
        Calcula matriz de homografía píxeles → metros (con sistema multi-frame)
        
        Args:
            image: Frame de referencia
            manual_corners: Opcional - esquinas manuales 4x2 [x,y]
            frame_number: Número de frame actual
            
        Returns:
            True si homografía calculada exitosamente
        """
        # Esquinas en imagen (2D)
        if manual_corners is not None:
            src_points = manual_corners.astype(np.float32)
        else:
            src_points = self.find_field_corners(image)
        
        if src_points is None:
            print("⚠ No se pudieron detectar esquinas del campo")
            return False
        
        self.field_corners_2d = src_points
        
        # Esquinas en campo real (3D, vista top-down)
        # Asumimos que vemos campo completo o casi completo
        # Coordenadas en metros: (0,0) = esquina inferior izquierda
        dst_points = np.array([
            [0, self.field_dims.width],                      # Top-left
            [self.field_dims.length, self.field_dims.width], # Top-right
            [self.field_dims.length, 0],                     # Bottom-right
            [0, 0]                                           # Bottom-left
        ], dtype=np.float32)
        
        self.field_corners_3d = dst_points
        
        # Calcular homografía
        H, _ = cv2.findHomography(src_points, dst_points)
        
        if H is not None:
            # Calcular calidad
            quality = self._compute_homography_quality(H, src_points)
            
            # Guardar candidato
            self.homography_candidates.append((H, quality, frame_number))
            self.calibration_frames.append(frame_number)
            self.last_calibration_frame = frame_number
            
            # Si es la mejor hasta ahora, actualizarla
            if self.best_homography is None or quality > self.best_homography[1]:
                self.best_homography = (H, quality, frame_number)
                self.homography_matrix = H
                print(f"✓ Nueva mejor homografía (calidad: {quality:.3f}, frame: {frame_number})")
            else:
                print(f"✓ Homografía calculada (calidad: {quality:.3f}, frame: {frame_number})")
            
            return True
        else:
            print("⚠ Error calculando homografía")
            return False
    
    def pixel_to_meters(self, point_2d: np.ndarray) -> Optional[np.ndarray]:
        """
        Convierte punto en píxeles a coordenadas en metros
        
        Args:
            point_2d: Array [x, y] o Nx2
            
        Returns:
            Coordenadas en metros [x, y] o Nx2, None si no hay homografía
        """
        if self.homography_matrix is None:
            return None
        
        # Asegurar formato correcto
        point_2d = np.array(point_2d, dtype=np.float32)
        
        if point_2d.ndim == 1:
            point_2d = point_2d.reshape(1, 2)
        
        # Aplicar homografía
        point_3d = cv2.perspectiveTransform(
            point_2d.reshape(-1, 1, 2),
            self.homography_matrix
        )
        
        result = point_3d.reshape(-1, 2)
        return result[0] if len(result) == 1 else result
    
    def meters_to_pixel(self, point_3d: np.ndarray) -> Optional[np.ndarray]:
        """Convierte coordenadas en metros a píxeles (inversa)"""
        if self.homography_matrix is None:
            return None
        
        inv_h = np.linalg.inv(self.homography_matrix)
        
        point_3d = np.array(point_3d, dtype=np.float32)
        if point_3d.ndim == 1:
            point_3d = point_3d.reshape(1, 2)
        
        point_2d = cv2.perspectiveTransform(
            point_3d.reshape(-1, 1, 2),
            inv_h
        )
        
        result = point_2d.reshape(-1, 2)
        return result[0] if len(result) == 1 else result
    
    def create_topdown_view(self, output_size: Tuple[int, int] = (1050, 680)) -> np.ndarray:
        """
        Crea imagen del campo vista desde arriba (vista cenital)
        
        Args:
            output_size: (width, height) en píxeles (10px = 1m)
            
        Returns:
            Imagen BGR del campo dibujado
        """
        w, h = output_size
        field_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Color verde césped
        field_img[:] = (34, 139, 34)  # Forest green
        
        # Escala: píxeles por metro
        scale_x = w / self.field_dims.length
        scale_y = h / self.field_dims.width
        
        def to_px(x_m, y_m):
            """Convierte metros a píxeles en imagen top-down"""
            return (int(x_m * scale_x), int(h - y_m * scale_y))
        
        # Color de líneas
        line_color = (255, 255, 255)
        line_thickness = 2
        
        # Perímetro del campo
        cv2.rectangle(field_img, to_px(0, 0), to_px(self.field_dims.length, self.field_dims.width),
                     line_color, line_thickness)
        
        # Línea central
        mid_x = self.field_dims.length / 2
        cv2.line(field_img, to_px(mid_x, 0), to_px(mid_x, self.field_dims.width),
                line_color, line_thickness)
        
        # Círculo central
        center = to_px(mid_x, self.field_dims.width / 2)
        radius = int(self.field_dims.center_circle_radius * scale_x)
        cv2.circle(field_img, center, radius, line_color, line_thickness)
        cv2.circle(field_img, center, 3, line_color, -1)  # Punto central
        
        # Áreas de penalti (ambos lados)
        for side in [0, self.field_dims.length]:
            if side == 0:
                # Área izquierda
                x1 = 0
                x2 = self.field_dims.penalty_area_length
            else:
                # Área derecha
                x1 = self.field_dims.length - self.field_dims.penalty_area_length
                x2 = self.field_dims.length
            
            y_center = self.field_dims.width / 2
            y1 = y_center - self.field_dims.penalty_area_width / 2
            y2 = y_center + self.field_dims.penalty_area_width / 2
            
            cv2.rectangle(field_img, to_px(x1, y1), to_px(x2, y2),
                         line_color, line_thickness)
            
            # Área pequeña (goal area)
            if side == 0:
                gx2 = self.field_dims.goal_area_length
            else:
                gx2 = x1
                x1 = self.field_dims.length - self.field_dims.goal_area_length
            
            gy1 = y_center - self.field_dims.goal_area_width / 2
            gy2 = y_center + self.field_dims.goal_area_width / 2
            
            cv2.rectangle(field_img, to_px(x1, gy1), to_px(gx2, gy2),
                         line_color, line_thickness)
        
        return field_img
    
    def project_positions_to_topdown(self, 
                                    positions_2d: np.ndarray,
                                    topdown_size: Tuple[int, int] = (1050, 680)) -> np.ndarray:
        """
        Proyecta posiciones de imagen a coordenadas en vista top-down
        
        Args:
            positions_2d: Nx2 array de posiciones [x,y] en píxeles de imagen original
            topdown_size: Tamaño de imagen top-down
            
        Returns:
            Nx2 array de posiciones [x,y] en píxeles de imagen top-down
        """
        # Convertir a metros
        positions_3d = self.pixel_to_meters(positions_2d)
        
        if positions_3d is None:
            return None
        
        # Asegurar que sea 2D
        if positions_3d.ndim == 1:
            positions_3d = positions_3d.reshape(1, -1)
        
        # Escalar a píxeles de top-down
        w, h = topdown_size
        scale_x = w / self.field_dims.length
        scale_y = h / self.field_dims.width
        
        topdown_positions = np.zeros_like(positions_3d)
        topdown_positions[:, 0] = positions_3d[:, 0] * scale_x
        topdown_positions[:, 1] = h - positions_3d[:, 1] * scale_y
        
        return topdown_positions.astype(np.int32)
    
    def refine_homography(self, min_quality: float = 0.3) -> bool:
        """
        Refina homografía usando promedio ponderado de mejores candidatos
        
        Args:
            min_quality: Calidad mínima para incluir candidato
            
        Returns:
            True si se refinó exitosamente
        """
        if len(self.homography_candidates) < 2:
            return False
        
        # Filtrar por calidad
        good_candidates = [(H, q, f) for H, q, f in self.homography_candidates if q >= min_quality]
        
        if len(good_candidates) < 2:
            print(f"⚠ Solo {len(good_candidates)} candidatos de calidad suficiente")
            return False
        
        # Ordenar por calidad
        good_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Usar top 5 mejores
        top_candidates = good_candidates[:min(5, len(good_candidates))]
        
        # Promedio ponderado por calidad
        weights = np.array([q for _, q, _ in top_candidates])
        weights = weights / weights.sum()
        
        H_refined = np.zeros((3, 3))
        for (H, q, f), w in zip(top_candidates, weights):
            H_refined += w * H
        
        # Normalizar
        H_refined = H_refined / H_refined[2, 2]
        
        self.homography_matrix = H_refined
        print(f"✓ Homografía refinada con {len(top_candidates)} candidatos")
        print(f"  Calidades: {[f'{q:.3f}' for _, q, _ in top_candidates]}")
        
        return True
    
    def should_recalibrate(self, frame_number: int) -> bool:
        """
        Determina si debe intentar recalibrar en este frame
        """
        if self.last_calibration_frame < 0:
            return True
        
        frames_since_last = frame_number - self.last_calibration_frame
        return frames_since_last >= self.min_calibration_interval
    
    def is_calibrated(self) -> bool:
        """Verifica si el sistema está calibrado"""
        return self.homography_matrix is not None
    
    def get_calibration_info(self) -> dict:
        """Retorna información sobre la calibración actual"""
        info = {
            'calibrated': self.is_calibrated(),
            'field_dims': f"{self.field_dims.length}x{self.field_dims.width}m",
            'has_homography': self.homography_matrix is not None,
            'num_calibrations': len(self.homography_candidates),
            'calibration_frames': self.calibration_frames
        }
        
        if self.best_homography is not None:
            info['best_quality'] = self.best_homography[1]
            info['best_frame'] = self.best_homography[2]
        
        return info
