"""
Classic Field Calibration - Pipeline Completo de Calibración Clásica
=====================================================================

Este módulo integra todos los componentes del pipeline de calibración clásica:
1. Detección de líneas con acumulación temporal
2. Estimación de homografía desde líneas detectadas
3. Zonificación táctica del campo
4. Proyección de posiciones de jugadores a zonas

Diseñado para ser compatible con la interfaz de FieldCalibration existente
para facilitar la integración en el sistema principal.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from modules.field_line_detector import FieldLineDetector, LineSegment
from modules.homography_estimator import HomographyEstimator, FieldDimensions
from modules.field_zones import FieldZoneManager, FieldZone


class ClassicFieldCalibration:
    """
    Pipeline completo de calibración clásica del campo de fútbol.
    
    Arquitectura:
    - Detector de líneas con acumulación temporal
    - Estimador de homografía robusto (RANSAC)
    - Sistema de zonificación táctica
    - Proyección de jugadores a zonas
    
    Compatible con la interfaz de FieldCalibration para integración fácil.
    """
    
    def __init__(
        self,
        field_dims: FieldDimensions = None,
        temporal_window: int = 30,
        grid_cols: int = 6,
        grid_rows: int = 3,
        calibration_interval: int = 10,
        min_frames_for_calibration: int = 15,
        debug: bool = False
    ):
        """
        Args:
            field_dims: Dimensiones del campo
            temporal_window: Ventana temporal para acumulación (frames)
            grid_cols: Columnas del grid de zonificación
            grid_rows: Filas del grid de zonificación
            calibration_interval: Cada cuántos frames intentar calibrar
            min_frames_for_calibration: Mínimo de frames acumulados antes de calibrar
            debug: Modo debug
        """
        self.field_dims = field_dims or FieldDimensions()
        self.calibration_interval = calibration_interval
        # No permitir que el mínimo requerido supere la ventana temporal
        # (por ejemplo, `temporal_window` suele ser 30)
        self.min_frames_for_calibration = min(min_frames_for_calibration, temporal_window)
        self.debug = debug
        
        # Inicializar componentes
        self.line_detector = FieldLineDetector(
            temporal_window=temporal_window,
            debug=debug
        )
        
        self.homography_estimator = HomographyEstimator(
            field_dims=self.field_dims,
            debug=debug
        )
        
        self.zone_manager = FieldZoneManager(
            field_length=self.field_dims.length,
            field_width=self.field_dims.width,
            grid_cols=grid_cols,
            grid_rows=grid_rows
        )
        
        # Estado
        self.homography_matrix = None
        self.homography_inverse = None
        self.is_calibrated = False
        self.calibration_confidence = 0.0
        self.frame_count = 0
        self.last_mask = None
        self.last_lines = []
        
        # Estabilización temporal de homografía
        self.smoothed_homography = None
        self.smoothing_alpha = 0.2  # Factor de suavizado (0.1=muy suave, 1.0=sin filtro)
        
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Procesa un frame y actualiza la calibración si es necesario.
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            True si la calibración fue exitosa/actualizada
        """
        self.frame_count += 1
        
        # 1. Detectar líneas con acumulación temporal
        mask, lines = self.line_detector.process_frame(frame)
        self.last_mask = mask
        self.last_lines = lines
        
        # 2. Intentar calibrar cada N frames (o si aún no está calibrado)
        should_calibrate = (
            not self.is_calibrated or
            self.frame_count % self.calibration_interval == 0
        )
        
        if should_calibrate and len(self.line_detector.mask_buffer) >= self.min_frames_for_calibration:
            success = self._attempt_calibration(frame, lines)
            if success:
                return True
        
        return False
    
    def _attempt_calibration(
        self,
        frame: np.ndarray,
        lines: List[LineSegment]
    ) -> bool:
        """
        Intenta calibrar el campo usando las líneas detectadas.
        
        Args:
            frame: Frame actual
            lines: Líneas detectadas
            
        Returns:
            True si la calibración fue exitosa
        """
        if len(lines) < 4:
            if self.debug:
                print(f"Frame {self.frame_count}: Insuficientes líneas ({len(lines)})")
            return False
        
        # Comprobar que hay suficientes intersecciones antes de intentar estimar
        intersections = self.homography_estimator.find_line_intersections(lines)
        if self.debug:
            print(f"Frame {self.frame_count}: Encontradas {len(intersections)} intersecciones (pre-check)")

        # Requerir un mínimo razonable de intersecciones
        required_intersections = 8
        if len(intersections) < required_intersections:
            if self.debug:
                print(f"Frame {self.frame_count}: Insuficientes intersecciones ({len(intersections)} < {required_intersections}), salto calibración")
            return False

        # Estimar homografía
        homography = self.homography_estimator.estimate(
            lines, frame.shape[:2]
        )

        if homography is None:
            # Fallback: si ya teníamos una homografía válida, conservarla
            if self.homography_matrix is not None:
                if self.debug:
                    print(f"Frame {self.frame_count}: Fallo en estimación de homografía — manteniendo homografía previa")
                return False
            if self.debug:
                print(f"Frame {self.frame_count}: Fallo en estimación de homografía y sin homografía previa")
            return False
        
        # Validar homografía
        is_valid, confidence = self.homography_estimator.validate_homography(
            homography, frame.shape[:2]
        )
        
        if not is_valid:
            if self.debug:
                print(f"Frame {self.frame_count}: Homografía inválida")
            return False
        
        # Estabilización temporal
        if self.smoothed_homography is None:
            self.smoothed_homography = homography.copy()
        else:
            # Suavizar solo si la nueva homografía es similar
            # (evitar saltos bruscos por detecciones erróneas)
            center_pt = np.array([[[frame.shape[1] / 2, frame.shape[0] / 2]]], dtype=np.float32)
            old_center = cv2.perspectiveTransform(center_pt, self.smoothed_homography)
            new_center = cv2.perspectiveTransform(center_pt, homography)
            dist = np.linalg.norm(old_center - new_center)
            
            # Si la diferencia es pequeña (< 10m), suavizar
            if dist < 10.0:
                self.smoothed_homography = (
                    self.smoothing_alpha * homography +
                    (1 - self.smoothing_alpha) * self.smoothed_homography
                )
            else:
                # Cambio brusco: puede ser cambio de cámara, usar nueva homografía directamente
                self.smoothed_homography = homography.copy()
        
        # Actualizar estado
        self.homography_matrix = self.smoothed_homography
        try:
            self.homography_inverse = np.linalg.inv(self.homography_matrix)
        except:
            self.homography_inverse = None
        
        self.is_calibrated = True
        self.calibration_confidence = confidence
        
        if self.debug:
            print(f"Frame {self.frame_count}: Calibración exitosa (confianza: {confidence:.2f})")
        
        return True
    
    def pixel_to_meters(self, point_2d: np.ndarray) -> Optional[np.ndarray]:
        """
        Convierte un punto de la imagen (píxeles) a coordenadas del campo (metros).
        
        Args:
            point_2d: [x, y] en píxeles o array Nx2
            
        Returns:
            [x, y] en metros o array Nx2, o None si no está calibrado
        """
        if not self.is_calibrated or self.homography_matrix is None:
            return None
        
        point_2d = np.array(point_2d, dtype=np.float32)
        if point_2d.ndim == 1:
            point_2d = point_2d.reshape(1, 2)
        
        try:
            point_3d = cv2.perspectiveTransform(
                point_2d.reshape(-1, 1, 2),
                self.homography_matrix
            )
            result = point_3d.reshape(-1, 2)
            return result[0] if len(result) == 1 else result
        except Exception as e:
            if self.debug:
                print(f"Error en pixel_to_meters: {e}")
            return None
    
    def meters_to_pixel(self, point_3d: np.ndarray) -> Optional[np.ndarray]:
        """
        Convierte un punto del campo (metros) a coordenadas de la imagen (píxeles).
        
        Args:
            point_3d: [x, y] en metros o array Nx2
            
        Returns:
            [x, y] en píxeles o array Nx2, o None si no está calibrado
        """
        if not self.is_calibrated or self.homography_inverse is None:
            return None
        
        point_3d = np.array(point_3d, dtype=np.float32)
        if point_3d.ndim == 1:
            point_3d = point_3d.reshape(1, 2)
        
        try:
            point_2d = cv2.perspectiveTransform(
                point_3d.reshape(-1, 1, 2),
                self.homography_inverse
            )
            result = point_2d.reshape(-1, 2)
            return result[0] if len(result) == 1 else result
        except Exception as e:
            if self.debug:
                print(f"Error en meters_to_pixel: {e}")
            return None
    
    def get_player_zone(
        self,
        player_position_pixel: np.ndarray
    ) -> Optional[Tuple[FieldZone, Dict[str, any]]]:
        """
        Obtiene la zona táctica donde se encuentra un jugador.
        
        Args:
            player_position_pixel: [x, y] posición del jugador en píxeles
            
        Returns:
            (FieldZone, info_adicional) o None si fuera del campo/no calibrado
        """
        if not self.is_calibrated:
            return None
        
        # Convertir a metros
        position_meters = self.pixel_to_meters(player_position_pixel)
        if position_meters is None:
            return None
        
        # Obtener zona
        zone = self.zone_manager.get_zone(position_meters)
        if zone is None:
            return None
        
        # Información adicional
        info = {
            'position_meters': position_meters.tolist(),
            'position_pixel': player_position_pixel.tolist(),
            'zone_id': zone.zone_id,
            'zone_name': zone.name,
            'zone_type': zone.zone_type.value,
            'tactical_info': zone.tactical_info
        }
        
        return zone, info
    
    def draw_projected_pitch(
        self,
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Dibuja el campo proyectado sobre la imagen usando la homografía.
        
        Args:
            image: Frame BGR
            color: Color de las líneas (BGR)
            thickness: Grosor de las líneas
            
        Returns:
            Imagen con overlay del campo
        """
        if not self.is_calibrated or self.homography_inverse is None:
            return image
        
        img_copy = image.copy()
        
        try:
            def to_px(points_3d):
                """Convierte puntos del campo a píxeles"""
                points_3d = np.array(points_3d, dtype=np.float32)
                if points_3d.ndim == 1:
                    points_3d = points_3d.reshape(1, 2)
                points_2d = cv2.perspectiveTransform(
                    points_3d.reshape(-1, 1, 2),
                    self.homography_inverse
                )
                return points_2d.reshape(-1, 2).astype(np.int32)
            
            # 1. Perímetro del campo
            corners = np.array([
                [0, 0],
                [self.field_dims.length, 0],
                [self.field_dims.length, self.field_dims.width],
                [0, self.field_dims.width]
            ])
            cv2.polylines(img_copy, [to_px(corners)], True, color, thickness)
            
            # 2. Línea del medio campo
            center_line = np.array([
                [self.field_dims.length / 2, 0],
                [self.field_dims.length / 2, self.field_dims.width]
            ])
            cv2.line(img_copy,
                    tuple(to_px([self.field_dims.length / 2, 0])[0]),
                    tuple(to_px([self.field_dims.length / 2, self.field_dims.width])[0]),
                    color, thickness)
            
            # 3. Círculo central
            center = np.array([self.field_dims.length / 2, self.field_dims.width / 2])
            angles = np.linspace(0, 2 * np.pi, 30)
            circle_pts = np.array([
                [center[0] + self.field_dims.center_circle_radius * np.cos(a),
                 center[1] + self.field_dims.center_circle_radius * np.sin(a)]
                for a in angles
            ])
            cv2.polylines(img_copy, [to_px(circle_pts)], True, color, thickness)
            
            # 4. Áreas de penalti
            for x_start in [0, self.field_dims.length - self.field_dims.penalty_area_length]:
                y_top = (self.field_dims.width - self.field_dims.penalty_area_width) / 2
                y_bottom = y_top + self.field_dims.penalty_area_width
                x_end = x_start + self.field_dims.penalty_area_length
                
                pts = np.array([
                    [x_start, y_top],
                    [x_end, y_top],
                    [x_end, y_bottom],
                    [x_start, y_bottom]
                ])
                cv2.polylines(img_copy, [to_px(pts)], True, color, thickness)
            
        except Exception as e:
            if self.debug:
                print(f"Error dibujando campo: {e}")
        
        return img_copy
    
    def get_zone_info(self, position_meters: np.ndarray) -> Tuple[int, str]:
        """
        Compatibilidad con interfaz existente.
        
        Args:
            position_meters: [x, y] en metros
            
        Returns:
            (zone_id, zone_name)
        """
        zone = self.zone_manager.get_zone(position_meters)
        if zone is None:
            return 0, "Out of Field"
        return zone.zone_id, zone.name
    
    def reset(self):
        """Reinicia el estado de calibración"""
        self.line_detector.reset()
        self.homography_matrix = None
        self.homography_inverse = None
        self.is_calibrated = False
        self.calibration_confidence = 0.0
        self.frame_count = 0
        self.smoothed_homography = None
    
    def get_calibration_status(self) -> bool:
        """Retorna True si el campo está calibrado"""
        return self.is_calibrated
    
    def get_debug_visualization(self, frame: np.ndarray) -> np.ndarray:
        """
        Genera visualización completa de debug.
        
        Args:
            frame: Frame original
            
        Returns:
            Imagen con overlays de debug
        """
        vis = self.line_detector.get_debug_visualization(frame)
        
        if self.is_calibrated:
            vis = self.draw_projected_pitch(vis)
            
            # Mostrar información de calibración
            cv2.putText(vis, f"Calibrated (conf: {self.calibration_confidence:.2f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"Lines detected: {len(self.last_lines)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(vis, "Calibrating...",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis, f"Frames accumulated: {len(self.line_detector.mask_buffer)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return vis

