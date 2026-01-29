"""
Field Calibrator - Calibración de homografía usando keypoints
==============================================================

Estima la matriz de homografía H entre imagen y campo real utilizando
los keypoints detectados por el modelo YOLO custom.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from modules.field_model_keypoints import FieldModel


class FieldCalibratorKeypoints:
    """
    Calibrador de campo usando keypoints detectados.
    
    Pipeline:
    1. Recibir keypoints en imagen (x_img, y_img)
    2. Matchear con keypoints del FieldModel (x_field, y_field)
    3. Estimar homografía H con RANSAC
    4. Proyectar puntos imagen → campo
    """
    
    def __init__(self, 
                 field_model: Optional[FieldModel] = None,
                 min_keypoints: int = 4,
                 ransac_threshold: float = 5.0,
                 confidence: float = 0.99):
        """
        Args:
            field_model: Modelo del campo con coordenadas de keypoints.
                        Si None, se usa FieldModel por defecto.
            min_keypoints: Mínimo de keypoints para estimar H (4 mínimo teórico)
            ransac_threshold: Threshold de RANSAC en píxeles (outlier rejection)
            confidence: Nivel de confianza de RANSAC (0-1)
        """
        self.field_model = field_model or FieldModel()
        self.min_keypoints = min_keypoints
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
        
        # Estado de calibración
        self.H = None  # Matriz de homografía 3x3
        self.H_inv = None  # Homografía inversa (campo → imagen)
        self.inliers_mask = None  # Máscara de inliers del último RANSAC
        self.matched_keypoints = {}  # Keypoints usados en la última calibración
        self.reprojection_error = None  # Error medio de reproyección
        
        # Sistema acumulativo de keypoints
        self.accumulated_keypoints = {}  # Acumula keypoints detectados a lo largo del tiempo
        self.keypoint_confidence = {}  # Confianza de cada keypoint acumulado
        self.max_accumulated_age = 300  # Frames máximos para mantener un keypoint
        self.keypoint_age = {}  # Edad de cada keypoint (frames desde última detección)
        
        # Historia de calibraciones (para filtrado temporal)
        self.calibration_history = []
        self.max_history = 5
    
    def estimate_homography(self, 
                           keypoints_img: Dict[str, Tuple[float, float]],
                           use_temporal_filtering: bool = True) -> bool:
        """
        Estima la homografía a partir de keypoints detectados en imagen.
        
        Sistema acumulativo: acumula keypoints de múltiples frames hasta
        tener suficientes para estimar la homografía.
        
        Args:
            keypoints_img: Dict {keypoint_name: (x_img, y_img)}
            use_temporal_filtering: Si True, promedia con calibraciones previas
            
        Returns:
            True si la calibración fue exitosa, False si falló
        """
        # 1. Actualizar keypoints acumulados con los nuevos detectados
        self._update_accumulated_keypoints(keypoints_img)
        
        # 2. Limpiar keypoints antiguos
        self._age_keypoints()
        
        # 3. Intentar estimar homografía con keypoints acumulados
        matched = self._match_keypoints(self.accumulated_keypoints)
        
        # Debug: mostrar matching
        if len(self.accumulated_keypoints) >= self.min_keypoints and len(matched) < self.min_keypoints:
            print(f"⚠ Problema de matching: {len(self.accumulated_keypoints)} acumulados → {len(matched)} matcheados")
            print(f"   Keypoints acumulados: {list(self.accumulated_keypoints.keys())[:5]}...")
            print(f"   Keypoints matcheados: {list(matched.keys())}")
        
        if len(matched) < self.min_keypoints:
            # No hay suficientes keypoints acumulados aún
            return False
        
        # 2. Preparar arrays para cv2.findHomography
        pts_img = []
        pts_field = []
        
        for kp_name, (x_img, y_img) in matched.items():
            coords_field = self.field_model.get_keypoint_coords(kp_name)
            if coords_field is not None:
                pts_img.append([x_img, y_img])
                pts_field.append(list(coords_field))
        
        pts_img = np.array(pts_img, dtype=np.float32)
        pts_field = np.array(pts_field, dtype=np.float32)
        
        # 3. Estimar homografía con RANSAC
        H, mask = cv2.findHomography(
            pts_img, 
            pts_field, 
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.confidence
        )
        
        if H is None:
            # Silenciar warning, es normal cuando hay pocos keypoints
            return False
        
        # 4. Validar calidad de la homografía
        num_inliers = np.sum(mask) if mask is not None else 0
        inlier_ratio = num_inliers / len(pts_img) if len(pts_img) > 0 else 0
        
        if num_inliers < self.min_keypoints:
            # Silenciar warning, es normal durante la fase inicial
            return False
        
        # 5. Calcular error de reproyección
        error = self._compute_reprojection_error(H, pts_img, pts_field, mask)
        
        if error > 20.0:  # Error muy alto (20 píxeles promedio)
            print(f"⚠ Error de reproyección alto: {error:.2f} px")
            return False
        
        # 6. Actualizar estado
        self.H = H
        self.H_inv = np.linalg.inv(H) if H is not None else None
        self.inliers_mask = mask
        self.matched_keypoints = matched
        self.reprojection_error = error
        
        # 7. Filtrado temporal (suavizar cambios bruscos)
        if use_temporal_filtering:
            self.calibration_history.append(H.copy())
            if len(self.calibration_history) > self.max_history:
                self.calibration_history.pop(0)
            
            # Promediar matrices de homografía recientes
            if len(self.calibration_history) >= 2:
                self.H = np.mean(self.calibration_history, axis=0)
                self.H_inv = np.linalg.inv(self.H)
        
        # Log de éxito (solo si es la primera vez o cada 100 calibraciones)
        if not hasattr(self, '_calibration_count'):
            self._calibration_count = 0
        
        self._calibration_count += 1
        
        # Mostrar solo la primera calibración, luego silenciar
        # (el sistema ya muestra resúmenes cada 30 frames en pruebatrackequipo.py)
        # if self._calibration_count == 1 or self._calibration_count % 100 == 0:
        #     print(f"✓ Calibración exitosa: {num_inliers}/{len(pts_img)} inliers, "
        #           f"error={error:.2f}px")
        
        return True
    
    def _update_accumulated_keypoints(self, new_keypoints: Dict[str, Tuple[float, float]]):
        """
        Actualiza el buffer de keypoints acumulados con nuevas detecciones.
        
        Args:
            new_keypoints: Keypoints recién detectados {name: (x, y)}
        """
        # Debug: mostrar keypoints detectados (primera vez que se detecta cada uno)
        new_detections = []
        for name, (x, y) in new_keypoints.items():
            if name not in self.accumulated_keypoints:
                new_detections.append(name)
            # Actualizar o agregar keypoint
            self.accumulated_keypoints[name] = (x, y)
            self.keypoint_age[name] = 0  # Resetear edad
        
        # Log de nuevos keypoints
        if new_detections:
            print(f"   Nuevos keypoints detectados: {new_detections}")
    
    def _age_keypoints(self):
        """
        Incrementa la edad de keypoints y elimina los muy antiguos.
        """
        to_remove = []
        
        for name in self.keypoint_age:
            self.keypoint_age[name] += 1
            
            # Eliminar si es muy antiguo
            if self.keypoint_age[name] > self.max_accumulated_age:
                to_remove.append(name)
        
        for name in to_remove:
            del self.accumulated_keypoints[name]
            del self.keypoint_age[name]
    
    def _match_keypoints(self, 
                        keypoints_img: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """
        Matchea keypoints detectados con los del modelo de campo.
        
        Usa estrategias de fuzzy matching para lidiar con diferencias
        en los nombres de keypoints entre el modelo de Roboflow y nuestro FieldModel.
        
        Args:
            keypoints_img: Keypoints detectados {name: (x, y)}
            
        Returns:
            Dict {standardized_name: (x, y)} con keypoints matcheados
        """
        matched = {}
        
        # Obtener mapping de nombres
        detected_names = list(keypoints_img.keys())
        name_mapping = self.field_model.match_keypoint_names(detected_names)
        
        # Aplicar mapping
        for detected_name, (x, y) in keypoints_img.items():
            if detected_name in name_mapping:
                standard_name = name_mapping[detected_name]
                matched[standard_name] = (x, y)
        
        # Priorizar keypoints críticos (esquinas, áreas)
        critical_kps = self.field_model.get_critical_keypoints()
        matched_critical = {k: v for k, v in matched.items() if k in critical_kps}
        
        # Si tenemos suficientes keypoints críticos, usarlos; si no, usar todos
        if len(matched_critical) >= self.min_keypoints:
            return matched_critical
        else:
            return matched
    
    def _compute_reprojection_error(self,
                                   H: np.ndarray,
                                   pts_img: np.ndarray,
                                   pts_field: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> float:
        """
        Calcula el error medio de reproyección.
        
        Args:
            H: Matriz de homografía
            pts_img: Puntos en imagen (N, 2)
            pts_field: Puntos en campo (N, 2)
            mask: Máscara de inliers (si None, usa todos)
            
        Returns:
            Error medio en píxeles
        """
        if mask is None:
            mask = np.ones(len(pts_img), dtype=bool)
        
        # Reproyectar puntos de imagen a campo
        pts_img_h = np.hstack([pts_img, np.ones((len(pts_img), 1))])  # Homogeneous
        pts_field_proj = (H @ pts_img_h.T).T
        pts_field_proj = pts_field_proj[:, :2] / pts_field_proj[:, 2:3]  # Normalize
        
        # Calcular distancias solo para inliers
        inlier_pts_field = pts_field[mask.ravel() > 0]
        inlier_pts_proj = pts_field_proj[mask.ravel() > 0]
        
        if len(inlier_pts_field) == 0:
            return float('inf')
        
        errors = np.linalg.norm(inlier_pts_field - inlier_pts_proj, axis=1)
        return np.mean(errors)
    
    def image_to_field(self, 
                      x_img: float, 
                      y_img: float) -> Optional[Tuple[float, float]]:
        """
        Proyecta un punto de la imagen al sistema de coordenadas del campo.
        
        Args:
            x_img, y_img: Coordenadas en imagen (píxeles)
            
        Returns:
            (X_field, Y_field) en coordenadas del campo (metros),
            o None si no hay calibración válida
        """
        if self.H is None:
            return None
        
        # Punto homogéneo
        pt_img = np.array([x_img, y_img, 1.0])
        
        # Aplicar homografía
        pt_field_h = self.H @ pt_img
        
        # Normalizar coordenadas homogéneas
        if pt_field_h[2] != 0:
            x_field = pt_field_h[0] / pt_field_h[2]
            y_field = pt_field_h[1] / pt_field_h[2]
            return (x_field, y_field)
        else:
            return None
    
    def field_to_image(self,
                      x_field: float,
                      y_field: float) -> Optional[Tuple[float, float]]:
        """
        Proyecta un punto del campo a la imagen (homografía inversa).
        
        Útil para overlay de anotaciones.
        
        Args:
            x_field, y_field: Coordenadas en campo (metros)
            
        Returns:
            (x_img, y_img) en píxeles, o None si no hay calibración
        """
        if self.H_inv is None:
            return None
        
        pt_field = np.array([x_field, y_field, 1.0])
        pt_img_h = self.H_inv @ pt_field
        
        if pt_img_h[2] != 0:
            x_img = pt_img_h[0] / pt_img_h[2]
            y_img = pt_img_h[1] / pt_img_h[2]
            return (x_img, y_img)
        else:
            return None
    
    def has_valid_calibration(self) -> bool:
        """Retorna True si hay una calibración válida."""
        return self.H is not None
    
    def get_calibration_info(self) -> dict:
        """
        Retorna información detallada de la última calibración.
        
        Útil para debugging y logging.
        """
        return {
            'has_calibration': self.has_valid_calibration(),
            'num_keypoints': len(self.matched_keypoints),
            'keypoint_names': list(self.matched_keypoints.keys()),
            'reprojection_error': self.reprojection_error,
            'num_inliers': int(np.sum(self.inliers_mask)) if self.inliers_mask is not None else 0,
            'homography_matrix': self.H.tolist() if self.H is not None else None
        }
    
    def visualize_calibration(self,
                             frame: np.ndarray,
                             keypoints_img: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """
        Visualiza la calibración sobre un frame.
        
        Dibuja:
        - Keypoints detectados
        - Líneas del campo reproyectadas
        - Info de calibración
        
        Args:
            frame: Frame BGR
            keypoints_img: Keypoints detectados en imagen
            
        Returns:
            Frame anotado
        """
        vis = frame.copy()
        
        # 1. Dibujar keypoints detectados
        for name, (x, y) in keypoints_img.items():
            color = (0, 255, 0) if name in self.matched_keypoints else (0, 0, 255)
            cv2.circle(vis, (int(x), int(y)), 5, color, -1)
            cv2.putText(vis, name[:10], (int(x)+10, int(y)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 2. Reproyectar líneas del campo si hay calibración
        if self.has_valid_calibration():
            self._draw_field_lines(vis)
        
        # 3. Info de calibración
        info_text = [
            f"Keypoints: {len(self.matched_keypoints)}/{len(keypoints_img)}",
            f"Error: {self.reprojection_error:.2f}px" if self.reprojection_error else "N/A",
            f"Status: {'OK' if self.has_valid_calibration() else 'FAIL'}"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(vis, text, (10, y_offset + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return vis
    
    def _draw_field_lines(self, img: np.ndarray):
        """Dibuja las líneas principales del campo reproyectadas."""
        if not self.has_valid_calibration():
            return
        
        # Líneas a dibujar (en coordenadas de campo)
        L = self.field_model.field_length
        W = self.field_model.field_width
        
        lines = [
            # Borde del campo
            [(0, 0), (L, 0)],
            [(L, 0), (L, W)],
            [(L, W), (0, W)],
            [(0, W), (0, 0)],
            # Línea central
            [(L/2, 0), (L/2, W)],
        ]
        
        # Reproyectar y dibujar
        for (x1, y1), (x2, y2) in lines:
            pt1_img = self.field_to_image(x1, y1)
            pt2_img = self.field_to_image(x2, y2)
            
            if pt1_img and pt2_img:
                cv2.line(img,
                        (int(pt1_img[0]), int(pt1_img[1])),
                        (int(pt2_img[0]), int(pt2_img[1])),
                        (0, 255, 255), 2)


if __name__ == '__main__':
    """
    Ejemplo de uso del calibrador.
    """
    # Simular keypoints detectados
    keypoints_img = {
        'corner_bottom_left': (100, 500),
        'corner_bottom_right': (1800, 480),
        'corner_top_left': (150, 50),
        'corner_top_right': (1750, 70),
        'center': (950, 270),
        'penalty_left': (400, 270),
    }
    
    # Crear calibrador
    field_model = FieldModel()
    calibrator = FieldCalibratorKeypoints(field_model)
    
    # Estimar homografía
    success = calibrator.estimate_homography(keypoints_img)
    
    if success:
        print("✓ Calibración exitosa!")
        print(calibrator.get_calibration_info())
        
        # Proyectar un punto de ejemplo
        x_img, y_img = 950, 270  # Centro de imagen
        x_field, y_field = calibrator.image_to_field(x_img, y_img)
        print(f"\nProyección: imagen({x_img}, {y_img}) → campo({x_field:.1f}, {y_field:.1f})")
    else:
        print("✗ Calibración falló")
