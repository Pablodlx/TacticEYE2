"""
Homography Estimator - Estimación de Homografía desde Líneas Detectadas
========================================================================

Este módulo estima la homografía imagen → campo de fútbol usando las líneas
detectadas. Funciona incluso cuando no están visibles las 4 esquinas del campo,
usando intersecciones de líneas y correspondencias geométricas conocidas.

Estrategia:
1. Detectar intersecciones de líneas (puntos de interés)
2. Identificar líneas conocidas del campo (medio, áreas, perímetro)
3. Establecer correspondencias con modelo del campo
4. Estimar homografía con RANSAC (robusto a outliers)
5. Validar y refinar la homografía

Diseñado para funcionar con información parcial (media cancha visible).
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from modules.field_line_detector import LineSegment


@dataclass
class FieldDimensions:
    """Dimensiones estándar de un campo de fútbol (FIFA)"""
    length: float = 105.0  # metros
    width: float = 68.0    # metros
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.32
    goal_area_length: float = 5.5
    goal_area_width: float = 18.32
    center_circle_radius: float = 9.15


@dataclass
class LineIntersection:
    """Intersección entre dos líneas"""
    point: np.ndarray  # [x, y] en imagen
    line1_idx: int
    line2_idx: int
    confidence: float = 1.0


class HomographyEstimator:
    """
    Estima homografía desde líneas detectadas usando correspondencias geométricas.
    
    No requiere ver las 4 esquinas del campo. Funciona con:
    - Media cancha visible
    - Solo líneas centrales
    - Áreas parcialmente visibles
    """
    
    def __init__(
        self,
        field_dims: FieldDimensions = None,
        ransac_threshold: float = 5.0,
        ransac_max_iters: int = 2000,
        min_correspondences: int = 4,
        debug: bool = False
    ):
        """
        Args:
            field_dims: Dimensiones del campo
            ransac_threshold: Umbral para RANSAC (píxeles)
            ransac_max_iters: Máximo de iteraciones RANSAC
            min_correspondences: Mínimo de correspondencias necesarias
            debug: Modo debug
        """
        self.field_dims = field_dims or FieldDimensions()
        self.ransac_threshold = ransac_threshold
        self.ransac_max_iters = ransac_max_iters
        self.min_correspondences = min_correspondences
        self.debug = debug
        
        # Generar modelo del campo (líneas conocidas en coordenadas del mundo)
        self.field_model = self._generate_field_model()
        
        # Estado
        self.last_homography = None
        self.last_confidence = 0.0
        
    def _generate_field_model(self) -> Dict[str, List[np.ndarray]]:
        """
        Genera modelo geométrico del campo en coordenadas del mundo (metros).
        
        Retorna diccionario con líneas conocidas:
        - 'perimeter': Líneas del perímetro
        - 'center': Línea del medio campo
        - 'penalty_areas': Líneas de áreas de penalti
        - 'goal_areas': Líneas de áreas de portería
        - 'center_circle': Puntos del círculo central
        """
        model = {
            'perimeter': [],
            'center': [],
            'penalty_areas': [],
            'goal_areas': [],
            'center_circle': []
        }
        
        # Perímetro (4 líneas)
        # Abajo (Y=0)
        model['perimeter'].append(np.array([[0, 0], [self.field_dims.length, 0]]))
        # Arriba (Y=width)
        model['perimeter'].append(np.array([[0, self.field_dims.width], 
                                           [self.field_dims.length, self.field_dims.width]]))
        # Izquierda (X=0)
        model['perimeter'].append(np.array([[0, 0], [0, self.field_dims.width]]))
        # Derecha (X=length)
        model['perimeter'].append(np.array([[self.field_dims.length, 0], 
                                           [self.field_dims.length, self.field_dims.width]]))
        
        # Línea del medio campo (vertical)
        center_x = self.field_dims.length / 2
        model['center'].append(np.array([[center_x, 0], [center_x, self.field_dims.width]]))
        
        # Áreas de penalti (rectángulos)
        for x_start in [0, self.field_dims.length - self.field_dims.penalty_area_length]:
            # Línea superior del área
            y_top = (self.field_dims.width - self.field_dims.penalty_area_width) / 2
            y_bottom = y_top + self.field_dims.penalty_area_width
            x_end = x_start + self.field_dims.penalty_area_length
            
            model['penalty_areas'].append(np.array([[x_start, y_top], [x_end, y_top]]))
            model['penalty_areas'].append(np.array([[x_start, y_bottom], [x_end, y_bottom]]))
            model['penalty_areas'].append(np.array([[x_start, y_top], [x_start, y_bottom]]))
            model['penalty_areas'].append(np.array([[x_end, y_top], [x_end, y_bottom]]))
        
        # Áreas de portería (similar)
        for x_start in [0, self.field_dims.length - self.field_dims.goal_area_length]:
            y_top = (self.field_dims.width - self.field_dims.goal_area_width) / 2
            y_bottom = y_top + self.field_dims.goal_area_width
            x_end = x_start + self.field_dims.goal_area_length
            
            model['goal_areas'].append(np.array([[x_start, y_top], [x_end, y_top]]))
            model['goal_areas'].append(np.array([[x_start, y_bottom], [x_end, y_bottom]]))
            model['goal_areas'].append(np.array([[x_start, y_top], [x_start, y_bottom]]))
            model['goal_areas'].append(np.array([[x_end, y_top], [x_end, y_bottom]]))
        
        # Círculo central (discretizado en puntos)
        center = np.array([center_x, self.field_dims.width / 2])
        angles = np.linspace(0, 2 * np.pi, 32)
        for angle in angles:
            point = center + self.field_dims.center_circle_radius * np.array([
                np.cos(angle), np.sin(angle)
            ])
            model['center_circle'].append(point)
        
        return model
    
    def find_line_intersections(
        self,
        lines: List[LineSegment]
    ) -> List[LineIntersection]:
        """
        Encuentra intersecciones entre líneas detectadas.
        
        Las intersecciones son puntos de interés que pueden corresponder
        a esquinas del campo, esquinas de áreas, etc.
        
        Args:
            lines: Lista de líneas detectadas
            
        Returns:
            Lista de intersecciones encontradas
        """
        intersections = []
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], start=i+1):
                # Calcular intersección de dos líneas
                p1 = np.array([line1.x1, line1.y1])
                p2 = np.array([line1.x2, line1.y2])
                p3 = np.array([line2.x1, line2.y1])
                p4 = np.array([line2.x2, line2.y2])
                
                # Intersección de segmentos usando álgebra lineal
                # Línea 1: p1 + t*(p2-p1)
                # Línea 2: p3 + s*(p4-p3)
                d1 = p2 - p1
                d2 = p4 - p3
                
                # Resolver sistema: p1 + t*d1 = p3 + s*d2
                # t*d1 - s*d2 = p3 - p1
                A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
                b = p3 - p1
                
                try:
                    t, s = np.linalg.solve(A, b)
                    
                    # Verificar que la intersección está dentro de ambos segmentos
                    if 0 <= t <= 1 and 0 <= s <= 1:
                        intersection_point = p1 + t * d1
                        
                        # Verificar que no es muy cercana a una ya encontrada
                        is_duplicate = False
                        for existing in intersections:
                            dist = np.linalg.norm(intersection_point - existing.point)
                            if dist < 10.0:  # 10 píxeles de tolerancia
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            intersections.append(LineIntersection(
                                point=intersection_point,
                                line1_idx=i,
                                line2_idx=j,
                                confidence=1.0
                            ))
                except np.linalg.LinAlgError:
                    # Líneas paralelas o casi paralelas
                    continue
        
        return intersections
    
    def identify_field_lines(
        self,
        lines: List[LineSegment],
        image_shape: Tuple[int, int]
    ) -> Dict[str, List[int]]:
        """
        Identifica qué líneas detectadas corresponden a líneas conocidas del campo.
        
        Usa heurísticas basadas en:
        - Longitud
        - Ángulo
        - Posición en la imagen
        
        Args:
            lines: Líneas detectadas
            image_shape: (height, width)
            
        Returns:
            Diccionario con índices de líneas identificadas por tipo
        """
        h, w = image_shape
        identified = {
            'center': [],      # Línea del medio campo (vertical)
            'horizontal': [],  # Líneas horizontales (perímetro, áreas)
            'vertical': [],    # Líneas verticales (perímetro, áreas)
            'diagonal': []     # Líneas diagonales (áreas)
        }
        
        for i, line in enumerate(lines):
            angle = line.angle
            
            # Línea vertical (medio campo o perímetro lateral)
            if 80 < angle < 100:
                # Si está cerca del centro horizontal de la imagen, probablemente es el medio
                mid_x = (line.x1 + line.x2) / 2
                if abs(mid_x - w/2) < w * 0.15:  # 15% de tolerancia
                    identified['center'].append(i)
                else:
                    identified['vertical'].append(i)
            
            # Línea horizontal (perímetro o áreas)
            elif angle < 15 or angle > 165:
                identified['horizontal'].append(i)
            
            # Línea diagonal (áreas de penalti)
            elif (30 < angle < 60) or (120 < angle < 150):
                identified['diagonal'].append(i)
        
        return identified
    
    def establish_correspondences(
        self,
        lines: List[LineSegment],
        intersections: List[LineIntersection],
        identified: Dict[str, List[int]],
        image_shape: Tuple[int, int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Establece correspondencias entre puntos de la imagen y puntos del modelo del campo.
        
        Estrategia:
        1. Usar intersecciones como puntos de correspondencia
        2. Usar puntos medios de líneas identificadas
        3. Combinar ambos para tener suficientes correspondencias
        
        Args:
            lines: Líneas detectadas
            intersections: Intersecciones encontradas
            identified: Líneas identificadas por tipo
            
        Returns:
            (puntos_imagen, puntos_modelo) como arrays Nx2
        """
        image_points = []
        model_points = []

        # Imagen dimensiones para mapeos proporcionales
        h_img, w_img = image_shape
        
        # === ESTRATEGIA 1: Intersecciones ===
        # Las intersecciones pueden corresponder a esquinas del campo o áreas
        for inter in intersections[:20]:  # Limitar a las 20 mejores
            image_points.append(inter.point)

        # Si no tenemos puntos modelo asignados para estas intersecciones,
        # generar correspondencias aproximadas mapeando la posición de la
        # intersección en la imagen a coordenadas del campo proporcionalmente.
        # Esto aporta diversidad en X/Y cuando las heurísticas posteriores
        # asignan muchos puntos con la misma coordenada Y.
        if len(model_points) < len(image_points):
            for i in range(len(model_points), len(image_points)):
                pt = image_points[i]
                # Normalizar según tamaño de imagen y mapear a dimensiones del campo
                x_ratio = np.clip(pt[0] / w_img, 0.0, 1.0)
                y_ratio = np.clip(pt[1] / h_img, 0.0, 1.0)
                model_x = x_ratio * self.field_dims.length
                model_y = y_ratio * self.field_dims.width
                model_points.append(np.array([model_x, model_y]))
            # Por ahora, no sabemos exactamente qué punto del modelo es
            # Lo resolveremos con RANSAC
        
        # === ESTRATEGIA 2: Puntos medios de líneas identificadas ===
        # Línea del medio campo
        if identified['center']:
            center_line_idx = identified['center'][0]
            center_line = lines[center_line_idx]
            mid_point = np.array([
                (center_line.x1 + center_line.x2) / 2,
                (center_line.y1 + center_line.y2) / 2
            ])
            image_points.append(mid_point)
            
            # En el modelo, el medio campo está en X=52.5, Y variable
            # Usamos el punto medio vertical del campo
            model_points.append(np.array([self.field_dims.length / 2, 
                                         self.field_dims.width / 2]))
        
        # Líneas horizontales (probablemente perímetro o áreas)
        h_img, w_img = image_shape
        for idx in identified['horizontal'][:4]:  # Máximo 4
            line = lines[idx]
            mid_point = np.array([
                (line.x1 + line.x2) / 2,
                (line.y1 + line.y2) / 2
            ])
            image_points.append(mid_point)

            # Estimación aproximada de coordenadas en el modelo:
            # - model_y se estima por la posición vertical relativa
            # - model_x se estima proporcionalmente a la posición horizontal
            y_ratio = mid_point[1] / h_img
            if y_ratio < 0.3:
                model_y = 0.0
            elif y_ratio > 0.7:
                model_y = self.field_dims.width
            else:
                model_y = self.field_dims.width / 2

            # Estimar X en el modelo según la posición X en la imagen
            x_ratio = np.clip(mid_point[0] / w_img, 0.0, 1.0)
            model_x = x_ratio * self.field_dims.length

            model_points.append(np.array([model_x, model_y]))
        
        # Si no tenemos suficientes correspondencias, generar candidatos del modelo
        if len(image_points) < self.min_correspondences:
            # Generar puntos candidatos del modelo (esquinas, centro, etc.)
            model_candidates = [
                np.array([0, 0]),
                np.array([self.field_dims.length, 0]),
                np.array([self.field_dims.length, self.field_dims.width]),
                np.array([0, self.field_dims.width]),
                np.array([self.field_dims.length / 2, self.field_dims.width / 2]),
                np.array([0, self.field_dims.width / 2]),
                np.array([self.field_dims.length, self.field_dims.width / 2]),
            ]
            
            # Usar intersecciones como puntos imagen
            for i, inter in enumerate(intersections[:len(model_candidates)]):
                image_points.append(inter.point)
                model_points.append(model_candidates[i % len(model_candidates)])

        # Asegurar que image_points y model_points tengan la misma longitud.
        # Durante la recolección inicial se pueden añadir solo puntos imagen
        # (por ejemplo, al iterar intersecciones) sin su correspondiente
        # punto modelo, lo que provoca desajustes y errores de indexación
        # posterior en RANSAC. Truncamos al menor tamaño común.
        if len(image_points) != len(model_points):
            n = min(len(image_points), len(model_points))
            if n == 0:
                return np.array([]), np.array([])
            image_points = image_points[:n]
            model_points = model_points[:n]

        return np.array(image_points), np.array(model_points)
    
    def estimate_homography_ransac(
        self,
        image_points: np.ndarray,
        model_points: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Estima homografía usando RANSAC para robustez ante outliers.
        
        Args:
            image_points: Puntos en la imagen (Nx2)
            model_points: Puntos correspondientes en el modelo (Nx2)
            
        Returns:
            Matriz de homografía 3x3 o None si falla
        """
        if len(image_points) < 4:
            return None
        
        # RANSAC para encontrar la mejor homografía
        best_homography = None
        best_inliers = 0
        
        for _ in range(self.ransac_max_iters):
            # Seleccionar 4 puntos aleatorios
            indices = np.random.choice(len(image_points), 4, replace=False)
            src_pts = image_points[indices]
            dst_pts = model_points[indices]
            
            try:
                # Calcular homografía con estos 4 puntos
                H = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Validar homografía
                if H is None or np.any(np.isnan(H)) or np.any(np.isinf(H)):
                    continue
                
                # Contar inliers
                # Transformar todos los puntos imagen al modelo
                ones = np.ones((len(image_points), 1))
                image_homogeneous = np.hstack([image_points, ones])
                transformed = (H @ image_homogeneous.T).T
                transformed = transformed[:, :2] / transformed[:, 2:3]
                
                # Calcular distancias
                errors = np.linalg.norm(transformed - model_points, axis=1)
                inliers = np.sum(errors < self.ransac_threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_homography = H
            except:
                continue
        
        # Si encontramos una buena homografía, refinarla con todos los inliers
        if best_homography is not None and best_inliers >= 4:
            # Recalcular con todos los inliers
            ones = np.ones((len(image_points), 1))
            image_homogeneous = np.hstack([image_points, ones])
            transformed = (best_homography @ image_homogeneous.T).T
            transformed = transformed[:, :2] / transformed[:, 2:3]
            errors = np.linalg.norm(transformed - model_points, axis=1)
            inlier_mask = errors < self.ransac_threshold
            
            if np.sum(inlier_mask) >= 4:
                inlier_image = image_points[inlier_mask]
                inlier_model = model_points[inlier_mask]
                best_homography = cv2.findHomography(
                    inlier_image, inlier_model,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_threshold,
                    maxIters=self.ransac_max_iters
                )[0]

        # Fallback: si no encontramos homografía con el muestreo aleatorio,
        # intentar directamente cv2.findHomography sobre todas las correspondencias
        # usando RANSAC — a veces el muestreo previo no converge pero
        # findHomography sí puede encontrar un buen consenso.
        if best_homography is None and len(image_points) >= 4:
            try:
                H_fallback = cv2.findHomography(
                    image_points.astype(np.float32),
                    model_points.astype(np.float32),
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_threshold,
                    maxIters=self.ransac_max_iters
                )[0]
                if H_fallback is not None and not (np.any(np.isnan(H_fallback)) or np.any(np.isinf(H_fallback))):
                    best_homography = H_fallback
            except Exception:
                pass

        return best_homography
    
    def validate_homography(
        self,
        homography: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[bool, float]:
        """
        Valida que la homografía es razonable.
        
        Checks:
        - Determinante positivo (no refleja)
        - Condición numérica razonable
        - Esquinas transformadas dentro de límites esperados
        
        Args:
            homography: Matriz de homografía 3x3
            image_shape: (height, width)
            
        Returns:
            (es_válida, confianza)
        """
        if homography is None:
            return False, 0.0
        
        # Check 1: Determinante positivo (evitar reflejos)
        det = np.linalg.det(homography[:2, :2])
        if det <= 0:
            return False, 0.0
        
        # Check 2: Condición numérica
        cond = np.linalg.cond(homography)
        if cond > 1e6:
            return False, 0.0
        
        # Check 3: Transformar esquinas de la imagen
        h, w = image_shape
        corners_image = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype=np.float32)
        
        corners_world = cv2.perspectiveTransform(
            corners_image.reshape(-1, 1, 2),
            homography
        ).reshape(-1, 2)
        
        # Las esquinas deben estar dentro de límites razonables del campo
        # (con margen para campos parcialmente visibles)
        margin = 20.0  # metros de margen
        x_min, x_max = -margin, self.field_dims.length + margin
        y_min, y_max = -margin, self.field_dims.width + margin
        
        valid_corners = 0
        for corner in corners_world:
            if x_min <= corner[0] <= x_max and y_min <= corner[1] <= y_max:
                valid_corners += 1
        
        # Al menos 2 esquinas deben ser válidas (campo parcialmente visible OK)
        if valid_corners < 2:
            return False, 0.0
        
        # Calcular confianza basada en número de esquinas válidas y condición
        confidence = (valid_corners / 4.0) * (1.0 / (1.0 + cond / 1e5))
        
        return True, confidence
    
    def estimate(
        self,
        lines: List[LineSegment],
        image_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Estima homografía completa desde líneas detectadas.
        
        Pipeline:
        1. Encontrar intersecciones
        2. Identificar líneas del campo
        3. Establecer correspondencias
        4. Estimar con RANSAC
        5. Validar
        
        Args:
            lines: Líneas detectadas
            image_shape: (height, width)
            
        Returns:
            Matriz de homografía 3x3 o None
        """
        if len(lines) < 4:
            return None
        
        # 1. Encontrar intersecciones
        intersections = self.find_line_intersections(lines)
        
        if self.debug:
            print(f"Encontradas {len(intersections)} intersecciones")
        
        # 2. Identificar líneas del campo
        identified = self.identify_field_lines(lines, image_shape)
        
        # 3. Establecer correspondencias
        image_points, model_points = self.establish_correspondences(
            lines, intersections, identified, image_shape
        )

        # Debug: informar sobre correspondencias encontradas
        if self.debug:
            try:
                print(f"Debug: image_points.shape={getattr(image_points, 'shape', None)}, "
                      f"model_points.shape={getattr(model_points, 'shape', None)}")
                if len(image_points) > 0 and len(model_points) > 0:
                    print(f"Debug: sample image_points (first 5): {image_points[:5]}")
                    print(f"Debug: sample model_points (first 5): {model_points[:5]}")
            except Exception:
                pass

        if len(image_points) < self.min_correspondences:
            if self.debug:
                print(f"Fallo en establish_correspondences: correspondencias insuficientes ({len(image_points)})")
            return None
        
        # 4. Estimar homografía con RANSAC
        homography = self.estimate_homography_ransac(image_points, model_points)
        
        if homography is None:
            return None
        
        # 5. Validar
        is_valid, confidence = self.validate_homography(homography, image_shape)
        
        if not is_valid:
            return None
        
        self.last_homography = homography
        self.last_confidence = confidence
        
        return homography

