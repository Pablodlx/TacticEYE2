"""
Field Line Detector - Detección Clásica de Líneas con Acumulación Temporal
===========================================================================

Este módulo implementa un detector robusto de líneas del campo de fútbol usando
técnicas clásicas de visión por computador (OpenCV). La clave es la acumulación
temporal de máscaras para eliminar oclusiones causadas por jugadores y árbitros.

Arquitectura:
1. Segmentación de líneas blancas (HSV/LAB)
2. Exclusión del césped (máscara verde)
3. Acumulación temporal de máscaras (ventana deslizante)
4. Detección de líneas (LSD + Hough como respaldo)
5. Filtrado geométrico de líneas relevantes

Diseñado para ser reemplazable por una red de segmentación deep learning
sin romper el pipeline (interfaz consistente).
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
from dataclasses import dataclass


@dataclass
class LineSegment:
    """Representa un segmento de línea detectado"""
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle: float  # En grados, 0-180
    confidence: float = 1.0


class FieldLineDetector:
    """
    Detector de líneas del campo con acumulación temporal.
    
    Estrategia:
    - Acumula máscaras de líneas durante N frames
    - Las líneas del campo aparecen consistentemente
    - Los jugadores se mueven y desaparecen de la acumulación
    - Resultado: máscara limpia de líneas del campo
    """
    def __init__(
        self,
        temporal_window: int = 30,
        min_line_length: float = 20.0,
        max_line_gap: float = 10.0,
        use_lsd: bool = True,
        use_hough: bool = True,
        hough_threshold: int = 50,
        line_merge_threshold: float = 5.0,
        min_component_area: int = 15,
        min_component_aspect: float = 2.0,
        debug: bool = False,
    ):
        """
        Args:
            temporal_window: Número de frames para acumular máscaras (default: 30)
            min_line_length: Longitud mínima de línea en píxeles
            max_line_gap: Máximo gap permitido en línea segmentada
            use_lsd: Usar Line Segment Detector (más preciso, más lento)
            use_hough: Usar HoughLinesP como respaldo/complemento
            hough_threshold: Umbral para HoughLinesP
            line_merge_threshold: Distancia para fusionar líneas similares
            debug: Mostrar imágenes intermedias
        """
        self.temporal_window = temporal_window
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.use_lsd = use_lsd
        self.use_hough = use_hough
        self.hough_threshold = hough_threshold
        self.line_merge_threshold = line_merge_threshold
        self.min_component_area = min_component_area
        self.min_component_aspect = min_component_aspect
        self.debug = debug
        
        # Buffer circular para acumulación temporal
        self.mask_buffer: deque = deque(maxlen=temporal_window)
        
        # Estadísticas
        self.frame_count = 0
        self.last_mask = None
        self.last_lines = []
        
    def extract_lines_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae máscara binaria de líneas blancas del campo.
        
        Pipeline:
        1. Convertir a espacio de color apropiado (HSV/LAB)
        2. Segmentar líneas blancas
        3. Excluir áreas que no son césped
        4. Filtrado morfológico
        
        Args:
            image: Frame BGR de OpenCV
            
        Returns:
            Máscara binaria (uint8) donde 255 = línea detectada
        """
        h, w = image.shape[:2]
        
        # Reducir resolución para velocidad (mantener aspect ratio)
        scale_factor = 0.5 if w > 1280 else 1.0
        if scale_factor < 1.0:
            small = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
        else:
            small = image.copy()
        
        h_small, w_small = small.shape[:2]
        
        # === PASO 1: MÁSCARA DE CÉSPED ===
        # Solo buscamos líneas dentro del área de césped
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        # Rango HSV para césped (verde)
        # H: 35-85 (verde), S: 40-255, V: 40-255
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Dilatar para incluir bordes donde puede haber líneas
        # Reducir kernel para evitar incluir demasiadas regiones externas
        kernel_grass = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        grass_mask = cv2.dilate(grass_mask, kernel_grass, iterations=1)
        
        # === PASO 2: DETECCIÓN DE LÍNEAS BLANCAS ===
        # Método 1: Top-hat morfológico (destaca líneas claras sobre fondo oscuro)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Top-hat con kernel horizontal (para líneas horizontales)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        tophat_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_h)
        
        # Top-hat con kernel vertical (para líneas verticales)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        tophat_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_v)
        
        # Combinar ambos
        tophat_combined = cv2.max(tophat_h, tophat_v)
        
        # Método 2: Segmentación por color (blanco en HSV)
        # Blanco: S bajo (< 50), V alto (> 200)
        lower_white_hsv = np.array([0, 0, 200])
        upper_white_hsv = np.array([180, 50, 255])
        white_mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)
        
        # Método 3: LAB color space (mejor para detectar blancos)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # En LAB: L alto (> 190) indica blanco (relajar umbral ligeramente)
        _, white_mask_lab = cv2.threshold(l_channel, 190, 255, cv2.THRESH_BINARY)
        
        # Combinar todos los métodos
        combined_mask = cv2.bitwise_or(
            cv2.bitwise_or(white_mask_hsv, white_mask_lab),
            cv2.threshold(tophat_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        )
        
        # Aplicar máscara de césped (solo líneas dentro del campo)
        lines_mask = cv2.bitwise_and(combined_mask, grass_mask)
        
        # === PASO 3: FILTRADO MORFOLÓGICO ===
        # Eliminar ruido pequeño
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_OPEN, kernel_noise, iterations=1)
        
        # Conectar líneas cercanas
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_CLOSE, kernel_connect, iterations=2)
        
        # Escalar de vuelta si fue reducido
        if scale_factor < 1.0:
            lines_mask = cv2.resize(lines_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # ==== FILTRADO POR COMPONENTES (AREA + ELONGACIÓN) ====
        # Eliminamos blobs que no son lineales (p. ej. brazos, jugadores),
        # conservando componentes alargadas que representan líneas blancas.
        mask_proc = lines_mask.copy()
        contours, _ = cv2.findContours(mask_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keep_mask = np.zeros_like(mask_proc)

        kept = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, ww, hh = cv2.boundingRect(cnt)
            # evitar división por cero
            min_side = min(ww, hh) if min(ww, hh) > 0 else 1
            aspect = max(ww, hh) / float(min_side)

            # calcular convex hull solidity como medida adicional
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            # Condición de línea: aceptar si
            #  - componente suficientemente grande (area), OR
            #  - componente muy alargada (aspect >= min_component_aspect)
            # Además requerimos cierta solidez para evitar blobs muy sueltos
            if (area >= self.min_component_area or aspect >= self.min_component_aspect) and solidity > 0.15:
                cv2.drawContours(keep_mask, [cnt], -1, 255, -1)
                kept += 1

        if self.debug:
            print(f"FieldLineDetector: contours_total={len(contours)}, contours_kept={kept}")

        # Reemplazar mask por la máscara filtrada
        lines_mask = keep_mask

        return lines_mask
    
    def accumulate_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Acumula máscaras en buffer temporal y retorna máscara acumulada.
        
        La idea: las líneas del campo son estáticas (aparecen en todos los frames),
        mientras que los jugadores se mueven (aparecen y desaparecen).
        Al acumular, las líneas se refuerzan y el ruido se reduce.
        
        Args:
            mask: Máscara binaria del frame actual
            
        Returns:
            Máscara acumulada (normalizada 0-255)
        """
        self.mask_buffer.append(mask.copy())
        self.frame_count += 1
        
        if len(self.mask_buffer) == 0:
            return mask
        
        # Sumar todas las máscaras del buffer
        accumulated = np.zeros_like(mask, dtype=np.float32)
        for m in self.mask_buffer:
            accumulated += (m.astype(np.float32) / 255.0)
        
        # Normalizar: cuanto más frames, más confianza
        # Usamos umbral adaptativo: línea debe aparecer en al menos 30% de frames
        threshold_ratio = 0.3
        min_occurrences = max(1, int(len(self.mask_buffer) * threshold_ratio))
        
        accumulated_binary = (accumulated >= min_occurrences).astype(np.uint8) * 255
        
        return accumulated_binary
    
    def detect_lines(self, mask: np.ndarray) -> List[LineSegment]:
        """
        Detecta segmentos de línea en la máscara acumulada.
        
        Usa LSD (Line Segment Detector) como método principal y HoughLinesP
        como respaldo/complemento.
        
        Args:
            mask: Máscara binaria acumulada
            
        Returns:
            Lista de LineSegment detectados
        """
        lines_detected = []
        
        # === MÉTODO 1: Line Segment Detector (LSD) ===
        # LSD es más preciso pero requiere opencv-contrib
        if self.use_lsd:
            try:
                lsd = cv2.createLineSegmentDetector()
                lines_lsd, width, prec, nfa = lsd.detect(mask)
                
                if lines_lsd is not None:
                    for line in lines_lsd:
                        x1, y1, x2, y2 = line[0]
                        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        if length >= self.min_line_length:
                            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                            lines_detected.append(LineSegment(
                                x1=float(x1), y1=float(y1),
                                x2=float(x2), y2=float(y2),
                                length=float(length),
                                angle=float(angle)
                            ))
            except Exception as e:
                if self.debug:
                    print(f"LSD falló: {e}, usando Hough como respaldo")
        
        # === MÉTODO 2: Probabilistic Hough Transform ===
        # Más robusto pero menos preciso que LSD
        if self.use_hough or len(lines_detected) < 5:
            lines_hough = cv2.HoughLinesP(
                mask,
                rho=1,
                theta=np.pi / 180,
                threshold=self.hough_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            
            if lines_hough is not None:
                for line in lines_hough:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    if length >= self.min_line_length:
                        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                        
                        # Evitar duplicados con LSD
                        is_duplicate = False
                        if self.use_lsd:
                            for existing in lines_detected:
                                dist1 = np.sqrt((x1 - existing.x1)**2 + (y1 - existing.y1)**2)
                                dist2 = np.sqrt((x2 - existing.x2)**2 + (y2 - existing.y2)**2)
                                if dist1 < self.line_merge_threshold and dist2 < self.line_merge_threshold:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            lines_detected.append(LineSegment(
                                x1=float(x1), y1=float(y1),
                                x2=float(x2), y2=float(y2),
                                length=float(length),
                                angle=float(angle)
                            ))
        
        # Filtrar y fusionar líneas similares
        lines_detected = self._merge_similar_lines(lines_detected)
        
        return lines_detected
    
    def _merge_similar_lines(self, lines: List[LineSegment]) -> List[LineSegment]:
        """
        Fusiona líneas muy similares (mismo ángulo, cercanas).
        Reduce redundancia y mejora estabilidad.
        """
        if len(lines) <= 1:
            return lines
        
        merged = []
        used = [False] * len(lines)
        
        for i, line1 in enumerate(lines):
            if used[i]:
                continue
            
            group = [line1]
            used[i] = True
            
            for j, line2 in enumerate(lines[i+1:], start=i+1):
                if used[j]:
                    continue
                
                # Verificar si son similares
                angle_diff = abs(line1.angle - line2.angle)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                
                # Mismo ángulo (tolerancia 5 grados)
                if angle_diff < 5.0:
                    # Calcular distancia entre líneas
                    # Usar distancia del punto medio
                    mid1 = np.array([(line1.x1 + line1.x2) / 2, (line1.y1 + line1.y2) / 2])
                    mid2 = np.array([(line2.x1 + line2.x2) / 2, (line2.y1 + line2.y2) / 2])
                    dist = np.linalg.norm(mid1 - mid2)
                    
                    if dist < self.line_merge_threshold * 3:
                        group.append(line2)
                        used[j] = True
            
            # Fusionar grupo: tomar la línea más larga
            best_line = max(group, key=lambda l: l.length)
            merged.append(best_line)
        
        return merged
    
    def filter_field_lines(
        self,
        lines: List[LineSegment],
        image_shape: Tuple[int, int]
    ) -> List[LineSegment]:
        """
        Filtra líneas para mantener solo las relevantes del campo.
        
        Criterios:
        - Longitud mínima
        - Ángulos esperados (horizontal, vertical, diagonales del área)
        - Ubicación dentro del campo
        
        Args:
            lines: Lista de líneas detectadas
            image_shape: (height, width) de la imagen
            
        Returns:
            Líneas filtradas
        """
        h, w = image_shape
        filtered = []
        
        for line in lines:
            # Filtro 1: Longitud mínima
            if line.length < self.min_line_length:
                continue
            
            # Filtro 2: Ángulo esperado
            # Líneas del campo suelen ser:
            # - Horizontales (0-10° o 170-180°)
            # - Verticales (80-100°)
            # - Diagonales del área (30-60° o 120-150°)
            angle = line.angle
            
            is_horizontal = angle < 15 or angle > 165
            is_vertical = 75 < angle < 105
            is_diagonal = (30 < angle < 60) or (120 < angle < 150)
            
            if not (is_horizontal or is_vertical or is_diagonal):
                continue
            
            # Filtro 3: Debe estar dentro de los márgenes de la imagen
            margin = 0.05  # 5% de margen
            if (line.x1 < -w * margin or line.x1 > w * (1 + margin) or
                line.x2 < -w * margin or line.x2 > w * (1 + margin) or
                line.y1 < -h * margin or line.y1 > h * (1 + margin) or
                line.y2 < -h * margin or line.y2 > h * (1 + margin)):
                continue
            
            filtered.append(line)
        
        return filtered
    
    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, List[LineSegment]]:
        """
        Procesa un frame completo: extrae máscara, acumula y detecta líneas.
        
        Args:
            image: Frame BGR de OpenCV
            
        Returns:
            (máscara_acumulada, líneas_detectadas)
        """
        # 1. Extraer máscara de líneas del frame actual
        current_mask = self.extract_lines_mask(image)
        
        # 2. Acumular con buffer temporal
        accumulated_mask = self.accumulate_mask(current_mask)
        
        # 3. Detectar líneas en máscara acumulada
        lines = self.detect_lines(accumulated_mask)
        
        # 4. Filtrar líneas relevantes
        lines = self.filter_field_lines(lines, image.shape[:2])
        
        # Guardar para debug/visualización
        self.last_mask = accumulated_mask
        self.last_lines = lines
        
        return accumulated_mask, lines
    
    def reset(self):
        """Reinicia el buffer temporal (útil cuando cambia la cámara)"""
        self.mask_buffer.clear()
        self.frame_count = 0
        self.last_mask = None
        self.last_lines = []
    
    def get_debug_visualization(self, image: np.ndarray) -> np.ndarray:
        """
        Genera visualización de debug con líneas detectadas.
        
        Args:
            image: Frame original
            
        Returns:
            Imagen con overlay de líneas detectadas
        """
        vis = image.copy()
        
        if self.last_mask is not None:
            # Mostrar máscara acumulada como overlay semitransparente
            mask_colored = cv2.applyColorMap(self.last_mask, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
        
        # Dibujar líneas detectadas
        for line in self.last_lines:
            cv2.line(vis, 
                    (int(line.x1), int(line.y1)),
                    (int(line.x2), int(line.y2)),
                    (0, 255, 0), 2)
            
            # Etiqueta con ángulo
            mid_x = int((line.x1 + line.x2) / 2)
            mid_y = int((line.y1 + line.y2) / 2)
            cv2.putText(vis, f"{line.angle:.0f}°",
                       (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis

