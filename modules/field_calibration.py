"""
Field Calibration - Sistema Industrial (Dictionary Search + ECC)
==============================================================
Enfoque robusto basado en búsqueda de patrones y refinamiento geométrico.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

@dataclass
class FieldDimensions:
    length: float = 105.0
    width: float = 68.0
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.32
    goal_area_length: float = 5.5
    goal_area_width: float = 18.32
    center_circle_radius: float = 9.15

class FieldCalibration:
    def __init__(self):
        self.field_dims = FieldDimensions()
        
        # 1. Generar el "Mundo Ideal" (Template)
        self.world_template = self._generate_world_template()
        
        # 2. Generar Diccionario de Vistas (Candidatos de Homografía)
        self.view_dictionary = self._generate_view_dictionary()
        
        self.homography_matrix = None
        self.last_mask = None
        self.best_view_name = "None"
        
        # Estabilización temporal
        self.smoothed_homography = None
        self.alpha = 0.15 # Factor de suavizado (0.1 = muy suave, 1.0 = sin filtro)

    def _generate_world_template(self, scale: int = 10) -> np.ndarray:
        """
        Genera una imagen binaria perfecta del campo (vista cenital).
        Scale: píxeles por metro.
        """
        w = int(self.field_dims.length * scale)
        h = int(self.field_dims.width * scale)
        img = np.zeros((h, w), dtype=np.uint8)
        
        # Funciones auxiliares de dibujo
        def to_px(x, y): return (int(x * scale), int(h - y * scale))
        
        # Color línea
        color = 255
        th = 2 # Grosor línea en el template (en metros sería th/scale)
        
        # Perímetro
        cv2.rectangle(img, to_px(0, 0), to_px(self.field_dims.length, self.field_dims.width), color, th)
        # Medio campo
        cv2.line(img, to_px(52.5, 0), to_px(52.5, 68), color, th)
        # Círculo
        center = to_px(52.5, 34)
        radius = int(self.field_dims.center_circle_radius * scale)
        cv2.circle(img, center, radius, color, th)
        
        # Áreas
        for x_start in [0, 105 - 16.5]:
            pts = [
                (x_start, 13.84), 
                (x_start + 16.5 if x_start == 0 else 105, 13.84),
                (x_start + 16.5 if x_start == 0 else 105, 54.16),
                (x_start, 54.16)
            ]
            px_pts = np.array([to_px(*p) for p in pts], np.int32)
            # Polylines necesita lista de arrays
            cv2.polylines(img, [px_pts], isClosed=True, color=color, thickness=th)

        return img

    def _generate_view_dictionary(self) -> List[Dict]:
        """
        Genera un diccionario EXTENSO de vistas sintéticas (Grid Search).
        Cubre: Pan (X), Zoom (Ancho) y Tilt (Altura/Perspectiva).
        """
        views = []
        src_w, src_h = 1280, 720
        src_pts = np.array([[0, src_h], [src_w, src_h], [src_w, 0], [0, 0]], dtype=np.float32)

        # 1. PAN (Posición X): Desde el área izquierda (16.5m) hasta la derecha (88.5m)
        # Paso de ~5 metros
        pans = np.linspace(16.5, 88.5, 15)

        # 2. ZOOM (Ancho visible en banda cercana Y=0): De 30m a 80m
        zooms = np.linspace(30, 80, 6)

        # 3. TILT / ALTURA (Factor de Perspectiva): Relación Ancho Lejano / Ancho Cercano
        # Factor 1.0 = Vista Satélite (Paralela)
        # Factor 1.5 = Cámara Alta (Estadio Olímpico)
        # Factor 2.5 = Cámara Baja (Estadio Inglés pequeño)
        tilts = np.linspace(1.2, 2.4, 4)

        print(f"📚 Generando diccionario denso: {len(pans)} Pans x {len(zooms)} Zooms x {len(tilts)} Alturas...")
        
        count = 0
        for x_center in pans:
            for width_near in zooms:
                for perspective_factor in tilts:
                    # Calculamos geometría del trapecio de visión en el mundo
                    
                    # Ancho en banda cercana (Y=0)
                    x_near_min = x_center - width_near/2
                    x_near_max = x_center + width_near/2
                    
                    # Ancho en banda lejana (Y=68) - Afectado por perspectiva
                    width_far = width_near * perspective_factor
                    x_far_min = x_center - width_far/2
                    x_far_max = x_center + width_far/2
                    
                    dst_poly = np.array([
                        [x_near_min, 0],   # Bottom-Left (World)
                        [x_near_max, 0],   # Bottom-Right (World)
                        [x_far_max, 68],   # Top-Right (World)
                        [x_far_min, 68]    # Top-Left (World)
                    ], dtype=np.float32)
                    
                    try:
                        H, _ = cv2.findHomography(src_pts, dst_poly)
                        if H is not None:
                            H_inv = np.linalg.inv(H)
                            # Máscara simulada (320x180 para velocidad)
                            mask_sim = cv2.warpPerspective(self.world_template, H_inv, (320, 180))
                            
                            # Validar que la máscara no esté vacía (cámara mirando fuera)
                            if cv2.countNonZero(mask_sim) > 100:
                                views.append({
                                    "name": f"P:{x_center:.0f} Z:{width_near:.0f} T:{perspective_factor:.1f}",
                                    "H": H,
                                    "H_inv": H_inv,
                                    "mask_sim": mask_sim
                                })
                                count += 1
                    except: pass
        
        print(f"✅ Diccionario generado con {len(views)} vistas válidas.")
        return views

    def _get_lines_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Segmentación AVANZADA (Relaxed):
        """
        small = cv2.resize(image, (320, 180))
        h, w = small.shape[:2]
        
        # --- PASO 0: Contexto de Césped ---
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        grass_roi = cv2.dilate(green_mask, np.ones((15, 15), np.uint8))
        
        # --- PASO 1: Candidatos Blancos ---
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_th)
        _, binary_white = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        lower_white = np.array([0, 0, 80])
        upper_white = np.array([180, 80, 255])
        white_color_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        candidates = cv2.bitwise_and(binary_white, white_color_mask)
        candidates = cv2.bitwise_and(candidates, grass_roi)
        
        # --- PASO 2: Filtro Geométrico (RELAJADO) ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidates, connectivity=8)
        cleaned_mask = np.zeros_like(candidates)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            if area < 5: continue # Ruido muy pequeño
            
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                
                # RELAJADO: Ratio > 1.5 (antes 2.5) acepta líneas más cortas/gruesas
                # Y permitimos cosas pequeñas si son MUY alargadas (líneas lejanas)
                if aspect_ratio < 1.5 and area < 50:
                    continue # Sigue siendo probable basura (media)
            
            cleaned_mask[labels == i] = 255
            
        # --- PASO 3: Conexión Final ---
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        final_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # --- PASO 4: FILTRO HOUGH (Eliminador Supremo de Ruido) ---
        # Solo lo que es una LÍNEA RECTA sobrevive. Lo demás (ruido) muere.
        lines = cv2.HoughLinesP(
            final_mask,
            rho=1,
            theta=np.pi/180,
            threshold=25,        # Umbral bajo para pillar líneas tenues
            minLineLength=15,    # Mínimo 15px de largo (en 320p es bastante)
            maxLineGap=10        # Permitir pequeños huecos
        )
        
        hough_mask = np.zeros_like(final_mask)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 2)
        
        # Engrosar un poco para facilitar el matching
        hough_mask = cv2.dilate(hough_mask, np.ones((3,3), np.uint8))
        
        return hough_mask

    def _refine_view(self, mask_real: np.ndarray, coarse_view: Dict) -> Dict:
        """
        Paso de Refinamiento Local:
        Toma la mejor vista 'basta' y busca variaciones finas alrededor de ella
        para ajustar la homografía al metro exacto.
        """
        try:
            # Extraer parámetros de la vista basta
            # "P:37 Z:40 T:1.5"
            parts = coarse_view["name"].split()
            base_pan = float(parts[0].split(":")[1])
            base_zoom = float(parts[1].split(":")[1])
            base_tilt = float(parts[2].split(":")[1])
        except:
            return coarse_view # Fallback

        src_w, src_h = 1280, 720
        src_pts = np.array([[0, src_h], [src_w, src_h], [src_w, 0], [0, 0]], dtype=np.float32)
        
        # Búsqueda fina: +/- 2.5 metros en pasos de 0.5 metros
        fine_pans = np.linspace(base_pan - 2.5, base_pan + 2.5, 6)
        fine_zooms = np.linspace(base_zoom - 2.5, base_zoom + 2.5, 6)
        
        # Mantenemos el Tilt fijo por ahora (es el más difícil de estimar)
        
        best_refined_iou = 0
        best_refined_view = coarse_view
        
        for p in fine_pans:
            for z in fine_zooms:
                # Recalcular geometría (código duplicado de generate_view, optimizable)
                half_width_bottom = (z * 0.4) / 2
                half_width_top = (z * 1.0) / 2
                
                bl_x = p - half_width_bottom
                br_x = p + half_width_bottom
                tl_x = p - half_width_top
                tr_x = p + half_width_top
                
                # Ajustar ancho lejano por Tilt
                width_far = z * base_tilt
                x_far_min = p - width_far/2
                x_far_max = p + width_far/2
                
                dst_poly = np.array([
                    [x_near_min := p - z/2, 0],
                    [x_near_max := p + z/2, 0],
                    [x_far_max, 68],
                    [x_far_min, 68]
                ], dtype=np.float32)
                
                try:
                    H, _ = cv2.findHomography(src_pts, dst_poly)
                    if H is not None:
                        H_inv = np.linalg.inv(H)
                        mask_sim = cv2.warpPerspective(self.world_template, H_inv, (320, 180))
                        
                        # IoU Rápido
                        intersection = cv2.bitwise_and(mask_real, mask_sim)
                        if cv2.countNonZero(intersection) < 20: continue
                        
                        union = cv2.bitwise_or(mask_real, mask_sim)
                        iou = cv2.countNonZero(intersection) / cv2.countNonZero(union)
                        
                        if iou > best_refined_iou:
                            best_refined_iou = iou
                            best_refined_view = {
                                "name": f"P:{p:.1f} Z:{z:.1f} T:{base_tilt:.1f}",
                                "H": H
                            }
                except: pass
                
        return best_refined_view

    def get_zone_info(self, position_meters: np.ndarray) -> Tuple[int, str]:
        """
        Retorna (Zone ID, Nombre) para una posición (x, y) en metros.
        Grid 6x3 (18 Zonas).
        """
        if position_meters is None: return 0, "Unknown"
        
        x, y = position_meters
        
        # Limites
        if x < 0 or x > 105 or y < 0 or y > 68: return 0, "Out of Field"
        
        # Grid 6 columnas x 3 filas
        col_width = 105.0 / 6
        row_height = 68.0 / 3
        
        col = int(x / col_width) # 0-5
        row = int(y / row_height) # 0-2 (0=Abajo, 1=Centro, 2=Arriba)
        
        # Corregir índices max
        col = min(col, 5)
        row = min(row, 2)
        
        # ID Zona (1-18)
        # Convención: Zonas 1-6 (Defensa/Abajo), ..., 13-18 (Ataque/Arriba)?
        # Convención táctica común: 1-3 (Defensa), 4-6 (Medio), 7-9 (Ataque) por Carriles
        # Pero usaremos numeración secuencial simple Grid 18:
        # Fila 2 (Arriba/Lejos): 13 14 15 16 17 18
        # Fila 1 (Centro):       07 08 09 10 11 12
        # Fila 0 (Abajo/Cerca):  01 02 03 04 05 06
        
        zone_id = (row * 6) + col + 1
        
        # Nombres especiales
        name = f"Zone {zone_id}"
        if zone_id == 14: name += " (The Hole)"
        elif zone_id in [1, 7, 13]: name += " (Left Flank)"
        elif zone_id in [6, 12, 18]: name += " (Right Flank)"
        elif zone_id in [3, 4]: name += " (Central Def)"
        elif zone_id in [15, 16]: name += " (Central Att)"
        
        return zone_id, name

    def compute_homography(self, image: np.ndarray, frame_number: int = 0, side_hint: int = 0, tilt_hint: int = 0) -> bool:
        """
        Paso 1: Extraer máscara de líneas.
        Paso 2: Comparar con Diccionario.
        Paso 3: Refinar Localmente (NUEVO).
        """
        # 1. Segmentación
        mask_real = self._get_lines_mask(image)
        self.last_mask = mask_real 
        
        # 2. Dictionary Search
        best_iou = 0
        best_candidate = None
        
        for view in self.view_dictionary:
            # Filtro por Pistas (Hints)
            try:
                parts = view["name"].split() 
                pan_val = float(parts[0].split(":")[1])
                tilt_val = float(parts[2].split(":")[1])
                
                # Filtro Lado
                if side_hint == -1 and pan_val > 55: continue 
                if side_hint == 1 and pan_val < 50: continue
                
                # Filtro Altura/Tilt
                if tilt_hint == -1 and tilt_val > 1.65: continue 
                if tilt_hint == 1 and tilt_val < 1.65: continue  

            except: pass

            intersection = cv2.bitwise_and(mask_real, view["mask_sim"])
            if cv2.countNonZero(intersection) < 50: continue

            union = cv2.bitwise_or(mask_real, view["mask_sim"])
            iou = cv2.countNonZero(intersection) / cv2.countNonZero(union)
            
            if iou > best_iou:
                best_iou = iou
                best_candidate = view
        
        if best_candidate is None or best_iou < 0.04: 
            return False

        # --- 3. REFINAMIENTO LOCAL (NUEVO) ---
        # Ajuste fino alrededor del ganador basto
        refined_candidate = self._refine_view(mask_real, best_candidate)
        
        self.best_view_name = refined_candidate["name"]
        initial_H = refined_candidate["H"]
        
        # 4. Estabilización Temporal
        current_H = initial_H
        if self.smoothed_homography is None:
            self.smoothed_homography = current_H
        else:
            center_pt = np.array([[[640, 360]]], dtype=np.float32)
            p1 = cv2.perspectiveTransform(center_pt, self.smoothed_homography)
            p2 = cv2.perspectiveTransform(center_pt, current_H)
            dist = np.linalg.norm(p1 - p2)
            
            if dist < 30.0:
                self.smoothed_homography = self.alpha * current_H + (1 - self.alpha) * self.smoothed_homography
            else:
                self.smoothed_homography = 0.05 * current_H + 0.95 * self.smoothed_homography

        self.homography_matrix = self.smoothed_homography
        return True

    def pixel_to_meters(self, point_2d: np.ndarray) -> Optional[np.ndarray]:
        if self.homography_matrix is None: return None
        point_2d = np.array(point_2d, dtype=np.float32)
        if point_2d.ndim == 1: point_2d = point_2d.reshape(1, 2)
        try:
            point_3d = cv2.perspectiveTransform(point_2d.reshape(-1, 1, 2), self.homography_matrix)
            return point_3d.reshape(-1, 2)[0] if len(point_3d) == 1 else point_3d.reshape(-1, 2)
        except: return None

    def draw_projected_pitch(self, image: np.ndarray, color=(0, 255, 255), thickness=2) -> np.ndarray:
        if self.homography_matrix is None: return image
        img_copy = image.copy()
        try:
            inv_h = np.linalg.inv(self.homography_matrix)
            
            def to_px(points_3d):
                points_3d = np.array(points_3d, dtype=np.float32)
                if points_3d.ndim == 1: points_3d = points_3d.reshape(1, 2)
                points_2d = cv2.perspectiveTransform(points_3d.reshape(-1, 1, 2), inv_h)
                return points_2d.reshape(-1, 2).astype(np.int32)

            # Dibujar elementos principales
            # 1. Perímetro
            corners = np.array([[0, 0], [105, 0], [105, 68], [0, 68]])
            cv2.polylines(img_copy, [to_px(corners)], True, color, thickness)
            
            # 2. Medio
            cv2.line(img_copy, tuple(to_px([52.5, 0])[0]), tuple(to_px([52.5, 68])[0]), color, thickness)
            
            # 3. Círculo
            center = np.array([52.5, 34])
            angles = np.linspace(0, 2*np.pi, 30)
            circle_pts = [[center[0] + 9.15*np.cos(a), center[1] + 9.15*np.sin(a)] for a in angles]
            cv2.polylines(img_copy, [to_px(circle_pts)], True, color, thickness)
            
            # 4. Areas
            for x in [0, 105-16.5]:
                pts = [[x, 13.84], [x+16.5 if x==0 else 105, 13.84],
                       [x+16.5 if x==0 else 105, 54.16], [x, 54.16]]
                cv2.polylines(img_copy, [to_px(pts)], True, color, thickness)

        except Exception as e:
            pass # Evitar crash por singularidad en dibujo
            
        return img_copy
        
    def refine_homography(self, min_quality=0.3): pass # No necesario con este método
    def should_recalibrate(self, f): return True
    def is_calibrated(self): return self.homography_matrix is not None