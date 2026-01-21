"""
TeamClassifierV2 - Clasificador robusto de equipos con anti-green masking, anti-text edges y features LAB.

Diseñado para trabajar con jugadores lejanos donde el ROI se contamina con césped y dorsales.

Estrategia:
1. Extraer ROI del torso (evitar cabeza/piernas).
2. Eliminar píxeles verdes (césped) con máscara HSV.
3. Eliminar bordes fuertes (dorsales/números) con detección de edges + dilatación.
4. Extraer features LAB (a*, b*) de píxeles de tela válidos (no-verde AND no-edges).
5. Acumular features por track (no por frame) con throttling para reducir coste.
6. KMeans k=2 sobre features agregados de diferentes tracks.
7. Votación temporal para estabilidad.
8. Detección de árbitros por class_id y color.

Mejoras anti-dorsal:
- Usa Laplacian/Sobel para detectar bordes fuertes (números/texto).
- Dilata máscara de bordes para eliminar completamente los dorsales.
- Funciona independiente del color del dorsal (azul/amarillo/blanco).

Author: TacticEYE2 Team
"""

import cv2
import numpy as np
from collections import deque, defaultdict
from typing import Optional, Tuple, List, Dict
import os


class TeamClassifierV2:
    """
    Clasificador de equipos robusto usando LAB color space con anti-green y anti-text masking.
    
    Features:
    - Anti-green mask para eliminar césped del ROI
    - Anti-text mask basada en bordes para eliminar dorsales (independiente de color)
    - Features LAB (a*, b*) robustos a iluminación
    - Clustering por track (no por frame) para evitar sobrerrepresentación
    - Throttling: actualiza color cada N frames para reducir coste computacional
    - Detección de árbitros por class_id y color
    - Votación temporal para estabilidad
    - Modo debug con visualización de ROIs y máscaras
    """
    
    def __init__(
        self,
        # Green removal parameters
        green_h_low: int = 35,
        green_h_high: int = 95,
        green_s_min: int = 40,
        green_v_min: int = 40,
        min_non_green_ratio: float = 0.05,
        
        # Anti-text/edges parameters (for removing jersey numbers regardless of color)
        edge_thresh: int = 18,
        edge_dilate: int = 3,
        edge_dilate_iter: int = 2,
        min_cloth_pixels: int = 150,
        
        # Feature extraction
        roi_top_frac: float = 0.15,
        roi_bottom_frac: float = 0.50,
        roi_left_frac: float = 0.20,
        roi_right_frac: float = 0.80,
        
        # LAB feature parameters
        use_L_channel: bool = True,
        L_weight: float = 0.5,
        
        # Track-based sampling
        min_track_samples: int = 3,
        max_track_feature_history: int = 20,
        min_bbox_area_frac: float = 0.001,
        min_bbox_width: int = 15,
        min_bbox_height: int = 30,
        min_roi_size: int = 12,
        
        # Performance throttling
        update_every_n: int = 5,
        
        # KMeans initialization
        kmeans_min_tracks: int = 12,
        kmeans_min_samples_per_track: int = 2,
        
        # Voting and confirmation
        vote_history: int = 5,
        
        # Referee detection
        referee_detection: bool = True,
        referee_black_L_max: int = 80,
        referee_black_chroma_max: float = 30.0,
        referee_yellow_h_min: int = 20,
        referee_yellow_h_max: int = 60,
        referee_yellow_s_min: int = 150,
        
        # Debug mode
        save_debug_rois: bool = False,
        debug_rois_dir: str = "debug_rois_v2pro"
    ):
        """
        Inicializa el clasificador de equipos V2 Pro con anti-text.
        
        Args:
            green_h_low: Hue mínimo para detectar verde (0-180)
            green_h_high: Hue máximo para detectar verde (0-180)
            green_s_min: Saturación mínima para detectar verde (0-255)
            green_v_min: Value mínimo para detectar verde (0-255)
            min_non_green_ratio: Ratio mínimo de píxeles no-verdes para aceptar muestra
            edge_thresh: Umbral para detectar bordes fuertes (dorsales)
            edge_dilate: Tamaño del kernel para dilatar bordes
            edge_dilate_iter: Número de iteraciones de dilatación
            min_cloth_pixels: Mínimo de píxeles de tela válidos tras masking
            roi_top_frac: Fracción superior del bbox para ROI torso
            roi_bottom_frac: Fracción inferior del bbox para ROI torso
            roi_left_frac: Fracción izquierda del bbox para ROI torso
            roi_right_frac: Fracción derecha del bbox para ROI torso
            use_L_channel: Si incluir canal L en features LAB
            L_weight: Peso del canal L si se usa
            min_track_samples: Muestras mínimas por track para usar en KMeans
            max_track_feature_history: Máximo de features guardadas por track
            min_bbox_area_frac: Área mínima del bbox como fracción del frame
            min_bbox_width: Ancho mínimo del bbox en píxeles
            min_bbox_height: Alto mínimo del bbox en píxeles
            min_roi_size: Tamaño mínimo del ROI (ancho o alto) para procesar
            update_every_n: Actualizar color cada N frames por track
            kmeans_min_tracks: Mínimo de tracks diferentes para inicializar KMeans
            kmeans_min_samples_per_track: Mínimo de muestras por track para considerarlo
            vote_history: Tamaño del buffer de votación temporal
            referee_detection: Activar detección de árbitros por color
            referee_black_L_max: L máximo para detectar negro de árbitro
            referee_black_chroma_max: Chroma máximo para negro de árbitro
            referee_yellow_h_min: H mínimo para amarillo de árbitro
            referee_yellow_h_max: H máximo para amarillo de árbitro
            referee_yellow_s_min: S mínimo para amarillo de árbitro
            save_debug_rois: Guardar ROIs y máscaras para debug
            debug_rois_dir: Directorio para guardar ROIs de debug
        """
        # Green removal
        self.green_h_low = green_h_low
        self.green_h_high = green_h_high
        self.green_s_min = green_s_min
        self.green_v_min = green_v_min
        self.min_non_green_ratio = min_non_green_ratio
        
        # Anti-text/edges (solves dorsals in any color)
        self.edge_thresh = edge_thresh
        self.edge_dilate = edge_dilate
        self.edge_dilate_iter = edge_dilate_iter
        self.min_cloth_pixels = min_cloth_pixels
        
        # ROI extraction
        self.roi_top_frac = roi_top_frac
        self.roi_bottom_frac = roi_bottom_frac
        self.roi_left_frac = roi_left_frac
        self.roi_right_frac = roi_right_frac
        
        # LAB features
        self.use_L_channel = use_L_channel
        self.L_weight = L_weight
        
        # Track-based sampling
        self.min_track_samples = min_track_samples
        self.max_track_feature_history = max_track_feature_history
        self.min_bbox_area_frac = min_bbox_area_frac
        self.min_bbox_width = min_bbox_width
        self.min_bbox_height = min_bbox_height
        self.min_roi_size = min_roi_size
        
        # Performance throttling
        self.update_every_n = update_every_n
        
        # KMeans
        self.kmeans_min_tracks = kmeans_min_tracks
        self.kmeans_min_samples_per_track = kmeans_min_samples_per_track
        
        # Voting
        self.vote_history = vote_history
        
        # Referee detection
        self.referee_detection = referee_detection
        self.referee_black_L_max = referee_black_L_max
        self.referee_black_chroma_max = referee_black_chroma_max
        self.referee_yellow_h_min = referee_yellow_h_min
        self.referee_yellow_h_max = referee_yellow_h_max
        self.referee_yellow_s_min = referee_yellow_s_min
        
        # Debug
        self.save_debug_rois = save_debug_rois
        self.debug_rois_dir = debug_rois_dir
        if self.save_debug_rois:
            os.makedirs(self.debug_rois_dir, exist_ok=True)
        
        # State
        self.track_features: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_track_feature_history))
        self.track_votes: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vote_history))
        self.track_team: Dict[int, int] = {}  # confirmed team assignments
        self.track_frame_count: Dict[int, int] = defaultdict(int)  # frames processed per track
        
        # KMeans state
        self.kmeans_initialized = False
        self.cluster_centers: Optional[np.ndarray] = None
        self.cluster_to_team: Dict[int, int] = {}
        
        # Frame tracking for debug
        self.frame_count = 0
        
    def reset(self):
        """Reinicia el estado del clasificador."""
        self.track_features.clear()
        self.track_votes.clear()
        self.track_team.clear()
        self.track_frame_count.clear()
        self.kmeans_initialized = False
        self.cluster_centers = None
        self.cluster_to_team.clear()
        self.frame_count = 0
        
    def is_ready(self) -> bool:
        """Retorna True si KMeans está inicializado."""
        return self.kmeans_initialized
    
    def _extract_torso_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extrae ROI del torso desde el bbox.
        
        Args:
            image: Frame completo (BGR)
            bbox: (x1, y1, x2, y2)
            
        Returns:
            ROI del torso (BGR) o None si es inválido
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Check minimum size
        if w < self.min_bbox_width or h < self.min_bbox_height:
            return None
        
        # Check minimum area
        frame_area = image.shape[0] * image.shape[1]
        bbox_area = w * h
        if bbox_area < self.min_bbox_area_frac * frame_area:
            return None
        
        # Extract torso region
        top = int(y1 + h * self.roi_top_frac)
        bottom = int(y1 + h * self.roi_bottom_frac)
        left = int(x1 + w * self.roi_left_frac)
        right = int(x1 + w * self.roi_right_frac)
        
        # Clip to image bounds
        top = max(0, min(top, image.shape[0] - 1))
        bottom = max(top + 1, min(bottom, image.shape[0]))
        left = max(0, min(left, image.shape[1] - 1))
        right = max(left + 1, min(right, image.shape[1]))
        
        roi = image[top:bottom, left:right].copy()
        
        if roi.size == 0:
            return None
        
        # Check ROI size is large enough (gating for distant players)
        if roi.shape[0] < self.min_roi_size or roi.shape[1] < self.min_roi_size:
            return None
            
        return roi
    
    def _create_non_green_mask(self, roi_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Crea máscara de píxeles no-verdes (anti-green).
        
        Args:
            roi_bgr: ROI en BGR
            
        Returns:
            (mask_non_green, ratio_non_green)
            mask_non_green: máscara binaria (255=no-verde, 0=verde)
            ratio_non_green: fracción de píxeles no-verdes
        """
        # Convert to HSV
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Create green mask
        # Green: H in [green_h_low, green_h_high], S > green_s_min, V > green_v_min
        mask_green = cv2.inRange(
            roi_hsv,
            np.array([self.green_h_low, self.green_s_min, self.green_v_min]),
            np.array([self.green_h_high, 255, 255])
        )
        
        # Invert to get non-green
        mask_non_green = cv2.bitwise_not(mask_green)
        
        # Light morphology to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_non_green = cv2.morphologyEx(mask_non_green, cv2.MORPH_OPEN, kernel)
        mask_non_green = cv2.morphologyEx(mask_non_green, cv2.MORPH_CLOSE, kernel)
        
        # Calculate ratio
        total_pixels = mask_non_green.size
        non_green_pixels = np.count_nonzero(mask_non_green)
        ratio_non_green = non_green_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return mask_non_green, ratio_non_green
    
    def _build_cloth_mask(self, roi_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construye máscara de tela válida eliminando césped y bordes (dorsales/números).
        
        Esta función resuelve el problema de dorsales de cualquier color (azul, amarillo, blanco)
        al detectar bordes fuertes (números/texto) independientemente de su color.
        
        Args:
            roi_bgr: ROI en BGR
            
        Returns:
            (mask_non_green, mask_edges, mask_cloth, used_mask)
            mask_non_green: máscara anti-verde (255=no-verde)
            mask_edges: máscara de bordes detectados (255=borde)
            mask_cloth: máscara de tela (no-verde AND no-bordes)
            used_mask: máscara final usada (cloth o fallback a non_green)
        """
        # Step 1: Anti-green mask
        mask_non_green, _ = self._create_non_green_mask(roi_bgr)
        
        # Step 2: Detect edges (jersey numbers/text regardless of color)
        # Convert to grayscale
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Use Laplacian for edge detection (detects high-frequency details like text)
        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian).astype(np.uint8)
        
        # Threshold to create edge mask
        _, mask_edges = cv2.threshold(laplacian_abs, self.edge_thresh, 255, cv2.THRESH_BINARY)
        
        # Dilate edges to ensure we remove all text/numbers
        if self.edge_dilate > 0 and self.edge_dilate_iter > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.edge_dilate, self.edge_dilate))
            mask_edges = cv2.dilate(mask_edges, kernel, iterations=self.edge_dilate_iter)
        
        # Step 3: Combine masks - cloth = non_green AND NOT edges
        mask_cloth = cv2.bitwise_and(mask_non_green, cv2.bitwise_not(mask_edges))
        
        # Step 4: Check if we have enough cloth pixels
        cloth_pixels = np.count_nonzero(mask_cloth)
        
        if cloth_pixels < self.min_cloth_pixels:
            # Fallback: use non-green mask (better than nothing)
            used_mask = mask_non_green
        else:
            # Use cloth mask (best option)
            used_mask = mask_cloth
        
        return mask_non_green, mask_edges, mask_cloth, used_mask
    
    def _extract_lab_features(self, roi_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae features LAB (a*, b*) de píxeles enmascarados.
        
        Args:
            roi_bgr: ROI en BGR
            mask: Máscara binaria (255=válido)
            
        Returns:
            Feature vector: [a_median, b_median] o [L_median*weight, a_median, b_median]
            None si no hay píxeles válidos
        """
        # Convert to LAB
        roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        
        # Extract masked pixels
        masked_pixels = roi_lab[mask > 0]
        
        if len(masked_pixels) == 0:
            return None
        
        # Compute median of each channel
        L_median = np.median(masked_pixels[:, 0])
        a_median = np.median(masked_pixels[:, 1])
        b_median = np.median(masked_pixels[:, 2])
        
        if self.use_L_channel:
            # Include L with weight
            feature = np.array([L_median * self.L_weight, a_median, b_median], dtype=np.float32)
        else:
            # Only a*, b*
            feature = np.array([a_median, b_median], dtype=np.float32)
        
        return feature
    
    def _is_referee_by_color(self, roi_bgr: np.ndarray) -> bool:
        """
        Detecta si el ROI corresponde a un árbitro por color (negro o amarillo).
        
        Args:
            roi_bgr: ROI en BGR
            
        Returns:
            True si parece árbitro
        """
        if not self.referee_detection:
            return False
        
        # Convert to LAB for black detection
        roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        L_median = np.median(roi_lab[:, :, 0])
        a_median = np.median(roi_lab[:, :, 1])
        b_median = np.median(roi_lab[:, :, 2])
        chroma = np.sqrt((a_median - 128)**2 + (b_median - 128)**2)
        
        # Black referee: low L, low chroma
        if L_median < self.referee_black_L_max and chroma < self.referee_black_chroma_max:
            return True
        
        # Yellow referee: convert to HSV
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        H_median = np.median(roi_hsv[:, :, 0])
        S_median = np.median(roi_hsv[:, :, 1])
        
        # Yellow: H in [20, 60], high S
        if (self.referee_yellow_h_min <= H_median <= self.referee_yellow_h_max and
            S_median >= self.referee_yellow_s_min):
            return True
        
        return False
    
    def _save_debug_roi(self, track_id: int, roi_bgr: np.ndarray, 
                        mask_non_green: np.ndarray, mask_edges: np.ndarray,
                        mask_cloth: np.ndarray, used_mask: np.ndarray,
                        feature: Optional[np.ndarray], decision: str):
        """
        Guarda ROI, máscaras y features para debug.
        
        Args:
            track_id: ID del track
            roi_bgr: ROI original
            mask_non_green: Máscara anti-verde
            mask_edges: Máscara de bordes
            mask_cloth: Máscara de tela
            used_mask: Máscara usada finalmente
            feature: Feature LAB extraído
            decision: Decisión tomada (ej: "team_0", "referee", "skip")
        """
        if not self.save_debug_rois:
            return
        
        try:
            # Create filename prefix
            prefix = f"frame{self.frame_count:06d}_track{track_id:03d}"
            
            # Save original ROI
            roi_path = os.path.join(self.debug_rois_dir, f"{prefix}_roi.png")
            cv2.imwrite(roi_path, roi_bgr)
            
            # Save mask_non_green
            mask_ng_path = os.path.join(self.debug_rois_dir, f"{prefix}_mask_non_green.png")
            cv2.imwrite(mask_ng_path, mask_non_green)
            
            # Save mask_edges
            mask_edges_path = os.path.join(self.debug_rois_dir, f"{prefix}_mask_edges.png")
            cv2.imwrite(mask_edges_path, mask_edges)
            
            # Save mask_cloth
            mask_cloth_path = os.path.join(self.debug_rois_dir, f"{prefix}_mask_cloth.png")
            cv2.imwrite(mask_cloth_path, mask_cloth)
            
            # Save masked ROI with used_mask
            masked_roi = cv2.bitwise_and(roi_bgr, roi_bgr, mask=used_mask)
            masked_path = os.path.join(self.debug_rois_dir, f"{prefix}_masked.png")
            cv2.imwrite(masked_path, masked_roi)
            
            # Save feature and decision to txt
            info_path = os.path.join(self.debug_rois_dir, f"{prefix}_info.txt")
            with open(info_path, 'w') as f:
                f.write(f"Track ID: {track_id}\n")
                f.write(f"Frame: {self.frame_count}\n")
                f.write(f"Decision: {decision}\n")
                f.write(f"ROI shape: {roi_bgr.shape}\n")
                f.write(f"Non-green pixels: {np.count_nonzero(mask_non_green)}\n")
                f.write(f"Edge pixels: {np.count_nonzero(mask_edges)}\n")
                f.write(f"Cloth pixels: {np.count_nonzero(mask_cloth)}\n")
                f.write(f"Used mask: {'cloth' if np.array_equal(used_mask, mask_cloth) else 'non_green'}\n")
                if feature is not None:
                    f.write(f"Feature: {feature}\n")
                else:
                    f.write("Feature: None\n")
        except Exception as e:
            # Don't crash on debug save errors
            pass
    
    def _run_kmeans(self) -> bool:
        """
        Ejecuta KMeans sobre features acumulados de diferentes tracks.
        
        Returns:
            True si se inicializó correctamente
        """
        # Collect features from tracks that have enough samples
        track_features_list = []
        track_ids_used = []
        
        for track_id, features_deque in self.track_features.items():
            if len(features_deque) >= self.kmeans_min_samples_per_track:
                # Compute median feature for this track
                features_array = np.array(list(features_deque), dtype=np.float32)
                median_feature = np.median(features_array, axis=0)
                track_features_list.append(median_feature)
                track_ids_used.append(track_id)
        
        # Check if we have enough tracks
        if len(track_features_list) < self.kmeans_min_tracks:
            return False
        
        # Prepare data for KMeans
        X = np.array(track_features_list, dtype=np.float32)
        
        # Run KMeans k=2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            X, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        
        self.cluster_centers = centers
        
        # Map clusters to teams (arbitrarily assign 0 and 1)
        # Strategy: assign cluster 0 -> team 0, cluster 1 -> team 1
        # (Could use heuristic like "cluster with lower a* = team 0" but keep it simple)
        self.cluster_to_team = {0: 0, 1: 1}
        
        self.kmeans_initialized = True
        
        # Log initialization
        print(f"[TeamClassifierV2] KMeans inicializado con {len(track_features_list)} tracks "
              f"({len(track_ids_used)} track IDs)")
        for i, center in enumerate(centers):
            team_id = self.cluster_to_team[i]
            print(f"  Centro cluster {i} (team {team_id}): {center}")
        
        # Check for cluster collapse (centers too close)
        distance = np.linalg.norm(centers[0] - centers[1])
        threshold = 15.0 if self.use_L_channel else 10.0
        
        print(f"  Distancia entre clusters: {distance:.2f}")
        
        if distance < threshold:
            print(f"  [WARNING] ⚠️ Clusters muy cercanos (distancia={distance:.2f} < {threshold})!")
            print(f"  Los equipos tienen colores MUY similares en espacio LAB.")
            print(f"  Sugerencias:")
            print(f"    1. Aumentar --tc-kmeans-min-tracks (actual: {self.kmeans_min_tracks})")
            print(f"    2. Ajustar umbrales anti-green si hay contaminación de césped")
            print(f"    3. Reducir --tc-min-non-green-ratio si rechaza demasiadas muestras")
            print(f"    4. Activar --tc-use-L para incluir luminosidad en features")
        else:
            print(f"  ✓ Separación de clusters aceptable")
        
        return True
    
    def _assign_team_to_feature(self, feature: np.ndarray) -> int:
        """
        Asigna equipo a un feature usando KMeans.
        
        Args:
            feature: Feature LAB [a, b] o [L*w, a, b]
            
        Returns:
            Team ID (0 o 1)
        """
        if not self.kmeans_initialized or self.cluster_centers is None:
            return -1
        
        # Compute distance to each center
        distances = [np.linalg.norm(feature - center) for center in self.cluster_centers]
        closest_cluster = int(np.argmin(distances))
        team_id = self.cluster_to_team.get(closest_cluster, -1)
        
        return team_id
    
    def add_detection(self, track_id: int, bbox: Tuple[int, int, int, int], 
                      image: np.ndarray, class_id: Optional[int] = None):
        """
        Procesa una detección y actualiza el estado del track.
        
        Args:
            track_id: ID del track
            bbox: (x1, y1, x2, y2)
            image: Frame completo (BGR)
            class_id: ID de clase YOLO (2=referee)
        """
        self.frame_count += 1
        
        # Referee by class_id
        if class_id == 2:
            self.track_team[track_id] = -1
            return
        
        # Throttling: only update color every N frames per track
        self.track_frame_count[track_id] += 1
        if self.track_frame_count[track_id] % self.update_every_n != 0:
            # Skip color update this frame
            return
        
        # Extract torso ROI
        roi_bgr = self._extract_torso_roi(image, bbox)
        if roi_bgr is None:
            # Skip: bbox too small or ROI too small (distant player)
            return
        
        # Check referee by color
        if self._is_referee_by_color(roi_bgr):
            self.track_team[track_id] = -1
            return
        
        # Build cloth mask (anti-green + anti-text/edges)
        mask_non_green, mask_edges, mask_cloth, used_mask = self._build_cloth_mask(roi_bgr)
        
        # Check if we have enough valid pixels
        valid_pixels = np.count_nonzero(used_mask)
        if valid_pixels < self.min_cloth_pixels:
            # Not enough cloth pixels after removing green and edges
            if self.save_debug_rois:
                self._save_debug_roi(track_id, roi_bgr, mask_non_green, mask_edges, 
                                    mask_cloth, used_mask, None, 
                                    f"skip_cloth_pixels_{valid_pixels}")
            return
        
        # Extract LAB features from cloth pixels only
        feature = self._extract_lab_features(roi_bgr, used_mask)
        if feature is None:
            if self.save_debug_rois:
                self._save_debug_roi(track_id, roi_bgr, mask_non_green, mask_edges,
                                    mask_cloth, used_mask, None, "skip_no_feature")
            return
        
        # Add feature to track history
        self.track_features[track_id].append(feature)
        
        # Try to initialize KMeans if not ready
        if not self.kmeans_initialized:
            initialized = self._run_kmeans()
            if not initialized:
                # Not enough data yet
                if self.save_debug_rois:
                    self._save_debug_roi(track_id, roi_bgr, mask_non_green, mask_edges,
                                        mask_cloth, used_mask, feature, "waiting_kmeans")
                return
        
        # Assign team using current feature
        team_vote = self._assign_team_to_feature(feature)
        
        if team_vote >= 0:
            # Add vote
            self.track_votes[track_id].append(team_vote)
            
            # Confirm team by majority vote
            if len(self.track_votes[track_id]) >= self.vote_history:
                votes = list(self.track_votes[track_id])
                # Count votes
                count_0 = votes.count(0)
                count_1 = votes.count(1)
                
                if count_0 > count_1:
                    self.track_team[track_id] = 0
                    decision = "team_0_confirmed"
                elif count_1 > count_0:
                    self.track_team[track_id] = 1
                    decision = "team_1_confirmed"
                else:
                    # Tie: keep previous or wait
                    decision = "tie_vote"
            else:
                decision = f"voting_{team_vote}"
        else:
            decision = "no_team"
        
        if self.save_debug_rois:
            self._save_debug_roi(track_id, roi_bgr, mask_non_green, mask_edges,
                                mask_cloth, used_mask, feature, decision)
    
    def get_team(self, track_id: int) -> int:
        """
        Obtiene el equipo confirmado de un track.
        
        Args:
            track_id: ID del track
            
        Returns:
            Team ID: 0, 1, o -1 (referee/unknown)
        """
        return self.track_team.get(track_id, -1)


# Comando de test sugerido (comentario):
# timeout 90 python pruebatrackequipo.py sample_match.mp4 --model weights/best.pt --imgsz 640 --conf 0.30 --max-det 200 --reid --tc-kmeans-min-tracks 10 --tc-vote-history 4 --tc-use-L --tc-L-weight 0.5
#
# Para debug con ROIs (visualizar máscaras):
# timeout 90 python pruebatrackequipo.py sample_match.mp4 --model weights/best.pt --imgsz 640 --conf 0.30 --max-det 200 --reid --tc-kmeans-min-tracks 10 --tc-vote-history 4 --tc-use-L --tc-L-weight 0.5 --tc-save-rois --tc-rois-dir debug_antitext
