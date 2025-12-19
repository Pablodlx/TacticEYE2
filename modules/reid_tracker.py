"""
ReID Tracker - Re-identificación de jugadores con persistencia de IDs
======================================================================
Mantiene IDs consistentes incluso cuando jugadores salen de pantalla 30-60s
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
from modules.team_classifier import TeamClassifier


@dataclass
class TrackState:
    """Estado de un track individual"""
    track_id: int
    last_seen: float
    last_bbox: np.ndarray
    feature_history: deque  # Últimas N features para matching robusto
    trajectory: deque  # Últimas posiciones
    team_id: int = -1  # 0=local, 1=visitante, 2=árbitro
    color_signature: np.ndarray = None
    original_class: int = 0  # Clase del modelo YOLO (0=player, 2=referee, 3=goalkeeper)


class OSNetFeatureExtractor(nn.Module):
    """
    Feature Extractor simplificado tipo OSNet
    Para producción, usar torchreid.models.build_model('osnet_x1_0', num_classes=1000)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Usando ResNet18 como backbone (ligero y rápido)
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        
        # Remover última FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 512
        
        self.eval()
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        # L2 normalize
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features


class ReIDTracker:
    """
    Tracker avanzado con Re-Identificación
    Combina ByteTrack + Features ReID para mantener IDs persistentes
    """
    
    def __init__(self, 
                 max_age: int = 150,  # 150 frames (~5s @ 30fps) sin detección
                 max_lost_time: float = 120.0,  # 120s máximo fuera de pantalla
                 feature_buffer_size: int = 20,
                 similarity_threshold: float = 0.5):
        """
        Args:
            max_age: Frames máximos sin detección antes de marcar como perdido
            max_lost_time: Segundos máximos para mantener track inactivo
            feature_buffer_size: Cuántas features guardar por track
            similarity_threshold: Umbral de similitud coseno para matching
        """
        self.max_age = max_age
        self.max_lost_time = max_lost_time
        self.feature_buffer_size = feature_buffer_size
        self.similarity_threshold = similarity_threshold
        
        # Estado del tracker
        self.active_tracks: Dict[int, TrackState] = {}
        self.lost_tracks: Dict[int, TrackState] = {}
        self.next_track_id = 1
        
        # Feature extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = OSNetFeatureExtractor(pretrained=True).to(self.device)
        self.feature_extractor.eval()
        
        # Transforms para ReID (tamaño estándar 256x128)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ ReID Tracker inicializado en {self.device}")
        # Team color signatures (list of HSV centroids) for team 0 and 1
        self.team_signatures = []  # list of np.array([h,s,v]) centroids
        self.team_update_alpha = 0.2
        # Initialization buffer for automatic team clustering
        self.team_init_buffer = []  # list of color signatures
        self.team_initialized = False
        self.team_init_samples = 40  # number of samples to collect before clustering
        self.team_distance_threshold = 25.0
        # Additional parameters to improve robustness
        self.team_min_chroma = 8.0  # minimal chroma (sqrt(a^2+b^2)) to consider sample
        self.team_min_bbox_area_ratio = 0.002  # minimal bbox area relative to image
        self.team_confirm_samples = 3  # samples per track to confirm team
        # Team classifier (lazy init on first frame)
        self.team_classifier: Optional[TeamClassifier] = None
        
    def extract_features(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Extrae features ReID de múltiples bboxes
        
        Args:
            image: Frame BGR de OpenCV
            bboxes: Array Nx4 [x1,y1,x2,y2]
            
        Returns:
            Features array NxD
        """
        if len(bboxes) == 0:
            return np.array([])
        
        crops = []
        h, w = image.shape[:2]
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            # Clamp a dimensiones de imagen
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                # Bbox inválida, usar feature dummy
                crops.append(torch.zeros(3, 256, 128))
                continue
                
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crops.append(torch.zeros(3, 256, 128))
                continue
                
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = self.transform(crop)
            crops.append(crop_tensor)
        
        if not crops:
            return np.array([])
        
        # Batch inference
        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(batch)
        
        return features.cpu().numpy()

    def compute_color_signature(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Compute a simple color signature (mean HSV) inside the bbox center area.

        Uses a central crop of the bbox to avoid grass and background.
        Returns an array [H,S,V]."""
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return np.array([0.0, 0.0, 0.0])

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([0.0, 0.0, 0.0])

        # use center 60% area
        ch, cw = crop.shape[:2]
        margin_h = int(ch * 0.2)
        margin_w = int(cw * 0.2)
        core = crop[margin_h:ch - margin_h or ch, margin_w:cw - margin_w or cw]
        if core.size == 0:
            core = crop
        # Use LAB color space for better perceptual distances
        lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB)
        l_med = np.median(lab[:, :, 0])
        a_med = np.median(lab[:, :, 1])
        b_med = np.median(lab[:, :, 2])
        return np.array([l_med, a_med, b_med], dtype=np.float32)

    def _chroma(self, lab_sig: np.ndarray) -> float:
        """Return chroma magnitude from LAB signature (a,b)."""
        return float(np.linalg.norm(lab_sig[1:3]))

    def _add_team_sample(self, color_sig: np.ndarray, bbox: np.ndarray, image_shape: tuple):
        """Añade una firma de color al buffer de inicialización y ejecuta clustering si hay suficientes muestras."""
        if color_sig is None or np.all(color_sig == 0):
            return
        if self.team_initialized:
            return
        # check chroma
        chroma = self._chroma(color_sig)
        if chroma < self.team_min_chroma:
            return
        # check bbox area ratio
        img_h, img_w = image_shape[:2]
        bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if (bbox_area / (img_h * img_w)) < self.team_min_bbox_area_ratio:
            return
        self.team_init_buffer.append(color_sig.copy())
        if len(self.team_init_buffer) >= self.team_init_samples:
            X = np.vstack(self.team_init_buffer).astype(np.float32)
            idx0 = 0
            idx1 = int(np.argmax([np.linalg.norm(X[i] - X[idx0]) for i in range(len(X))]))
            c0 = X[idx0].copy()
            c1 = X[idx1].copy()
            for _ in range(10):
                d0 = np.linalg.norm(X - c0, axis=1)
                d1 = np.linalg.norm(X - c1, axis=1)
                labels = (d1 < d0).astype(int)
                if labels.sum() == 0 or labels.sum() == len(X):
                    break
                c0 = X[labels == 0].mean(axis=0)
                c1 = X[labels == 1].mean(axis=0)
            self.team_signatures = [c0, c1]
            self.team_initialized = True
            self.team_init_buffer = []
            print(f"ReID: team clustering completed — centroids L*a*b: {self.team_signatures[0]} | {self.team_signatures[1]}")

    def assign_team(self, color_sig: np.ndarray) -> int:
        """Assign a team id (0 or 1) based on current team signatures.

        If signatures are empty, create first centroid -> team 0.
        If one exists and new signature is sufficiently different, create team 1.
        Otherwise assign to nearest centroid.
        """
        # ignore invalid signatures
        if color_sig is None or np.all(color_sig == 0):
            return -1

        def dist(a, b):
            return np.linalg.norm(a - b)

        # If not yet initialized, accumulate signatures and run a simple k-means
        if not self.team_initialized:
            # add to buffer
            self.team_init_buffer.append(color_sig.copy())
            if len(self.team_init_buffer) >= self.team_init_samples:
                # run simple kmeans k=2 on collected signatures
                X = np.vstack(self.team_init_buffer).astype(np.float32)
                # init centroids: pick two farthest points
                idx0 = 0
                idx1 = int(np.argmax([np.linalg.norm(X[i] - X[idx0]) for i in range(len(X))]))
                c0 = X[idx0].copy()
                c1 = X[idx1].copy()
                for _ in range(10):
                    d0 = np.linalg.norm(X - c0, axis=1)
                    d1 = np.linalg.norm(X - c1, axis=1)
                    labels = (d1 < d0).astype(int)
                    if labels.sum() == 0 or labels.sum() == len(X):
                        break
                    c0 = X[labels == 0].mean(axis=0)
                    c1 = X[labels == 1].mean(axis=0)

                self.team_signatures = [c0, c1]
                self.team_initialized = True
                # clear buffer to save memory
                self.team_init_buffer = []
                print(f"ReID: team clustering completed — centroids L*a*b: {self.team_signatures[0]} | {self.team_signatures[1]}")
                # decide assignment for current signature
                d0 = dist(color_sig, self.team_signatures[0])
                d1 = dist(color_sig, self.team_signatures[1])
                return 0 if d0 <= d1 else 1
            else:
                # Not enough data yet -> unknown
                return -1

        # After initialization, assign to nearest centroid only if distance below threshold
        if len(self.team_signatures) >= 2:
            dists = [dist(color_sig, c) for c in self.team_signatures[:2]]
            best = int(np.argmin(dists))
            if min(dists) > self.team_distance_threshold:
                # too far from known teams -> unknown
                print(f"ReID: team assignment unknown (min dist {min(dists):.1f} > threshold {self.team_distance_threshold})")
                return -1
            # update centroid
            self.team_signatures[best] = (1 - self.team_update_alpha) * self.team_signatures[best] + self.team_update_alpha * color_sig
            # debug print for assignments
            print(f"ReID: assign team {best} (distances {dists[0]:.1f},{dists[1]:.1f})")
            return best

        # fallback
        return -1
    
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Similitud coseno entre dos features"""
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6)
    
    def match_with_lost_tracks(self, features: np.ndarray, bboxes: np.ndarray) -> Dict[int, int]:
        """
        Intenta recuperar tracks perdidos usando ReID
        
        Returns:
            Dict mapping: detection_idx -> recovered_track_id
        """
        if len(features) == 0 or len(self.lost_tracks) == 0:
            return {}
        
        current_time = time.time()
        recoveries = {}
        
        for det_idx, det_feat in enumerate(features):
            best_match_id = None
            best_similarity = self.similarity_threshold
            
            for track_id, track in list(self.lost_tracks.items()):
                # Verificar si no ha pasado demasiado tiempo
                if current_time - track.last_seen > self.max_lost_time:
                    del self.lost_tracks[track_id]
                    continue
                
                # Comparar con features históricas del track
                for hist_feat in track.feature_history:
                    sim = self.compute_similarity(det_feat, hist_feat)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match_id = track_id
            
            if best_match_id is not None:
                recoveries[det_idx] = best_match_id
        
        return recoveries
    
    def update(self, 
               image: np.ndarray,
               detections: np.ndarray,
               scores: np.ndarray,
               classes: np.ndarray) -> List[Tuple[int, np.ndarray, int]]:
        """
        Actualiza tracker con nuevas detecciones
        
        Args:
            image: Frame completo BGR
            detections: Nx4 array [x1,y1,x2,y2]
            scores: N array de confidencias
            classes: N array de class IDs (0=player, 1=ball, 2=referee, 3=goalkeeper)
            
        Returns:
            Lista de (track_id, bbox, class_id) para tracks activos
        """
        current_time = time.time()
        
        # Separar jugadores/árbitros/porteros de balón
        player_mask = (classes == 0) | (classes == 2) | (classes == 3)  # player, referee, goalkeeper
        player_dets = detections[player_mask]
        player_scores = scores[player_mask]
        player_classes = classes[player_mask]  # Mantener clases originales

        # initialize TeamClassifier lazily
        if self.team_classifier is None:
            self.team_classifier = TeamClassifier(image.shape,
                                                 init_samples=self.team_init_samples,
                                                 min_chroma=self.team_min_chroma,
                                                 distance_threshold=self.team_distance_threshold,
                                                 confirm_samples=self.team_confirm_samples)
        
        # Extraer features solo de jugadores
        features = self.extract_features(image, player_dets)
        
        # Intentar recuperar tracks perdidos
        recoveries = self.match_with_lost_tracks(features, player_dets)
        
        # TODO: Aquí normalmente iría el matching ByteTrack (IoU + Kalman)
        # Por simplicidad, usamos matching greedy basado en features + IoU
        
        # Matching simple: asociar detecciones a tracks activos
        matched_tracks = set()
        unmatched_dets = []
        results = []
        
        for det_idx, (bbox, feat) in enumerate(zip(player_dets, features)):
            current_class = int(player_classes[det_idx])
            # Verificar si esta detección recupera un track perdido
            if det_idx in recoveries:
                track_id = recoveries[det_idx]
                if track_id in self.lost_tracks:
                    track = self.lost_tracks.pop(track_id)
                    self.active_tracks[track_id] = track
                else:
                    continue
                
                # Actualizar track
                track.last_bbox = bbox
                track.last_seen = current_time
                track.feature_history.append(feat)
                track.trajectory.append(self._bbox_center(bbox))
                track.original_class = current_class
                
                matched_tracks.add(track_id)
                results.append((track_id, bbox, current_class))
                continue
            
            # Buscar mejor match con tracks activos (IoU + feature similarity)
            best_track_id = None
            best_score = 0.0
            
            for track_id, track in self.active_tracks.items():
                if track_id in matched_tracks:
                    continue
                
                # IoU
                iou = self._compute_iou(bbox, track.last_bbox)
                
                # Feature similarity
                if len(track.feature_history) > 0:
                    feat_sim = max([self.compute_similarity(feat, h) 
                                   for h in track.feature_history])
                else:
                    feat_sim = 0.0
                
                # Score combinado (80% feature, 20% IoU)
                combined_score = 0.8 * feat_sim + 0.2 * iou
                
                if combined_score > best_score and combined_score > 0.4:
                    best_score = combined_score
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Match encontrado
                track = self.active_tracks[best_track_id]
                track.last_bbox = bbox
                track.last_seen = current_time
                track.feature_history.append(feat)
                # update team classifier with this detection and possibly confirm team
                try:
                    if self.team_classifier is not None:
                        self.team_classifier.add_detection(best_track_id, bbox, image, class_id=current_class)
                        team_assigned = self.team_classifier.get_team(best_track_id)
                        if team_assigned != -1:
                            track.team_id = team_assigned
                except Exception:
                    pass
                track.trajectory.append(self._bbox_center(bbox))
                track.original_class = current_class
                
                matched_tracks.add(best_track_id)
                results.append((best_track_id, bbox, current_class))
            else:
                # Nueva detección sin match
                unmatched_dets.append((bbox, feat, current_class))
        
        # Crear nuevos tracks para detecciones no asociadas
        for bbox, feat, cls in unmatched_dets:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            # initial team unknown (referee/ball handled below)
            color_sig = self.compute_color_signature(image, bbox)
            assigned_team = -1
            if cls in (2,):  # referee -> no team
                assigned_team = -1
            elif cls == 1:  # ball
                assigned_team = -1

            track = TrackState(
                track_id=track_id,
                last_seen=current_time,
                last_bbox=bbox,
                feature_history=deque([feat], maxlen=self.feature_buffer_size),
                trajectory=deque([self._bbox_center(bbox)], maxlen=300),  # 10s @ 30fps
                original_class=cls,
                color_signature=color_sig,
                # add a small color history to confirm team over multiple frames
                # reuse feature_history size for limit
                
                team_id=assigned_team
            )

            # add per-track color history container
            track.color_history = deque([color_sig], maxlen=self.team_confirm_samples)
            
            self.active_tracks[track_id] = track
            # register sample with TeamClassifier and possibly confirm
            try:
                if self.team_classifier is not None:
                    self.team_classifier.add_detection(track_id, bbox, image, class_id=cls)
                    team_assigned = self.team_classifier.get_team(track_id)
                    if team_assigned != -1:
                        track.team_id = team_assigned
            except Exception:
                pass

            results.append((track_id, bbox, cls))
        
        # Mover tracks no actualizados a lost_tracks
        for track_id in list(self.active_tracks.keys()):
            if track_id not in matched_tracks:
                track = self.active_tracks[track_id]
                if current_time - track.last_seen > self.max_age / 30.0:  # Asumir 30 fps
                    self.lost_tracks[track_id] = self.active_tracks.pop(track_id)
        
        # Procesar balón (ID único = 0)
        ball_mask = classes == 1  # class 1 = ball
        if np.any(ball_mask):
            ball_bbox = detections[ball_mask][0]
            results.append((0, ball_bbox, 1))  # ID=0 para balón
        
        return results
    
    def _bbox_center(self, bbox: np.ndarray) -> np.ndarray:
        """Calcula centro de bbox"""
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    
    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calcula IoU entre dos bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def get_track_trajectory(self, track_id: int, seconds: float = 10.0) -> Optional[np.ndarray]:
        """Obtiene trayectoria de un track en los últimos N segundos"""
        if track_id in self.active_tracks:
            track = self.active_tracks[track_id]
            return np.array(list(track.trajectory))
        return None
    
    def get_active_tracks_count(self) -> Dict[str, int]:
        """Retorna conteo de tracks activos y perdidos"""
        return {
            'active': len(self.active_tracks),
            'lost': len(self.lost_tracks),
            'total_ids': self.next_track_id - 1
        }
