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
            
            track = TrackState(
                track_id=track_id,
                last_seen=current_time,
                last_bbox=bbox,
                feature_history=deque([feat], maxlen=self.feature_buffer_size),
                trajectory=deque([self._bbox_center(bbox)], maxlen=300),  # 10s @ 30fps
                original_class=cls
            )
            
            self.active_tracks[track_id] = track
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
