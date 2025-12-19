"""
Team Classifier - Diferenciación automática de equipos por color de camiseta
============================================================================
Usa K-means clustering para asignar jugadores a equipos
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from typing import Dict, Tuple, Optional


class TeamClassifier:
    """
    Clasifica jugadores en equipos basándose en colores de camiseta
    """
    
    def __init__(self, n_teams: int = 3):
        """
        Args:
            n_teams: Número de equipos/grupos (local, visitante, árbitro)
        """
        self.n_teams = n_teams
        self.team_colors = None  # Se calculará dinámicamente
        self.track_team_votes: Dict[int, list] = defaultdict(list)
        self.stable_assignments: Dict[int, int] = {}
        
    def extract_jersey_color(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extrae color dominante de la camiseta (zona superior del jugador)
        
        Args:
            image: Frame BGR
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Color promedio en HSV [H, S, V]
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # Clamp
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0, 0])
        
        # Crop jugador
        player_crop = image[y1:y2, x1:x2]
        
        if player_crop.size == 0:
            return np.array([0, 0, 0])
        
        # Zona de camiseta: 20-50% de altura (torso superior)
        crop_h = player_crop.shape[0]
        jersey_roi = player_crop[int(crop_h * 0.2):int(crop_h * 0.5), :]
        
        if jersey_roi.size == 0:
            return np.array([0, 0, 0])
        
        # Convertir a HSV (mejor para colores)
        hsv = cv2.cvtColor(jersey_roi, cv2.COLOR_BGR2HSV)
        
        # Crear máscara para eliminar blancos/negros/grises extremos
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 200])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        mask = cv2.bitwise_not(mask)
        
        # Extraer píxeles válidos
        valid_pixels = hsv[mask > 0]
        
        if len(valid_pixels) < 10:
            # No hay suficientes píxeles, usar promedio simple
            return hsv.reshape(-1, 3).mean(axis=0)
        
        # Color dominante = mediana de píxeles válidos
        dominant_color = np.median(valid_pixels, axis=0)
        
        return dominant_color
    
    def classify_teams_batch(self, 
                            image: np.ndarray, 
                            tracks: list) -> Dict[int, int]:
        """
        Clasifica todos los tracks visibles en equipos USANDO EL MODELO YOLO
        
        Args:
            image: Frame BGR
            tracks: Lista de (track_id, bbox, class_id)
            
        Returns:
            Dict {track_id: team_id} donde:
              - 0, 1 = jugadores de campo por color
              - 2 = árbitros (del modelo)
              - 3 = porteros (se agrupan por color con su equipo)
        """
        if not tracks:
            return {}
        
        assignments = {}
        player_tracks = []  # players (0) y goalkeepers (3) se clasifican por color
        
        for tid, bbox, cls in tracks:
            if cls == 2:  # Referee - DIRECTO DEL MODELO
                assignments[tid] = 2
                # Mantener votación para estabilidad
                self.track_team_votes[tid].append(2)
                if len(self.track_team_votes[tid]) > 30:
                    self.track_team_votes[tid].pop(0)
                self.stable_assignments[tid] = 2
            else:  # Players (0) y Goalkeepers (3) - clasificar por color
                player_tracks.append((tid, bbox, cls))
        
        if len(player_tracks) < 2:
            for tid, _, _ in player_tracks:
                assignments[tid] = 0
            return assignments
        
        # Extraer colores para K-means
        colors = []
        track_ids = []
        original_classes = []
        
        for track_id, bbox, cls in player_tracks:
            color = self.extract_jersey_color(image, bbox)
            colors.append(color)
            track_ids.append(track_id)
            original_classes.append(cls)
        
        colors = np.array(colors)
        
        # K-means para 2 equipos
        n_clusters = min(2, len(player_tracks))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors)
        
        self.team_colors = kmeans.cluster_centers_
        
        # Asignar con votación
        for track_id, label, orig_cls in zip(track_ids, labels, original_classes):
            team_id = int(label)  # 0 o 1
            
            # Sistema de votación (últimos 30 frames)
            self.track_team_votes[track_id].append(team_id)
            if len(self.track_team_votes[track_id]) > 30:
                self.track_team_votes[track_id].pop(0)
            
            # Asignar por mayoría de votos
            votes = self.track_team_votes[track_id]
            if len(votes) >= 5:  # Mínimo 5 observaciones
                stable_team = Counter(votes).most_common(1)[0][0]
                self.stable_assignments[track_id] = stable_team
                assignments[track_id] = stable_team
            else:
                assignments[track_id] = team_id
        
        return assignments
    
    def _identify_referee_cluster(self, cluster_centers: np.ndarray) -> int:
        """
        Identifica cuál cluster corresponde al árbitro
        Árbitros típicamente: negro (bajo V) o colores poco saturados
        """
        scores = []
        for i, center in enumerate(cluster_centers):
            h, s, v = center
            # Score: bajo V y/o baja S indica árbitro
            score = (255 - v) + (255 - s) * 0.5
            scores.append(score)
        
        return int(np.argmax(scores))
    
    def get_team_id(self, track_id: int) -> int:
        """Obtiene team_id estable para un track"""
        return self.stable_assignments.get(track_id, -1)
    
    def get_team_color_bgr(self, team_id: int) -> Optional[Tuple[int, int, int]]:
        """Retorna color BGR para visualización de un equipo"""
        if self.team_colors is None or team_id >= len(self.team_colors):
            colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Verde, Rojo, Amarillo
            return colors[team_id] if team_id < 3 else (255, 255, 255)
        
        hsv_color = self.team_colors[team_id].astype(np.uint8).reshape(1, 1, 3)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        return tuple(map(int, bgr_color))
    
    def reset_voting(self, track_id: int):
        """Resetea votación para un track (útil cuando reaparece)"""
        if track_id in self.track_team_votes:
            self.track_team_votes[track_id].clear()
        if track_id in self.stable_assignments:
            del self.stable_assignments[track_id]
