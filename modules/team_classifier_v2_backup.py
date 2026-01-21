"""
TeamClassifier V2: Clasificación robusta de equipos usando HSV + K-means

Características:
- Clustering K-means en espacio HSV sobre colores de camiseta
- Identificación automática de árbitros (negro/amarillo)
- Sistema de votación para estabilidad de asignaciones
- Diseño simple y mantenible
"""
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Optional, Tuple


class TeamClassifierV2:
    """
    Clasificador robusto de equipos usando análisis de color HSV.
    
    Workflow:
    1. Extrae región de torso (parte superior central del bbox)
    2. Calcula color dominante en espacio HSV (mediana H, S, V)
    3. Acumula muestras hasta tener suficientes
    4. Ejecuta K-means (k=2) para encontrar 2 clusters de equipos
    5. Identifica árbitros por patrón de color específico
    6. Usa sistema de votación temporal para confirmar asignaciones
    """
    
    def __init__(
        self,
        min_samples: int = 60,
        min_saturation: float = 20.0,
        min_value: float = 30.0,
        vote_history: int = 5,
        referee_detection: bool = True
    ):
        """
        Args:
            min_samples: Mínimas muestras antes de ejecutar K-means
            min_saturation: Saturación mínima para considerar muestra válida (0-255)
            min_value: Valor mínimo para evitar negros puros (0-255)
            vote_history: Número de frames para sistema de votación
            referee_detection: Si True, identifica árbitros automáticamente
        """
        self.min_samples = min_samples
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.vote_history = vote_history
        self.referee_detection = referee_detection
        
        # Buffers para clustering
        self.color_samples = []  # Lista de (H, S, V) tuples
        self.team_centers = None  # Centroides K-means: np.array shape (2, 3)
        self.initialized = False
        
        # Votación por track
        self.track_votes = defaultdict(lambda: deque(maxlen=self.vote_history))
        self.track_team = {}  # track_id -> team (0, 1, or -1 for referee/unknown)
        
        # Estadísticas
        self.frames_processed = 0
        
    def _extract_jersey_region(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae región del torso (camiseta) desde bbox.
        
        Args:
            image: Frame BGR
            bbox: [x1, y1, x2, y2]
            
        Returns:
            ROI del torso o None si inválido
        """
        x1, y1, x2, y2 = map(int, bbox)
        h_img, w_img = image.shape[:2]
        
        # Clips
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        
        # Extraer parte superior central (torso, evitando cabeza y piernas)
        # Vertical: desde 15% hasta 50% de la altura
        # Horizontal: desde 20% hasta 80% del ancho
        ty1 = y1 + int(0.15 * bbox_h)
        ty2 = y1 + int(0.50 * bbox_h)
        tx1 = x1 + int(0.20 * bbox_w)
        tx2 = x1 + int(0.80 * bbox_w)
        
        # Validar
        if ty2 <= ty1 or tx2 <= tx1:
            return None
            
        roi = image[ty1:ty2, tx1:tx2]
        
        # Requiere tamaño mínimo
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            return None
            
        return roi
    
    def _get_dominant_color_hsv(self, roi: np.ndarray) -> Tuple[float, float, float]:
        """
        Calcula color dominante de un ROI en espacio HSV usando mediana.
        
        Args:
            roi: Imagen BGR del torso
            
        Returns:
            (H, S, V) tupla con valores medianos
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Tomar región central (70%) para evitar bordes
        h, w = hsv.shape[:2]
        margin_h = max(1, int(0.15 * h))
        margin_w = max(1, int(0.15 * w))
        core = hsv[margin_h:h-margin_h, margin_w:w-margin_w]
        
        if core.size == 0:
            core = hsv
            
        # Calcular mediana por canal
        h_median = np.median(core[:, :, 0])
        s_median = np.median(core[:, :, 1])
        v_median = np.median(core[:, :, 2])
        
        return (h_median, s_median, v_median)
    
    def _is_referee_color(self, hsv: Tuple[float, float, float]) -> bool:
        """
        Detecta si un color HSV corresponde a árbitro (negro o amarillo típicamente).
        
        Args:
            hsv: (H, S, V) tuple
            
        Returns:
            True si parece color de árbitro
        """
        if not self.referee_detection:
            return False
            
        h, s, v = hsv
        
        # Negro: muy bajo V y baja S (más estricto)
        if v < 45 and s < 60:
            return True
            
        # Amarillo/Verde fluorescente: H en rango 25-55, muy alta S (más estricto)
        if 25 <= h <= 55 and s > 150 and v > 120:
            return True
            
        return False
    
    def _run_kmeans(self):
        """
        Ejecuta K-means sobre muestras acumuladas para encontrar 2 equipos.
        """
        if len(self.color_samples) < self.min_samples:
            return False
            
        # Preparar datos
        X = np.array(self.color_samples, dtype=np.float32)
        
        # K-means con k=2
        K = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        attempts = 10
        
        ret, labels, centers = cv2.kmeans(
            X, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
        )
        
        self.team_centers = centers  # Shape (2, 3) - HSV
        
        # Mapear clusters a equipos: más oscuro (menor V) = team 0
        v_values = centers[:, 2]
        idx_team0 = int(np.argmin(v_values))
        idx_team1 = 1 - idx_team0
        
        # Guardar mapeo (lo usaremos al asignar)
        self.cluster_to_team = {idx_team0: 0, idx_team1: 1}
        
        self.initialized = True
        
        print(f"[TeamClassifierV2] K-means inicializado con {len(self.color_samples)} muestras")
        print(f"  Centro team 0 (HSV): {centers[idx_team0]} (índice cluster {idx_team0})")
        print(f"  Centro team 1 (HSV): {centers[idx_team1]} (índice cluster {idx_team1})")
        print(f"  Mapeo cluster->team: {self.cluster_to_team}")
        
        return True
    
    def add_detection(
        self,
        track_id: int,
        bbox: np.ndarray,
        image: np.ndarray,
        class_id: Optional[int] = None
    ):
        """
        Procesa una detección y actualiza asignación de equipo.
        
        Args:
            track_id: ID del track
            bbox: [x1, y1, x2, y2]
            image: Frame BGR
            class_id: Clase YOLO (0=player, 1=ball, 2=referee, 3=goalkeeper)
        """
        self.frames_processed += 1
        
        # Ignorar ball
        if class_id == 1:
            return
            
        # Si es referee (class 2), asignar directamente
        if class_id == 2:
            self.track_team[track_id] = -1  # -1 = referee/NA
            return
            
        # Extraer región de torso
        roi = self._extract_jersey_region(image, bbox)
        if roi is None:
            return
            
        # Calcular color dominante
        hsv_color = self._get_dominant_color_hsv(roi)
        h, s, v = hsv_color
        
        # Filtrar muestras de baja calidad (poco saturadas o muy oscuras)
        if s < self.min_saturation or v < self.min_value:
            return
            
        # Detectar árbitro por color
        if self._is_referee_color(hsv_color):
            self.track_team[track_id] = -1
            return
            
        # Si aún no tenemos clusters, acumular muestras
        if not self.initialized:
            self.color_samples.append(hsv_color)
            
            # Intentar inicializar K-means
            if len(self.color_samples) >= self.min_samples:
                self._run_kmeans()
                
            return
            
        # Con clusters inicializados, asignar equipo
        # Calcular distancia a cada centro
        hsv_array = np.array(hsv_color, dtype=np.float32)
        distances = np.linalg.norm(self.team_centers - hsv_array, axis=1)
        
        # Cluster más cercano
        cluster_idx = int(np.argmin(distances))
        team = self.cluster_to_team[cluster_idx]
        
        # Sistema de votación
        self.track_votes[track_id].append(team)
        
        # Confirmar si hay mayoría en el historial
        votes = list(self.track_votes[track_id])
        if len(votes) >= max(2, self.vote_history // 2):
            # Contar votos
            counts = np.bincount(votes)
            confirmed_team = int(np.argmax(counts))
            self.track_team[track_id] = confirmed_team
    
    def get_team(self, track_id: int) -> int:
        """
        Obtiene equipo asignado a un track.
        
        Args:
            track_id: ID del track
            
        Returns:
            0 o 1 para equipos, -1 para referee/unknown
        """
        return self.track_team.get(track_id, -1)
    
    def is_ready(self) -> bool:
        """Indica si el clasificador está inicializado y listo."""
        return self.initialized
    
    def reset(self):
        """Reinicia el clasificador."""
        self.color_samples = []
        self.team_centers = None
        self.initialized = False
        self.track_votes.clear()
        self.track_team.clear()
        self.frames_processed = 0
