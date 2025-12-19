import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Optional, Tuple


class TeamClassifier:
    """
    Classifica jugadores en dos equipos usando color del torso + clustering.

    API principal:
    - add_detection(track_id, bbox, image, class_id=None, homography=None)
    - get_team(track_id) -> int (0/1 or -1 unknown)
    - ready() -> bool (True cuando clustering inicial completado)

    Diseño:
    - Extrae región de torso desde la bbox (zona central-superior)
    - Convierte a LAB, calcula firma robusta (mediana L,a,b)
    - Acumula firmas y lanza kmeans k=2 cuando hay suficientes muestras
    - Asigna equipos por distancia a centroides; confirma por historial temporal
    - Usa homografía (si se provee) como prior posicional para estabilizar izquierdas/derechas
    """

    def __init__(self,
                 image_shape: Tuple[int,int],
                 init_samples: int = 40,
                 min_chroma: float = 6.0,
                 distance_threshold: float = 40.0,
                 confirm_samples: int = 3,
                 side_prior_alpha: float = 0.2):
        self.img_h, self.img_w = image_shape[:2]
        self.init_samples = init_samples
        self.min_chroma = min_chroma
        self.distance_threshold = distance_threshold
        self.confirm_samples = confirm_samples
        self.side_prior_alpha = side_prior_alpha

        # buffers
        self.sample_buffer = []  # list of (lab_sig, bbox_center_x)
        self.team_centroids = None  # np.array shape (2,3) in LAB
        self.team_assigned_side = None  # which centroid corresponds to left/right

        # per-track histories
        self.color_history = defaultdict(lambda: deque(maxlen=self.confirm_samples))
        self.team_history = defaultdict(lambda: deque(maxlen=self.confirm_samples))
        self.track_team = {}  # confirmed team per track

    # ------------------ color / torso helpers ------------------
    def _extract_torso(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.img_w - 1, x2), min(self.img_h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        h = y2 - y1
        w = x2 - x1
        # torso: upper 60% height, central 60% width
        ty1 = y1 + int(0.12 * h)
        ty2 = y1 + int(0.60 * h)
        tx1 = x1 + int(0.2 * w)
        tx2 = x1 + int(0.8 * w)
        if ty2 <= ty1 or tx2 <= tx1:
            return None
        torso = image[ty1:ty2, tx1:tx2]
        return torso

    def _lab_signature(self, roi: np.ndarray) -> np.ndarray:
        # take core center 70% to avoid edges; compute median per channel
        h, w = roi.shape[:2]
        mh = max(1, int(0.15 * h))
        mw = max(1, int(0.15 * w))
        core = roi[mh:h-mh, mw:w-mw] if h > 2*mh and w > 2*mw else roi
        lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB)
        l_med = np.median(lab[:,:,0])
        a_med = np.median(lab[:,:,1])
        b_med = np.median(lab[:,:,2])
        return np.array([l_med, a_med, b_med], dtype=np.float32)

    def _chroma(self, lab_sig: np.ndarray) -> float:
        return float(np.linalg.norm(lab_sig[1:3]))

    # ------------------ clustering ------------------
    def _run_kmeans(self):
        # Run kmeans on sample_buffer's lab signatures
        X = np.array([s[0] for s in self.sample_buffer], dtype=np.float32)
        if len(X) < 2:
            return False
        # use cv2.kmeans
        K = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        attempts = 5
        ret, labels, centers = cv2.kmeans(X, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        self.team_centroids = centers  # shape (2,3)
        # determine which centroid is left/right using average bbox x in samples
        avg_x = [0.0, 0.0]
        counts = [0,0]
        for (lab, cx), lbl in zip(self.sample_buffer, labels.flatten()):
            avg_x[int(lbl)] += cx
            counts[int(lbl)] += 1
        for i in range(2):
            if counts[i]>0:
                avg_x[i] /= counts[i]
            else:
                avg_x[i] = self.img_w/2
        # team_assigned_side[0] = index of centroid that is left side (smaller x)
        left_idx = int(np.argmin(avg_x))
        right_idx = 1 - left_idx
        self.team_assigned_side = (left_idx, right_idx)
        return True

    # ------------------ public API ------------------
    def add_detection(self, track_id: int, bbox: np.ndarray, image: np.ndarray,
                      class_id: Optional[int] = None, homography: Optional[np.ndarray] = None):
        """
        Añade una detección (por frame) para su evaluación.

        - track_id: id del track (persistente en el tracker)
        - bbox: [x1,y1,x2,y2]
        - image: frame BGR
        - class_id: clase YOLO (para ignorar ball/referee)
        - homography: optional 3x3 homography para proyectar centro
        """
        # ignore ball/referee if known
        if class_id is not None and int(class_id) in (1,2):
            return

        torso = self._extract_torso(image, bbox)
        if torso is None:
            return
        lab_sig = self._lab_signature(torso)
        chroma = self._chroma(lab_sig)
        # ignore low-chroma samples
        if chroma < self.min_chroma:
            return

        cx = (bbox[0] + bbox[2]) / 2.0
        # store sample for initial clustering
        if self.team_centroids is None:
            self.sample_buffer.append((lab_sig, cx))
            if len(self.sample_buffer) >= self.init_samples:
                self._run_kmeans()
            return

        # after centroids exist, add to per-track history and try to confirm
        self.color_history[track_id].append(lab_sig)
        # use median of history to test
        arr = np.vstack(list(self.color_history[track_id]))
        med = np.median(arr, axis=0)
        # distances to centroids
        dists = np.linalg.norm(self.team_centroids - med[None,:], axis=1)
        best = int(np.argmin(dists))
        if dists[best] > self.distance_threshold:
            # too far -> mark unknown for now
            self.team_history[track_id].append(-1)
        else:
            # map centroid index to team id (0 = home/left, 1 = away/right)
            left_idx, right_idx = self.team_assigned_side
            team = 0 if best == left_idx else 1
            self.team_history[track_id].append(team)

        # confirm if history majority agrees
        vals = [v for v in self.team_history[track_id] if v != -1]
        if len(vals) >= self.confirm_samples:
            # majority vote
            team_confirmed = int(np.bincount(vals).argmax())
            prev = self.track_team.get(track_id, None)
            if prev is None or prev != team_confirmed:
                # apply hysteresis: only change if confirmed
                self.track_team[track_id] = team_confirmed

    def get_team(self, track_id: int) -> int:
        return int(self.track_team.get(track_id, -1))

    def ready(self) -> bool:
        return self.team_centroids is not None

    def reset(self):
        self.sample_buffer = []
        self.team_centroids = None
        self.team_assigned_side = None
        self.color_history.clear()
        self.team_history.clear()
        self.track_team.clear()
