import cv2
import numpy as np
import os
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
                 init_samples: int = 60,
                 min_chroma: float = 4.0,
                 distance_threshold: float = 40.0,
                 confirm_samples: int = 5,
                 roi_expansion: float = 0.15,
                 clahe_clip: float = 2.0,
                 save_rois: bool = False,
                 rois_dir: str = 'team_rois',
                 pos_init_samples: Optional[int] = None,
                 area_seed_threshold: Optional[float] = None,
                 use_hue_hist: bool = True,
                 hue_bins: int = 12,
                 hue_weight: float = 1.0,
                 bootstrap_frames: int = 60,
                 bootstrap_force: bool = False):
        self.img_h, self.img_w = image_shape[:2]
        self.init_samples = init_samples
        self.min_chroma = min_chroma
        self.distance_threshold = distance_threshold
        self.confirm_samples = confirm_samples
        self.roi_expansion = roi_expansion
        self.clahe_clip = clahe_clip
        # fraction of bbox height to use as lower limit for torso (avoid dorsal numbers)
        self.torso_lower_frac = 0.45
        # apply slight blur to torso core before computing color signature to reduce small high-contrast numbers/logos
        self.jersey_denoise_ksize = (5, 5)
        self.save_rois = save_rois
        self.rois_dir = rois_dir
        if self.save_rois:
            os.makedirs(self.rois_dir, exist_ok=True)
        # positional seeding: collect centers even if color low
        if pos_init_samples is not None:
            self.pos_init_samples = int(pos_init_samples)
        else:
            self.pos_init_samples = max(30, int(self.init_samples * 0.5))
        self.pos_samples = []  # list of cx values
        self.pos_initialized = False
        self.pos_centers = None
        # accept large ROIs as seeds even if chroma low
        if area_seed_threshold is not None:
            self.area_seed_threshold = float(area_seed_threshold)
        else:
            self.area_seed_threshold = 0.02 * (self.img_h * self.img_w)  # 2% of frame area
        # hue histogram features for robustness across angles
        self.use_hue_hist = bool(use_hue_hist)
        self.hue_bins = int(hue_bins)
        self.hue_weight = float(hue_weight)
        # bootstrap across frames when many wide-shot frames available
        self.bootstrap_frames = int(bootstrap_frames)
        self.bootstrap_force = bool(bootstrap_force)
        self.bootstrap_buffer = []  # list of feat vectors for global bootstrap
        self.frame_count = 0
        # small-ROI handling and track-median clustering
        self.min_roi_pixels = 32 * 32
        self.resize_small_to = (64, 64)
        # minimum roi side (pixels) below which sample is considered low-quality
        self.min_roi_side = 32
        # whether to reject tiny ROIs from seeding clustering (still kept for bootstrap)
        self.reject_small_rois = True
        self.cluster_on_track_medians = True
        self.track_median_min_samples = max(4, int(self.confirm_samples))
        self.feat_history = defaultdict(lambda: deque(maxlen=self.track_median_min_samples*3))
        # threshold to fallback to positional assignment when color centroids are too close
        self.centroid_lab_separation_threshold = 12.0
        # ensemble weights: combine color vs position when deciding team
        self.color_weight = 1.0
        self.position_weight = 0.8
        # prefer color-based mapping of centroids to team ids instead of positional mapping
        self.prefer_color_for_mapping = True
        # hysteresis: number of frames to hold a confirmed team before allowing switches
        self.hold_frames = 12
        # require new color distance to be this factor better than current to allow switch during hold
        self.change_distance_factor = 0.85
        # track last confirmation frame per track
        self.last_confirm_frame = {}

        # buffers
        # store tuples (feat_vec, lab_sig, bbox_center_x)
        self.sample_buffer = []
        self.team_centroids = None  # np.array shape (2,3) in LAB
        self.team_assigned_side = None  # which centroid corresponds to left/right

        # per-track histories
        # color_history now stores full feature vectors for median
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
        # torso: upper area but allow expansion to include more jersey
        ty1 = y1 + int(0.12 * h)
        ty2 = y1 + int(self.torso_lower_frac * h)
        tx1 = x1 + int(0.2 * w)
        tx2 = x1 + int(0.8 * w)
        # expand ROI by fraction
        expand_x = int(self.roi_expansion * w)
        expand_y = int(self.roi_expansion * h)
        tx1 = max(x1, tx1 - expand_x)
        tx2 = min(x2, tx2 + expand_x)
        ty1 = max(y1, ty1 - expand_y)
        ty2 = min(y2, ty2 + expand_y)
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
        # if roi is tiny, upscale to reduce noise (but caller can still mark it low-quality)
        if roi.shape[0] * roi.shape[1] < self.min_roi_pixels:
            roi = cv2.resize(roi, self.resize_small_to, interpolation=cv2.INTER_LINEAR)
        # reduce small high-contrast artifacts (numbers/logos) with a mild blur
        try:
            core = cv2.GaussianBlur(core, self.jersey_denoise_ksize, 0)
        except Exception:
            pass
        # apply CLAHE on L channel to reduce lighting variation
        lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8,8))
        l_eq = clahe.apply(l)
        lab[:, :, 0] = l_eq
        l_med = np.median(lab[:, :, 0])
        a_med = np.median(lab[:, :, 1])
        b_med = np.median(lab[:, :, 2])
        return np.array([l_med, a_med, b_med], dtype=np.float32)

    def _hue_hist(self, roi: np.ndarray) -> np.ndarray:
        # resize small rois to stabilize histogram
        if roi.shape[0] * roi.shape[1] < self.min_roi_pixels:
            roi = cv2.resize(roi, self.resize_small_to, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        # compute histogram over hue [0,180)
        hist = cv2.calcHist([h], [0], None, [self.hue_bins], [0, 180])
        hist = hist.flatten().astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist

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
        self.team_centroids = centers  # shape (2,D)
        # map centroids to team ids. Prefer color-based mapping (darker L -> team 0),
        # otherwise fall back to positional mapping using average bbox x in samples.
        try:
            centers_lab = centers[:, :3]
            if self.prefer_color_for_mapping:
                # assign centroid with lower L (darker) to team 0
                l_vals = centers_lab[:, 0]
                idx_team0 = int(np.argmin(l_vals))
                idx_team1 = 1 - idx_team0
                self.team_assigned_side = (idx_team0, idx_team1)
            else:
                avg_x = [0.0, 0.0]
                counts = [0,0]
                for (feat, lab, cx), lbl in zip(self.sample_buffer, labels.flatten()):
                    avg_x[int(lbl)] += cx
                    counts[int(lbl)] += 1
                for i in range(2):
                    if counts[i]>0:
                        avg_x[i] /= counts[i]
                    else:
                        avg_x[i] = self.img_w/2
                left_idx = int(np.argmin(avg_x))
                right_idx = 1 - left_idx
                self.team_assigned_side = (left_idx, right_idx)
        except Exception:
            # fallback: default mapping
            self.team_assigned_side = (0,1)
        try:
            print(f"[TeamClassifier] KMeans centers (LAB part): {self.team_centroids[:, :3].tolist()} assigned_side={self.team_assigned_side}")
        except Exception:
            pass
        return True

    def _run_kmeans_on_track_medians(self):
        # build medians per track from feat_history
        medians = []
        tids = []
        for tid, deq in self.feat_history.items():
            if len(deq) >= self.track_median_min_samples:
                arr = np.vstack(list(deq))
                med = np.median(arr, axis=0)
                medians.append(med)
                tids.append(tid)
        if len(medians) < 2:
            return False
        X = np.array(medians, dtype=np.float32)
        try:
            K = 2
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
            attempts = 5
            ret, labels, centers = cv2.kmeans(X, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
            self.team_centroids = centers
            # prefer color-based mapping of centroids to team ids
            try:
                centers_lab = centers[:, :3]
                if self.prefer_color_for_mapping:
                    l_vals = centers_lab[:, 0]
                    idx_team0 = int(np.argmin(l_vals))
                    idx_team1 = 1 - idx_team0
                    self.team_assigned_side = (idx_team0, idx_team1)
                else:
                    avg_x = [0.0, 0.0]
                    counts = [0, 0]
                    for (feat, lab, cx) in self.sample_buffer:
                        dists = np.linalg.norm(self.team_centroids - feat[None, :], axis=1)
                        lbl = int(np.argmin(dists))
                        avg_x[lbl] += cx
                        counts[lbl] += 1
                    for i in range(2):
                        if counts[i] > 0:
                            avg_x[i] /= counts[i]
                        else:
                            avg_x[i] = self.img_w / 2
                    left_idx = int(np.argmin(avg_x))
                    right_idx = 1 - left_idx
                    self.team_assigned_side = (left_idx, right_idx)
            except Exception:
                self.team_assigned_side = (0,1)
            return True
        except Exception:
            return False

    def _run_bootstrap_kmeans(self):
        if len(self.bootstrap_buffer) < 2:
            return False
        X = np.array(self.bootstrap_buffer, dtype=np.float32)
        try:
            K = 2
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
            attempts = 5
            ret, labels, centers = cv2.kmeans(X, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
            self.team_centroids = centers
            # prefer color-based mapping when possible
            try:
                centers_lab = centers[:, :3]
                if self.prefer_color_for_mapping:
                    l_vals = centers_lab[:, 0]
                    idx_team0 = int(np.argmin(l_vals))
                    idx_team1 = 1 - idx_team0
                    self.team_assigned_side = (idx_team0, idx_team1)
                else:
                    # we don't have bbox x here; rely on positional split if available
                    pass
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _run_positional_init(self):
        # run 1D kmeans on collected cx positions to determine left/right split
        if len(self.pos_samples) < 2:
            return False
        X = np.array(self.pos_samples, dtype=np.float32).reshape(-1, 1)
        K = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        attempts = 3
        try:
            ret, labels, centers = cv2.kmeans(X, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
            centers = centers.flatten()
            left_idx = int(np.argmin(centers))
            right_idx = 1 - left_idx
            self.pos_centers = centers
            self.pos_initialized = True
            # store a simple split x midpoint for quick checks
            self.pos_split_x = float(centers.mean())
            return True
        except Exception:
            return False

    def _maybe_save_roi(self, roi: np.ndarray, track_id: int, idx: int = 0):
        if not self.save_rois:
            return
        try:
            fname = f"roi_tid{track_id}_{idx:06d}.png"
            path = os.path.join(self.rois_dir, fname)
            cv2.imwrite(path, roi)
        except Exception:
            pass

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
        # compute hue histogram feature
        hue_hist = self._hue_hist(torso) if self.use_hue_hist else None
        # build feature vector: LAB + (hue_hist * weight)
        if self.use_hue_hist and hue_hist is not None:
            feat = np.concatenate([lab_sig, hue_hist * self.hue_weight]).astype(np.float32)
        else:
            feat = lab_sig.astype(np.float32)
        # optionally save ROI for debugging
        if self.save_rois:
            self._maybe_save_roi(torso, track_id)
            # determine ROI quality (size in pixels and side lengths)
            roi_h, roi_w = torso.shape[:2]
            roi_pixels = roi_w * roi_h
            small_side = (roi_w < self.min_roi_side) or (roi_h < self.min_roi_side)
            low_quality_roi = (roi_pixels < self.min_roi_pixels) or small_side

            # do not drop immediately; allow large-area seeds or hue-based samples
            if chroma < self.min_chroma and not ( (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) >= self.area_seed_threshold ):
                # if hue provided, still proceed (we may rely on hue hist)
                if not self.use_hue_hist:
                    return

        cx = (bbox[0] + bbox[2]) / 2.0
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # increment frame counter for bootstrap logic
        self.frame_count += 1

        # store sample for initial clustering
        if self.team_centroids is None:
            # always collect positional samples for wide-shot initialization
            self.pos_samples.append(cx)
            if len(self.pos_samples) >= self.pos_init_samples and not self.pos_initialized:
                self._run_positional_init()

            # accept area-seed samples even if chroma low
            # decide whether to add to sample_buffer: reject very small/low-chroma ROIs
            accept_sample = False
            if area >= self.area_seed_threshold:
                accept_sample = True
            elif chroma >= self.min_chroma:
                # if chroma is acceptable but ROI very small, require a stronger chroma
                if not (self.reject_small_rois and low_quality_roi):
                    accept_sample = True
                else:
                    # if chroma is much higher than threshold, allow it
                    if chroma >= (self.min_chroma * 1.5):
                        accept_sample = True
            elif (self.use_hue_hist and hue_hist is not None):
                # accept hue-based sample even if chroma slightly low
                accept_sample = True

            if accept_sample:
                self.sample_buffer.append((feat, lab_sig, cx))
            # always push to bootstrap buffer (trimmed) to allow global bootstrap
            if len(self.bootstrap_buffer) < max(self.bootstrap_frames * 3, 300):
                self.bootstrap_buffer.append(feat)
            else:
                # keep last N
                self.bootstrap_buffer = self.bootstrap_buffer[-max(self.bootstrap_frames, 300):]  
            # cap buffer size to avoid unbounded memory
            if len(self.sample_buffer) > max(self.init_samples * 3, 300):
                self.sample_buffer = self.sample_buffer[-max(self.init_samples, 300):]
            # if we have enough color samples, run kmeans
            if len(self.sample_buffer) >= self.init_samples:
                self._run_kmeans()
                # if positional was initialized earlier, map centroids to sides
                if self.pos_initialized and self.team_centroids is not None:
                    # compute avg x per centroid using stored sample_buffer
                    avg_x = [0.0, 0.0]
                    counts = [0, 0]
                    for (feat2, lab, cx2) in [s for s in self.sample_buffer]:
                        # assign to nearest centroid in feat space
                        dists = np.linalg.norm(self.team_centroids - feat2[None, :], axis=1)
                        lbl = int(np.argmin(dists))
                        avg_x[lbl] += cx2
                        counts[lbl] += 1
                    for i in range(2):
                        if counts[i] > 0:
                            avg_x[i] /= counts[i]
                        else:
                            avg_x[i] = self.img_w / 2
                    left_idx = int(np.argmin(avg_x))
                    right_idx = 1 - left_idx
                    self.team_assigned_side = (left_idx, right_idx)
            # if not enough color samples but we've seen many frames, try bootstrap
            if (not self.team_centroids) and (self.frame_count >= self.bootstrap_frames):
                ok = self._run_bootstrap_kmeans()
                if ok and self.pos_initialized:
                    # map centroids to left/right using positional centers mean
                    # assign sample_buffer avg_x if available, else use pos_split_x
                    if len(self.sample_buffer) > 0:
                        avg_x = [0.0, 0.0]
                        counts = [0, 0]
                        for (feat2, lab, cx2) in [s for s in self.sample_buffer]:
                            dists = np.linalg.norm(self.team_centroids - feat2[None, :], axis=1)
                            lbl = int(np.argmin(dists))
                            avg_x[lbl] += cx2
                            counts[lbl] += 1
                        for i in range(2):
                            if counts[i] > 0:
                                avg_x[i] /= counts[i]
                            else:
                                avg_x[i] = self.img_w / 2
                        left_idx = int(np.argmin(avg_x))
                        right_idx = 1 - left_idx
                        self.team_assigned_side = (left_idx, right_idx)
                # if bootstrap forced, optionally clear buffers
                if self.bootstrap_force and self.team_centroids is not None:
                    self.sample_buffer = []
            return
            return

        # after centroids exist, add to per-track history and try to confirm
        # Always keep per-track history (used for median voting), even if ROI was small;
        # we rejected small ROIs only from seeding to avoid polluting initial clustering.
        self.color_history[track_id].append(lab_sig)
        # store full feature in feat_history for per-track medians
        self.feat_history[track_id].append(feat)
        # use median of history to test (LAB only)
        arr = np.vstack(list(self.color_history[track_id]))
        med = np.median(arr, axis=0)
        # distances to centroids computed only on LAB channels (first 3 dims)
        try:
            centers_lab = self.team_centroids[:, :3]
        except Exception:
            centers_lab = self.team_centroids
        # compute color distances in full feature space if available
        try:
            centers_feat = self.team_centroids
            med_feat = feat if feat is not None else med
            color_dists = np.linalg.norm(centers_feat - med_feat[None, :], axis=1)
        except Exception:
            # fallback to LAB-only distances
            color_dists = np.linalg.norm(centers_lab - med[None, :], axis=1)

        best = int(np.argmin(color_dists))
        # positional split (used as fallback)
        split_x = getattr(self, 'pos_split_x', self.img_w / 2)
        if color_dists[best] > self.distance_threshold * 2:
            # color very far -> try positional fallback for low-quality ROIs
            if low_quality_roi or chroma < self.min_chroma:
                fallback_team = 0 if cx < split_x else 1
                self.team_history[track_id].append(fallback_team)
                # continue to scoring below using fallback appended value
            else:
                self.team_history[track_id].append(-1)
                return
        # build scores per team: higher is better
        scores = []
        # compute centroid separation (LAB) to decide whether to trust color strongly
        try:
            centers_lab = self.team_centroids[:, :3]
            sep = float(np.linalg.norm(centers_lab[0] - centers_lab[1]))
        except Exception:
            sep = 0.0
        # if centroids are well separated, down-weight positional prior
        if sep > self.centroid_lab_separation_threshold:
            # scale down position influence (smaller -> favor color)
            pos_weight_effective = max(0.05, self.position_weight * (self.centroid_lab_separation_threshold / sep))
        else:
            pos_weight_effective = self.position_weight
        for i in range(2):
            # color score: inverse distance (smaller dist -> higher score)
            cd = float(color_dists[i]) if color_dists[i] > 0 else 1e-6
            color_score = 1.0 / cd
            # position score: for team i, prefer left (i==0) or right (i==1)
            pos_pref = 1.0 if (i == 0 and cx < split_x) or (i == 1 and cx >= split_x) else 0.0
            score = self.color_weight * color_score + pos_weight_effective * pos_pref
            scores.append(score)

        team = int(np.argmax(scores))
        self.team_history[track_id].append(team)
        # if we don't have a confirmed team yet for this track, set a provisional one
        if track_id not in self.track_team:
            self.track_team[track_id] = int(team)

        # confirm if history majority agrees (allow shorter confirmation for new tracks)
        vals = [v for v in self.team_history[track_id] if v != -1]
        confirm_thresh = max(1, int(self.confirm_samples // 2))
        if len(vals) >= confirm_thresh:
            # majority vote
            team_confirmed = int(np.bincount(vals).argmax())
            prev = self.track_team.get(track_id, None)
            # determine whether we're within hold period
            last_fr = self.last_confirm_frame.get(track_id, -99999)
            in_hold = (self.frame_count - last_fr) < self.hold_frames
            allow_change = True
            if prev is not None and prev != team_confirmed and in_hold:
                # require stronger color evidence to switch during hold
                try:
                    # map team id to centroid index
                    if self.team_assigned_side is not None:
                        left_idx, right_idx = self.team_assigned_side
                        prev_centroid_idx = left_idx if prev == 0 else right_idx
                        new_centroid_idx = left_idx if team_confirmed == 0 else right_idx
                    else:
                        prev_centroid_idx = prev
                        new_centroid_idx = team_confirmed
                    prev_dist = float(color_dists[prev_centroid_idx])
                    new_dist = float(color_dists[new_centroid_idx])
                    # allow change only if new_dist significantly better than prev_dist
                    if not (new_dist < prev_dist * self.change_distance_factor):
                        allow_change = False
                except Exception:
                    allow_change = False

            if prev is None or prev != team_confirmed:
                if allow_change:
                    self.track_team[track_id] = team_confirmed
                    self.last_confirm_frame[track_id] = self.frame_count
        # if still no global centroids and configured, try clustering on track medians
        if self.team_centroids is None and self.cluster_on_track_medians:
            # run when we have enough tracks with medians
            ok = self._run_kmeans_on_track_medians()
            if ok:
                # assign newly discovered centroids to existing tracks if possible
                for tid, deq in list(self.feat_history.items()):
                    if len(deq) == 0:
                        continue
                    arr = np.vstack(list(deq))
                    med = np.median(arr, axis=0)
                    dists = np.linalg.norm(self.team_centroids - med[None, :], axis=1)
                    best = int(np.argmin(dists))
                    # map centroid index to team id (0 = home/left, 1 = away/right)
                    left_idx, right_idx = self.team_assigned_side if self.team_assigned_side is not None else (0,1)
                    team = 0 if best == left_idx else 1
                    self.track_team[tid] = team

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
        # reset positional seeds
        self.pos_samples = []
        self.pos_initialized = False
        self.pos_centers = None
        try:
            delattr(self, 'pos_split_x')
        except Exception:
            pass
