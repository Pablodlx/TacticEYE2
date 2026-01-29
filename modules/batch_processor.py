"""
Batch Processor
===============

Procesamiento de micro-batches de frames.
Orquesta el pipeline completo: detección, tracking, clasificación, posesión.
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from ultralytics import YOLO

# Imports de módulos existentes
from modules.reid_tracker import ReIDTracker
from modules.team_classifier_v2 import TeamClassifierV2
from modules.possession_tracker_v2 import PossessionTrackerV2
from modules.match_state import MatchState, TrackerState, TeamClassifierState, PossessionState

# Imports de módulos de calibración y tracking espacial
from modules.field_keypoints_yolo import FieldKeypointsYOLO
from modules.field_model_keypoints import FieldModel as FieldModelKeypoints
from modules.field_calibrator_keypoints import FieldCalibratorKeypoints
from modules.field_model import FieldModel, FieldDimensions, ZoneModel
from modules.spatial_possession_tracker import SpatialPossessionTracker

# Sistema de heatmaps con resolución de flip
from modules.field_heatmap_system import (
    FIELD_POINTS,
    FIELD_LENGTH,
    FIELD_WIDTH,
    HeatmapAccumulator,
    estimate_homography_with_flip_resolution,
    project_points,
    project_points_by_triangulation
)

# Jerarquía de prioridad de keypoints (mayor = más fiable)
# Basada en las 15 clases del modelo field_kp_merged_fast
KEYPOINT_PRIORITY = {
    # Intersecciones de línea central (prioridad máxima)
    'midline_top_intersection': 100,
    'midline_bottom_intersection': 100,
    
    # Círculo central (muy fiables)
    'halfcircle_top': 95,
    'halfcircle_bottom': 95,
    
    # Intersecciones arco-área (arcos de penalti)
    'top_arc_area_intersection': 90,
    'bottom_arc_area_intersection': 90,
    
    # Área grande (big box) - partes internas
    'bigarea_top_inner': 80,
    'bigarea_bottom_inner': 80,
    
    # Área grande (big box) - partes externas
    'bigarea_top_outter': 75,
    'bigarea_bottom_outter': 75,
    
    # Área pequeña (small box) - partes internas
    'smallarea_top_inner': 70,
    'smallarea_bottom_inner': 70,
    
    # Área pequeña (small box) - partes externas
    'smallarea_top_outter': 65,
    'smallarea_bottom_outter': 65,
    
    # Esquinas (menor prioridad por oclusión frecuente)
    'corner': 40,
}


@dataclass
class ChunkOutput:
    """
    Salida del procesamiento de un chunk.
    
    Contiene todos los resultados que se pueden persistir o servir via API.
    """
    batch_idx: int
    start_frame: int
    end_frame: int
    
    # Detecciones por frame
    detections_by_frame: Dict[int, Dict[str, Any]]
    
    # Posiciones de jugadores (para heatmaps, trayectorias)
    player_positions: List[Dict[str, Any]]
    
    # Eventos detectados (pases, cambios de posesión)
    events: List[Dict[str, Any]]
    
    # Estadísticas del chunk
    chunk_stats: Dict[str, Any]
    
    # Timestamp de procesamiento
    processing_time_ms: float


class BatchProcessor:
    """
    Procesador de micro-batches.
    
    Recibe un chunk de frames y el estado actual, ejecuta el pipeline completo
    y retorna el estado actualizado + outputs del chunk.
    """
    
    def __init__(
        self,
        model_path: str = "weights/best.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        # Parámetros del tracker
        max_age: int = 30,
        max_lost_time: float = 120.0,
        # Parámetros del team classifier
        kmeans_min_tracks: int = 10,
        vote_history: int = 4,
        use_L: bool = True,
        L_weight: float = 0.5,
        # Parámetros de posesión
        possession_distance: float = 50.0,
        # Parámetros de calibración espacial
        enable_spatial_tracking: bool = True,
        zone_partition_type: str = 'thirds_lanes',
        zone_nx: int = 6,
        zone_ny: int = 4,
        enable_heatmaps: bool = True,
        heatmap_resolution: Tuple[int, int] = (50, 34)
    ):
        """
        Inicializa el procesador de batches.
        
        Args:
            model_path: Ruta al modelo YOLO
            device: 'cuda' o 'cpu'
            conf_threshold: Umbral de confianza para detecciones
            iou_threshold: Umbral IoU para NMS
            imgsz: Tamaño de imagen para inferencia
            ... (resto de parámetros del pipeline)
        """
        
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        
        # Cargar modelo YOLO
        print(f"Cargando modelo YOLO desde {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Parámetros guardados para reinicializar módulos
        self.tracker_params = {
            'max_age': max_age,
            'max_lost_time': max_lost_time
        }
        
        self.classifier_params = {
            'kmeans_min_tracks': kmeans_min_tracks,
            'vote_history': vote_history,
            'use_L_channel': use_L,
            'L_weight': L_weight
        }
        
        # Parámetros de calibración espacial
        self.enable_spatial_tracking = enable_spatial_tracking
        self.spatial_params = {
            'zone_partition_type': zone_partition_type,
            'zone_nx': zone_nx,
            'zone_ny': zone_ny,
            'enable_heatmaps': enable_heatmaps,
            'heatmap_resolution': heatmap_resolution
        }
        
        # Módulos del pipeline (se inicializan por match)
        self.tracker: Optional[ReIDTracker] = None
        self.team_classifier: Optional[TeamClassifierV2] = None
        self.possession_tracker: Optional[PossessionTrackerV2] = None
        
        # Módulos de calibración espacial
        self.field_calibrator: Optional[FieldCalibrator] = None
        self.spatial_tracker: Optional[SpatialPossessionTracker] = None
    
    def initialize_modules(self, match_state: MatchState):
        """
        Inicializa o restaura los módulos del pipeline desde el estado.
        
        Args:
            match_state: Estado del partido (puede contener estado previo)
        """
        
        # 1. Tracker ReID
        self.tracker = ReIDTracker(**self.tracker_params)
        
        # Restaurar estado del tracker si existe
        if match_state.tracker_state.next_id > 0:
            self.tracker.next_id = match_state.tracker_state.next_id
            # Restaurar tracks activos (simplificado)
            # En producción: restaurar features, bboxes, etc.
            for track_id, track_data in match_state.tracker_state.active_tracks.items():
                # Placeholder: aquí restaurarías el estado completo del track
                pass
        
        # 2. Team Classifier
        self.team_classifier = TeamClassifierV2(**self.classifier_params)
        
        # Restaurar estado del classifier
        if match_state.team_classifier_state.is_trained:
            # Restaurar asignaciones de equipos
            for player_id, team_id in match_state.team_classifier_state.player_team_map.items():
                self.team_classifier.team_assignments[int(player_id)] = team_id
            
            # Restaurar colores de equipos (si existen)
            if match_state.team_classifier_state.team_colors:
                # Aquí restaurarías el modelo KMeans si lo guardaste
                pass
        
        # 3. Possession Tracker
        # PossessionTrackerV2 solo necesita fps, que se pasa al llamar a update()
        self.possession_tracker = PossessionTrackerV2()
        
        # Restaurar estado de posesión
        self.possession_tracker.current_possession_team = match_state.possession_state.current_team
        self.possession_tracker.current_possession_player = match_state.possession_state.current_player
        self.possession_tracker.total_frames_by_team = match_state.possession_state.frames_by_team.copy()
        self.possession_tracker.passes_by_team = match_state.possession_state.passes_by_team.copy()
        
        # 4. Calibración de campo y tracking espacial
        if self.enable_spatial_tracking:
            print("✓ Inicializando calibración automática de campo (YOLO Custom)...")
            
            # Detector de keypoints con modelo YOLO custom
            try:
                self.keypoints_detector = FieldKeypointsYOLO(
                    model_path="weights/field_kp_merged_fast/weights/best.pt",
                    confidence_threshold=0.25,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                self.use_keypoints = True
                print("  ✓ Detector YOLO custom inicializado")
            except Exception as e:
                print(f"  ⚠ Error inicializando detector YOLO: {e}")
                self.keypoints_detector = None
                self.use_keypoints = False
            
            # Modelo de campo para keypoints
            if self.use_keypoints:
                self.field_model_keypoints = FieldModelKeypoints(
                    field_length=105.0,
                    field_width=68.0,
                    use_normalized=False
                )
                self.field_calibrator_keypoints = FieldCalibratorKeypoints(
                    field_model=self.field_model_keypoints,
                    min_keypoints=4,
                    ransac_threshold=5.0,
                    confidence=0.99
                )
            else:
                self.field_calibrator_keypoints = None
            
            # FieldModel básico para ZoneModel
            field_dims = FieldDimensions(length=105.0, width=68.0)
            basic_field_model = FieldModel(dimensions=field_dims)
            
            # Inicializar modelo de zonas
            zone_model = ZoneModel(
                field_model=basic_field_model,
                partition_type=self.spatial_params['zone_partition_type'],
                nx=self.spatial_params['zone_nx'],
                ny=self.spatial_params['zone_ny']
            )
            
            # Usar el calibrador de keypoints como principal
            self.field_calibrator = self.field_calibrator_keypoints
            
            # Inicializar tracker espacial
            self.spatial_tracker = SpatialPossessionTracker(
                calibrator=self.field_calibrator,
                zone_model=zone_model,
                enable_heatmaps=self.spatial_params['enable_heatmaps'],
                heatmap_resolution=self.spatial_params['heatmap_resolution']
            )
            
            print(f"  - Modo: Keypoints acumulativos (YOLO Custom Model)")
            print(f"  - Modelo de zonas: {self.spatial_params['zone_partition_type']}")
            print(f"  - Número de zonas: {zone_model.num_zones}")
            print(f"  - Heatmaps: {'Activados' if self.spatial_params['enable_heatmaps'] else 'Desactivados'}")
            
            # Inicializar acumulador de heatmaps con resolución de flip
            heatmap_res = self.spatial_params['heatmap_resolution']
            self.heatmap_accumulator = HeatmapAccumulator(
                field_length=105.0,
                field_width=68.0,
                nx=heatmap_res[0],
                ny=heatmap_res[1]
            )
            print(f"  - Acumulador de heatmaps: {heatmap_res[0]}x{heatmap_res[1]} celdas")
        else:
            self.keypoints_detector = None
            self.field_model_keypoints = None
            self.field_calibrator = None
            self.field_calibrator_keypoints = None
            self.spatial_tracker = None
            self.heatmap_accumulator = None
            self.use_keypoints = False
    
    def save_modules_state(self, match_state: MatchState):
        """
        Guarda el estado de los módulos en MatchState.
        
        Args:
            match_state: Estado del partido a actualizar
        """
        
        # 1. Guardar estado del tracker
        match_state.tracker_state.next_id = self.tracker.next_track_id
        match_state.tracker_state.last_frame_idx = match_state.total_frames_processed
        
        # Guardar tracks activos (simplificado)
        # En producción: guardar features, bboxes, historial, etc.
        match_state.tracker_state.active_tracks = {}
        for track_id, track in self.tracker.active_tracks.items():
            match_state.tracker_state.active_tracks[track_id] = {
                'id': track_id,
                'team_id': track.team_id,
                # Más datos según necesites...
            }
        
        # 2. Guardar estado del classifier
        match_state.team_classifier_state.player_team_map = {
            int(k): int(v) for k, v in self.team_classifier.track_team.items()
        }
        
        if self.team_classifier.kmeans_initialized:
            match_state.team_classifier_state.is_trained = True
            # Guardar centros de clusters
            if self.team_classifier.cluster_centers is not None:
                centers = self.team_classifier.cluster_centers
                match_state.team_classifier_state.team_colors = {
                    i: centers[i].tolist() for i in range(len(centers))
                }
        
        # 3. Guardar estado de posesión
        match_state.possession_state.current_team = self.possession_tracker.current_possession_team
        match_state.possession_state.current_player = self.possession_tracker.current_possession_player or -1
        match_state.possession_state.frames_by_team = self.possession_tracker.total_frames_by_team.copy()
        match_state.possession_state.passes_by_team = self.possession_tracker.passes_by_team.copy()
        match_state.possession_state.last_frame_idx = match_state.total_frames_processed
    
    def _create_annotated_frame(self, frame: np.ndarray, tracked_objects: list, frame_idx: int, keypoints: dict = None) -> np.ndarray:
        """
        Crea un frame anotado con cajas de detección, labels y keypoints.
        
        Args:
            frame: Frame BGR
            tracked_objects: Lista de objetos detectados
            frame_idx: Número de frame
            keypoints: Diccionario de keypoints detectados {nombre: (x, y)}
            
        Returns:
            Frame anotado
        """
        import cv2
        
        # Colores por clase y equipo
        team_colors = {
            0: (0, 255, 0),      # Equipo 0: Verde
            1: (255, 0, 0),      # Equipo 1: Azul
            -1: (128, 128, 128)  # Sin equipo: Gris
        }
        class_colors = {
            'ball': (0, 255, 255),     # Amarillo
            'referee': (255, 255, 255), # Blanco
            'player': None,             # Usar color de equipo
            'goalkeeper': None          # Usar color de equipo
        }
        
        for obj in tracked_objects:
            bbox = obj['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Determinar color
            class_name = obj['class_name']
            if class_name in class_colors and class_colors[class_name] is not None:
                color = class_colors[class_name]
            else:
                team_id = obj.get('team_id', -1)
                color = team_colors.get(team_id, (255, 255, 255))
            
            # Dibujar caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Preparar label
            track_id = obj['track_id']
            team_id = obj.get('team_id', -1)
            
            if class_name == 'ball':
                label = "BALL"
            elif class_name == 'referee':
                label = f"REF #{track_id}"
            else:
                team_label = f"T{team_id}" if team_id >= 0 else "?"
                label = f"{team_label} #{track_id}"
            
            # Dibujar label con fondo
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        # Visualizar keypoints detectados con bounding boxes
        if keypoints:
            keypoint_box_size = 15
            
            for kp_name, (kp_x, kp_y) in keypoints.items():
                # Obtener prioridad del keypoint
                priority = KEYPOINT_PRIORITY.get(kp_name, 50)
                
                # Color según prioridad
                if priority >= 90:  # ALTA
                    kp_color = (0, 255, 255)  # Amarillo
                elif priority >= 75:  # Media-Alta
                    kp_color = (0, 200, 255)  # Naranja
                elif priority >= 65:  # Media
                    kp_color = (255, 150, 0)  # Azul claro
                else:  # Baja
                    kp_color = (180, 180, 180)  # Gris
                
                # Dibujar bounding box
                x1 = int(kp_x - keypoint_box_size)
                y1 = int(kp_y - keypoint_box_size)
                x2 = int(kp_x + keypoint_box_size)
                y2 = int(kp_y + keypoint_box_size)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), kp_color, 2)
                
                # Label del keypoint
                kp_label = kp_name.replace('_', ' ').title()
                if len(kp_label) > 20:
                    parts = kp_label.split()
                    if len(parts) > 2:
                        kp_label = f"{parts[0][0]}{parts[1][0]}{parts[-1][:3]}"
                
                # Fondo para el texto
                label_size = cv2.getTextSize(kp_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(frame, (x1, y1 - 18), (x1 + label_size[0] + 4, y1 - 2),
                             kp_color, -1)
                
                # Texto
                cv2.putText(frame, kp_label, (x1 + 2, y1 - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Punto central
                cv2.circle(frame, (int(kp_x), int(kp_y)), 3, (255, 255, 255), -1)
        
        # Añadir info del frame en la esquina
        info_text = f"Frame: {frame_idx}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def process_chunk(
        self,
        match_state: MatchState,
        frames: List[np.ndarray],
        start_frame_idx: int,
        fps: float,
        visualize_interval: int = 15,
        send_visualization: callable = None
    ) -> Tuple[MatchState, ChunkOutput]:
        """
        Procesa un micro-batch de frames.
        
        Args:
            match_state: Estado actual del partido
            frames: Lista de frames del chunk (BGR)
            start_frame_idx: Índice global del primer frame del chunk
            fps: FPS del video
            visualize_interval: Cada cuántos frames enviar visualización
            send_visualization: Función callback para enviar frame anotado
        
        Returns:
            (match_state actualizado, outputs del chunk)
        
        Pipeline:
            1. Detección YOLO en cada frame
            2. Tracking ReID para mantener IDs
            3. Clasificación de equipos
            4. Detección de posesión y pases
            5. Generación de outputs
        """
        
        import time
        start_time = time.time()
        
        # Inicializar módulos si es la primera vez
        if self.tracker is None:
            self.initialize_modules(match_state)
        
        # Estructuras de salida
        detections_by_frame = {}
        player_positions = []
        events = []
        
        # Acumulador de posiciones PROYECTADAS para heatmap (media de posiciones de campo por batch)
        player_field_positions_accumulator = {}  # {track_id: {'team_id': int, 'field_positions': [(x,y), ...]}}
        best_homography_batch = None
        best_homography_is_flipped = False
        best_homography_num_kp = 0  # Número de keypoints de la mejor homografía
        best_frame_keypoints = None  # Keypoints del frame con mejor homografía
        
        # Procesar cada frame del chunk
        current_keypoints = None
        for i, frame in enumerate(frames):
            frame_idx = start_frame_idx + i
            
            # 1. DETECCIÓN YOLO
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device
            )[0]
            
            # Parsear detecciones
            boxes = []
            scores = []
            cls_ids = []
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes_data = results.boxes.xyxy.cpu().numpy()
                scores_data = results.boxes.conf.cpu().numpy()
                cls_data = results.boxes.cls.cpu().numpy().astype(int)
                
                for box, score, cls_id in zip(boxes_data, scores_data, cls_data):
                    boxes.append(box)
                    scores.append(score)
                    cls_ids.append(cls_id)
            
            boxes = np.array(boxes) if boxes else np.empty((0, 4))
            scores = np.array(scores) if scores else np.array([])
            cls_ids = np.array(cls_ids) if cls_ids else np.array([])
            
            # 2. TRACKING ReID
            tracked_tuples = self.tracker.update(frame, boxes, scores, cls_ids)
            
            # Convertir tuplas (track_id, bbox, class_id) a diccionarios
            class_names = {0: 'player', 1: 'ball', 2: 'referee', 3: 'goalkeeper'}
            tracked_objects = []
            for track_id, bbox, class_id in tracked_tuples:
                tracked_objects.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'class_id': class_id,
                    'class_name': class_names.get(class_id, 'unknown')
                })
            
            # 3. CLASIFICACIÓN DE EQUIPOS
            # TeamClassifierV2 maneja entrenamiento y clasificación automáticamente
            for obj in tracked_objects:
                if obj['class_name'] in ('player', 'goalkeeper'):
                    # add_detection extrae features, entrena KMeans si es necesario, y clasifica
                    bbox_tuple = tuple(map(int, obj['bbox']))
                    self.team_classifier.add_detection(
                        track_id=obj['track_id'],
                        bbox=bbox_tuple,
                        image=frame,
                        class_id=obj['class_id']
                    )
                    # Obtener equipo asignado
                    obj['team_id'] = self.team_classifier.get_team(obj['track_id'])
                else:
                    # Referee o ball
                    obj['team_id'] = -1
            
            # 3.5. DETECCIÓN DE KEYPOINTS (antes de visualización)
            if self.use_keypoints and self.keypoints_detector is not None and frame_idx % 3 == 0:
                try:
                    current_keypoints = self.keypoints_detector.detect_keypoints(frame)
                except Exception as e:
                    if frame_idx % 60 == 0:
                        print(f"[Keypoints] Error frame {frame_idx}: {e}")
                    current_keypoints = None
            
            # Enviar frame anotado periódicamente
            if send_visualization and (i % visualize_interval == 0 or i == len(frames) - 1):
                annotated_frame = self._create_annotated_frame(frame.copy(), tracked_objects, frame_idx, current_keypoints)
                send_visualization(annotated_frame, frame_idx)
            
            # 4. DETECCIÓN DE POSESIÓN
            ball_owner_team = -1
            ball_owner_id = -1
            ball_bbox = None
            
            # Encontrar el balón
            for obj in tracked_objects:
                if obj['class_name'] == 'ball':
                    ball_bbox = obj['bbox']
                    break
            
            # Encontrar el jugador más cercano al balón
            if ball_bbox is not None:
                ball_center = np.array([
                    (ball_bbox[0] + ball_bbox[2]) / 2,
                    (ball_bbox[1] + ball_bbox[3]) / 2
                ])
                
                min_dist = float('inf')
                closest_player = None
                
                for obj in tracked_objects:
                    if obj['class_name'] == 'player':
                        player_center = np.array([
                            (obj['bbox'][0] + obj['bbox'][2]) / 2,
                            (obj['bbox'][1] + obj['bbox'][3]) / 2
                        ])
                        
                        dist = np.linalg.norm(ball_center - player_center)
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_player = obj
                
                if closest_player and min_dist < 60:  # 60 píxeles de distancia máxima
                    ball_owner_team = closest_player.get('team_id', -1)
                    ball_owner_id = closest_player['track_id']
            
            # Guardar estado anterior para detectar cambios
            prev_possession_team = self.possession_tracker.current_possession_team
            prev_possession_player = self.possession_tracker.current_possession_player
            
            # Actualizar posesión (no retorna nada)
            self.possession_tracker.update(
                frame_idx,
                ball_owner_team if ball_owner_team >= 0 else None,
                ball_owner_id if ball_owner_id >= 0 else None
            )
            
            # 4.5. TRACKING ESPACIAL (si está habilitado)
            spatial_info = {}
            if self.enable_spatial_tracking and self.spatial_tracker is not None:
                # Calibrar menos frecuentemente
                # Calibración con keypoints acumulativos
                # Usar keypoints ya detectados si existen
                if self.use_keypoints and self.keypoints_detector is not None and current_keypoints is not None:
                    try:
                        # Siempre intentar acumular keypoints (aunque sean 0)
                        success = self.field_calibrator_keypoints.estimate_homography(current_keypoints)
                        
                        # Log solo cada 30 frames Y solo si hay calibración exitosa
                        if frame_idx % 30 == 0 and success:
                            num_detected = len(current_keypoints)
                            num_accumulated = len(self.field_calibrator_keypoints.accumulated_keypoints)
                            info = self.field_calibrator_keypoints.get_calibration_info()
                            print(f"[Calibración] ✓ Frame {frame_idx}: {num_detected} detectados, "
                                  f"{num_accumulated} acumulados, calibrado con {info['num_keypoints']} keypoints "
                                  f"(error={info['reprojection_error']:.2f}px)")
                            
                    except Exception as e:
                        if frame_idx % 60 == 0:
                            print(f"[Calibración] ✗ Frame {frame_idx}: {e}")
                
                # Acumular posiciones en píxeles durante el batch (proyección al final)
                if self.heatmap_accumulator is not None and current_keypoints:
                    try:
                        # Convertir formato de keypoints al esperado por el sistema
                        frame_keypoints = [
                            {"cls_name": name, "xy": coords, "conf": 0.9}
                            for name, coords in current_keypoints.items()
                        ]
                        
                        # Estimar homografía con resolución de flip (guardar mejor del batch)
                        H, is_flipped = estimate_homography_with_flip_resolution(
                            frame_keypoints,
                            FIELD_POINTS,
                            min_points=3,
                            conf_threshold=0.3
                        )
                        
                        # Guardar mejor homografía del batch (la que tenga más keypoints)
                        if H is not None:
                            num_keypoints_current = len(frame_keypoints)
                            if best_homography_batch is None:
                                best_homography_batch = H
                                best_homography_is_flipped = is_flipped
                                best_homography_num_kp = num_keypoints_current
                                best_frame_keypoints = frame_keypoints  # Guardar keypoints también
                            elif num_keypoints_current > best_homography_num_kp:
                                # Actualizar si esta homografía tiene más keypoints (más confiable)
                                best_homography_batch = H
                                best_homography_is_flipped = is_flipped
                                best_homography_num_kp = num_keypoints_current
                                best_frame_keypoints = frame_keypoints
                        
                        # Acumular posiciones en píxeles por ID (no proyectar aún)
                        for obj in tracked_objects:
                            if obj['class_name'] == 'player' and obj.get('team_id', -1) >= 0:
                                track_id = obj['track_id']
                                team_id = obj['team_id']
                                bbox = obj['bbox']
                                center_x = (bbox[0] + bbox[2]) / 2
                                center_y = (bbox[1] + bbox[3]) / 2
                                
                                # PROYECTAR frame a frame usando keypoints del FRAME ACTUAL (no acumulados)
                                # Esto evita problemas cuando la cámara cambia de ángulo
                                if current_keypoints and len(current_keypoints) >= 2:
                                    # Convertir diccionario de keypoints a lista para project_points_by_triangulation
                                    # current_keypoints es {name: (x, y)}, necesitamos [{"cls_name": name, "xy": (x,y), "conf": 1.0}]
                                    current_keypoints_list = [
                                        {"cls_name": name, "xy": coords, "conf": 1.0}
                                        for name, coords in current_keypoints.items()
                                    ]
                                    
                                    # Usar keypoints detectados en este frame específico
                                    pixel_pos = [(center_x, center_y)]
                                    field_coords = project_points_by_triangulation(
                                        pixel_pos,
                                        current_keypoints_list,  # Lista de keypoints del frame actual
                                        FIELD_POINTS,
                                        best_homography_is_flipped,
                                        min_references=2,
                                        max_references=4
                                    )
                                    
                                    # Si la proyección es válida, acumular posición de campo
                                    if field_coords is not None and len(field_coords) > 0:
                                        field_pos = field_coords[0]
                                        if not np.isnan(field_pos).any():
                                            # Inicializar acumulador para este ID si no existe
                                            if track_id not in player_field_positions_accumulator:
                                                player_field_positions_accumulator[track_id] = {
                                                    'team_id': team_id,
                                                    'field_positions': []
                                                }
                                            
                                            # Acumular posición PROYECTADA (en coordenadas de campo)
                                            player_field_positions_accumulator[track_id]['field_positions'].append(
                                                (field_pos[0], field_pos[1])
                                            )
                    
                    except Exception as e:
                        if frame_idx % 60 == 0:
                            print(f"[Heatmap] Error frame {frame_idx}: {e}")
                
                # Actualizar tracker espacial
                spatial_state = self.spatial_tracker.update(
                    ball_pos=ball_bbox[:2] if ball_bbox is not None else None,
                    players=tracked_objects,
                    frame=None  # Ya no pasamos frame aquí, calibramos arriba
                )
                
                spatial_info = {
                    'field_position': spatial_state.get('field_position'),
                    'zone_id': spatial_state.get('zone_id', -1),
                    'zone_name': self.spatial_tracker.zone_model.get_zone_name(
                        spatial_state.get('zone_id', -1)
                    ) if spatial_state.get('zone_id', -1) >= 0 else 'unknown',
                    'calibration_valid': spatial_state.get('calibration_valid', False),
                    'position_fallback': spatial_state.get('position_fallback', False)
                }
            
            # Detectar cambio de equipo
            new_possession_team = self.possession_tracker.current_possession_team
            if prev_possession_team is not None and new_possession_team != prev_possession_team:
                events.append({
                    'type': 'possession_change',
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    'from_team': prev_possession_team,
                    'to_team': new_possession_team,
                    'player_id': ball_owner_id
                })
            
            # Detectar pase (cambio de jugador en el mismo equipo)
            new_possession_player = self.possession_tracker.current_possession_player
            if (prev_possession_player is not None and 
                new_possession_player is not None and
                new_possession_player != prev_possession_player and
                new_possession_team == prev_possession_team):
                events.append({
                    'type': 'pass',
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    'team': new_possession_team,
                    'from_player': prev_possession_player,
                    'to_player': new_possession_player
                })
            
            # 5. GUARDAR OUTPUTS
            detections_by_frame[frame_idx] = {
                'frame': frame_idx,
                'objects': tracked_objects,
                'ball_owner': ball_owner_id,
                'possession_team': ball_owner_team,
                'spatial_info': spatial_info  # Añadir info espacial
            }
            
            # Guardar posiciones de jugadores (con info espacial si disponible)
            for obj in tracked_objects:
                if obj['class_name'] == 'player':
                    pos_data = {
                        'frame': frame_idx,
                        'timestamp': frame_idx / fps,
                        'player_id': obj['track_id'],
                        'team_id': obj.get('team_id', -1),
                        'bbox': obj['bbox'].tolist(),
                        'position': [(obj['bbox'][0] + obj['bbox'][2]) / 2,
                                   (obj['bbox'][1] + obj['bbox'][3]) / 2]
                    }
                    
                    # Añadir posición de campo si está disponible y es el poseedor
                    if (self.enable_spatial_tracking and 
                        obj['track_id'] == ball_owner_id and 
                        spatial_info.get('field_position') is not None):
                        pos_data['field_position'] = spatial_info['field_position']
                        pos_data['zone_id'] = spatial_info.get('zone_id', -1)
                    
                    player_positions.append(pos_data)
        
        # ====================================================================
        # ACUMULACIÓN DE POSICIONES MEDIAS EN HEATMAP (al final del batch)
        # MÉTODO: Media de posiciones proyectadas frame a frame
        # ====================================================================
        if self.heatmap_accumulator is not None and player_field_positions_accumulator:
            try:
                # Calcular posición media de campo por cada ID (media de posiciones ya proyectadas)
                valid_players = []
                
                for track_id, data in player_field_positions_accumulator.items():
                    field_positions = np.array(data['field_positions'])
                    
                    # Media de posiciones de campo durante el batch
                    avg_x_field = np.mean(field_positions[:, 0])
                    avg_y_field = np.mean(field_positions[:, 1])
                    
                    valid_players.append({
                        'team_id': data['team_id'],
                        'track_id': track_id,
                        'xy': (avg_x_field, avg_y_field),
                        'num_samples': len(field_positions)
                    })
                
                # Acumular en heatmap
                if valid_players:
                    for player in valid_players:
                        # Agregar directamente al grid del heatmap
                        team_id = player['team_id']
                        x_field, y_field = player['xy']
                        
                        # Convertir a índices de grid
                        ix = int(x_field / FIELD_LENGTH * self.heatmap_accumulator.nx)
                        iy = int(y_field / FIELD_WIDTH * self.heatmap_accumulator.ny)
                        
                        # Clipping
                        ix = np.clip(ix, 0, self.heatmap_accumulator.nx - 1)
                        iy = np.clip(iy, 0, self.heatmap_accumulator.ny - 1)
                        
                        # Acumular
                        if team_id == 0:
                            self.heatmap_accumulator.counts_team0[iy, ix] += 1
                        elif team_id == 1:
                            self.heatmap_accumulator.counts_team1[iy, ix] += 1
                    
                    # Incrementar contador de frames una sola vez por batch
                    self.heatmap_accumulator.num_frames += 1
                    
                    # DEBUG: Imprimir posiciones medias del batch cada 200 frames
                    if start_frame_idx % 200 == 0 and len(valid_players) > 0:
                        print(f"\n{'='*70}")
                        print(f"[Heatmap Debug] Posiciones MEDIAS (proyectadas frame a frame)")
                        print(f"  Frames: {start_frame_idx}-{start_frame_idx + len(frames) - 1}")
                        print(f"{'='*70}")
                        print(f"Sistema de referencia:")
                        print(f"  X = Longitudinal (0-{FIELD_LENGTH}m, línea gol izq→der)")
                        print(f"  Y = Lateral (0-{FIELD_WIDTH}m, banda abajo→arriba)")
                        print(f"  Centro campo = ({FIELD_LENGTH/2:.1f}, {FIELD_WIDTH/2:.1f})m")
                        print(f"\nJugadores válidos: {len(valid_players)}")
                        print(f"{'-'*70}")
                        
                        for player in valid_players:
                            track_id = player['track_id']
                            team_name = f"Team {player['team_id']}"
                            x_field, y_field = player['xy']
                            num_samples = player['num_samples']
                            
                            # Zona aproximada
                            if x_field < FIELD_LENGTH / 3:
                                zone_x = "Defensa"
                            elif x_field < 2 * FIELD_LENGTH / 3:
                                zone_x = "Mediocampo"
                            else:
                                zone_x = "Ataque"
                            
                            if y_field < FIELD_WIDTH / 3:
                                zone_y = "Derecha"
                            elif y_field < 2 * FIELD_WIDTH / 3:
                                zone_y = "Centro"
                            else:
                                zone_y = "Izquierda"
                            
                            print(f"  ID #{track_id} ({team_name}) [Media de {num_samples} frames proyectados]:")
                            print(f"    Campo: X={x_field:.2f}m, Y={y_field:.2f}m")
                            print(f"    Zona: {zone_x} - {zone_y}")
                        
                        print(f"{'='*70}\n")
            
            except Exception as e:
                import traceback
                print(f"[Heatmap] Error proyectando por triangulación: {e}")
                traceback.print_exc()
        
        # Actualizar estado del partido
        end_frame_idx = start_frame_idx + len(frames) - 1
        batch_idx = start_frame_idx // len(frames)
        
        match_state.update_progress(batch_idx, len(frames))
        self.save_modules_state(match_state)
        
        # Calcular estadísticas del chunk
        chunk_stats = {
            'frames_processed': len(frames),
            'detections_count': sum(len(d['objects']) for d in detections_by_frame.values()),
            'events_count': len(events),
            'possession_team': self.possession_tracker.current_possession_team,
            'possession_player': self.possession_tracker.current_possession_player,
        }
        
        # Añadir estadísticas espaciales si están habilitadas
        if self.enable_spatial_tracking and self.spatial_tracker is not None:
            spatial_stats = self.spatial_tracker.get_spatial_statistics()
            zone_stats = self.spatial_tracker.get_zone_statistics()
            
            chunk_stats['spatial'] = {
                'calibration_valid': self.field_calibrator.has_valid_calibration(),
                'possession_by_zone': spatial_stats.get('possession_by_zone', {}),
                'zone_percentages': spatial_stats.get('zone_percentages', {}),
                'zone_partition_type': zone_stats.get('partition_type', 'unknown'),
                'num_zones': zone_stats.get('num_zones', 0)
            }
            
            # No incluir heatmaps en chunk_stats (son muy grandes)
            # Se exportarán al final del análisis
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Crear output del chunk
        output = ChunkOutput(
            batch_idx=batch_idx,
            start_frame=start_frame_idx,
            end_frame=end_frame_idx,
            detections_by_frame=detections_by_frame,
            player_positions=player_positions,
            events=events,
            chunk_stats=chunk_stats,
            processing_time_ms=processing_time_ms
        )
        
        return match_state, output


# ============================================================================
# Utilidades para outputs
# ============================================================================

def _convert_numpy_to_json(obj):
    """
    Convierte recursivamente numpy arrays a listas para serialización JSON.
    
    Args:
        obj: Objeto a convertir
        
    Returns:
        Objeto convertido (arrays → listas)
    """
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def save_chunk_output(match_id: str, output: ChunkOutput, output_dir: str = "outputs_streaming"):
    """
    Guarda los outputs de un chunk a disco.
    
    Args:
        match_id: ID del partido
        output: Outputs del chunk
        output_dir: Directorio base de outputs
    """
    import os
    import json
    
    # Crear directorio del partido
    match_dir = os.path.join(output_dir, match_id)
    os.makedirs(match_dir, exist_ok=True)
    
    # Convertir numpy arrays a listas antes de guardar
    detections_json = _convert_numpy_to_json(output.detections_by_frame)
    positions_json = _convert_numpy_to_json(output.player_positions)
    events_json = _convert_numpy_to_json(output.events)
    
    # Guardar detecciones
    detections_file = os.path.join(match_dir, f"detections_batch_{output.batch_idx:04d}.json")
    with open(detections_file, 'w') as f:
        json.dump(detections_json, f)
    
    # Guardar posiciones
    positions_file = os.path.join(match_dir, f"positions_batch_{output.batch_idx:04d}.json")
    with open(positions_file, 'w') as f:
        json.dump(positions_json, f)
    
    # Guardar eventos
    if output.events:
        events_file = os.path.join(match_dir, f"events_batch_{output.batch_idx:04d}.json")
        with open(events_file, 'w') as f:
            json.dump(events_json, f)
    
    # Guardar estadísticas del chunk
    stats_file = os.path.join(match_dir, f"stats_batch_{output.batch_idx:04d}.json")
    with open(stats_file, 'w') as f:
        json.dump({
            'batch_idx': output.batch_idx,
            'start_frame': output.start_frame,
            'end_frame': output.end_frame,
            'chunk_stats': output.chunk_stats,
            'processing_time_ms': output.processing_time_ms
        }, f, indent=2)


def load_match_outputs(match_id: str, output_dir: str = "outputs_streaming") -> List[ChunkOutput]:
    """
    Carga todos los outputs de un partido.
    
    Returns:
        Lista de ChunkOutputs ordenados por batch_idx
    """
    import os
    import json
    import glob
    
    match_dir = os.path.join(output_dir, match_id)
    if not os.path.exists(match_dir):
        return []
    
    outputs = []
    
    # Encontrar todos los archivos de stats
    stats_files = sorted(glob.glob(os.path.join(match_dir, "stats_batch_*.json")))
    
    for stats_file in stats_files:
        with open(stats_file, 'r') as f:
            stats_data = json.load(f)
        
        batch_idx = stats_data['batch_idx']
        
        # Cargar archivos correspondientes
        detections_file = os.path.join(match_dir, f"detections_batch_{batch_idx:04d}.json")
        positions_file = os.path.join(match_dir, f"positions_batch_{batch_idx:04d}.json")
        events_file = os.path.join(match_dir, f"events_batch_{batch_idx:04d}.json")
        
        with open(detections_file, 'r') as f:
            detections = json.load(f)
        
        with open(positions_file, 'r') as f:
            positions = json.load(f)
        
        events = []
        if os.path.exists(events_file):
            with open(events_file, 'r') as f:
                events = json.load(f)
        
        output = ChunkOutput(
            batch_idx=batch_idx,
            start_frame=stats_data['start_frame'],
            end_frame=stats_data['end_frame'],
            detections_by_frame=detections,
            player_positions=positions,
            events=events,
            chunk_stats=stats_data['chunk_stats'],
            processing_time_ms=stats_data['processing_time_ms']
        )
        
        outputs.append(output)
    
    return outputs


def export_spatial_heatmaps(processor: BatchProcessor, 
                           output_path: str = "heatmaps.npz"):
    """
    Exporta los heatmaps espaciales generados durante el análisis.
    
    Args:
        processor: BatchProcessor con spatial_tracker inicializado
        output_path: Ruta donde guardar los heatmaps
    
    Returns:
        True si se exportaron correctamente, False si no hay datos
    """
    if not processor.enable_spatial_tracking or processor.spatial_tracker is None:
        print("⚠ Tracking espacial no está habilitado")
        return False
    
    # Debug: Estado del TeamClassifier antes de exportar heatmaps
    if processor.team_classifier:
        tc = processor.team_classifier
        print(f"\n[TeamClassifier DEBUG]")
        print(f"  KMeans initialized: {tc.kmeans_initialized}")
        print(f"  Total tracks: {len(tc.track_team)}")
        
        # Contar cuántos jugadores por equipo
        team_counts = {-1: 0, 0: 0, 1: 0}
        for tid, team_id in tc.track_team.items():
            team_counts[team_id] = team_counts.get(team_id, 0) + 1
        
        print(f"  Distribución de equipos:")
        print(f"    Team -1 (referee/unknown): {team_counts.get(-1, 0)} jugadores")
        print(f"    Team 0: {team_counts.get(0, 0)} jugadores")
        print(f"    Team 1: {team_counts.get(1, 0)} jugadores")
        
        if tc.kmeans_initialized:
            print(f"  Cluster centers:")
            for i, center in enumerate(tc.cluster_centers):
                team = tc.cluster_to_team.get(i, -1)
                print(f"    Cluster {i} (Team {team}): {center}")
    
    # Exportar heatmaps del sistema clásico
    heatmap_0 = processor.spatial_tracker.export_heatmap(team_id=0, normalize=True)
    heatmap_1 = processor.spatial_tracker.export_heatmap(team_id=1, normalize=True)
    
    if heatmap_0 is None or heatmap_1 is None:
        print("⚠ No se generaron heatmaps (heatmaps deshabilitados)")
        return False
    
    # Logging para debug
    print(f"[Heatmaps] Team 0: shape={heatmap_0.shape}, sum={heatmap_0.sum():.2f}, max={heatmap_0.max():.2f}")
    print(f"[Heatmaps] Team 1: shape={heatmap_1.shape}, sum={heatmap_1.sum():.2f}, max={heatmap_1.max():.2f}")
    
    # Exportar heatmaps del nuevo sistema con resolución de flip (si está disponible)
    heatmap_flip_0 = None
    heatmap_flip_1 = None
    if hasattr(processor, 'heatmap_accumulator') and processor.heatmap_accumulator is not None:
        heatmap_flip_0 = processor.heatmap_accumulator.get_heatmap(0, normalize='max')
        heatmap_flip_1 = processor.heatmap_accumulator.get_heatmap(1, normalize='max')
        
        print(f"[Heatmaps Flip] Team 0: shape={heatmap_flip_0.shape}, sum={heatmap_flip_0.sum():.2f}, max={heatmap_flip_0.max():.2f}")
        print(f"[Heatmaps Flip] Team 1: shape={heatmap_flip_1.shape}, sum={heatmap_flip_1.sum():.2f}, max={heatmap_flip_1.max():.2f}")
        print(f"[Heatmaps Flip] Frames procesados: {processor.heatmap_accumulator.num_frames}")
    
    # Obtener estadísticas espaciales
    spatial_stats = processor.spatial_tracker.get_spatial_statistics()
    zone_stats = processor.spatial_tracker.get_zone_statistics()
    
    # Guardar en formato NPZ
    save_data = {
        'team_0_heatmap': heatmap_0,
        'team_1_heatmap': heatmap_1,
        'possession_by_zone_team_0': spatial_stats['possession_by_zone'][0],
        'possession_by_zone_team_1': spatial_stats['possession_by_zone'][1],
        'zone_percentages_team_0': spatial_stats['zone_percentages'][0],
        'zone_percentages_team_1': spatial_stats['zone_percentages'][1],
        'metadata': {
            'resolution': processor.spatial_params['heatmap_resolution'],
            'partition_type': zone_stats['partition_type'],
            'num_zones': zone_stats['num_zones'],
            'field_dims': (105.0, 68.0)  # Dimensiones del campo en metros
        }
    }
    
    # Agregar heatmaps con resolución de flip si están disponibles
    if heatmap_flip_0 is not None and heatmap_flip_1 is not None:
        save_data['team_0_heatmap_flip'] = heatmap_flip_0
        save_data['team_1_heatmap_flip'] = heatmap_flip_1
        save_data['heatmap_flip_frames'] = processor.heatmap_accumulator.num_frames
    
    np.savez(output_path, **save_data)
    
    print(f"✓ Heatmaps guardados en: {output_path}")
    return True

