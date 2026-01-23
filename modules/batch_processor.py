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
        possession_distance: float = 50.0
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
        
        # Módulos del pipeline (se inicializan por match)
        self.tracker: Optional[ReIDTracker] = None
        self.team_classifier: Optional[TeamClassifierV2] = None
        self.possession_tracker: Optional[PossessionTrackerV2] = None
    
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
    
    def _create_annotated_frame(self, frame: np.ndarray, tracked_objects: list, frame_idx: int) -> np.ndarray:
        """
        Crea un frame anotado con cajas de detección y labels.
        
        Args:
            frame: Frame BGR
            tracked_objects: Lista de objetos detectados
            frame_idx: Número de frame
            
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
        
        # Procesar cada frame del chunk
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
            
            # Generar visualización periódicamente
            if send_visualization and (i % visualize_interval == 0 or i == len(frames) - 1):
                annotated = self._create_annotated_frame(frame.copy(), tracked_objects, frame_idx)
                send_visualization(annotated, frame_idx)
            
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
                'possession_team': ball_owner_team
            }
            
            # Guardar posiciones de jugadores
            for obj in tracked_objects:
                if obj['class_name'] == 'player':
                    player_positions.append({
                        'frame': frame_idx,
                        'timestamp': frame_idx / fps,
                        'player_id': obj['track_id'],
                        'team_id': obj.get('team_id', -1),
                        'bbox': obj['bbox'].tolist(),
                        'position': [(obj['bbox'][0] + obj['bbox'][2]) / 2,
                                   (obj['bbox'][1] + obj['bbox'][3]) / 2]
                    })
        
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
