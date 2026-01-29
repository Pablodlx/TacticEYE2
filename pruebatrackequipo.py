"""
TacticEYE2 - Pipeline Simplificado
Tracking con ReID + Clasificación de Equipos + Detección de Posesión

Uso:
    python pruebatrackequipo.py video.mp4 --model weights/best.pt --reid

Funcionalidades:
- Detección YOLO (player, ball, referee, goalkeeper)
- Tracking con Re-Identificación (ReID)
- Clasificación automática de equipos (V2/V3)
- Detección de posesión del balón (V2)

Parámetros principales:
    --reid                    Activar tracking con ReID
    --use-v3                  Usar TeamClassifierV3
    --possession-distance     Distancia máxima para posesión (default: 60px)
    --no-show                 Sin visualización (más rápido)
    --output                  Guardar video procesado
"""

import argparse
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    print("Error: ultralytics no disponible. Instala con `pip install ultralytics`.")
    raise

try:
    from modules.reid_tracker import ReIDTracker
    REID_AVAILABLE = True
except Exception:
    REID_AVAILABLE = False
    print("⚠️  ReID Tracker no disponible")

try:
    from modules.possession_tracker_v2 import PossessionTrackerV2
    POSSESSION_AVAILABLE = True
except Exception:
    POSSESSION_AVAILABLE = False
    print("⚠️  PossessionTrackerV2 no disponible")
 
try:
    from modules.team_classifier_v2 import TeamClassifierV2
    TEAMCLASS_AVAILABLE = True
except Exception:
    TEAMCLASS_AVAILABLE = False
    print("⚠️  TeamClassifierV2 no disponible")

try:
    from modules.team_classifier_v3 import TeamClassifierV3
    TEAMCLASS_V3_AVAILABLE = True
except Exception:
    TEAMCLASS_V3_AVAILABLE = False

try:
    from modules.field_keypoints_yolo import FieldKeypointsYOLO
    from modules.field_model_keypoints import FieldModel as FieldModelKeypoints
    from modules.field_calibrator_keypoints import FieldCalibratorKeypoints
    from modules.field_heatmap_system import (
        FIELD_POINTS,
        FIELD_LENGTH,
        FIELD_WIDTH,
        estimate_homography_with_flip_resolution,
        project_points_by_triangulation
    )
    import torch
    CALIBRATION_AVAILABLE = True
except Exception:
    CALIBRATION_AVAILABLE = False
    print("⚠️  Field Calibration system no disponible")


# Jerarquía de prioridad de keypoints (mayor = más fiable)
KEYPOINT_PRIORITY = {
    # Alta prioridad: Centro del campo e intersecciones con áreas
    'midline_top_intersection': 100,
    'midline_bottom_intersection': 100,
    'halfcircle_top': 95,
    'halfcircle_bottom': 95,
    'left_penalty_arc_intersection_top': 90,
    'left_penalty_arc_intersection_bottom': 90,
    'right_penalty_arc_intersection_top': 90,
    'right_penalty_arc_intersection_bottom': 90,
    
    # Media-alta prioridad: Áreas grandes y pequeñas (intersecciones)
    'left_bigbox_top_inner': 80,
    'left_bigbox_bottom_inner': 80,
    'right_bigbox_top_inner': 80,
    'right_bigbox_bottom_inner': 80,
    'left_bigbox_top_outer': 75,
    'left_bigbox_bottom_outer': 75,
    'right_bigbox_top_outer': 75,
    'right_bigbox_bottom_outer': 75,
    
    # Media prioridad: Áreas pequeñas
    'left_smallbox_top_inner': 70,
    'left_smallbox_bottom_inner': 70,
    'right_smallbox_top_inner': 70,
    'right_smallbox_bottom_inner': 70,
    'left_smallbox_top_outer': 65,
    'left_smallbox_bottom_outer': 65,
    'right_smallbox_top_outer': 65,
    'right_smallbox_bottom_outer': 65,
    
    # Baja prioridad: Esquinas (menos fiables)
    'corner_left_top': 40,
    'corner_left_bottom': 40,
    'corner_right_top': 40,
    'corner_right_bottom': 40,
}


def get_keypoint_score(kp_name, distance_px, max_distance=500.0):
    """
    Calcula un score combinando prioridad del keypoint y distancia.
    Score más alto = mejor referencia.
    """
    # Prioridad base del keypoint
    priority = KEYPOINT_PRIORITY.get(kp_name, 50)  # Default: prioridad media
    
    # Factor de distancia (normalizado, inverso: más cerca = mejor)
    # Distancia se normaliza a [0, 1] y se invierte
    distance_factor = max(0, 1 - (distance_px / max_distance))
    
    # Score combinado: prioridad * factor_distancia
    # Esto favorece keypoints de alta prioridad, pero penaliza si están muy lejos
    score = priority * (0.3 + 0.7 * distance_factor)  # 30% prioridad, 70% distancia
    
    return score


def main():
    p = argparse.ArgumentParser(description='TacticEYE2 - Pipeline simplificado')
    
    # Video y modelo
    p.add_argument('video', help='Ruta al video a analizar')
    p.add_argument('--model', default='yolov8n.pt', help='Modelo YOLO')
    
    # Detección YOLO
    p.add_argument('--imgsz', type=int, default=640, help='Tamaño de imagen')
    p.add_argument('--conf', type=float, default=0.35, help='Umbral de confianza')
    p.add_argument('--max-det', type=int, default=100, help='Máximo de detecciones')
    
    # ReID Tracker
    p.add_argument('--reid', action='store_true', help='Activar ReID tracker')
    
    # TeamClassifier V2
    p.add_argument('--tc-kmeans-min-tracks', type=int, default=12)
    p.add_argument('--tc-vote-history', type=int, default=4)
    p.add_argument('--tc-use-L', action='store_true', default=True)
    p.add_argument('--tc-L-weight', type=float, default=0.5)
    
    # TeamClassifier V3
    p.add_argument('--use-v3', action='store_true', help='Usar TeamClassifierV3')
    p.add_argument('--v3-recalibrate', type=int, default=300, help='Recalibrar cada N frames')
    p.add_argument('--v3-variance', dest='v3_variance', action='store_true', default=True)
    p.add_argument('--no-v3-variance', dest='v3_variance', action='store_false')
    p.add_argument('--v3-adaptive-thresh', dest='v3_adaptive_thresh', action='store_true', default=True)
    p.add_argument('--no-v3-adaptive-thresh', dest='v3_adaptive_thresh', action='store_false')
    p.add_argument('--v3-hysteresis', dest='v3_hysteresis', action='store_true', default=True)
    p.add_argument('--no-v3-hysteresis', dest='v3_hysteresis', action='store_false')
    
    # Possession detection
    p.add_argument('--possession-distance', type=int, default=60,
                   help='Distancia máxima en píxeles para posesión del balón')
    
    # Field Calibration
    p.add_argument('--calibrate', action='store_true', help='Activar calibración del campo')
    p.add_argument('--keypoints-model', default='weights/field_kp_merged_fast/weights/best.pt',
                   help='Modelo YOLO de keypoints del campo')
    p.add_argument('--keypoints-conf', type=float, default=0.25,
                   help='Umbral de confianza para keypoints')
    p.add_argument('--show-keypoints', action='store_true',
                   help='Mostrar keypoints detectados en el frame')
    
    # Salida
    p.add_argument('--no-show', action='store_true', help='No mostrar ventana')
    p.add_argument('--output', default=None, help='Guardar video procesado')
    
    args = p.parse_args()

    # Cargar modelo YOLO
    model = YOLO(args.model)
    try:
        names = model.names
        print("Model classes:")
        for cid, name in names.items():
            print(f"  {cid}: {name}")
    except Exception:
        names = None

    print("Iniciando detección YOLO...")
    
    # Inicializar tracker con ReID
    tracker = None
    if args.reid and REID_AVAILABLE:
        tracker = ReIDTracker(max_age=30, max_lost_time=120.0)
        print(f"✓ ReID Tracker inicializado en {tracker.device}")
    
    # Inicializar PossessionTracker V2
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    possession = None
    if POSSESSION_AVAILABLE:
        possession = PossessionTrackerV2(fps=fps, hysteresis_frames=5)
        print(f"[PossessionV2] Inicializado (fps={fps}, hysteresis=5 frames)")
    
    # Inicializar TeamClassifier
    team_classifier = None
    if tracker and TEAMCLASS_AVAILABLE:
        if args.use_v3 and TEAMCLASS_V3_AVAILABLE:
            team_classifier = TeamClassifierV3(
                recalibrate_every_n_frames=args.v3_recalibrate,
                use_variance_features=args.v3_variance,
                use_adaptive_edge_threshold=args.v3_adaptive_thresh,
                use_temporal_hysteresis=args.v3_hysteresis
            )
            print("[TeamClassifierV3] Inicializado")
        else:
            team_classifier = TeamClassifierV2(
                kmeans_min_tracks=args.tc_kmeans_min_tracks,
                vote_history=args.tc_vote_history,
                use_L_channel=args.tc_use_L,
                L_weight=args.tc_L_weight
            )
            print(f"[TeamClassifierV2] Inicializado en modo LAB+AntiGreen")
        
        tracker.team_classifier = team_classifier
    
    # Estructura para almacenar datos de heatmap basados en keypoints
    heatmap_data = {
        'team_0': [],
        'team_1': [],
        'frames': [],
        'keypoints_detected': []
    }
    
    # Acumulador temporal de posiciones para triangulación (por frame)
    frame_player_positions = []  # Para proyectar en batch
    current_frame_keypoints_list = None
    is_flipped = False
    last_valid_keypoints = None  # Persistir últimos keypoints válidos
    last_valid_flip = False  # Persistir última orientación válida
    accumulated_kp_for_triangulation = {}  # Acumular keypoints detectados a lo largo de frames
    last_triangulation_update_frame = 0
    
    # Inicializar sistema de calibración
    keypoints_detector = None
    field_calibrator = None
    accumulated_keypoints = {}
    calibration_success = False
    homography_matrix = None
    current_frame_keypoints_list = None  # Para triangulación
    is_flipped = False  # Orientación del campo
    
    if args.calibrate and CALIBRATION_AVAILABLE:
        try:
            keypoints_detector = FieldKeypointsYOLO(
                model_path=args.keypoints_model,
                confidence_threshold=args.keypoints_conf,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            field_model = FieldModelKeypoints(
                field_length=105.0,
                field_width=68.0,
                use_normalized=False
            )
            
            field_calibrator = FieldCalibratorKeypoints(
                field_model=field_model,
                min_keypoints=4,
                ransac_threshold=5.0
            )
            
            print(f"✓ Sistema de calibración inicializado")
            print(f"  - Modelo: {args.keypoints_model}")
            print(f"  - Confidence: {args.keypoints_conf}")
            print(f"  - Device: {keypoints_detector.device}")
        except Exception as e:
            print(f"⚠️  Error inicializando calibración: {e}")
            keypoints_detector = None
    
    # Abrir video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir {args.video}")
        return
    
    # Configurar writer si se solicita output
    writer = None
    if args.output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        print(f"Guardando video en: {args.output}")
    
    frame_no = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_no += 1
        
        # Sistema de calibración con keypoints (detección continua frame a frame)
        current_keypoints = {}
        if keypoints_detector is not None:
            # Detectar keypoints en cada frame (o cada N frames para performance)
            if frame_no % 3 == 0:  # Cada 3 frames para balance entre precisión y velocidad
                current_keypoints = keypoints_detector.detect_keypoints(frame)
                
                # Acumular keypoints detectados para triangulación
                # Mantener ventana móvil de últimos 150 frames (~5 segundos a 30fps)
                MAX_KP_AGE = 150
                for name, coords in current_keypoints.items():
                    accumulated_kp_for_triangulation[name] = {
                        'xy': coords,
                        'frame': frame_no
                    }
                
                # Limpiar keypoints antiguos
                accumulated_kp_for_triangulation = {
                    name: data for name, data in accumulated_kp_for_triangulation.items()
                    if frame_no - data['frame'] <= MAX_KP_AGE
                }
                
                # Convertir keypoints acumulados a formato lista para triangulación
                current_frame_keypoints_list = [
                    {"cls_name": name, "xy": data['xy'], "conf": 0.9}
                    for name, data in accumulated_kp_for_triangulation.items()
                ]
                
                # DEBUG: Mostrar cuando se detectan keypoints
                if current_keypoints:
                    kp_freshness = {
                        name: f"{'🆕' if frame_no - data['frame'] == 0 else f'🕐{frame_no - data['frame']}f'}"
                        for name, data in accumulated_kp_for_triangulation.items()
                    }
                    if frame_no <= 100 or len(current_frame_keypoints_list) >= 3:
                        # Mostrar coordenadas de keypoints detectados
                        kp_coords_debug = {}
                        for name, data in accumulated_kp_for_triangulation.items():
                            x_px, y_px = data['xy']
                            kp_coords_debug[name] = f"({x_px:.0f},{y_px:.0f})"
                        print(f"[DEBUG Frame {frame_no}] KPs frescos: {list(current_keypoints.keys())} | "
                              f"Total acumulados: {len(current_frame_keypoints_list)}")
                        if len(current_frame_keypoints_list) >= 1:
                            print(f"  Píxeles detectados: {kp_coords_debug}")
                
                # Determinar orientación del campo usando heurística geométrica
                # Reducir min_points a 2 para videos con pocos keypoints visibles
                if len(current_frame_keypoints_list) >= 2:
                    _, is_flipped = estimate_homography_with_flip_resolution(
                        current_frame_keypoints_list,
                        FIELD_POINTS,
                        min_points=2,  # Reducido de 3 a 2
                        conf_threshold=0.3
                    )
                    # Guardar como último válido
                    last_valid_keypoints = current_frame_keypoints_list
                    last_valid_flip = is_flipped
                    last_triangulation_update_frame = frame_no
                    
                    if frame_no <= 100 or (frame_no - last_triangulation_update_frame == 0 and frame_no % 30 == 0):
                        print(f"[TRIANGULACIÓN Frame {frame_no}] ✓ Sistema actualizado con {len(current_frame_keypoints_list)} keypoints, flip={is_flipped}")
                
                # Acumular para calibración (opcional, mantener ventana móvil)
                MAX_AGE = 300
                for name, (x, y) in current_keypoints.items():
                    if name not in accumulated_keypoints:
                        accumulated_keypoints[name] = []
                    accumulated_keypoints[name].append((x, y, frame_no))
            else:
                # Usar últimos keypoints válidos si no se detectan nuevos
                current_frame_keypoints_list = last_valid_keypoints
                is_flipped = last_valid_flip
            
            # Limpiar keypoints antiguos (hacer siempre, no solo cuando se detectan nuevos)
            if accumulated_keypoints:
                MAX_AGE = 300
                for name in list(accumulated_keypoints.keys()):
                    accumulated_keypoints[name] = [
                        (x, y, f) for x, y, f in accumulated_keypoints[name]
                        if frame_no - f <= MAX_AGE
                    ]
                    if not accumulated_keypoints[name]:
                        del accumulated_keypoints[name]
            
            # Promediar keypoints para calibración (opcional)
            averaged_keypoints = {}
            for name, points in accumulated_keypoints.items():
                if points:
                    xs = [x for x, y, f in points]
                    ys = [y for x, y, f in points]
                    averaged_keypoints[name] = (np.mean(xs), np.mean(ys))
            
            # Intentar calibración cada 60 frames (mantener homografía actualizada)
            if frame_no % 60 == 0 and len(averaged_keypoints) >= 4:
                success = field_calibrator.estimate_homography(averaged_keypoints)
                if success:
                    if not calibration_success:
                        print(f"[Calibración] ✓ Completada en frame {frame_no}: {len(field_calibrator.matched_keypoints)}/{len(averaged_keypoints)} keypoints, error={field_calibrator.reprojection_error:.2f}px")
                    calibration_success = True
                    homography_matrix = field_calibrator.H
        else:
            # Si no hay detector de keypoints, resetear
            current_frame_keypoints_list = None
        
        # Mostrar estado de calibración (solo info)
        if keypoints_detector is not None:
            cal_text = f"Campo: Calibrado ({len(current_keypoints)} kp)" if calibration_success else f"Calibrando... ({len(current_keypoints)} kp)"
            cal_color = (0, 255, 0) if calibration_success else (0, 165, 255)
            cv2.putText(frame, cal_text, (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cal_color, 2)
            
            # Registrar keypoints detectados en este frame
            if current_keypoints:
                heatmap_data['keypoints_detected'].append({
                    'frame': frame_no,
                    'keypoints': list(current_keypoints.keys()),
                    'count': len(current_keypoints)
                })
            
            # Registrar información del frame
            heatmap_data['frames'].append({
                'number': frame_no,
                'calibrated': calibration_success,
                'keypoints_count': len(current_keypoints)
            })
        
        # Mostrar estadísticas de pases cada 1000 frames
        if frame_no > 0 and frame_no % 1000 == 0 and possession is not None:
            stats = possession.get_possession_stats()
            passes = stats.get('passes', {})
            print(f"\n[Pases @ frame {frame_no}]")
            for team_id in sorted(passes.keys()):
                print(f"  Team {team_id}: {passes[team_id]} pases")
            print()
        
        # Detección YOLO
        results = model(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=args.max_det,
            verbose=False
        )
        
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            if not args.no_show:
                cv2.imshow('TacticEYE2', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if writer:
                writer.write(frame)
            continue
        
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        
        # Tracking con ReID
        if tracker:
            tracks = tracker.update(frame, boxes, scores, cls_ids)
            
            # PASO 1: Acumular posiciones de jugadores en píxeles
            frame_player_positions = []
            for tid, bbox, cid in tracks:
                if cid in (0, 3):  # Solo jugadores y porteros
                    x1, y1, x2, y2 = map(int, bbox)
                    center_x = (x1 + x2) / 2.0
                    center_y = y2  # Pies del jugador
                    
                    team_id = -1
                    try:
                        team_id = tracker.active_tracks[tid].team_id
                    except Exception:
                        pass
                    
                    frame_player_positions.append({
                        'tid': tid,
                        'team_id': team_id,
                        'cid': cid,
                        'pixel_pos': (center_x, center_y),
                        'bbox': bbox
                    })
            
            # PASO 2: Proyectar todas las posiciones con TRIANGULACIÓN
            field_positions_map = {}  # {tid: (x_field, y_field)}
            
            # DEBUG: Verificar estado antes de proyectar
            if frame_no % 5 == 0:
                num_kps_before_projection = len(current_frame_keypoints_list) if current_frame_keypoints_list else 0
                num_players = len(frame_player_positions)
                print(f"[DEBUG PRE-PROYECCIÓN Frame {frame_no}] KPs disponibles: {num_kps_before_projection}, Jugadores: {num_players}")
            
            if current_frame_keypoints_list and len(frame_player_positions) > 0:
                try:
                    pixel_positions = [p['pixel_pos'] for p in frame_player_positions]
                    field_coords_array = project_points_by_triangulation(
                        pixel_positions,
                        current_frame_keypoints_list,
                        FIELD_POINTS,
                        is_flipped,
                        min_references=1,  # Permitir 1 keypoint para proyección básica
                        max_references=4
                    )
                    
                    # Mapear resultados
                    for i, player_data in enumerate(frame_player_positions):
                        if not np.isnan(field_coords_array[i]).any():
                            field_positions_map[player_data['tid']] = tuple(field_coords_array[i])
                            
                            # Guardar para heatmap
                            if player_data['team_id'] in (0, 1):
                                heatmap_data[f"team_{player_data['team_id']}"].append({
                                    'frame': frame_no,
                                    'track_id': player_data['tid'],
                                    'position': tuple(field_coords_array[i]),
                                    'method': 'triangulation'
                                })
                except Exception as e:
                    if frame_no % 100 == 0:
                        print(f"[Triangulación] Error frame {frame_no}: {e}")
            
            # DEBUG: Imprimir posiciones cada 5 frames
            if frame_no % 5 == 0 and (field_positions_map or frame_player_positions):
                print(f"\n{'='*70}")
                print(f"[Frame {frame_no}] Posiciones de Elementos Detectados")
                print(f"{'='*70}")
                print(f"Sistema de coordenadas:")
                print(f"  X = Longitudinal (0-{FIELD_LENGTH}m, línea gol izq→der)")
                print(f"  Y = Lateral (0-{FIELD_WIDTH}m, banda abajo→arriba)")
                print(f"  Centro campo = ({FIELD_LENGTH/2:.1f}, {FIELD_WIDTH/2:.1f})m")
                
                num_kps = len(current_frame_keypoints_list) if current_frame_keypoints_list else 0
                kp_source = ""
                if num_kps > 0:
                    # Verificar si son keypoints del frame actual o reutilizados
                    if frame_no % 3 == 0:
                        kp_source = " (detectados en este frame)"
                    else:
                        kp_source = " (reutilizados del último válido)"
                
                print(f"  Keypoints detectados: {num_kps}{kp_source}")
                if num_kps > 0:
                    print(f"  Orientación flip: {is_flipped}")
                else:
                    print(f"  ⚠️  Sin keypoints → Sin proyección posible")
                
                print(f"\nElementos proyectados: {len(field_positions_map)}/{len(frame_player_positions)}")
                print(f"{'-'*70}")
                
                # Agrupar por equipo para mejor visualización
                team_0_players = []
                team_1_players = []
                other_players = []
                
                for player_data in frame_player_positions:
                    tid = player_data['tid']
                    team_id = player_data['team_id']
                    px_pos = player_data['pixel_pos']
                    field_pos = field_positions_map.get(tid)
                    
                    player_info = {
                        'tid': tid,
                        'px': px_pos,
                        'field': field_pos
                    }
                    
                    if team_id == 0:
                        team_0_players.append(player_info)
                    elif team_id == 1:
                        team_1_players.append(player_info)
                    else:
                        other_players.append(player_info)
                
                # Imprimir Team 0
                if team_0_players:
                    print(f"\n🟢 TEAM 0 ({len(team_0_players)} jugadores):")
                    for p in team_0_players:
                        if p['field']:
                            x_f, y_f = p['field']
                            # Zona aproximada
                            if x_f < FIELD_LENGTH / 3:
                                zone = "Defensa"
                            elif x_f < 2 * FIELD_LENGTH / 3:
                                zone = "Mediocampo"
                            else:
                                zone = "Ataque"
                            
                            print(f"  ID #{p['tid']:3d} | Píxeles: ({p['px'][0]:4.0f}, {p['px'][1]:4.0f}) "
                                  f"→ Campo: ({x_f:5.1f}m, {y_f:5.1f}m) | {zone}")
                        else:
                            print(f"  ID #{p['tid']:3d} | Píxeles: ({p['px'][0]:4.0f}, {p['px'][1]:4.0f}) → ❌ No proyectado")
                
                # Imprimir Team 1
                if team_1_players:
                    print(f"\n🔴 TEAM 1 ({len(team_1_players)} jugadores):")
                    for p in team_1_players:
                        if p['field']:
                            x_f, y_f = p['field']
                            # Zona aproximada
                            if x_f < FIELD_LENGTH / 3:
                                zone = "Defensa"
                            elif x_f < 2 * FIELD_LENGTH / 3:
                                zone = "Mediocampo"
                            else:
                                zone = "Ataque"
                            
                            print(f"  ID #{p['tid']:3d} | Píxeles: ({p['px'][0]:4.0f}, {p['px'][1]:4.0f}) "
                                  f"→ Campo: ({x_f:5.1f}m, {y_f:5.1f}m) | {zone}")
                        else:
                            print(f"  ID #{p['tid']:3d} | Píxeles: ({p['px'][0]:4.0f}, {p['px'][1]:4.0f}) → ❌ No proyectado")
                
                # Imprimir otros (sin equipo asignado)
                if other_players:
                    print(f"\n⚪ SIN EQUIPO ({len(other_players)} jugadores):")
                    for p in other_players:
                        if p['field']:
                            x_f, y_f = p['field']
                            print(f"  ID #{p['tid']:3d} | Píxeles: ({p['px'][0]:4.0f}, {p['px'][1]:4.0f}) "
                                  f"→ Campo: ({x_f:5.1f}m, {y_f:5.1f}m)")
                        else:
                            print(f"  ID #{p['tid']:3d} | Píxeles: ({p['px'][0]:4.0f}, {p['px'][1]:4.0f}) → ❌ No proyectado")
                
                print(f"{'='*70}\n")
            
            # PASO 3: Dibujar tracks con información de campo
            for tid, bbox, cid in tracks:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Obtener team_id
                team_id = -1
                try:
                    team_id = tracker.active_tracks[tid].team_id
                except Exception:
                    pass
                
                # Obtener coordenadas de campo si están disponibles
                field_coords = field_positions_map.get(tid, None)
                
                # Color por clase y equipo
                if cid == 2:  # referee
                    color = (0, 140, 255)
                    team_label = 'REF'
                elif cid == 1:  # ball
                    color = (255, 255, 255)
                    team_label = 'NA'
                else:  # player/goalkeeper
                    if team_id == 0:
                        color = (0, 200, 0)  # Team 0 - verde
                    elif team_id == 1:
                        color = (255, 0, 0)  # Team 1 - azul (BGR)
                    else:
                        color = (200, 100, 0)
                    team_label = 'NA' if team_id is None or team_id < 0 else str(team_id)
                
                # Label con información según modo debug
                if args.show_keypoints and field_coords:
                    # Mostrar coordenadas de campo por triangulación
                    label = f"ID:{tid} T{team_label} ({field_coords[0]:.1f},{field_coords[1]:.1f})m"
                else:
                    label = f"ID:{tid} {names.get(cid, cid) if names else cid} T{team_label}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Detección de posesión
            if possession is not None:
                import math
                
                # PASO 1: Buscar posición del balón
                ball_pos = None
                ball_pos_field = None
                for tid, bbox, cid in tracks:
                    if cid == 1:  # ball
                        x1, y1, x2, y2 = map(int, bbox)
                        ball_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        
                        # Transformar a coordenadas del campo si está calibrado
                        if calibration_success and homography_matrix is not None:
                            point = np.array([[ball_pos[0], ball_pos[1]]], dtype=np.float32).reshape(-1, 1, 2)
                            transformed = cv2.perspectiveTransform(point, homography_matrix)
                            ball_pos_field = tuple(transformed[0][0])
                        break
                
                # PASO 2: Buscar jugador más cercano al balón
                ball_owner_team = None
                ball_owner_id = None
                min_distance = float('inf')
                min_distance_field = float('inf')
                
                if ball_pos is not None:
                    for tid, bbox, cid in tracks:
                        if cid not in (0, 3):  # Solo jugadores y porteros
                            continue
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        player_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        
                        # Distancia en píxeles (para visualización)
                        dist_px = math.hypot(player_pos[0] - ball_pos[0], player_pos[1] - ball_pos[1])
                        
                        # Si está calibrado, usar distancia en coordenadas del campo (metros)
                        if calibration_success and homography_matrix is not None and ball_pos_field is not None:
                            point = np.array([[player_pos[0], y2]], dtype=np.float32).reshape(-1, 1, 2)
                            transformed = cv2.perspectiveTransform(point, homography_matrix)
                            player_pos_field = tuple(transformed[0][0])
                            dist_field = math.hypot(
                                player_pos_field[0] - ball_pos_field[0],
                                player_pos_field[1] - ball_pos_field[1]
                            )
                            
                            if dist_field < min_distance_field:
                                min_distance_field = dist_field
                                min_distance = dist_px  # Guardar también distancia en píxeles
                                ball_owner_id = tid
                                try:
                                    ball_owner_team = tracker.active_tracks[tid].team_id
                                except Exception:
                                    ball_owner_team = None
                        else:
                            # Usar distancia en píxeles si no está calibrado
                            if dist_px < min_distance:
                                min_distance = dist_px
                                ball_owner_id = tid
                                try:
                                    ball_owner_team = tracker.active_tracks[tid].team_id
                                except Exception:
                                    ball_owner_team = None
                
                # PASO 3: Validar distancia
                if calibration_success and ball_pos_field is not None:
                    # Usar distancia en metros (típicamente 2-3 metros máximo)
                    max_possession_distance_m = 3.0
                    if min_distance_field > max_possession_distance_m:
                        ball_owner_team = None
                        ball_owner_id = None
                else:
                    # Usar distancia en píxeles
                    max_possession_distance = args.possession_distance
                    if min_distance > max_possession_distance:
                        ball_owner_team = None
                        ball_owner_id = None
                
                # PASO 4: Filtrar team_id inválidos (referees = -1)
                if ball_owner_team is not None and ball_owner_team < 0:
                    ball_owner_team = None
                
                # Actualizar PossessionTrackerV2
                possession.update(frame_no, ball_owner_team, ball_owner_id)
                
                # Visualizar estadísticas de posesión
                stats = possession.get_possession_stats()
                current = possession.get_current_possession()
                
                current_team = current['team']
                team_text = f"Team {current_team}" if current_team is not None else "N/A"
                stext1 = f"Possession: {team_text}"
                
                p0 = stats['possession_percent'].get(0, 0.0)
                p1 = stats['possession_percent'].get(1, 0.0)
                stext2 = f"Total: Team0={p0:.1f}% Team1={p1:.1f}%"
                
                s0 = stats['possession_seconds'].get(0, 0.0)
                s1 = stats['possession_seconds'].get(1, 0.0)
                stext3 = f"Time: T0={s0:.1f}s T1={s1:.1f}s"
                
                # Mostrar distancia según modo
                if calibration_success and ball_pos_field is not None:
                    stext4 = f"Max dist: 3.0m (campo calibrado)"
                else:
                    stext4 = f"Max dist: {args.possession_distance}px"
                
                cv2.putText(frame, stext1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, stext2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,255,100), 2)
                cv2.putText(frame, stext3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
                cv2.putText(frame, stext4, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
                
                # Visualizar jugador con posesión
                if ball_owner_id is not None and ball_pos is not None:
                    for tid, bbox, cid in tracks:
                        if tid == ball_owner_id:
                            x1, y1, x2, y2 = map(int, bbox)
                            player_center = (int((x1 + x2)/2), int((y1 + y2)/2))
                            ball_center = (int(ball_pos[0]), int(ball_pos[1]))
                            
                            # Rectángulo amarillo
                            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 255), 3)
                            
                            # Línea al balón
                            cv2.line(frame, player_center, ball_center, (0, 255, 255), 2)
                            
                            # Distancia (mostrar en metros si está calibrado)
                            if calibration_success and ball_pos_field is not None:
                                dist_text = f"{min_distance_field:.2f}m"
                            else:
                                dist_text = f"{min_distance:.0f}px"
                            
                            mid_point = (int((player_center[0] + ball_center[0])/2), 
                                        int((player_center[1] + ball_center[1])/2))
                            cv2.putText(frame, dist_text, mid_point, 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                            # Etiqueta
                            cv2.putText(frame, 'POSSESSION', (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            break
        
        else:
            # Sin tracker: solo dibujar detecciones
            for i, (box, conf, cid) in enumerate(zip(boxes, scores, cls_ids)):
                x1, y1, x2, y2 = map(int, box)
                class_label = names.get(cid, cid) if names else cid
                label = f"{class_label}:{conf:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1-6,0)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Visualizar keypoints detectados con bounding boxes
        if args.show_keypoints and current_keypoints:
            # Obtener prioridades para codificar color
            keypoint_box_size = 15  # Tamaño del box alrededor del keypoint
            
            for kp_name, (kp_x, kp_y) in current_keypoints.items():
                # Obtener prioridad del keypoint
                priority = KEYPOINT_PRIORITY.get(kp_name, 50)
                
                # Color según prioridad
                if priority >= 90:  # ALTA (centro del campo)
                    kp_color = (0, 255, 255)  # Amarillo brillante
                elif priority >= 75:  # Media-Alta (áreas grandes)
                    kp_color = (0, 200, 255)  # Naranja
                elif priority >= 65:  # Media (áreas pequeñas)
                    kp_color = (255, 150, 0)  # Azul claro
                else:  # Baja (esquinas)
                    kp_color = (180, 180, 180)  # Gris
                
                # Dibujar bounding box centrado en el keypoint
                x1 = int(kp_x - keypoint_box_size)
                y1 = int(kp_y - keypoint_box_size)
                x2 = int(kp_x + keypoint_box_size)
                y2 = int(kp_y + keypoint_box_size)
                
                # Box del keypoint
                cv2.rectangle(frame, (x1, y1), (x2, y2), kp_color, 2)
                
                # Label del keypoint (nombre corto)
                kp_label = kp_name.replace('_', ' ').title()
                # Acortar nombres muy largos
                if len(kp_label) > 20:
                    parts = kp_label.split()
                    if len(parts) > 2:
                        kp_label = f"{parts[0][0]}{parts[1][0]}{parts[-1][:3]}"
                
                # Fondo semi-transparente para el texto
                label_size = cv2.getTextSize(kp_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(frame, (x1, y1 - 18), (x1 + label_size[0] + 4, y1 - 2), 
                             kp_color, -1)
                
                # Texto del label
                cv2.putText(frame, kp_label, (x1 + 2, y1 - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Punto central
                cv2.circle(frame, (int(kp_x), int(kp_y)), 3, (255, 255, 255), -1)
        
        # Mostrar/guardar
        if not args.no_show:
            cv2.imshow('TacticEYE2', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if writer:
            writer.write(frame)
        
        if frame_no % 100 == 0:
            print(f'Procesados {frame_no} frames...')
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Exportar datos de heatmap basados en keypoints
    if keypoints_detector is not None and (heatmap_data['team_0'] or heatmap_data['team_1']):
        import json
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'outputs_heatmap'
        os.makedirs(output_dir, exist_ok=True)
        
        heatmap_file = os.path.join(output_dir, f'heatmap_data_{timestamp}.json')
        
        # Preparar resumen
        summary = {
            'total_frames': frame_no,
            'team_0_positions': len(heatmap_data['team_0']),
            'team_1_positions': len(heatmap_data['team_1']),
            'keypoints_frames': len(heatmap_data['keypoints_detected']),
            'calibration_success': calibration_success
        }
        
        export_data = {
            'summary': summary,
            'team_0': heatmap_data['team_0'],
            'team_1': heatmap_data['team_1'],
            'keypoints_detected': heatmap_data['keypoints_detected'],
            'frames_info': heatmap_data['frames']
        }
        
        with open(heatmap_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n{'='*70}")
        print('HEATMAP DATA EXPORTED')
        print('='*70)
        print(f"\nArchivo: {heatmap_file}")
        print(f"  - Team 0: {summary['team_0_positions']} posiciones registradas")
        print(f"  - Team 1: {summary['team_1_positions']} posiciones registradas")
        print(f"  - Frames con keypoints: {summary['keypoints_frames']}")
        print(f"  - Campo calibrado: {'Sí' if summary['calibration_success'] else 'No'}")
        print('='*70)
    
    # Resumen final de posesión
    if possession is not None:
        print('\n' + '='*70)
        print('POSSESSION SUMMARY (PossessionTrackerV2)')
        print('='*70)
        
        stats = possession.get_possession_stats()
        
        print(f"\nTotal frames processed: {stats['total_frames']}")
        print(f"Total time: {stats['total_seconds']:.2f} seconds")
        
        print("\nPossession by team:")
        for team_id in range(2):
            frames = stats['possession_frames'].get(team_id, 0)
            seconds = stats['possession_seconds'].get(team_id, 0.0)
            percent = stats['possession_percent'].get(team_id, 0.0)
            print(f"  Team {team_id}: {frames} frames ({seconds:.1f}s) = {percent:.1f}%")
        
        total_assigned = sum(stats['possession_frames'].values())
        print(f"\nValidation:")
        print(f"  Frames assigned: {total_assigned}/{stats['total_frames']}")
        coverage = total_assigned/stats['total_frames']*100 if stats['total_frames'] > 0 else 0
        print(f"  Coverage: {coverage:.1f}%")
        
        timeline = possession.get_possession_timeline()
        if timeline:
            print(f"\nPossession timeline ({len(timeline)} segments):")
            for i, (start, end, team) in enumerate(timeline[:3]):
                duration = end - start
                print(f"  Segment {i+1}: Frames {start}-{end} ({duration}f) → Team {team}")
            if len(timeline) > 6:
                print(f"  ... ({len(timeline)-6} segments omitted) ...")
            for i, (start, end, team) in enumerate(timeline[-3:], len(timeline)-2):
                if i >= 3:
                    duration = end - start
                    print(f"  Segment {i}: Frames {start}-{end} ({duration}f) → Team {team}")
        
        print('='*70)
    
    # Resumen final de calibración
    if keypoints_detector is not None:
        print('\n' + '='*70)
        print('FIELD CALIBRATION SUMMARY')
        print('='*70)
        
        if calibration_success:
            print(f"\n✓ Calibración exitosa")
            print(f"  - Keypoints acumulados: {len(accumulated_keypoints)}")
            print(f"  - Keypoints matched: {len(field_calibrator.matched_keypoints)}")
            print(f"  - Reprojection error: {field_calibrator.reprojection_error:.2f}px")
            print(f"\n  Homography matrix:")
            for row in homography_matrix:
                print(f"    [{row[0]:9.6f}, {row[1]:9.6f}, {row[2]:9.6f}]")
            
            print(f"\n  Keypoints detectados:")
            for name in sorted(accumulated_keypoints.keys()):
                count = len(accumulated_keypoints[name])
                x, y = averaged_keypoints[name]
                print(f"    - {name}: ({x:.1f}, {y:.1f}) [{count} detecciones]")
        else:
            print(f"\n⚠ Calibración no completada")
            print(f"  - Keypoints acumulados: {len(accumulated_keypoints)}")
            print(f"  - Se requieren al menos 4 keypoints para calibrar")
            if accumulated_keypoints:
                print(f"\n  Keypoints detectados:")
                for name in sorted(accumulated_keypoints.keys()):
                    count = len(accumulated_keypoints[name])
                    print(f"    - {name}: [{count} detecciones]")
        
        print('='*70)
    
    print('\nDetección finalizada')


if __name__ == '__main__':
    main()
