"""
Script de prueba: ejecutar solo detección YOLO sobre un vídeo

Uso:
    python example_yolo_detection.py input.mp4 --model weights/best.pt --classes 0 32 --imgsz 640 --conf 0.35

Opciones relevantes:
- `--classes`: lista de ids de clases a mostrar (ej. 0 para personas). Si no se pasa, muestra todas.
- `--imgsz`: tamaño de entrada para YOLO (reduce NMS time).
- `--max-det`: máximo detecciones por frame.
- `--conf`: umbral de confianza.
- `--no-show`: no mostrar ventanas (útil en servidor).
- `--output`: ruta para guardar video con cajas.

Requiere `ultralytics` y `opencv-python`.
"""
"""
Script de prueba: ejecutar solo detección YOLO sobre un vídeo

Uso:
    python example_yolo_detection.py input.mp4 --model weights/best.pt --classes 0 32 --imgsz 640 --conf 0.35

Opciones relevantes:
- `--classes`: lista de ids de clases a mostrar (ej. 0 para personas). Si no se pasa, muestra todas.
- `--imgsz`: tamaño de entrada para YOLO (reduce NMS time).
- `--max-det`: máximo detecciones por frame.
- `--conf`: umbral de confianza.
- `--no-show`: no mostrar ventanas (útil en servidor).
- `--output`: ruta para guardar video con cajas.

Requiere `ultralytics` y `opencv-python`.
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

try:
    from modules.possession_tracker_v2 import PossessionTrackerV2
    POSSESSION_AVAILABLE = True
except Exception:
    POSSESSION_AVAILABLE = False
 
try:
    from modules.team_classifier_v2 import TeamClassifierV2
    TEAMCLASS_AVAILABLE = True
except Exception:
    TEAMCLASS_AVAILABLE = False

try:
    from modules.team_classifier_v3 import TeamClassifierV3
    TEAMCLASS_V3_AVAILABLE = True
except Exception:
    TEAMCLASS_V3_AVAILABLE = False

def draw_boxes(frame, boxes, scores, cls_ids, names=None, det_ids=None, colors=None, default_color=(0,255,0)):
    """Dibuja cajas y muestra clase/conf y opcionalmente un ID por detección.

    - If det_ids is provided, it should align with boxes and be shown as `D{id}`.
    - If names is available, class name is shown.
    """
    for i, ((x1, y1, x2, y2), conf, cid) in enumerate(zip(boxes.astype(int), scores, cls_ids)):
        class_label = names.get(cid, cid) if names else cid
        det_label = f" D{det_ids[i]}" if det_ids is not None else ""
        label = f"{class_label}:{conf:.2f}{det_label}"
        # choose per-box color if provided
        if colors is not None:
            c = colors[i]
        else:
            c = default_color
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('--model', default='yolov8n.pt')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--max-det', type=int, default=100)
    p.add_argument('--classes', type=int, nargs='*', default=None, help='IDs de clases a mostrar')
    p.add_argument('--reid', action='store_true', help='Habilitar ReID tracker (IDs persistentes)')
    # TeamClassifier V2 Pro tuning args
    p.add_argument('--tc-kmeans-min-tracks', type=int, default=12, help='Mínimo de tracks para inicializar KMeans')
    p.add_argument('--tc-kmeans-min-samples', type=int, default=2, help='Mínimo de muestras por track para KMeans')
    p.add_argument('--tc-min-track-samples', type=int, default=3, help='Mínimo de muestras por track')
    p.add_argument('--tc-vote-history', type=int, default=5, help='Tamaño del buffer de votación')
    p.add_argument('--tc-green-h-low', type=int, default=35, help='Hue mínimo para detectar verde (0-180)')
    p.add_argument('--tc-green-h-high', type=int, default=95, help='Hue máximo para detectar verde (0-180)')
    p.add_argument('--tc-green-s-min', type=int, default=40, help='Saturación mínima para verde (0-255)')
    p.add_argument('--tc-green-v-min', type=int, default=40, help='Value mínimo para verde (0-255)')
    p.add_argument('--tc-min-non-green-ratio', type=float, default=0.05, help='Ratio mínimo de píxeles no-verdes')
    p.add_argument('--tc-min-bbox-area-frac', type=float, default=0.001, help='Área mínima bbox como fracción de frame')
    p.add_argument('--tc-use-L', action='store_true', default=True, help='Incluir canal L en features LAB')
    p.add_argument('--tc-L-weight', type=float, default=0.5, help='Peso del canal L si se usa')
    p.add_argument('--tc-save-rois', action='store_true', help='Guardar ROIs para debug')
    p.add_argument('--tc-rois-dir', type=str, default='debug_rois_v2pro', help='Directorio para ROIs de debug')
    # TeamClassifier V3 flags
    p.add_argument('--use-v3', action='store_true', help='Usar TeamClassifierV3 (recomendado)')
    p.add_argument('--v3-recalibrate', type=int, default=300, help='Recalibrar KMeans cada N frames (0=desactivar)')
    p.add_argument('--v3-variance', dest='v3_variance', action='store_true', default=True, help='Usar features de varianza (default: True)')
    p.add_argument('--no-v3-variance', dest='v3_variance', action='store_false', help='Desactivar features de varianza')
    p.add_argument('--v3-adaptive-thresh', dest='v3_adaptive_thresh', action='store_true', default=True, help='Umbral adaptativo de edges (default: True)')
    p.add_argument('--no-v3-adaptive-thresh', dest='v3_adaptive_thresh', action='store_false', help='Desactivar umbral adaptativo')
    p.add_argument('--v3-hysteresis', dest='v3_hysteresis', action='store_true', default=True, help='Activar histeresis temporal (default: True)')
    p.add_argument('--no-v3-hysteresis', dest='v3_hysteresis', action='store_false', help='Desactivar histeresis temporal')
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--output', default=None, help='Guardar video con detecciones')
    p.add_argument('--dump-csv', default=None, help='Guardar CSV de asignaciones por frame')
    
    # Possession detection parameters
    p.add_argument('--possession-distance', type=int, default=60, 
                   help='Distancia máxima en píxeles para posesión del balón (default: 60)')
    
    args = p.parse_args()

    model = YOLO(args.model)
    # Mostrar nombres/IDs de clases del modelo
    try:
        names = model.names
        print("Model classes:")
        for cid, name in names.items():
            print(f"  {cid}: {name}")
    except Exception:
        names = None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"No se pudo abrir {args.video}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_no = 0
    csv_file = None
    csv_writer = None
    if args.dump_csv:
        import csv
        csv_file = open(args.dump_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame','track_id','cx','tc_team','tr_team','class_id'])
    print('Iniciando detección YOLO...')

    # Inicializar ReID tracker opcional
    tracker = None
    if args.reid:
        if not REID_AVAILABLE:
            print('Warning: ReID module no disponible, ejecuto sin tracker')
        else:
            tracker = ReIDTracker()

    # Inicializar PossessionTrackerV2 (funciona independiente del ReID)
    possession = None
    if POSSESSION_AVAILABLE:
        # PossessionTrackerV2: reglas simples, todo el tiempo asignado a un equipo
        possession = PossessionTrackerV2(fps=fps, hysteresis_frames=5)
        print('[PossessionV2] Inicializado (fps={}, hysteresis=5 frames)'.format(fps))
    else:
        print('Warning: modules.possession_tracker_v2 no disponible; sin posesion')
    # Inicializar TeamClassifier (V3 opcional) si está disponible
    team_classifier = None
    if TEAMCLASS_AVAILABLE:
        try:
            # Parámetros comunes
            common_kwargs = dict(
                # Green removal
                green_h_low=args.tc_green_h_low,
                green_h_high=args.tc_green_h_high,
                green_s_min=args.tc_green_s_min,
                green_v_min=args.tc_green_v_min,
                min_non_green_ratio=args.tc_min_non_green_ratio,
                # Track sampling
                min_track_samples=args.tc_min_track_samples,
                min_bbox_area_frac=args.tc_min_bbox_area_frac,
                # KMeans / voting
                kmeans_min_tracks=args.tc_kmeans_min_tracks,
                kmeans_min_samples_per_track=args.tc_kmeans_min_samples,
                vote_history=args.tc_vote_history,
                # Debug
                save_debug_rois=args.tc_save_rois,
                debug_rois_dir=args.tc_rois_dir,
                # LAB features (kept for compatibility)
                use_L_channel=args.tc_use_L,
                L_weight=args.tc_L_weight,
                referee_detection=True
            )

            if args.use_v3 and TEAMCLASS_V3_AVAILABLE:
                v3_kwargs = dict(common_kwargs)
                v3_kwargs.update(dict(
                    use_variance_features=args.v3_variance,
                    adaptive_edge_threshold=args.v3_adaptive_thresh,
                    hysteresis_mode=args.v3_hysteresis,
                    recalibrate_every_n_frames=args.v3_recalibrate,
                    use_spatial_context=False
                ))
                team_classifier = TeamClassifierV3(**v3_kwargs)
                print('[TeamClassifierV3] Inicializado (LAB+V3 enhancements)')
            else:
                # Fallback V2
                team_classifier = TeamClassifierV2(**common_kwargs)
                print('[TeamClassifierV2] Inicializado en modo LAB+AntiGreen')

            # Informational prints
            print(f'  Green removal: H=[{args.tc_green_h_low},{args.tc_green_h_high}] S>={args.tc_green_s_min} V>={args.tc_green_v_min}')
            print(f'  Min non-green ratio: {args.tc_min_non_green_ratio}')
            print(f'  KMeans min tracks: {args.tc_kmeans_min_tracks}, min samples/track: {args.tc_kmeans_min_samples}')
            feat_desc = f"LAB (a*,b* + L*w={args.tc_L_weight})" if args.tc_use_L else "LAB (a*,b*)"
            print(f'  Features: {feat_desc}')
            print(f'  Features: {feat_desc}')
            debug_desc = f"enabled -> {args.tc_rois_dir}" if args.tc_save_rois else "disabled"
            print(f'  Debug ROIs: {debug_desc}')
            print(f'  Debug ROIs: {"enabled -> " + args.tc_rois_dir if args.tc_save_rois else "disabled"}')
        except Exception:
            team_classifier = None
            import traceback
            print('[ERROR] TeamClassifierV2 initialization failed:')
            traceback.print_exc()
    det_global = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        # Opcional: resize para velocidad antes de pasar a predict (pero mantener coords originales)
        resized = cv2.resize(frame, (args.imgsz, int(h * args.imgsz / w))) if args.imgsz and args.imgsz < max(w,h) else frame

        # predict
        results = model.predict(resized, conf=args.conf, imgsz=args.imgsz, max_det=args.max_det, verbose=False)
        # ultralytics retorna lista por batch
        res = results[0]

        # obtener boxes en coordenadas del resized frame, necesitamos mapear a original si redimensionamos
        if resized.shape[:2] != frame.shape[:2]:
            rh = frame.shape[0] / resized.shape[0]
            rw = frame.shape[1] / resized.shape[1]
        else:
            rh = rw = 1.0

        boxes = []
        scores = []
        cls_ids = []
        for box in getattr(res, 'boxes', []):
            xyxy = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
            conf = float(box.conf.cpu().numpy()[0])
            cid = int(box.cls.cpu().numpy()[0])

            if args.classes is not None and cid not in args.classes:
                continue

            x1, y1, x2, y2 = xyxy
            x1 = int(x1 * rw)
            x2 = int(x2 * rw)
            y1 = int(y1 * rh)
            y2 = int(y2 * rh)

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            cls_ids.append(cid)

        # Si tracker habilitado, actualizar y dibujar con IDs
        if tracker is not None and len(boxes) > 0:
            dets = np.array(boxes, dtype=np.float32)
            scores_arr = np.array(scores, dtype=np.float32)
            classes_arr = np.array(cls_ids, dtype=np.int32)
            tracks = tracker.update(frame, dets, scores_arr, classes_arr)

            # Dibujar tracks: (track_id, bbox, class_id)
            for tid, bbox, cid in tracks:
                x1, y1, x2, y2 = map(int, bbox)
                # Prefer TeamClassifier (color) when it has a confirmed team; otherwise use track's team_id
                team_id = None
                # try to read current tracker team if present
                try:
                    team_id = tracker.active_tracks[tid].team_id
                except Exception:
                    team_id = None

                # If TeamClassifier available, always feed it and prefer its confirmed assignment
                if TEAMCLASS_AVAILABLE and team_classifier is not None:
                    try:
                        team_classifier.add_detection(tid, (x1, y1, x2, y2), frame, class_id=cid)
                        t = team_classifier.get_team(tid)
                        if t >= 0:
                            # V2 usa votación interna, confiar en asignación cuando existe
                            team_id = int(t)
                            try:
                                tracker.active_tracks[tid].team_id = team_id
                            except Exception:
                                pass
                    except Exception:
                        pass

                # referee always shown as TNA (no team)
                if cid == 2:
                    color = (0, 200, 200)  # referee - yellow
                    team_label = 'NA'
                elif cid == 1:
                    # ball shown in white and TNA
                    color = (255, 255, 255)
                    team_label = 'NA'
                else:
                    if team_id == 0:
                        color = (0, 200, 0)  # local - green
                    elif team_id == 1:
                        color = (255, 0, 0)  # visitor - blue (BGR)
                    else:
                        color = (200, 100, 0)  # default
                    team_label = 'NA' if team_id is None or team_id < 0 else str(team_id)

                label = f"ID:{tid} {names.get(cid, cid) if names else cid} T{team_label}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            # Preparar datos para PossessionTrackerV2
            if possession is not None:
                import math
                
                # PASO 1: Buscar posición del balón
                ball_pos = None
                for tid, bbox, cid in tracks:
                    if cid == 1:  # ball
                        x1, y1, x2, y2 = map(int, bbox)
                        ball_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        break
                
                # PASO 2: Buscar jugador más cercano al balón
                ball_owner_team = None
                ball_owner_id = None
                min_distance = float('inf')
                
                if ball_pos is not None:
                    for tid, bbox, cid in tracks:
                        # Solo considerar jugadores y porteros
                        if cid not in (0, 3):
                            continue
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        player_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        
                        # Calcular distancia al balón
                        dist = math.hypot(player_pos[0] - ball_pos[0], player_pos[1] - ball_pos[1])
                        
                        # Encontrar el más cercano
                        if dist < min_distance:
                            min_distance = dist
                            ball_owner_id = tid
                            # Obtener team_id del jugador
                            try:
                                ball_owner_team = tracker.active_tracks[tid].team_id
                            except Exception:
                                ball_owner_team = None
                
                # PASO 3: Validar distancia - si está muy lejos, no hay posesión clara
                max_possession_distance = args.possession_distance
                if min_distance > max_possession_distance:
                    ball_owner_team = None
                    ball_owner_id = None
                
                # PASO 4: Filtrar team_id inválidos (referees = -1)
                # PossessionTrackerV2 solo acepta: 0, 1, o None
                if ball_owner_team is not None and ball_owner_team < 0:
                    ball_owner_team = None
                
                # Actualizar PossessionTrackerV2 con el equipo del poseedor
                possession.update(frame_no, ball_owner_team)
                
                # Obtener estadísticas y visualizar
                stats = possession.get_possession_stats()
                current = possession.get_current_possession()
                
                # Línea 1: Posesión actual
                current_team = current['team']
                team_text = f"Team {current_team}" if current_team is not None else "N/A"
                stext1 = f"Possession: {team_text}"
                
                # Línea 2: Porcentajes acumulados
                p0 = stats['possession_percent'].get(0, 0.0)
                p1 = stats['possession_percent'].get(1, 0.0)
                stext2 = f"Total: Team0={p0:.1f}% Team1={p1:.1f}%"
                
                # Línea 3: Tiempo en segundos
                s0 = stats['possession_seconds'].get(0, 0.0)
                s1 = stats['possession_seconds'].get(1, 0.0)
                stext3 = f"Time: T0={s0:.1f}s T1={s1:.1f}s"
                
                # Línea 4: Configuración de precisión
                stext4 = f"Max dist: {max_possession_distance}px"
                
                cv2.putText(frame, stext1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, stext2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,255,100), 2)
                cv2.putText(frame, stext3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
                cv2.putText(frame, stext4, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
                
                # Si se detectó un poseedor, marcar su caja y línea al balón
                if ball_owner_id is not None and ball_pos is not None:
                    for tid, bbox, cid in tracks:
                        try:
                            # Marcar SOLO el jugador con posesión (por ID)
                            if tid == ball_owner_id:
                                x1, y1, x2, y2 = map(int, bbox)
                                player_center = (int((x1 + x2)/2), int((y1 + y2)/2))
                                ball_center = (int(ball_pos[0]), int(ball_pos[1]))
                                
                                # Rectángulo amarillo alrededor del poseedor
                                cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 255), 3)
                                
                                # Línea entre jugador y balón
                                cv2.line(frame, player_center, ball_center, (0, 255, 255), 2)
                                
                                # Mostrar distancia en píxeles
                                dist_text = f"{min_distance:.0f}px"
                                mid_point = (int((player_center[0] + ball_center[0])/2), 
                                            int((player_center[1] + ball_center[1])/2))
                                cv2.putText(frame, dist_text, mid_point, 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                
                                # Etiqueta de posesión
                                cv2.putText(frame, 'POSSESSION', (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                break
                        except Exception:
                            pass
            # --- Runtime assignment report (Team vs Tracker) every 100 frames ---
            if frame_no % 100 == 0:
                try:
                    total_tracks = len(tracks)
                    assigned_by_tc = 0
                    assigned_by_tracker = 0
                    undecided = 0
                    tc_map = {}
                    for tid, bbox, cid in tracks:
                        tid = int(tid)
                        tc_team = None
                        if TEAMCLASS_AVAILABLE and team_classifier is not None:
                            try:
                                tc_team = team_classifier.get_team(tid)
                            except Exception:
                                tc_team = None
                        # tracker team_id if present
                        tr_team = None
                        try:
                            tr_team = tracker.active_tracks[tid].team_id
                        except Exception:
                            tr_team = None
                        if tc_team is not None and tc_team >= 0:
                            assigned_by_tc += 1
                        if tr_team is not None and tr_team >= 0:
                            assigned_by_tracker += 1
                        if (tc_team is None or tc_team < 0) and (tr_team is None or tr_team < 0):
                            undecided += 1
                        tc_map[tid] = {'tc': tc_team, 'tr': tr_team}
                    print(f"[REPORT] Frame {frame_no}: tracks={total_tracks} tc_assigned={assigned_by_tc} tr_assigned={assigned_by_tracker} undecided={undecided}")
                    # print up to 8 sample mappings
                    sample_items = list(tc_map.items())[:8]
                    for tid, m in sample_items:
                        print(f"  TID {tid}: TeamClassifier={m['tc']} Tracker={m['tr']}")
                    # dump full mapping to CSV if requested
                    if csv_writer is not None:
                        for tid, info in tc_map.items():
                            # find cx from tracks list
                            cx_val = None
                            clsid = ''
                            for ttid, bbox, cid in tracks:
                                if int(ttid) == int(tid):
                                    cx_val = (int(bbox[0]) + int(bbox[2]))/2.0
                                    clsid = cid
                                    break
                            csv_writer.writerow([frame_no, tid, cx_val, info.get('tc'), info.get('tr'), clsid])
                        try:
                            csv_file.flush()
                            import os
                            os.fsync(csv_file.fileno())
                        except Exception:
                            pass
                except Exception:
                    pass
                
        else:
            if boxes:
                # Asignar IDs únicos por detección cuando no hay ReID
                det_ids = [det_global + i for i in range(len(boxes))]
                det_global += len(boxes)
                # build per-box colors: white for ball (cid==1), default green otherwise
                box_colors = []
                for cid in cls_ids:
                    if cid == 1:
                        box_colors.append((255, 255, 255))
                    else:
                        box_colors.append((0, 255, 0))
                frame = draw_boxes(frame, np.array(boxes), scores, cls_ids, names=names, det_ids=det_ids, colors=box_colors)
                
                # PossessionTrackerV2 sin ReID: usar detecciones del frame actual
                if possession is not None:
                    ball_pos = None
                    ball_owner_team = None
                    min_distance = float('inf')
                    
                    # Encontrar balón y jugador más cercano
                    for i, b in enumerate(boxes):
                        x1, y1, x2, y2 = b
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        cid = cls_ids[i]
                        
                        if cid == 1:  # balón
                            ball_pos = (cx, cy)
                            continue
                        
                        if cid in (0, 3) and ball_pos is not None:  # jugador/portero
                            import math
                            dist = math.hypot(cx - ball_pos[0], cy - ball_pos[1])
                            
                            if dist < 150 and dist < min_distance:
                                min_distance = dist
                                # Intentar obtener team del TeamClassifier
                                tmp_id = -(i+1)
                                if TEAMCLASS_AVAILABLE and team_classifier is not None:
                                    try:
                                        team_classifier.add_detection(tmp_id, (x1, y1, x2, y2), frame, class_id=cid)
                                        t = team_classifier.get_team(tmp_id)
                                        if t >= 0:
                                            ball_owner_team = int(t)
                                    except Exception:
                                        pass
                    
                    # Filtrar team_id inválidos (referees = -1)
                    # PossessionTrackerV2 solo acepta: 0, 1, o None
                    if ball_owner_team is not None and ball_owner_team < 0:
                        ball_owner_team = None
                    
                    # Actualizar posesión
                    possession.update(frame_no, ball_owner_team)
                    
                    # Visualizar
                    stats = possession.get_possession_stats()
                    current = possession.get_current_possession()
                    
                    current_team = current['team']
                    team_text = f"Team {current_team}" if current_team is not None else "N/A"
                    stext1 = f"Possession: {team_text}"
                    
                    p0 = stats['possession_percent'].get(0, 0.0)
                    p1 = stats['possession_percent'].get(1, 0.0)
                    stext2 = f"Total: Team0={p0:.1f}% Team1={p1:.1f}%"
                    
                    cv2.putText(frame, stext1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, stext2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,255,100), 2)

        if not args.no_show:
            cv2.imshow('YOLO Detection', frame)
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
    
    # Imprimir totales de posesión (PossessionTrackerV2)
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
        
        # Validación
        total_assigned = sum(stats['possession_frames'].values())
        print(f"\nValidation:")
        print(f"  Frames assigned: {total_assigned}/{stats['total_frames']}")
        print(f"  Coverage: {total_assigned/stats['total_frames']*100 if stats['total_frames'] > 0 else 0:.1f}%")
        
        # Timeline (solo primeros y últimos segmentos)
        timeline = possession.get_possession_timeline()
        if timeline:
            print(f"\nPossession timeline ({len(timeline)} segments):")
            for i, (start, end, team) in enumerate(timeline[:3]):
                duration = end - start
                print(f"  Segment {i+1}: Frames {start}-{end} ({duration}f) → Team {team}")
            if len(timeline) > 6:
                print(f"  ... ({len(timeline)-6} segments omitted) ...")
            for i, (start, end, team) in enumerate(timeline[-3:], len(timeline)-2):
                if i >= 3:  # Solo si hay más de 6
                    duration = end - start
                    print(f"  Segment {i}: Frames {start}-{end} ({duration}f) → Team {team}")
        
        print('='*70)
    
    print('Detección finalizada')


if __name__ == '__main__':
    main()
