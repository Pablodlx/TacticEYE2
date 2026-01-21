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
    from modules.possession_tracker import PossessionTracker
    POSSESSION_AVAILABLE = True
except Exception:
    POSSESSION_AVAILABLE = False
 
try:
    from modules.team_classifier_v2 import TeamClassifierV2
    TEAMCLASS_AVAILABLE = True
except Exception:
    TEAMCLASS_AVAILABLE = False

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
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--output', default=None, help='Guardar video con detecciones')
    p.add_argument('--dump-csv', default=None, help='Guardar CSV de asignaciones por frame')
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

    # Inicializar PossessionTracker (funciona independiente del ReID)
    possession = None
    if POSSESSION_AVAILABLE:
        # usar valores por defecto; podemos exponerlos por CLI más adelante
        possession = PossessionTracker()
    else:
        print('Warning: modules.possession_tracker no disponible; sin posesion')
    # Inicializar TeamClassifier V2 Pro si está disponible
    team_classifier = None
    if TEAMCLASS_AVAILABLE:
        try:
            # Parámetros V2 Pro con anti-green masking y features LAB
            tc_kwargs = dict(
                # Green removal
                green_h_low=args.tc_green_h_low,
                green_h_high=args.tc_green_h_high,
                green_s_min=args.tc_green_s_min,
                green_v_min=args.tc_green_v_min,
                min_non_green_ratio=args.tc_min_non_green_ratio,
                # LAB features
                use_L_channel=args.tc_use_L,
                L_weight=args.tc_L_weight,
                # Track sampling
                min_track_samples=args.tc_min_track_samples,
                min_bbox_area_frac=args.tc_min_bbox_area_frac,
                # KMeans
                kmeans_min_tracks=args.tc_kmeans_min_tracks,
                kmeans_min_samples_per_track=args.tc_kmeans_min_samples,
                # Voting
                vote_history=args.tc_vote_history,
                # Referee detection
                referee_detection=True,
                # Debug
                save_debug_rois=args.tc_save_rois,
                debug_rois_dir=args.tc_rois_dir
            )
            team_classifier = TeamClassifierV2(**tc_kwargs)
            print('[TeamClassifierV2] Inicializado en modo LAB+AntiGreen')
            print(f'  Green removal: H=[{args.tc_green_h_low},{args.tc_green_h_high}] S>={args.tc_green_s_min} V>={args.tc_green_v_min}')
            print(f'  Min non-green ratio: {args.tc_min_non_green_ratio}')
            print(f'  KMeans min tracks: {args.tc_kmeans_min_tracks}, min samples/track: {args.tc_kmeans_min_samples}')
            feat_desc = f"LAB (a*,b* + L*w={args.tc_L_weight})" if args.tc_use_L else "LAB (a*,b*)"
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
            # Preparar datos para PossessionTracker: ball_pos y players
            if possession is not None:
                # buscar la posición del balón en los tracks (class id 1)
                ball_pos = None
                players_input = []
                for tid, bbox, cid in tracks:
                    x1, y1, x2, y2 = map(int, bbox)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    if cid == 1:
                        ball_pos = (cx, cy)
                        continue
                    # considerar players y goalkeepers como jugadores
                    if cid in (0, 3):
                        # intentar obtener team_id desde tracker.active_tracks
                        team_id = None
                        try:
                            team_id = tracker.active_tracks[tid].team_id
                        except Exception:
                            team_id = None
                        players_input.append({'track_id': tid, 'pos': (cx, cy), 'team_id': team_id})

                possession.update(ball_pos, players_input)
                pstate = possession.get_current_possession()
                # Dibujar estado de posesión en la esquina superior izquierda
                pp = pstate.get('possession_percent', {})
                player_p = pstate.get('player_percent', None)
                # Show percentages relative to total frames (so total frames == 100%)
                totals = pstate.get('total_assigned_percent_of_total_frames', pstate.get('total_assigned_percent', {}))
                stext = (f"Possession: {pstate['state']} team:{pstate['team']} "
                         f"P0:{pp.get('team_0',0):.2f} P1:{pp.get('team_1',0):.2f} "
                         f"cont:{pp.get('contested',0):.2f} none:{pp.get('none',0):.2f} ")
                if player_p is not None:
                    stext += f" player_pct:{player_p:.2f}"
                # Totales acumulados
                stext2 = f"Total P0:{totals.get(0,0):.2f} Total P1:{totals.get(1,0):.2f}"
                cv2.putText(frame, stext, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 2)
                cv2.putText(frame, stext2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)
                # Assigned team per-frame (interpolated)
                assigned = pstate.get('assigned_team', None)
                if assigned is not None:
                    atxt = f"Assigned team: {assigned}"
                else:
                    atxt = "Assigned team: -"
                cv2.putText(frame, atxt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)
                # Si hay player con posesión, remarcar su caja
                if pstate['player'] is not None:
                    try:
                        tid_pos = int(pstate['player'])
                        # buscar bbox en tracks
                        for tid, bbox, cid in tracks:
                            if tid == tid_pos:
                                x1, y1, x2, y2 = map(int, bbox)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                                cv2.putText(frame, 'POS', (x1, y1-22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
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
                # Cuando no hay ReID, podemos igualmente alimentar PossessionTracker con detecciones
                if possession is not None:
                    # hallar ball pos en detecciones
                    ball_pos = None
                    players_input = []
                    for i, b in enumerate(boxes):
                        x1, y1, x2, y2 = b
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        cid = cls_ids[i]
                        if cid == 1:
                            ball_pos = (cx, cy)
                        if cid in (0, 3):
                            # assign temporary negative ids to detections to track within frame
                            tmp_id = -(i+1)
                            team_id = None
                            if TEAMCLASS_AVAILABLE and team_classifier is not None:
                                try:
                                    team_classifier.add_detection(tmp_id, (x1, y1, x2, y2), frame, class_id=cid)
                                    t = team_classifier.get_team(tmp_id)
                                    if t >= 0:
                                        team_id = int(t)
                                except Exception:
                                    team_id = None
                            players_input.append({'track_id': tmp_id, 'pos': (cx, cy), 'team_id': team_id})
                    possession.update(ball_pos, players_input)
                    pstate = possession.get_current_possession()
                    pp = pstate.get('possession_percent', {})
                    player_p = pstate.get('player_percent', None)
                    totals = pstate.get('total_assigned_percent_of_total_frames', pstate.get('total_assigned_percent', pstate.get('total_possession_percent', {})))
                    stext = (f"Possession: {pstate['state']} team:{pstate['team']} "
                             f"P0:{pp.get('team_0',0):.2f} P1:{pp.get('team_1',0):.2f} "
                             f"cont:{pp.get('contested',0):.2f} none:{pp.get('none',0):.2f} ")
                    if player_p is not None:
                        stext += f" player_pct:{player_p:.2f}"
                    stext2 = f"Total P0:{totals.get(0,0):.2f} Total P1:{totals.get(1,0):.2f}"
                    cv2.putText(frame, stext, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 2)
                    cv2.putText(frame, stext2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)

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
    # Imprimir totales de posesión si el tracker está activo
    if possession is not None:
        final = possession.get_current_possession()
        totals_frames = final.get('total_assigned_frames', final.get('total_possession_frames', {}))
        pct_of_total = final.get('total_assigned_percent_of_total_frames', None)
        pct_of_assigned = final.get('total_assigned_percent', None)
        # Compute sums
        total_frames_processed = final.get('frame_index', 0)
        sum_assigned = (totals_frames.get(0, 0) + totals_frames.get(1, 0))

        print('--- Possession summary ---')
        print(f"Frames processed: {total_frames_processed}")
        print(f"Team 0 frames assigned: {totals_frames.get(0,0)}")
        print(f"Team 1 frames assigned: {totals_frames.get(1,0)}")
        print(f"Sum assigned frames: {sum_assigned}")

        # Percentage relative to total frames processed
        if pct_of_total is not None:
            print(f"Team 0 percent of TOTAL frames: {pct_of_total.get(0,0)*100:.2f}%")
            print(f"Team 1 percent of TOTAL frames: {pct_of_total.get(1,0)*100:.2f}%")
        else:
            if total_frames_processed > 0:
                print(f"Team 0 percent of TOTAL frames: {totals_frames.get(0,0)/total_frames_processed*100:.2f}%")
                print(f"Team 1 percent of TOTAL frames: {totals_frames.get(1,0)/total_frames_processed*100:.2f}%")

        # Percentage relative to assigned frames (sums to 100% across teams if sum_assigned>0)
        if sum_assigned > 0:
            if pct_of_assigned is not None:
                print(f"Team 0 percent of ASSIGNED frames: {pct_of_assigned.get(0,0)*100:.2f}%")
                print(f"Team 1 percent of ASSIGNED frames: {pct_of_assigned.get(1,0)*100:.2f}%")
            else:
                print(f"Team 0 percent of ASSIGNED frames: {totals_frames.get(0,0)/sum_assigned*100:.2f}%")
                print(f"Team 1 percent of ASSIGNED frames: {totals_frames.get(1,0)/sum_assigned*100:.2f}%")
    print('Detección finalizada')


if __name__ == '__main__':
    main()
