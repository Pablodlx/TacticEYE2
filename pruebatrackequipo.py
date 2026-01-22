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
            
            # Dibujar tracks
            for tid, bbox, cid in tracks:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Obtener team_id si está disponible
                team_id = -1
                try:
                    team_id = tracker.active_tracks[tid].team_id
                except Exception:
                    pass
                
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
                
                label = f"ID:{tid} {names.get(cid, cid) if names else cid} T{team_label}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Detección de posesión
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
                        if cid not in (0, 3):  # Solo jugadores y porteros
                            continue
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        player_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        dist = math.hypot(player_pos[0] - ball_pos[0], player_pos[1] - ball_pos[1])
                        
                        if dist < min_distance:
                            min_distance = dist
                            ball_owner_id = tid
                            try:
                                ball_owner_team = tracker.active_tracks[tid].team_id
                            except Exception:
                                ball_owner_team = None
                
                # PASO 3: Validar distancia
                max_possession_distance = args.possession_distance
                if min_distance > max_possession_distance:
                    ball_owner_team = None
                    ball_owner_id = None
                
                # PASO 4: Filtrar team_id inválidos (referees = -1)
                if ball_owner_team is not None and ball_owner_team < 0:
                    ball_owner_team = None
                
                # Actualizar PossessionTrackerV2
                possession.update(frame_no, ball_owner_team)
                
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
                
                stext4 = f"Max dist: {max_possession_distance}px"
                
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
                            
                            # Distancia
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
    
    print('Detección finalizada')


if __name__ == '__main__':
    main()
