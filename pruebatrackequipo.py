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
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--output', default=None, help='Guardar video con detecciones')
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
    print('Iniciando detección YOLO...')

    # Inicializar ReID tracker opcional
    tracker = None
    if args.reid:
        if not REID_AVAILABLE:
            print('Warning: ReID module no disponible, ejecuto sin tracker')
        else:
            tracker = ReIDTracker()

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
                # obtener team_id desde el track si está disponible
                team_id = None
                try:
                    team_id = tracker.active_tracks[tid].team_id
                except Exception:
                    team_id = None

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
    print('Detección finalizada')


if __name__ == '__main__':
    main()
