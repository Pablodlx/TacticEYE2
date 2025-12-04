import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict

# ==========================================
# CONFIGURACIÓN
# ==========================================
VIDEO_PATH = "data/raw/sample_match.mp4" 
MODEL_PLAYERS = "runs/snv3_clean_1280_L_FINETUNE/weights/best.pt"

# Radar
PITCH_WIDTH = 105
PITCH_HEIGHT = 68
SCALE = 6
W_RADAR, H_RADAR = int(PITCH_WIDTH * SCALE), int(PITCH_HEIGHT * SCALE)

# --- CONFIGURACIÓN DE SUAVIZADO ---
# Cuántos frames usamos para la media (Más alto = Más suave pero con un poco de lag)
SMOOTH_WINDOW = 5 

# ==========================================
# CALIBRACIÓN MANUAL (TU SOLUCIÓN ESTRELLA)
# ==========================================
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"📍 Punto {len(clicked_points)}: ({x}, {y})")

def manual_calibration(cap):
    ret, frame = cap.read()
    if not ret: return None, None

    print("\n--- 🖱️ CALIBRACIÓN MANUAL ---")
    print("Haz clic en las 4 esquinas del ÁREA GRANDE en orden:")
    print("1. Sup-Izq  ->  2. Sup-Dcha  ->  3. Inf-Dcha  ->  4. Inf-Izq")
    print("(Pulsa SPACE para confirmar)")

    cv2.namedWindow("CALIBRACION", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CALIBRACION", 1024, 600)
    cv2.setMouseCallback("CALIBRACION", mouse_callback)

    while True:
        display = frame.copy()
        
        for i, pt in enumerate(clicked_points):
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
            cv2.putText(display, str(i+1), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        if len(clicked_points) > 1:
            for i in range(len(clicked_points)-1):
                cv2.line(display, clicked_points[i], clicked_points[i+1], (0, 255, 0), 2)
        
        if len(clicked_points) == 4:
            cv2.line(display, clicked_points[3], clicked_points[0], (0, 255, 0), 2)
            cv2.putText(display, "PULSA SPACE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("CALIBRACION", display)
        key = cv2.waitKey(1)
        if key == 32 and len(clicked_points) == 4: break
        if key == ord('r'): clicked_points.clear()
        if key == ord('q'): exit()

    cv2.destroyWindow("CALIBRACION")
    
    # Coordenadas Reales Área Grande (Portería Derecha)
    real_pts = np.array([
        [88.5, 13.84], [105.0, 13.84], 
        [105.0, 54.16], [88.5, 54.16]
    ], dtype=np.float32)

    src = np.array(clicked_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = real_pts.reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst)
    return H

# ==========================================
# VISUALIZACIÓN + SUAVIZADO
# ==========================================
def draw_radar_bg():
    img = np.zeros((H_RADAR + 100, W_RADAR + 100, 3), dtype=np.uint8)
    img[:] = (34, 139, 34) 
    ox, oy = 50, 50
    lc = (255, 255, 255)
    
    cv2.rectangle(img, (ox, oy), (ox+W_RADAR, oy+H_RADAR), lc, 2)
    mid_x = int(52.5*SCALE)+ox
    cv2.line(img, (mid_x, oy), (mid_x, oy+H_RADAR), lc, 2)
    cv2.circle(img, (mid_x, int(34.0*SCALE)+oy), int(9.15*SCALE), lc, 2)
    
    aw, ah = int(16.5*SCALE), int(40.32*SCALE)
    ay = int((68-40.32)/2*SCALE)+oy
    cv2.rectangle(img, (ox+W_RADAR-aw, ay), (ox+W_RADAR, ay+ah), lc, 1)
    return img, (ox, oy)

def run_app():
    print("🚀 TacticEYE: Iniciando motor con suavizado...")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    H = manual_calibration(cap) # <-- Tu calibración manual
    
    if H is None: return

    print("✅ Matriz cargada. Trackeando jugadores...")
    model = YOLO(MODEL_PLAYERS)
    
    bg, (ox, oy) = draw_radar_bg()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.namedWindow("Radar", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision", 1024, 600)

    # --- MEMORIA DE JUGADORES (Para el suavizado) ---
    # Diccionario: ID -> Lista de las últimas posiciones
    player_history = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))

    while True:
        ok, frame = cap.read()
        if not ok: break

        # Tracking (BoT-SORT ya viene activado)
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
        radar = bg.copy()

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            tids = results.boxes.id.cpu().numpy()

            for box, cls, tid in zip(boxes, clss, tids):
                feet = np.array([[[(box[0]+box[2])/2, box[3]]]], dtype=np.float32)
                try:
                    # 1. Proyección Cruda
                    real = cv2.perspectiveTransform(feet, H)[0][0]
                    rx, ry = real[0], real[1]

                    # 2. Filtrado de Límites
                    if -5 < rx < 110 and -5 < ry < 75:
                        
                        # 3. SUAVIZADO (AÑADIR A HISTORIAL)
                        player_history[tid].append((rx, ry))
                        
                        # CALCULAR MEDIA DE POSICIONES
                        hist = np.array(player_history[tid])
                        avg_x = np.mean(hist[:, 0])
                        avg_y = np.mean(hist[:, 1])

                        # Coordenadas finales (Suaves)
                        cx, cy = int(avg_x*SCALE)+ox, int(avg_y*SCALE)+oy
                        
                        # Colores
                        cid = int(cls)
                        col = (200,200,200)
                        if cid == 0: col = (0,165,255) 
                        elif cid == 1: col = (0,0,255)
                        elif cid == 2: col = (0,255,255)
                        elif cid == 3: col = (255,0,0)
                        
                        # Dibujar Jugador
                        cv2.circle(radar, (cx, cy), 6, col, -1)
                        # Dibujar ID
                        if cid != 0:
                            cv2.putText(radar, str(int(tid)), (cx+5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                            
                        # OPTIONAL: Dibujar pequeña estela (últimos frames)
                        if len(hist) > 2:
                            prev_pt = (int(hist[0][0]*SCALE)+ox, int(hist[0][1]*SCALE)+oy)
                            cv2.line(radar, prev_pt, (cx, cy), (col[0], col[1], col[2], 100), 1)

                except: pass

        cv2.imshow("Vision", results.plot())
        cv2.imshow("Radar", radar)

        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_app()
