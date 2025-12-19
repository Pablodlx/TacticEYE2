import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict

from modules.field_calibration import FieldCalibration

# ==========================================
# CONFIGURACIÓN
# ==========================================
VIDEO_PATH = "sample_match.mp4" 
MODEL_PLAYERS = "weights/best.pt"

# Radar
PITCH_WIDTH = 105
PITCH_HEIGHT = 68
SCALE = 6
W_RADAR, H_RADAR = int(PITCH_WIDTH * SCALE), int(PITCH_HEIGHT * SCALE)

# --- CONFIGURACIÓN DE SUAVIZADO ---
# Cuántos frames usamos para la media (Más alto = Más suave pero con un poco de lag)
SMOOTH_WINDOW = 5 

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
import time

from modules.field_calibration import FieldCalibration

# ==========================================
# CONFIGURACIÓN
# ==========================================
VIDEO_PATH = "sample_match.mp4" 
MODEL_PLAYERS = "weights/best.pt"

# Radar
PITCH_WIDTH = 105
PITCH_HEIGHT = 68
SCALE = 6
W_RADAR, H_RADAR = int(PITCH_WIDTH * SCALE), int(PITCH_HEIGHT * SCALE)

# --- CONFIGURACIÓN DE SUAVIZADO ---
SMOOTH_WINDOW = 5 

# ==========================================
# VISUALIZACIÓN + SUAVIZADO
# ==========================================
def draw_radar_bg():
    img = np.zeros((H_RADAR + 100, W_RADAR + 100, 3), dtype=np.uint8)
    img[:] = (34, 139, 34) 
    ox, oy = 50, 50
    lc = (255, 255, 255)
    gc = (100, 200, 100) # Green Grid Color
    
    # Grid Táctico 6x3
    # Verticales
    for i in range(1, 6):
        x_m = i * (105.0 / 6)
        x_px = int(x_m * SCALE) + ox
        cv2.line(img, (x_px, oy), (x_px, oy + H_RADAR), gc, 1)
        
    # Horizontales
    for i in range(1, 3):
        y_m = i * (68.0 / 3)
        y_px = int(y_m * SCALE) + oy
        cv2.line(img, (ox, y_px), (ox + W_RADAR, y_px), gc, 1)
        
    # Marcar Zona 14 (Col 4, Row 1 - 0-indexed -> Col 3, Row 1?) 
    # Zona 14 es "The Hole": Central, Attacking Third.
    # En Grid 6x3: Row 1 (Centro), Col 4 (Ataque, 0-5).
    # ID = 1*6 + 4 + 1 = 11? Depende de la numeración.
    # En nuestra func: Row 1 (Centro vertical), Col 3 o 4.
    z14_x = int(((105/6)*4 + (105/6)/2) * SCALE) + ox
    z14_y = int(((68/3)*1 + (68/3)/2) * SCALE) + oy
    cv2.putText(img, "Z14", (z14_x-15, z14_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.rectangle(img, (ox, oy), (ox+W_RADAR, oy+H_RADAR), lc, 2)
    mid_x = int(52.5*SCALE)+ox
    cv2.line(img, (mid_x, oy), (mid_x, oy+H_RADAR), lc, 2)
    cv2.circle(img, (mid_x, int(34.0*SCALE)+oy), int(9.15*SCALE), lc, 2)
    
    aw, ah = int(16.5*SCALE), int(40.32*SCALE)
    ay = int((68-40.32)/2*SCALE)+oy
    cv2.rectangle(img, (ox+W_RADAR-aw, ay), (ox+W_RADAR, ay+ah), lc, 1)
    return img, (ox, oy)

def run_app():
    print("🚀 TacticEYE: Iniciando modo inteligente continuo...")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ No se puede abrir {VIDEO_PATH}")
        return

    # Inicializar Modelos
    model = YOLO(MODEL_PLAYERS)
    calib = FieldCalibration()
    
    bg, (ox, oy) = draw_radar_bg()
    
    cv2.namedWindow("Radar", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision", 1024, 600)

    # Memoria
    player_history = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
    frame_count = 0
    side_hint = 0 # 0=Auto, -1=Left, 1=Right
    tilt_hint = 0 # 0=Auto, -1=High Cam, 1=Low Cam
    
    ball_zone_text = ""
    
    print("🎥 Analizando video y buscando campo automáticamente...")

    while True:
        ok, frame = cap.read()
        if not ok: break
        
        frame_count += 1
        
        # --- 1. TRACKING JUGADORES (YOLO) ---
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
        
        # --- 2. CALIBRACIÓN INDUSTRIAL (Dictionary Match) ---
        # Intentar detectar en CADA frame (es rápido al ser baja resolución)
        if frame_count % 3 == 0:
            calib.compute_homography(frame, frame_number=frame_count, side_hint=side_hint, tilt_hint=tilt_hint)

        # --- 3. VISUALIZACIÓN ---
        # Si tenemos calibración, dibujar overlay en la visión principal
        if calib.is_calibrated():
            # Dibujar líneas amarillas del campo "imaginado" sobre el real
            frame = calib.draw_projected_pitch(frame, color=(0, 255, 255), thickness=2)
            
            s_str = "AUTO"
            if side_hint == -1: s_str = "LEFT"
            if side_hint == 1: s_str = "RIGHT"
            
            t_str = "AUTO"
            if tilt_hint == -1: t_str = "HIGH CAM (Flat)"
            if tilt_hint == 1: t_str = "LOW CAM (Deep)"
            
            cv2.putText(frame, f"VISTA: {calib.best_view_name}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"SIDE: {s_str} | TILT: {t_str}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            
            if ball_zone_text:
                cv2.putText(frame, ball_zone_text, (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        else:
            cv2.putText(frame, "STATUS: BUSCANDO EN DICCIONARIO...", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # DEBUG: Mostrar lo que ve la "IA" (Máscara de líneas)
        if calib.last_mask is not None:
            cv2.imshow("Segmentation (Debug)", calib.last_mask)

        radar = bg.copy()
        ball_zone_text = "" # Reset
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            tids = results.boxes.id.cpu().numpy()

            for box, cls, tid in zip(boxes, clss, tids):
                # Punto de contacto (pies)
                feet_px = [(box[0]+box[2])/2, box[3]]
                
                # Intentar proyectar si está calibrado
                if calib.is_calibrated():
                    # Usar la función de la clase (maneja internamente la matriz)
                    real = calib.pixel_to_meters(np.array(feet_px))
                    
                    if real is not None:
                        rx, ry = real[0], real[1]
                        
                        # Filtrar fantasmas fuera del campo (con margen de 5m)
                        if -5 < rx < 110 and -5 < ry < 75:
                            
                            # Detectar Zona del Balón
                            if int(cls) == 1: # Balón
                                zid, zname = calib.get_zone_info(real)
                                ball_zone_text = f"BALL: {zname}"
                            
                            # Suavizado
                            player_history[tid].append((rx, ry))
                            hist = np.array(player_history[tid])
                            avg_x, avg_y = np.mean(hist, axis=0)

                            # Mapear al Radar
                            cx, cy = int(avg_x*SCALE)+ox, int(avg_y*SCALE)+oy
                            
                            # Colores
                            cid = int(cls)
                            col = (200,200,200)
                            if cid == 0: col = (0,165,255)  # Jugador
                            elif cid == 1: col = (0,0,255)  # Balón
                            elif cid == 2: col = (0,255,255) # Arbitro
                            elif cid == 3: col = (255,0,0)  # Portero
                            
                            cv2.circle(radar, (cx, cy), 6, col, -1)
                            if cid != 0:
                                cv2.putText(radar, str(int(tid)), (cx+5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        cv2.imshow("Vision", frame) # Frame ya tiene el overlay pintado
        cv2.imshow("Radar", radar)

        key = cv2.waitKey(1)
        if key == ord('q'): break
        
        # CONTROLES MANUALES
        # Lado (Side)
        if key == 81 or key == ord('a'): # Izquierda
            side_hint = -1
        if key == 83 or key == ord('d'): # Derecha
            side_hint = 1
            
        # Altura (Tilt)
        if key == 82 or key == ord('w'): # Arriba = Cámara Alta (Menos persp)
            tilt_hint = -1
        if key == 84 or key == ord('s'): # Abajo = Cámara Baja (Más persp)
            tilt_hint = 1
            
        # Reset
        if key == 32 or key == ord('c'): # Space/C -> Auto
            side_hint = 0
            tilt_hint = 0
            print("🔄 Reset a AUTO")
            
        if key == ord('r'): # Reset calibración interna
            calib.homography_matrix = None
            calib.homography_candidates.clear()
            print("🔄 Reiniciando calibración...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_app()



if __name__ == "__main__":
    run_app()
