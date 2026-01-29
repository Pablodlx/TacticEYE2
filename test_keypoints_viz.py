#!/usr/bin/env python3
"""
Script simple para probar visualización de keypoints con bounding boxes
"""
import cv2
import numpy as np
from modules.field_keypoints_yolo import FieldKeypointsYOLO

# Prioridades de keypoints (copiado de pruebatrackequipo.py)
KEYPOINT_PRIORITY = {
    'midline_top_intersection': 100,
    'midline_bottom_intersection': 100,
    'halfcircle_top': 95,
    'halfcircle_bottom': 95,
    'left_penalty_arc_intersection_top': 90,
    'left_penalty_arc_intersection_bottom': 90,
    'right_penalty_arc_intersection_top': 90,
    'right_penalty_arc_intersection_bottom': 90,
    'left_bigbox_top_inner': 80,
    'left_bigbox_bottom_inner': 80,
    'right_bigbox_top_inner': 80,
    'right_bigbox_bottom_inner': 80,
    'left_bigbox_top_outer': 75,
    'left_bigbox_bottom_outer': 75,
    'right_bigbox_top_outer': 75,
    'right_bigbox_bottom_outer': 75,
    'left_smallbox_top_inner': 70,
    'left_smallbox_bottom_inner': 70,
    'right_smallbox_top_inner': 70,
    'right_smallbox_bottom_inner': 70,
    'left_smallbox_top_outer': 65,
    'left_smallbox_bottom_outer': 65,
    'right_smallbox_top_outer': 65,
    'right_smallbox_bottom_outer': 65,
    'corner_top_left': 40,
    'corner_top_right': 40,
    'corner_bottom_left': 40,
    'corner_bottom_right': 40,
    'corner_left_top': 40,
    'corner_right_top': 40,
}

def main():
    # Inicializar detector de keypoints
    print("Cargando detector de keypoints...")
    detector = FieldKeypointsYOLO(
        model_path='weights/field_kp_merged_fast/weights/best.pt',
        confidence_threshold=0.25,
        device='cuda'
    )
    
    # Abrir video
    video_path = 'prueba3.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {w}x{h} @ {fps:.1f} FPS")
    
    # Configurar output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('test_keypoints_boxes.mp4', fourcc, fps, (w, h))
    
    frame_count = 0
    max_frames = 300  # Procesar solo 300 frames (10 segundos)
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detectar keypoints cada 3 frames
        if frame_count % 3 == 0:
            keypoints = detector.detect_keypoints(frame)
            
            if keypoints:
                print(f"Frame {frame_count}: {len(keypoints)} keypoints detectados")
                
                # Visualizar keypoints con bounding boxes
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
        
        # Agregar contador de frames
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        writer.write(frame)
        
        if frame_count % 30 == 0:
            print(f"Procesados {frame_count} frames...")
    
    cap.release()
    writer.release()
    
    print(f"\n✓ Video guardado: test_keypoints_boxes.mp4")
    print(f"  Total frames procesados: {frame_count}")

if __name__ == '__main__':
    main()
