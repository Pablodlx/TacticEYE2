"""
Ejemplo de Uso - Pipeline de Calibración Clásica
=================================================

Este script demuestra cómo usar el pipeline completo de calibración clásica
con detección de jugadores YOLO y análisis de zonas tácticas.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

# Importar módulos del sistema
from modules.classic_field_calibration import ClassicFieldCalibration
from modules.field_zones import ZoneType

# YOLO (si está disponible)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO no disponible. Usando detecciones simuladas.")


def draw_player_with_zone(
    frame: np.ndarray,
    bbox: np.ndarray,
    zone_info: Dict,
    color: tuple = (0, 255, 0)
) -> np.ndarray:
    """
    Dibuja un jugador con información de zona.
    
    Args:
        frame: Frame BGR
        bbox: [x1, y1, x2, y2]
        zone_info: Información de zona del jugador
        color: Color del bbox
        
    Returns:
        Frame con overlay
    """
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Dibujar bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Información de zona
    zone_id = zone_info.get('zone_id', 0)
    zone_name = zone_info.get('zone_name', 'Unknown')
    zone_type = zone_info.get('zone_type', 'unknown')
    
    # Etiqueta
    label = f"Zone {zone_id}: {zone_name}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    # Fondo para texto
    cv2.rectangle(frame,
                 (x1, y1 - label_size[1] - 5),
                 (x1 + label_size[0], y1),
                 color, -1)
    
    cv2.putText(frame, label,
               (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def process_video_with_classic_calibration(
    video_path: str,
    yolo_model_path: str = None,
    output_path: str = None,
    show_debug: bool = True
):
    """
    Procesa un video completo con calibración clásica y análisis de zonas.
    
    Args:
        video_path: Ruta al video
        yolo_model_path: Ruta al modelo YOLO (opcional)
        output_path: Ruta para guardar video de salida (opcional)
        show_debug: Mostrar ventanas de debug
    """
    # 1. Inicializar componentes
    print("🚀 Inicializando pipeline de calibración clásica...")
    
    calibration = ClassicFieldCalibration(
        temporal_window=30,
        calibration_interval=10,
        grid_cols=6,
        grid_rows=3,
        debug=True
    )
    
    # YOLO (opcional)
    yolo_model = None
    if YOLO_AVAILABLE and yolo_model_path:
        print(f"📦 Cargando modelo YOLO: {yolo_model_path}")
        yolo_model = YOLO(yolo_model_path)
    elif YOLO_AVAILABLE:
        print("⚠️  No se proporcionó modelo YOLO. Usando detecciones simuladas.")
    
    # 2. Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}x{height} @ {fps}fps")
    
    # Writer de salida (opcional)
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Guardando salida en: {output_path}")
    
    # Estadísticas
    frame_count = 0
    players_zones = {}  # {frame: {player_id: zone_info}}
    
    print("\n▶️  Procesando video...")
    print("   Presiona 'q' para salir, 's' para guardar frame actual\n")
    
    # 3. Procesar frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 3.1. Calibración clásica
        calibration_updated = calibration.process_frame(frame)
        
        if calibration_updated:
            print(f"✅ Frame {frame_count}: Calibración actualizada "
                  f"(confianza: {calibration.calibration_confidence:.2f})")
        
        # 3.2. Detección de jugadores (YOLO o simulado)
        detections = []
        
        if yolo_model:
            results = yolo_model.predict(frame, conf=0.3, verbose=False)[0]
            for box in results.boxes:
                if box.cls == 0:  # Solo jugadores (clase 0)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf.cpu().numpy()[0]
                    detections.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'confidence': conf,
                        'class': int(box.cls.cpu().numpy()[0])
                    })
        else:
            # Detecciones simuladas (para testing sin YOLO)
            if frame_count % 30 == 0:  # Cada 30 frames
                h, w = frame.shape[:2]
                detections.append({
                    'bbox': np.array([w*0.3, h*0.5, w*0.35, h*0.6]),
                    'confidence': 0.8,
                    'class': 0
                })
                detections.append({
                    'bbox': np.array([w*0.6, h*0.4, w*0.65, h*0.5]),
                    'confidence': 0.75,
                    'class': 0
                })
        
        # 3.3. Proyección a zonas (si está calibrado)
        frame_zones = {}
        if calibration.is_calibrated:
            for i, det in enumerate(detections):
                bbox = det['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                player_pos = np.array([center_x, center_y])
                
                zone_info = calibration.get_player_zone(player_pos)
                if zone_info:
                    zone, info = zone_info
                    frame_zones[i] = info
                    
                    # Dibujar jugador con zona
                    color = (0, 255, 0) if det['class'] == 0 else (0, 255, 255)
                    frame = draw_player_with_zone(frame, bbox, info, color)
        
        players_zones[frame_count] = frame_zones
        
        # 3.4. Dibujar campo proyectado
        if calibration.is_calibrated:
            frame = calibration.draw_projected_pitch(frame, color=(0, 255, 255), thickness=2)
        
        # 3.5. Información de estado
        status_y = 30
        cv2.putText(frame, f"Frame: {frame_count}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if calibration.is_calibrated:
            cv2.putText(frame, f"Calibrated (conf: {calibration.calibration_confidence:.2f})",
                       (10, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Lines detected: {len(calibration.last_lines)}",
                       (10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Players in zones: {len(frame_zones)}",
                       (10, status_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Calibrating...",
                       (10, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Frames accumulated: {len(calibration.line_detector.mask_buffer)}",
                       (10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 3.6. Guardar frame
        if writer:
            writer.write(frame)
        
        # 3.7. Mostrar
        if show_debug:
            cv2.imshow('Calibration - Main', frame)
            
            # Ventana de debug
            debug_frame = calibration.get_debug_visualization(frame)
            cv2.imshow('Calibration - Debug', debug_frame)
            
            # Ventana de zonas (si está calibrado)
            if calibration.is_calibrated:
                zones_vis = calibration.zone_manager.visualize_zones(
                    frame.shape[:2],
                    calibration.homography_matrix
                )
                cv2.imshow('Calibration - Zones', zones_vis)
        
        # 3.8. Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⏹️  Detenido por usuario")
            break
        elif key == ord('s'):
            cv2.imwrite(f'frame_{frame_count}.jpg', frame)
            print(f"💾 Frame {frame_count} guardado")
        
        # Progress
        if frame_count % 100 == 0:
            print(f"   Procesados {frame_count} frames...")
    
    # 4. Estadísticas finales
    print("\n📊 Estadísticas:")
    print(f"   Total frames procesados: {frame_count}")
    print(f"   Frames calibrados: {sum(1 for f in range(1, frame_count+1) if calibration.is_calibrated)}")
    
    if calibration.is_calibrated:
        stats = calibration.zone_manager.get_zone_statistics()
        print(f"   Zonas configuradas: {stats['total_zones']} ({stats['grid_size']})")
    
    # 5. Limpiar
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Procesamiento completado")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ejemplo de calibración clásica de campo de fútbol'
    )
    parser.add_argument('video', type=str, help='Ruta al video de entrada')
    parser.add_argument('--model', type=str, default=None,
                       help='Ruta al modelo YOLO (opcional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Ruta para guardar video de salida (opcional)')
    parser.add_argument('--no-debug', action='store_true',
                       help='No mostrar ventanas de debug')
    
    args = parser.parse_args()
    
    # Verificar que el video existe
    if not Path(args.video).exists():
        print(f"❌ Error: El video {args.video} no existe")
        return
    
    # Procesar
    process_video_with_classic_calibration(
        video_path=args.video,
        yolo_model_path=args.model,
        output_path=args.output,
        show_debug=not args.no_debug
    )


if __name__ == '__main__':
    main()

