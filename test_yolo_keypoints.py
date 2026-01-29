#!/usr/bin/env python3
"""
Test rápido del sistema de keypoints YOLO custom
"""

import cv2
import sys
from modules.field_keypoints_yolo import FieldKeypointsYOLO
from modules.field_model_keypoints import FieldModel
from modules.field_calibrator_keypoints import FieldCalibratorKeypoints


def test_keypoints_system(video_path: str, max_frames: int = 300):
    """
    Prueba el sistema completo de detección y calibración con keypoints YOLO.
    """
    print("="*70)
    print("TEST: Sistema de Keypoints YOLO Custom")
    print("="*70)
    
    # 1. Inicializar detector
    print("\n1. Inicializando detector YOLO...")
    detector = FieldKeypointsYOLO(
        model_path="weights/field_kp_merged_fast/weights/best.pt",
        confidence_threshold=0.25
    )
    
    # 2. Inicializar modelo de campo
    print("2. Inicializando modelo de campo...")
    field_model = FieldModel(
        field_length=105.0,
        field_width=68.0,
        use_normalized=False
    )
    
    # 3. Inicializar calibrador
    print("3. Inicializando calibrador...")
    calibrator = FieldCalibratorKeypoints(
        field_model=field_model,
        min_keypoints=4,
        ransac_threshold=5.0
    )
    
    # 4. Procesar video
    print(f"\n4. Procesando video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"✗ Error abriendo video: {video_path}")
        return False
    
    frame_count = 0
    calibrations_successful = 0
    total_keypoints_detected = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar cada 30 frames
        if frame_count % 30 == 0:
            keypoints = detector.detect_keypoints(frame)
            total_keypoints_detected += len(keypoints)
            
            # Intentar calibrar
            success = calibrator.estimate_homography(keypoints)
            
            if success:
                calibrations_successful += 1
                info = calibrator.get_calibration_info()
                print(f"  Frame {frame_count}: ✓ Calibrado con {info['num_keypoints']} keypoints "
                      f"(error={info['reprojection_error']:.2f}px)")
            else:
                num_accumulated = len(calibrator.accumulated_keypoints)
                print(f"  Frame {frame_count}: {len(keypoints)} detectados, "
                      f"{num_accumulated} acumulados (necesita {calibrator.min_keypoints})")
        
        frame_count += 1
    
    cap.release()
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(f"Frames procesados: {frame_count}")
    print(f"Keypoints detectados total: {total_keypoints_detected}")
    print(f"Calibraciones exitosas: {calibrations_successful}")
    print(f"Keypoints acumulados: {len(calibrator.accumulated_keypoints)}")
    
    if calibrator.H is not None:
        print(f"\n✓ Sistema FUNCIONANDO - Homografía estimada correctamente")
        print(f"  Keypoints en calibración final: {len(calibrator.matched_keypoints)}")
        print(f"  Error de reproyección: {calibrator.reprojection_error:.2f}px")
        return True
    else:
        print(f"\n✗ Sistema NO calibrado - Necesita más keypoints")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_yolo_keypoints.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    success = test_keypoints_system(video_path, max_frames=300)
    
    sys.exit(0 if success else 1)
