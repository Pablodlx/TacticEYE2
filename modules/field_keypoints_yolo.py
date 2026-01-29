"""
Field Keypoints Detector - YOLO Custom Model
==============================================

Detecta keypoints del campo de fútbol usando modelo YOLO entrenado custom.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from ultralytics import YOLO


class FieldKeypointsYOLO:
    """
    Detector de keypoints del campo usando modelo YOLO custom.
    
    El modelo detecta keypoints específicos del campo de fútbol que luego
    se usan para calibración y homografía.
    """
    
    def __init__(self, 
                 model_path: str = "weights/field_kp_merged_fast/weights/best.pt",
                 confidence_threshold: float = 0.25,
                 device: str = "cuda"):
        """
        Args:
            model_path: Ruta al modelo YOLO entrenado (.pt)
            confidence_threshold: Umbral de confianza para detecciones
            device: 'cuda' o 'cpu'
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Cargar modelo
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            print(f"✓ Modelo de keypoints cargado: {model_path}")
            
            # Verificar clases del modelo
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"  Clases detectables: {len(self.class_names)}")
            else:
                self.class_names = {}
                
        except Exception as e:
            print(f"✗ Error cargando modelo de keypoints: {e}")
            raise
    
    def detect_keypoints(self, frame: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Detecta keypoints del campo en un frame.
        
        Args:
            frame: Frame BGR de la transmisión
            
        Returns:
            Dict {keypoint_name: (x, y)} con coordenadas en píxeles
        """
        keypoints = {}
        
        try:
            # Inferencia
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Procesar detecciones
            if len(results) > 0:
                result = results[0]
                
                # Extraer boxes y clases
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        # Obtener centro del bounding box como keypoint
                        x1, y1, x2, y2 = box.xyxy[0]
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Obtener clase
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Nombre del keypoint (clase)
                        if class_id in self.class_names:
                            keypoint_name = str(self.class_names[class_id])
                        else:
                            keypoint_name = str(class_id)
                        
                        # Guardar keypoint
                        keypoints[keypoint_name] = (float(center_x), float(center_y))
            
        except Exception as e:
            print(f"Error detectando keypoints: {e}")
        
        return keypoints
    
    def visualize_keypoints(self, 
                           frame: np.ndarray,
                           keypoints: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """
        Dibuja los keypoints detectados sobre el frame.
        
        Args:
            frame: Frame BGR
            keypoints: Dict {name: (x, y)}
            
        Returns:
            Frame con keypoints dibujados
        """
        frame_vis = frame.copy()
        
        for name, (x, y) in keypoints.items():
            # Dibujar círculo
            cv2.circle(frame_vis, (int(x), int(y)), 8, (0, 255, 0), -1)
            cv2.circle(frame_vis, (int(x), int(y)), 10, (255, 255, 255), 2)
            
            # Dibujar nombre
            cv2.putText(frame_vis, str(name), (int(x) + 15, int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame_vis, str(name), (int(x) + 15, int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame_vis


if __name__ == "__main__":
    """Test del detector"""
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python field_keypoints_yolo.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Crear detector
    detector = FieldKeypointsYOLO(
        model_path="weights/field_kp_merged_fast/weights/best.pt",
        confidence_threshold=0.25
    )
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar cada 30 frames
        if frame_count % 30 == 0:
            keypoints = detector.detect_keypoints(frame)
            
            print(f"\nFrame {frame_count}: {len(keypoints)} keypoints detectados")
            for name, (x, y) in list(keypoints.items())[:5]:
                print(f"  - {name}: ({x:.1f}, {y:.1f})")
            
            # Visualizar
            frame_vis = detector.visualize_keypoints(frame, keypoints)
            
            # Mostrar
            cv2.imshow('Keypoints', cv2.resize(frame_vis, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Procesados {frame_count} frames")
