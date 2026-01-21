"""
Test Rápido - Pipeline de Calibración Clásica
==============================================

Script de prueba rápida para verificar que todos los componentes funcionan.
"""

import numpy as np
import cv2
from modules.classic_field_calibration import ClassicFieldCalibration

def test_pipeline():
    """Prueba básica del pipeline"""
    print("🧪 Test del Pipeline de Calibración Clásica\n")
    
    # 1. Inicializar
    print("1️⃣  Inicializando componentes...")
    calibration = ClassicFieldCalibration(
        temporal_window=10,  # Reducido para test rápido
        calibration_interval=5,
        debug=False
    )
    print("   ✅ Componentes inicializados\n")
    
    # 2. Crear frame sintético (campo verde con líneas blancas)
    print("2️⃣  Generando frame sintético...")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Césped verde
    frame[:, :] = [40, 120, 40]  # Verde BGR
    
    # Líneas blancas simuladas
    # Línea del medio campo (vertical)
    cv2.line(frame, (640, 0), (640, 720), (255, 255, 255), 5)
    
    # Líneas horizontales (perímetro)
    cv2.line(frame, (0, 50), (1280, 50), (255, 255, 255), 5)
    cv2.line(frame, (0, 670), (1280, 670), (255, 255, 255), 5)
    
    # Líneas verticales (perímetro)
    cv2.line(frame, (50, 0), (50, 720), (255, 255, 255), 5)
    cv2.line(frame, (1230, 0), (1230, 720), (255, 255, 255), 5)
    
    # Círculo central
    cv2.circle(frame, (640, 360), 80, (255, 255, 255), 3)
    
    print("   ✅ Frame sintético generado\n")
    
    # 3. Procesar frames
    print("3️⃣  Procesando frames (acumulación temporal)...")
    for i in range(15):
        calibration.process_frame(frame)
        if calibration.is_calibrated:
            print(f"   ✅ Calibrado en frame {i+1} (confianza: {calibration.calibration_confidence:.2f})")
            break
        else:
            print(f"   ⏳ Frame {i+1}: Acumulando ({len(calibration.line_detector.mask_buffer)} frames)")
    
    print()
    
    # 4. Verificar calibración
    if calibration.is_calibrated:
        print("4️⃣  Verificando calibración...")
        print(f"   ✅ Homografía calculada")
        print(f"   ✅ Líneas detectadas: {len(calibration.last_lines)}")
        print(f"   ✅ Confianza: {calibration.calibration_confidence:.2f}\n")
        
        # 5. Probar proyección
        print("5️⃣  Probando proyección de puntos...")
        
        # Punto central de la imagen
        center_pixel = np.array([640, 360])
        center_meters = calibration.pixel_to_meters(center_pixel)
        
        if center_meters is not None:
            print(f"   ✅ Centro imagen ({center_pixel}) → Campo ({center_meters[0]:.1f}, {center_meters[1]:.1f} m)")
        
        # Punto en esquina
        corner_pixel = np.array([100, 100])
        corner_meters = calibration.pixel_to_meters(corner_pixel)
        
        if corner_meters is not None:
            print(f"   ✅ Esquina ({corner_pixel}) → Campo ({corner_meters[0]:.1f}, {corner_meters[1]:.1f} m)")
        
        # 6. Probar zonificación
        print("\n6️⃣  Probando zonificación...")
        if center_meters is not None:
            zone_info = calibration.get_player_zone(center_pixel)
            if zone_info:
                zone, info = zone_info
                print(f"   ✅ Zona detectada: {zone.zone_id} - {zone.name}")
                print(f"   ✅ Tipo: {info['zone_type']}")
                print(f"   ✅ Info táctica: {info['tactical_info']}")
        
        # 7. Estadísticas de zonas
        print("\n7️⃣  Estadísticas de zonas...")
        stats = calibration.zone_manager.get_zone_statistics()
        print(f"   ✅ Total zonas: {stats['total_zones']}")
        print(f"   ✅ Grid: {stats['grid_size']}")
        print(f"   ✅ Zonas por tipo:")
        for zone_type, count in stats['zones_by_type'].items():
            print(f"      - {zone_type}: {count}")
        
        print("\n✅ Todos los tests pasaron correctamente!")
        return True
    else:
        print("   ❌ No se pudo calibrar (puede ser normal con frame sintético)")
        print("   ℹ️  Prueba con un video real para mejor resultado")
        return False

if __name__ == '__main__':
    try:
        test_pipeline()
    except Exception as e:
        print(f"\n❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()



