#!/usr/bin/env python
"""
Script para diagnosticar por qué no se calibra el campo
"""

import cv2
import numpy as np
from modules.field_line_detector import FieldLineDetector

# Cargar un frame de prueba
cap = cv2.VideoCapture('sample_match2.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ No se pudo leer el video")
    exit(1)

print("="*60)
print("DIAGNÓSTICO DE CALIBRACIÓN DE CAMPO")
print("="*60)

# Crear detector con diferentes umbrales
print("\n1. PROBANDO CON UMBRALES ORIGINALES (200-255)")
detector = FieldLineDetector()

# Preprocesar
mask = detector.preprocess_image(frame)

# Contar píxeles blancos
white_pixels = np.sum(mask > 0)
total_pixels = mask.shape[0] * mask.shape[1]
percent = (white_pixels / total_pixels) * 100

print(f"   Frame shape: {frame.shape}")
print(f"   Píxeles blancos: {white_pixels:,} / {total_pixels:,} ({percent:.2f}%)")

# Detectar líneas
segments = detector.detect_lines(frame)
print(f"   Segmentos detectados: {len(segments)}")

if len(segments) > 0:
    lengths = [s.length for s in segments]
    print(f"   Longitudes: min={min(lengths):.1f}, max={max(lengths):.1f}, promedio={np.mean(lengths):.1f}")

# Clasificar
clusters = detector.detect_and_classify(frame)
print(f"   Líneas horizontales: {len(clusters.get('horizontal', []))}")
print(f"   Líneas verticales: {len(clusters.get('vertical', []))}")
print(f"   Líneas diagonales: {len(clusters.get('diagonal', []))}")
print(f"   TOTAL clasificadas: {sum(len(v) for v in clusters.values())}")

# Probar con umbrales más bajos
print("\n2. PROBANDO CON UMBRALES MÁS BAJOS (150-255)")
detector_low = FieldLineDetector(white_threshold_low=150)
mask_low = detector_low.preprocess_image(frame)
white_pixels_low = np.sum(mask_low > 0)
percent_low = (white_pixels_low / total_pixels) * 100
print(f"   Píxeles blancos: {white_pixels_low:,} ({percent_low:.2f}%)")

segments_low = detector_low.detect_lines(frame)
print(f"   Segmentos detectados: {len(segments_low)}")

clusters_low = detector_low.detect_and_classify(frame)
print(f"   Líneas H/V/D: {len(clusters_low.get('horizontal', []))}/{len(clusters_low.get('vertical', []))}/{len(clusters_low.get('diagonal', []))}")

# Probar con umbrales aún más bajos
print("\n3. PROBANDO CON UMBRALES MUY BAJOS (120-255)")
detector_verylow = FieldLineDetector(white_threshold_low=120, min_line_length=30)
segments_verylow = detector_verylow.detect_lines(frame)
print(f"   Segmentos detectados: {len(segments_verylow)}")

clusters_verylow = detector_verylow.detect_and_classify(frame)
print(f"   Líneas H/V/D: {len(clusters_verylow.get('horizontal', []))}/{len(clusters_verylow.get('vertical', []))}/{len(clusters_verylow.get('diagonal', []))}")

# Guardar visualización
print("\n4. GUARDANDO VISUALIZACIÓN")
cv2.imwrite('/tmp/field_detection_original.jpg', mask)
cv2.imwrite('/tmp/field_detection_low.jpg', mask_low)
print("   ✓ Guardado en /tmp/field_detection_*.jpg")

# Conclusión
print("\n" + "="*60)
print("CONCLUSIÓN:")
print("="*60)
if sum(len(v) for v in clusters.values()) >= 4:
    print("✅ El detector PUEDE encontrar suficientes líneas")
    print("   El problema puede estar en:")
    print("   - Matching de keypoints")
    print("   - Estimación de homografía")
    print("   - Umbral de confianza muy alto")
else:
    print("⚠️ El detector NO encuentra suficientes líneas")
    print(f"   Con umbral 200: {sum(len(v) for v in clusters.values())} líneas")
    print(f"   Con umbral 150: {sum(len(v) for v in clusters_low.values())} líneas")
    print(f"   Con umbral 120: {sum(len(v) for v in clusters_verylow.values())} líneas")
    print("\n   Recomendación: Bajar white_threshold_low a 120-150")
