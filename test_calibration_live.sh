#!/bin/bash
# Script de prueba para el sistema de calibración integrado en pruebatrackequipo.py

VIDEO="${1:-sample_match.mp4}"

echo "=========================================="
echo "  TacticEYE2 - Test Calibración Live"
echo "=========================================="
echo ""
echo "Video: $VIDEO"
echo ""
echo "Características activas:"
echo "  ✓ ReID Tracking"
echo "  ✓ Team Classification V3"
echo "  ✓ Possession Detection"
echo "  ✓ Field Calibration (YOLO Keypoints)"
echo ""
echo "Controles:"
echo "  - Presiona 'q' para salir"
echo "  - Los keypoints se mostrarán en amarillo"
echo "  - Estado de calibración en la parte inferior"
echo ""
echo "=========================================="
echo ""

python pruebatrackequipo.py "$VIDEO" \
    --model weights/best.pt \
    --reid \
    --use-v3 \
    --calibrate \
    --show-keypoints \
    --keypoints-model weights/field_kp_merged_fast/weights/best.pt \
    --keypoints-conf 0.25 \
    --possession-distance 60 \
    --conf 0.35

echo ""
echo "Test finalizado"
