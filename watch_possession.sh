#!/bin/bash
# Script para monitorear el procesamiento de posesión

while pgrep -f "pruebatrackequipo.py" > /dev/null; do
    clear
    echo "=== PROCESAMIENTO EN CURSO ==="
    echo ""
    tail -30 possession_output.txt | grep -E "(Procesados|Possession|Cambio)"
    echo ""
    echo "Esperando... (Ctrl+C para salir)"
    sleep 10
done

echo ""
echo "=== PROCESAMIENTO TERMINADO ==="
echo ""
echo "=== RESUMEN FINAL DE POSESIÓN ==="
tail -100 possession_output.txt | grep -A 50 "RESUMEN FINAL"
