#!/bin/bash

echo "🚀 Iniciando TacticEYE2 Web Application..."
echo ""
echo "📋 Verificando dependencias..."

# Verificar que existe el directorio weights
if [ ! -d "weights" ]; then
    echo "⚠️  Advertencia: No se encontró el directorio 'weights/'"
    echo "   Asegúrate de tener el modelo YOLO en weights/best.pt"
fi

# Verificar que existe best.pt
if [ ! -f "weights/best.pt" ]; then
    echo "❌ Error: No se encontró weights/best.pt"
    echo "   El modelo YOLO es necesario para el análisis"
    exit 1
fi

# Crear directorios necesarios
mkdir -p uploads outputs static templates

echo "✅ Verificación completada"
echo ""
echo "🌐 Iniciando servidor web en http://localhost:8000"
echo "   Presiona Ctrl+C para detener"
echo ""

# Iniciar aplicación
python app.py
