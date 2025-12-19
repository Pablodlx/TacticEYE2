# 🎯 TacticEYE2 - Sistema Completo de Análisis Táctico de Fútbol

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-11-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Sistema profesional de análisis táctico de partidos de fútbol con inteligencia artificial. Incluye tracking avanzado con Re-Identificación, calibración automática del campo, mapas de calor 3D, estadísticas en tiempo real y overlays profesionales tipo Wyscout.

![TacticEYE2 Demo](demo.gif)

## 🚀 Características Principales

### 1️⃣ **Tracking Avanzado con ReID**
- ✅ Re-identificación de jugadores usando features profundas (OSNet/ResNet)
- ✅ IDs persistentes por 30-60 segundos fuera de pantalla
- ✅ Matching basado en similitud de apariencia + IoU
- ✅ ID único para el balón

### 2️⃣ **Diferenciación Automática de Equipos**
- ✅ Clustering K-means en espacio HSV de colores de camiseta
- ✅ Identificación automática de árbitros
- ✅ Sistema de votación para estabilidad de asignaciones

### 3️⃣ **Calibración Automática del Campo**
- ✅ Detección automática de líneas del campo
- ✅ Cálculo de homografía 2D → 3D (píxeles → metros reales)
- ✅ Mapeo a campo FIFA estándar (105m × 68m)
- ✅ Vista cenital (top-down) del campo

### 4️⃣ **Mapas de Calor 3D en Tiempo Real**
- ✅ Heatmaps por equipo (local/visitante/árbitro)
- ✅ Heatmap del balón
- ✅ Actualización automática cada 5 segundos
- ✅ Histórico configurable (últimos 60 segundos)

### 5️⃣ **Overlay Profesional Tipo Wyscout**
- ✅ IDs encima de cada jugador
- ✅ Trayectorias recientes (últimos 10 segundos)
- ✅ Mini-mapa cenital en esquina
- ✅ Panel de estadísticas en vivo
- ✅ Velocidades individuales

### 6️⃣ **Estadísticas Avanzadas**
- ✅ **Posesión**: % de tiempo con balón por equipo
- ✅ **Pases**: Completados/intentados + precisión
- ✅ **Distancia**: Total recorrida por jugador y equipo
- ✅ **Velocidad**: Máxima y promedio (km/h)
- ✅ **Presión**: Alta/media/baja (zonas del campo)

### 7️⃣ **Exportación Completa**
- ✅ Vídeo con overlay profesional (MP4)
- ✅ CSV con posiciones 3D por frame
- ✅ JSON con eventos del partido
- ✅ JSON con resumen de estadísticas
- ✅ NPZ con datos de heatmaps
- ✅ JSON con trayectorias completas

## 📋 Requisitos

### Hardware Recomendado
- **GPU**: NVIDIA con CUDA (mínimo 6GB VRAM)
- **RAM**: 16GB mínimo
- **Almacenamiento**: 5GB libres

### Software
- Python 3.8+
- CUDA 11.8+ (para GPU)
- FFmpeg (para procesamiento de vídeo)

## 🔧 Instalación

### 1. Clonar repositorio
```bash
git clone https://github.com/Pablodlx/TacticEYE2.git
cd TacticEYE2
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar instalación
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🎮 Uso Rápido

### Análisis Básico
```bash
python analyze_match.py --video sample_match.mp4
```

### Análisis Completo con Opciones
```bash
python analyze_match.py \
    --video sample_match.mp4 \
    --model weights/best.pt \
    --output ./outputs \
    --conf 0.3 \
    --calibration-frame 100 \
    --max-frames 1000
```

### Análisis sin Preview (más rápido)
```bash
python analyze_match.py --video sample_match.mp4 --no-preview
```

## 📚 Parámetros de la Línea de Comandos

| Parámetro | Descripción | Valor por Defecto |
|-----------|-------------|-------------------|
| `--video` | Ruta al vídeo a analizar | **Requerido** |
| `--model` | Ruta al modelo YOLO | `weights/best.pt` |
| `--output` | Directorio de salida | `./outputs` |
| `--conf` | Umbral de confianza (0-1) | `0.3` |
| `--iou` | Umbral de IoU para NMS | `0.5` |
| `--calibration-frame` | Frame para calibración | `100` |
| `--no-preview` | Desactivar preview en vivo | `False` |
| `--max-frames` | Máximo de frames a procesar | `None` (todos) |

## 🏗️ Arquitectura del Sistema

```
TacticEYE2/
├── modules/
│   ├── reid_tracker.py           # Re-ID + Tracking
│   ├── team_classifier.py        # Clasificación de equipos
│   ├── field_calibration.py      # Calibración del campo
│   ├── heatmap_generator.py      # Generación de heatmaps
│   ├── match_statistics.py       # Cálculo de estadísticas
│   ├── professional_overlay.py   # Overlays visuales
│   └── data_exporter.py          # Exportación de datos
├── analyze_match.py              # Script principal
├── config.yaml                   # Configuración
├── requirements.txt              # Dependencias
├── weights/
│   └── best.pt                   # Modelo YOLO11l entrenado
└── outputs/                      # Resultados generados
```

## 📊 Salidas Generadas

Después del análisis, encontrarás en `./outputs/`:

```
outputs/
├── analyzed_sample_match.mp4      # Vídeo con overlay
├── positions_YYYYMMDD_HHMMSS.csv  # Posiciones 3D
├── events_YYYYMMDD_HHMMSS.json    # Eventos del partido
├── match_summary_YYYYMMDD_HHMMSS.json  # Resumen estadísticas
├── heatmaps_YYYYMMDD_HHMMSS.npz   # Datos de heatmaps
└── trajectories_YYYYMMDD_HHMMSS.json  # Trayectorias
```

### Ejemplo CSV (posiciones)
```csv
frame,timestamp,track_id,team_id,x_pixels,y_pixels,x_meters,y_meters,velocity_kmh
100,3.33,5,0,640,480,45.2,32.1,15.3
100,3.33,7,1,800,500,52.7,28.4,12.8
...
```

### Ejemplo JSON (eventos)
```json
{
  "events": [
    {
      "timestamp": 12.5,
      "frame": 375,
      "event_type": "pass",
      "team_id": 0,
      "player_id": 5,
      "x_meters": 45.2,
      "y_meters": 32.1,
      "success": true
    }
  ]
}
```

## 🎨 Overlays Visuales

### Mini-mapa Cenital
Vista top-down del campo con posiciones de todos los jugadores en tiempo real.

### Panel de Estadísticas
- Barra de posesión animada
- Pases completados/intentados por equipo
- Distancia total recorrida
- Precisión de pases en %

### IDs y Trayectorias
- ID numérico encima de cada jugador
- Color según equipo
- Líneas de trayectoria con degradado de opacidad
- Velocidad actual (km/h)

## 🔬 Módulos Técnicos

### ReID Tracker
- **Feature Extractor**: ResNet18 pre-entrenado
- **Dimensión de features**: 512D, L2-normalizadas
- **Matching**: Similitud coseno (70%) + IoU (30%)
- **Buffer**: Últimas 10 features por track

### Team Classifier
- **Algoritmo**: K-means en espacio HSV
- **ROI**: 20-50% de altura de bbox (zona de camiseta)
- **Filtrado**: Elimina blancos/negros extremos
- **Estabilidad**: Votación por mayoría en 30 frames

### Field Calibration
- **Detección**: Canny + Hough Line Transform
- **Máscara**: Segmentación de césped verde en HSV
- **Homografía**: OpenCV findHomography (RANSAC)
- **Resolución top-down**: 10 píxeles = 1 metro

### Match Statistics
- **Posesión**: Radio de 3m alrededor del balón
- **Pases**: Detección por velocidad del balón (>5 m/s)
- **Distancia**: Acumulación frame-a-frame
- **Velocidad**: Ventana deslizante de 30 frames

## ⚙️ Configuración Avanzada

Edita `config.yaml` para personalizar:

```yaml
# Sensibilidad del detector
model:
  conf_threshold: 0.3  # Bajar para más detecciones

# Persistencia de IDs
tracking:
  max_lost_time: 60.0  # Segundos fuera de pantalla

# Resolución de heatmaps
heatmaps:
  grid_resolution: 50  # Mayor = más detalle

# Overlay
overlay:
  trajectory_length: 300  # Frames de trayectoria
```

## 🐛 Solución de Problemas

### Error: "CUDA out of memory"
```bash
# Reducir tamaño de imagen
python analyze_match.py --video sample.mp4 --img-size 640
```

### Error: "No se detectan líneas del campo"
```bash
# Especificar frame diferente para calibración
python analyze_match.py --video sample.mp4 --calibration-frame 500
```

### Procesamiento muy lento
```bash
# Desactivar preview
python analyze_match.py --video sample.mp4 --no-preview
```

### IDs inconsistentes
```yaml
# En config.yaml, aumentar similarity_threshold
tracking:
  similarity_threshold: 0.7  # Más estricto
```

## 📈 Rendimiento

En GPU NVIDIA RTX 3080:
- **Resolución**: 1920×1080
- **FPS de procesamiento**: ~15 FPS
- **Tiempo real**: 2x (procesa 1 min en 2 min)

En CPU (Intel i7-12700K):
- **FPS de procesamiento**: ~3 FPS
- **Tiempo real**: 10x (procesa 1 min en 10 min)

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Detector de objetos
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Inspiración para tracking
- [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) - Re-identificación
- Comunidad de Computer Vision en fútbol

## 📧 Contacto

**PabloDLX**
- GitHub: [@Pablodlx](https://github.com/Pablodlx)
- Proyecto: [TacticEYE2](https://github.com/Pablodlx/TacticEYE2)

---

⭐ Si te gusta el proyecto, ¡dale una estrella en GitHub!

**TacticEYE2** - El mejor sistema de análisis táctico amateur del mundo 🚀⚽
