# ⚽ TacticEYE2 - Sistema de Análisis de Fútbol

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-11-green)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Sistema de análisis de partidos de fútbol con **interfaz web** y funcionalidades core: **Tracking con ReID**, **Clasificación de Equipos**, **Detección de Posesión** y **Contador de Pases**.

## 🎯 Funcionalidades Core

### 1️⃣ **Tracking con Re-Identificación (ReID)**
- ✅ Re-identificación de jugadores usando features profundas (OSNet)
- ✅ IDs persistentes durante todo el partido
- ✅ Matching basado en similitud visual + IoU
- ✅ Gestión de oclusiones y salidas de cámara

### 2️⃣ **Clasificación Automática de Equipos**
- ✅ **TeamClassifierV2**: Clustering K-means en espacio LAB con eliminación de verde
- ✅ Detección automática de árbitros
- ✅ Sistema de votación temporal para estabilidad

### 3️⃣ **Detección de Posesión del Balón (V2)**
- ✅ Sistema determinista (100% del tiempo asignado)
- ✅ Algoritmo de 3 pasos: detectar balón → encontrar jugador cercano → validar distancia
- ✅ Hiesteresis configurable (default: 5 frames)
- ✅ Distancia configurable (default: 60px)
- ✅ Timeline completo de cambios de posesión
- ✅ Estadísticas en tiempo real (%, frames, segundos)
- ✅ Visualización con rectángulo amarillo y línea al balón

### 4️⃣ **Contador de Pases** 🆕
- ✅ Detección automática de pases entre jugadores del mismo equipo
- ✅ Estadísticas acumuladas por equipo
- ✅ Visualización en tiempo real

### 5️⃣ **Interfaz Web** 🆕
- ✅ Aplicación web completa con FastAPI
- ✅ Subida de videos y análisis en tiempo real
- ✅ Gráficos interactivos con Chart.js
- ✅ WebSocket para actualizaciones en vivo
- ✅ Dashboard responsive con Bootstrap

## 📋 Requisitos

### Hardware
- **GPU**: NVIDIA con CUDA (recomendado, mínimo 6GB VRAM)
- **RAM**: 8GB mínimo
- **Almacenamiento**: 2GB libres

### Software
- Python 3.8+
- CUDA 11.8+ (para GPU)

## 🔧 Instalación

### 1. Clonar repositorio
```bash
git clone https://github.com/TuUsuario/TacticEYE2.git
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
pip install -r requirements_web.txt  # Para la aplicación web
```

### 4. Verificar instalación
```bash
python setup_check.py
```

## 🎮 Uso

### 🌐 Aplicación Web (Recomendado)

1. **Iniciar servidor:**
```bash
python app.py
```

Por defecto usa `8001` (para evitar conflictos con proyectos en `8000`) y, si está ocupado, usa automáticamente otro libre.

2. **Abrir en navegador:**
```
http://localhost:8001
```

Si quieres forzar un puerto concreto, ejecuta:
```bash
PORT=8010 python app.py
```

3. **Usar la interfaz:**
   - Subir video (.mp4, .avi, etc.)
   - El análisis comenzará automáticamente
   - Ver estadísticas en tiempo real:
     - Posesión por equipo (%)
     - Pases completados
     - Timeline de posesión
     - Gráficos interactivos

### 💻 Línea de Comandos

#### Comando Básico
```bash
python pruebatrackequipo.py video.mp4 --model weights/best.pt --reid
```

### Análisis con V3 (Recomendado)
```bash
python pruebatrackequipo.py video.mp4 \
    --model weights/best.pt \
    --reid \
    --use-v3 \
    --v3-recalibrate 300
```

**Nota:** TeamClassifierV3 no está disponible actualmente. Usar V2 por defecto.

### Posesión con Alta Precisión
```bash
python pruebatrackequipo.py video.mp4 \
    --model weights/best.pt \
    --reid \
    --possession-distance 40
```

### Sin Visualización (Más Rápido)
```bash
python pruebatrackequipo.py video.mp4 \
    --model weights/best.pt \
    --reid \
    --no-show \
    --output resultado.mp4
```

## 📚 Parámetros Disponibles

### Detección YOLO
| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--model` | Ruta al modelo YOLO | `yolov8n.pt` |
| `--imgsz` | Tamaño de imagen | `640` |
| `--conf` | Umbral de confianza | `0.35` |
| `--max-det` | Máximo detecciones | `100` |

### Tracking ReID
| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--reid` | Activar ReID tracker | `False` |

### Clasificación de Equipos V2
| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--tc-kmeans-min-tracks` | Tracks mínimos para KMeans | `12` |
| `--tc-vote-history` | Historial de votación | `4` |
| `--tc-use-L` | Usar canal L* | `True` |
| `--tc-L-weight` | Peso del canal L* | `0.5` |

### Clasificación de Equipos V3
| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--use-v3` | Usar TeamClassifierV3 | `False` |
| `--v3-recalibrate` | Recalibrar cada N frames | `300` |
| `--v3-variance` | Features de varianza | `True` |
| `--v3-adaptive-thresh` | Umbral adaptativo | `True` |
| `--v3-hysteresis` | Hiesteresis temporal | `True` |

### Detección de Posesión
| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--possession-distance` | Distancia máxima (píxeles) | `60` |

### Salida
| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--no-show` | Sin ventana de preview | `False` |
| `--output` | Guardar video procesado | `None` |

## 🏗️ Estructura del Proyecto

```
TacticEYE2/
├── modules/
│   ├── reid_tracker.py              # Tracking con Re-ID
│   ├── team_classifier.py           # Clasificación V1
│   ├── team_classifier_v2.py        # Clasificación V2
│   ├── team_classifier_v2_backup.py # Backup V2
│   ├── possession_tracker.py        # Posesión V1
│   └── possession_tracker_v2.py     # Posesión V2 ⭐
├── weights/
│   └── best.pt                      # Modelo YOLO entrenado
├── pruebatrackequipo.py             # Script principal ⭐
├── setup_check.py                   # Verificación
├── config.yaml                      # Configuración
└── requirements.txt                 # Dependencias
```

## 📊 Salida del Sistema

### Resumen de Posesión (Consola)
```
======================================================================
POSSESSION SUMMARY (PossessionTrackerV2)
======================================================================

Total frames processed: 900
Total time: 30.00 seconds

Possession by team:
  Team 0: 241 frames (8.0s) = 26.8%
  Team 1: 544 frames (18.1s) = 60.4%

Validation:
  Frames assigned: 785/900
  Coverage: 87.2%

Possession timeline (9 segments):
  Segment 1: Frames 116-487 (371f) → Team 1
  Segment 2: Frames 487-588 (101f) → Team 0
  ...
======================================================================
```

### Visualización en Tiempo Real
- **Rectángulo amarillo**: Jugador con posesión
- **Línea amarilla**: Conexión jugador-balón
- **Distancia**: Mostrada en píxeles
- **Estadísticas**: Posesión acumulada
- **Equipo 0**: Verde
- **Equipo 1**: Azul (rojo en BGR)
- **Árbitros**: Naranja

## 🎯 Ejemplos de Uso

### 1. Máxima Precisión
```bash
python pruebatrackequipo.py video.mp4 \
    --model weights/best.pt \
    --conf 0.40 \
    --reid \
    --use-v3 \
    --v3-recalibrate 150 \
    --possession-distance 40
```

### 2. Máxima Velocidad
```bash
python pruebatrackequipo.py video.mp4 \
    --model weights/best.pt \
    --conf 0.25 \
    --imgsz 416 \
    --reid \
    --no-show
```

### 3. Balance Óptimo (Recomendado)
```bash
python pruebatrackequipo.py video.mp4 \
    --model weights/best.pt \
    --conf 0.35 \
    --reid \
    --use-v3 \
    --possession-distance 60
```

## 🔍 Módulos Core

### ReID Tracker (`reid_tracker.py`)
Sistema de tracking con re-identificación:
- **Features OSNet**: Extracción de características profundas
- **IDs persistentes**: Mantiene IDs todo el partido
- **Recovery**: Re-identifica tras oclusiones
- **Matching**: Combina similitud visual + IoU espacial

### TeamClassifierV2 (`team_classifier_v2.py`)
Clasificación de equipos robusta:
- **Espacio LAB**: Clustering en espacio de color LAB
- **Anti-verde**: Eliminación automática del césped
- **Votación temporal**: Sistema de votación para estabilidad
- **Features**: LAB (a*,b* + L* weighted)

### TeamClassifierV3 (Opcional)
Sistema avanzado con:
- **Recalibración**: KMeans se re-entrena automáticamente
- **Features robustas**: Varianza + textura + edges
- **Hiesteresis**: Requiere múltiples frames para cambios
- **Adaptativo**: Ajusta umbrales según separación

### PossessionTrackerV2 (`possession_tracker_v2.py`)
Sistema determinista de posesión:
- **100% asignación**: Todo el tiempo a algún equipo
- **Hiesteresis**: 5 frames consecutivos para cambios
- **Timeline**: Segmentos completos con timestamps
- **Validación**: Cobertura automática
- **API simple**: `update()`, `get_possession_stats()`, `get_possession_timeline()`

## 🐛 Solución de Problemas

### Error: KeyError -1
**Solución**: El sistema filtra automáticamente team_id inválidos (referees).

### Clasificación incorrecta
**Solución**: Prueba TeamClassifierV3:
```bash
--use-v3 --v3-recalibrate 300
```

### Posesión con cambios rápidos
**Solución**: Reduce la distancia:
```bash
--possession-distance 40
```

### Procesamiento lento
**Solución**: Reduce resolución y desactiva preview:
```bash
--imgsz 416 --no-show
```

## 📝 Clases Detectadas

El modelo YOLO detecta:
- **0**: `player` - Jugador de campo
- **1**: `ball` - Balón
- **2**: `referee` - Árbitro
- **3**: `goalkeeper` - Portero

## 🚀 Roadmap (Próximas Funcionalidades)

Las siguientes funcionalidades se añadirán más adelante:
- 🔄 Calibración del campo (homografía 2D→3D)
- 🔄 Mapas de calor por equipo
- 🔄 Estadísticas avanzadas (distancias, velocidades)
- 🔄 Overlays profesionales
- 🔄 Exportación completa (CSV, JSON, NPZ)
- 🔄 Detección de eventos (pases, tiros)

## 🤝 Contribuir

Las contribuciones son bienvenidas:
1. Fork del repositorio
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## 🙏 Agradecimientos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid)
- Comunidad de Computer Vision

---

**Versión Simplificada v2.0** - Solo funcionalidades core: Tracking + Equipos + Posesión
