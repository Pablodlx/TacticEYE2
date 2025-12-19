# Guía de Calibración Clásica - Pipeline OpenCV

## 📋 Índice

1. [Arquitectura del Sistema](#arquitectura-del-sistema)
2. [Componentes Principales](#componentes-principales)
3. [Flujo de Procesamiento](#flujo-de-procesamiento)
4. [Integración con YOLO](#integración-con-yolo)
5. [Decisiones Técnicas](#decisiones-técnicas)
6. [Uso Práctico](#uso-práctico)
7. [Migración a Deep Learning](#migración-a-deep-learning)

---

## 🏗️ Arquitectura del Sistema

El pipeline de calibración clásica está diseñado en 4 módulos principales:

```
┌─────────────────────────────────────────────────────────────┐
│         ClassicFieldCalibration (Orquestador)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐    ┌──────────────────┐            │
│  │ FieldLineDetector │───▶│HomographyEstimator│            │
│  │                  │    │                  │            │
│  │ - Segmentación   │    │ - Intersecciones │            │
│  │ - Acumulación    │    │ - Correspondencias│            │
│  │ - Detección LSD  │    │ - RANSAC         │            │
│  └──────────────────┘    └──────────────────┘            │
│           │                        │                       │
│           └──────────┬─────────────┘                       │
│                      ▼                                     │
│            ┌──────────────────┐                            │
│            │ FieldZoneManager │                            │
│            │                  │                            │
│            │ - Grid 6x3       │                            │
│            │ - Clasificación  │                            │
│            │ - Zonificación   │                            │
│            └──────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Datos

```
Frame BGR
    │
    ▼
[FieldLineDetector]
    │
    ├─▶ Extracción de máscara (HSV/LAB)
    ├─▶ Acumulación temporal (buffer N frames)
    └─▶ Detección de líneas (LSD + Hough)
         │
         ▼
[HomographyEstimator]
    │
    ├─▶ Intersecciones de líneas
    ├─▶ Identificación de líneas conocidas
    ├─▶ Correspondencias imagen ↔ modelo
    └─▶ Estimación RANSAC
         │
         ▼
[FieldZoneManager]
    │
    ├─▶ División en grid (6x3 = 18 zonas)
    ├─▶ Clasificación táctica
    └─▶ Proyección de jugadores
```

---

## 🧩 Componentes Principales

### 1. FieldLineDetector

**Responsabilidad**: Detectar líneas blancas del campo con robustez ante oclusiones.

**Estrategias**:
- **Segmentación multi-espacio**: HSV (color blanco) + LAB (luminosidad) + Top-hat morfológico
- **Acumulación temporal**: Buffer circular de N frames (default: 30)
  - Las líneas del campo son estáticas → se refuerzan
  - Los jugadores se mueven → desaparecen de la acumulación
- **Detección dual**: LSD (preciso) + HoughLinesP (robusto)
- **Filtrado geométrico**: Longitud mínima, ángulos esperados

**Parámetros clave**:
```python
temporal_window=30          # Frames para acumular
min_line_length=20.0        # Píxeles mínimos
use_lsd=True                # Usar Line Segment Detector
use_hough=True              # Respaldo con Hough
```

### 2. HomographyEstimator

**Responsabilidad**: Estimar homografía imagen → campo desde líneas detectadas.

**Estrategias**:
- **Intersecciones**: Puntos de interés donde se cruzan líneas
- **Identificación heurística**: Reconocer líneas conocidas (medio, perímetro, áreas)
- **Correspondencias**: Establecer matches entre imagen y modelo del campo
- **RANSAC robusto**: Manejar outliers y correspondencias incorrectas
- **Validación**: Verificar que la homografía es razonable

**Funciona con información parcial**:
- ✅ Media cancha visible
- ✅ Solo líneas centrales
- ✅ Áreas parcialmente visibles
- ❌ Requiere mínimo 4 líneas detectadas

### 3. FieldZoneManager

**Responsabilidad**: Dividir el campo en zonas tácticas para análisis.

**Grid configurable**:
- Default: 6 columnas × 3 filas = 18 zonas
- Personalizable: `grid_cols`, `grid_rows`

**Clasificación táctica**:
- **Por profundidad**: Defensiva / Medio / Ataque
- **Por ancho**: Wing / Central
- **Áreas especiales**: Penalty Area / Goal Area / Center Circle

**Información por zona**:
```python
zone.zone_id          # ID único (1-18)
zone.name             # "Bottom Center-Left"
zone.bounds           # (x_min, y_min, x_max, y_max)
zone.center           # (x, y) en metros
zone.zone_type        # ZoneType enum
zone.tactical_info    # Dict con metadata táctica
```

### 4. ClassicFieldCalibration

**Responsabilidad**: Orquestar todo el pipeline y proporcionar interfaz unificada.

**Características**:
- Calibración continua cada N frames
- Estabilización temporal de homografía
- Compatible con interfaz existente (`FieldCalibration`)
- Proyección de jugadores a zonas

---

## 🔄 Flujo de Procesamiento

### Paso 1: Detección de Líneas

```python
# Por cada frame:
mask, lines = line_detector.process_frame(frame)

# Internamente:
# 1. Extraer máscara de líneas blancas
# 2. Acumular en buffer temporal
# 3. Detectar segmentos de línea (LSD/Hough)
# 4. Filtrar por geometría
```

### Paso 2: Estimación de Homografía

```python
# Cada N frames (o si no está calibrado):
homography = homography_estimator.estimate(lines, image_shape)

# Internamente:
# 1. Encontrar intersecciones de líneas
# 2. Identificar líneas conocidas del campo
# 3. Establecer correspondencias imagen ↔ modelo
# 4. Estimar con RANSAC
# 5. Validar homografía
```

### Paso 3: Zonificación

```python
# Una vez calibrado:
zone, info = calibration.get_player_zone(player_pixel_position)

# Internamente:
# 1. Convertir píxeles → metros
# 2. Buscar zona en grid
# 3. Retornar información táctica
```

---

## 🔗 Integración con YOLO

### Ejemplo Completo

```python
from modules.classic_field_calibration import ClassicFieldCalibration
from ultralytics import YOLO
import cv2

# 1. Inicializar componentes
yolo_model = YOLO('path/to/model.pt')
calibration = ClassicFieldCalibration(
    temporal_window=30,
    calibration_interval=10,
    debug=True
)

# 2. Procesar video
cap = cv2.VideoCapture('match.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 2.1. Detectar jugadores (YOLO)
    results = yolo_model.predict(frame, conf=0.3, verbose=False)[0]
    
    # 2.2. Calibrar campo (clásico)
    calibration.process_frame(frame)
    
    # 2.3. Procesar cada detección
    if calibration.is_calibrated:
        for box in results.boxes:
            # Obtener centro del bbox
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            player_pos = np.array([center_x, center_y])
            
            # Obtener zona táctica
            zone_info = calibration.get_player_zone(player_pos)
            if zone_info:
                zone, info = zone_info
                print(f"Jugador en zona {zone.zone_id}: {zone.name}")
                print(f"  Tipo: {info['zone_type']}")
                print(f"  Posición: {info['position_meters']}")
    
    # 2.4. Visualizar
    if calibration.is_calibrated:
        frame = calibration.draw_projected_pitch(frame)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Integración con Tracker Existente

```python
# En analyze_match.py o similar:

from modules.classic_field_calibration import ClassicFieldCalibration

class TacticEYE2:
    def __init__(self, ...):
        # ... código existente ...
        
        # Reemplazar o complementar FieldCalibration
        self.field_calibration = ClassicFieldCalibration(
            temporal_window=30,
            calibration_interval=10
        )
    
    def process_frame(self, frame):
        # 1. YOLO detection (existente)
        results = self.model.predict(frame, ...)
        
        # 2. Calibración clásica
        self.field_calibration.process_frame(frame)
        
        # 3. Tracking (existente)
        tracks = self.tracker.update(frame, boxes, scores, classes)
        
        # 4. Proyección a zonas (NUEVO)
        if self.field_calibration.is_calibrated:
            for track_id, bbox, class_id in tracks:
                center = self._get_bbox_center(bbox)
                zone_info = self.field_calibration.get_player_zone(center)
                # Usar zone_info para análisis táctico
```

---

## 🎯 Decisiones Técnicas

### ¿Por qué acumulación temporal?

**Problema**: Jugadores y árbitros ocultan líneas constantemente.

**Solución**: Acumular máscaras durante N frames.
- Las líneas del campo aparecen consistentemente → se refuerzan
- Los jugadores se mueven → desaparecen de la acumulación
- Resultado: Máscara limpia de líneas

**Trade-off**: Latencia inicial de N frames antes de calibrar.

### ¿Por qué LSD + Hough?

**LSD (Line Segment Detector)**:
- ✅ Más preciso
- ✅ Detecta segmentos completos
- ❌ Más lento
- ❌ Requiere opencv-contrib

**HoughLinesP**:
- ✅ Más robusto a ruido
- ✅ Más rápido
- ✅ Disponible en opencv estándar
- ❌ Menos preciso

**Combinación**: Usar LSD como principal, Hough como respaldo.

### ¿Por qué RANSAC para homografía?

**Problema**: Correspondencias pueden ser incorrectas (outliers).

**Solución**: RANSAC (Random Sample Consensus).
- Selecciona 4 puntos aleatorios
- Estima homografía
- Cuenta inliers
- Repite N veces
- Retorna mejor homografía

**Ventaja**: Robusto a hasta 50% de outliers.

### ¿Por qué grid 6x3?

**Análisis táctico típico**:
- 3 tercios verticales (defensa, medio, ataque)
- 2-3 carriles horizontales (izquierda, centro, derecha)

**Grid 6x3 = 18 zonas**:
- Suficiente granularidad para análisis táctico
- No demasiado fino (evita ruido)
- Estándar en análisis profesional

**Personalizable**: Puede cambiarse a 4x6, 5x3, etc.

### ¿Por qué no precisión milimétrica?

**Objetivo**: Análisis táctico por zonas, no tracking preciso.

**Ventajas**:
- Más robusto ante errores de calibración
- Más rápido (no requiere refinamiento fino)
- Suficiente para estadísticas tácticas

**Si se necesita precisión**: Puede refinarse con ECC o correspondencias manuales.

---

## 💻 Uso Práctico

### Configuración Básica

```python
from modules.classic_field_calibration import ClassicFieldCalibration

calibration = ClassicFieldCalibration(
    temporal_window=30,        # Acumular 30 frames
    calibration_interval=10,   # Intentar calibrar cada 10 frames
    grid_cols=6,               # 6 columnas
    grid_rows=3,               # 3 filas
    debug=True                 # Mostrar información
)
```

### Configuración para Cámaras Móviles

```python
# Cámara se mueve frecuentemente → reducir ventana temporal
calibration = ClassicFieldCalibration(
    temporal_window=15,        # Menos frames (más rápido)
    calibration_interval=5,   # Calibrar más frecuentemente
    smoothing_alpha=0.3        # Más suavizado
)
```

### Configuración para Alta Precisión

```python
# Más frames para mejor acumulación
calibration = ClassicFieldCalibration(
    temporal_window=60,        # Más frames
    calibration_interval=5,   # Calibrar frecuentemente
    min_frames_for_calibration=30  # Esperar más antes de calibrar
)
```

### Visualización de Debug

```python
# Obtener visualización completa
debug_frame = calibration.get_debug_visualization(frame)
cv2.imshow('Debug', debug_frame)

# Visualizar zonas proyectadas
if calibration.is_calibrated:
    zones_vis = calibration.zone_manager.visualize_zones(
        frame.shape[:2],
        calibration.homography_matrix
    )
    cv2.imshow('Zones', zones_vis)
```

---

## 🚀 Migración a Deep Learning

El pipeline está diseñado para permitir reemplazar la detección de líneas por una red de segmentación sin romper el resto del sistema.

### Interfaz Consistente

```python
# Actual (clásico)
mask, lines = line_detector.process_frame(frame)

# Futuro (deep learning)
class DeepLearningLineDetector:
    def process_frame(self, frame):
        # Usar red de segmentación
        mask = self.segmentation_model.predict(frame)
        lines = self.extract_lines_from_mask(mask)
        return mask, lines  # Misma interfaz
```

### Plan de Migración

1. **Fase 1**: Mantener detector clásico como respaldo
   ```python
   if deep_learning_available:
       mask, lines = dl_detector.process_frame(frame)
   else:
       mask, lines = classic_detector.process_frame(frame)
   ```

2. **Fase 2**: Híbrido (clásico + deep learning)
   ```python
   mask_classic, lines_classic = classic_detector.process_frame(frame)
   mask_dl, lines_dl = dl_detector.process_frame(frame)
   
   # Combinar resultados
   mask_combined = cv2.bitwise_or(mask_classic, mask_dl)
   lines_combined = merge_lines(lines_classic, lines_dl)
   ```

3. **Fase 3**: Solo deep learning (cuando sea suficientemente robusto)

### Ventajas del Diseño Modular

- ✅ `HomographyEstimator` y `FieldZoneManager` no cambian
- ✅ Solo se reemplaza `FieldLineDetector`
- ✅ Interfaz consistente facilita migración
- ✅ Puede probarse deep learning sin romper producción

---

## 📊 Métricas y Validación

### Indicadores de Calidad

```python
# Confianza de calibración
confidence = calibration.calibration_confidence  # 0.0 - 1.0

# Número de líneas detectadas
num_lines = len(calibration.last_lines)

# Frames acumulados
frames_accumulated = len(calibration.line_detector.mask_buffer)
```

### Validación Visual

1. **Máscara acumulada**: Debe mostrar líneas del campo sin jugadores
2. **Líneas detectadas**: Deben corresponder a líneas reales del campo
3. **Campo proyectado**: Debe alinearse con el campo real en la imagen
4. **Zonas**: Deben cubrir el campo visible correctamente

### Troubleshooting

**Problema**: No se detectan líneas
- ✅ Verificar iluminación (muy oscura/clara)
- ✅ Ajustar umbrales de segmentación
- ✅ Verificar que hay césped verde visible

**Problema**: Homografía incorrecta
- ✅ Aumentar `min_frames_for_calibration`
- ✅ Verificar que hay suficientes líneas (≥4)
- ✅ Ajustar `ransac_threshold`

**Problema**: Calibración inestable
- ✅ Aumentar `smoothing_alpha` (más suavizado)
- ✅ Reducir `calibration_interval` (calibrar más frecuentemente)

---

## 📚 Referencias

- **OpenCV Line Segment Detector**: https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
- **RANSAC**: Fischler & Bolles, 1981
- **Homography Estimation**: Hartley & Zisserman, Multiple View Geometry
- **Análisis Táctico**: Taki & Hasegawa, 2000

---

## 🎓 Conclusión

Este pipeline proporciona una solución robusta y explicable para calibración automática de campos de fútbol usando técnicas clásicas de visión por computador. Está diseñado para:

- ✅ Producción (robusto, rápido)
- ✅ Explicabilidad (cada paso es claro)
- ✅ Extensibilidad (fácil migrar a deep learning)
- ✅ Análisis táctico (zonificación, no precisión milimétrica)

Para preguntas o mejoras, consultar el código fuente y los comentarios técnicos en cada módulo.

