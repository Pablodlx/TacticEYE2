# Sistema de Calibración Automática y Mapeo Espacial de Posesión

## Arquitectura General

Este documento describe el diseño e implementación del sistema de calibración automática de campo y tracking espacial de posesión para TacticEYE2.

---

## 1. Visión General del Sistema

### 1.1 Objetivo

Calcular en qué zonas del campo tiene más posesión cada equipo, **sin calibración manual**, incluso con vistas parciales del campo (broadcast típico).

### 1.2 Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline de Análisis                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │      FieldLineDetector             │
         │  (Detección automática de líneas)  │
         └────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │      FieldCalibrator               │
         │  (Estimación de homografía)        │
         └────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │  SpatialPossessionTracker          │
         │  (Tracking + Reproyección)         │
         └────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │     ZoneModel + Heatmaps           │
         │  (Acumulación por zonas)           │
         └────────────────────────────────────┘
```

---

## 2. Detección Automática de Geometría del Campo

### 2.1 Pipeline de Detección (`FieldLineDetector`)

**Objetivo**: Detectar líneas blancas del campo sin anotación manual.

**Métodos**:

1. **Preprocesamiento**:
   - Filtro bilateral para reducir ruido
   - Umbralización adaptativa para detectar blanco (200-255)
   - Morfología (close + open) para limpiar y conectar líneas

2. **Detección de Líneas**:
   - **LSD (Line Segment Detector)** de OpenCV
   - Ventajas: rápido, preciso, sin parámetros críticos
   - Alternativa: Hough Transform con post-filtrado

3. **Clasificación y Agrupación**:
   - Agrupar por orientación: horizontal, vertical, diagonal
   - Fusionar segmentos colineales cercanos
   - Filtrar por longitud mínima

4. **Extracción de Keypoints**:
   - Intersecciones de líneas H×V → candidatos a esquinas
   - Clustering espacial para eliminar duplicados

**Código**:
```python
from modules.field_line_detector import FieldLineDetector

detector = FieldLineDetector(
    white_threshold_low=200,
    min_line_length=50
)

# Detectar líneas en frame
line_clusters = detector.detect_and_classify(frame)
# Retorna: {'horizontal': [...], 'vertical': [...], ...}

# Visualizar
vis = detector.visualize_lines(frame, line_clusters)
```

### 2.2 Matching con Modelo de Campo (`KeypointMatcher`)

**Estrategia**:

- **Heurísticas geométricas**: 
  - Broadcast típico → vista lateral elevada
  - Líneas horizontales superiores/inferiores → bandas
  - Líneas verticales laterales → líneas de gol/área
  
- **Ratios geométricos**:
  - Distancia entre área grande y pequeña (conocida)
  - Ángulo entre líneas (perpendiculares en el campo)

- **Fallback simple**:
  - Si detección falla → usar vista estándar asumida
  - Mapear región visible a porción de campo

**Nota**: En producción, se puede mejorar con:
- **Deep Learning**: Modelo pre-entrenado para detectar keypoints de campo
- **Template matching**: Comparar con templates de vistas típicas
- **Optimización global**: Ajustar todos los keypoints simultáneamente

---

## 3. Estimación y Estabilización de Homografía

### 3.1 FieldCalibrator

**Responsabilidades**:
- Estimar matriz H (3×3) de imagen → campo
- Mantener calibración temporal estable
- Manejar cambios de cámara/zoom

**Pipeline**:

1. **Detección de correspondencias**:
   ```python
   correspondences = [
       ((x_img, y_img), (x_field, y_field)),
       ...
   ]
   ```

2. **Estimación con RANSAC**:
   ```python
   H, mask = cv2.findHomography(
       img_pts, field_pts, 
       cv2.RANSAC, 
       ransacReprojThreshold=5.0
   )
   ```

3. **Filtrado temporal** (`HomographyFilter`):
   - Buffer de últimas N homografías
   - Promedio ponderado por confianza
   - Evita jitter cuando cámara está estática

4. **Métricas de confianza**:
   - Ratio de inliers / total
   - Error de reproyección medio
   - Número de correspondencias

**API**:
```python
from modules.field_calibration import FieldCalibrator

calibrator = FieldCalibrator(
    use_temporal_filter=True,
    min_confidence=0.5
)

# En cada frame (o cada N frames)
H = calibrator.estimate_homography(frame)

# Reproyectar punto
x_field, y_field = calibrator.image_to_field(x_img, y_img)

# Verificar si hay calibración válida
if calibrator.has_valid_calibration():
    ...
```

### 3.2 Manejo de Vistas Parciales

**Problema**: Cámara solo ve una porción del campo.

**Solución**:
- La homografía mapea **correctamente** la porción visible
- Zonas no visibles simplemente no acumulan tiempo
- Si el jugador con posesión sale de vista:
  - Opción 1: Usar última zona conocida (temporal)
  - Opción 2: No acumular espacialmente (solo global)

**Implementado**: Opción 1 con timeout de 1 segundo (~30 frames).

---

## 4. Reproyección de Posesión a Campo

### 4.1 Integración con Pipeline Existente

**Modificación mínima** en `batch_processor.py`:

```python
# En cada frame con poseedor identificado
if possession_tracker.current_player >= 0:
    # Obtener posición del jugador (centro-base del bbox)
    player_bbox = tracked_objects[player_idx]['bbox']
    x_img = (player_bbox[0] + player_bbox[2]) / 2
    y_img = player_bbox[3]  # Base del bbox (pies)
    
    # Reproyectar a campo
    field_pos = calibrator.image_to_field(x_img, y_img)
    
    # Actualizar tracker espacial
    spatial_tracker.update_position(field_pos, team_id)
```

### 4.2 Punto Representativo del Jugador

**¿Qué punto usar?**
- ✅ **Base del bbox** (x_center, y_bottom): representa posición de pies
- ❌ Centro del bbox: puede estar elevado si jugador salta
- ❌ Keypoint de modelo de pose: más preciso pero más caro

**Implementado**: Centro-base del bounding box.

---

## 5. Modelo de Zonas y Acumulación

### 5.1 ZoneModel

**Tipos de partición soportados**:

1. **Grid** (rejilla regular):
   - NxM zonas uniformes
   - Default: 6×4 (24 zonas)
   
2. **Thirds** (tercios):
   - Defensivo / Medio / Ofensivo
   - 3 zonas
   
3. **Thirds + Lanes** (tercios × carriles):
   - 3 tercios × 3 carriles (izq/centro/der)
   - 9 zonas

**API**:
```python
from modules.field_model import ZoneModel, FieldModel

field_model = FieldModel()  # Campo estándar FIFA

# Opción 1: Grid 6×4
zone_model = ZoneModel(
    field_model, 
    partition_type='grid',
    nx=6, ny=4
)

# Opción 2: Tercios + carriles
zone_model = ZoneModel(
    field_model,
    partition_type='thirds_lanes'
)

# Determinar zona de una posición
zone_id = zone_model.zone_from_xy(x_field, y_field, team_id)
zone_name = zone_model.get_zone_name(zone_id)
```

### 5.2 SpatialPossessionTracker

**Extiende** `PossessionTracker` con:

- `time_by_team_and_zone`: Array 2D (team × zone)
- `heatmaps`: Mapas de calor continuos (opcional)
- `last_field_pos`: Última posición conocida por equipo (fallback)

**Lógica de acumulación**:

```python
# Frame a frame:
if jugador_con_posesion:
    # Obtener posición en campo
    field_pos = calibrator.image_to_field(player_x, player_y)
    
    if field_pos is not None:
        # Determinar zona
        zone_id = zone_model.zone_from_xy(*field_pos, team_id)
        
        # Acumular tiempo (1 frame)
        time_by_team_and_zone[team_id][zone_id] += 1
        
        # Actualizar heatmap
        heatmap[team_id][heatmap_coords] += 1.0
        
        # Guardar como última posición válida
        last_field_pos[team_id] = field_pos
    
    else:
        # No hay calibración válida
        # Usar última posición si < 30 frames (fallback)
        if frames_since_last_valid < 30:
            zone_id = last_zone[team_id]
            time_by_team_and_zone[team_id][zone_id] += 1
```

**API completa**:
```python
from modules.spatial_possession_tracker import SpatialPossessionTracker

tracker = SpatialPossessionTracker(
    calibrator=calibrator,
    zone_model=zone_model,
    enable_heatmaps=True,
    heatmap_resolution=(50, 34)
)

# En cada frame
state = tracker.update(ball_pos, players, frame)

# Obtener estadísticas espaciales
spatial_stats = tracker.get_spatial_statistics()
# Retorna:
# {
#   'possession_by_zone': {0: [...], 1: [...]},
#   'zone_percentages': {0: [...], 1: [...]},
#   'heatmaps': {0: array(...), 1: array(...)}
# }

# Estadísticas por zona legibles
zone_stats = tracker.get_zone_statistics()
# {
#   'zones': [
#     {'zone_id': 0, 'zone_name': 'zone_0', 
#      'team_0_frames': 120, 'team_1_frames': 80, ...},
#     ...
#   ]
# }

# Exportar heatmap para visualización
heatmap_team0 = tracker.export_heatmap(team_id=0, normalize=True)
```

---

## 6. Integración con Pipeline Existente

### 6.1 Modificaciones en `batch_processor.py`

**Añadir calibrador y tracker espacial**:

```python
class BatchProcessor:
    def __init__(self, ...):
        # ... inicializaciones existentes ...
        
        # NUEVO: Calibrador de campo
        from modules.field_calibration import FieldCalibrator
        from modules.spatial_possession_tracker import SpatialPossessionTracker
        from modules.field_model import ZoneModel
        
        self.field_calibrator = FieldCalibrator(
            use_temporal_filter=True
        )
        
        zone_model = ZoneModel(
            self.field_calibrator.field_model,
            partition_type='thirds_lanes',
            nx=6, ny=4
        )
        
        self.spatial_tracker = SpatialPossessionTracker(
            calibrator=self.field_calibrator,
            zone_model=zone_model,
            enable_heatmaps=True
        )
    
    def process_chunk(self, ...):
        for i, frame in enumerate(frames):
            # ... tracking y detección existente ...
            
            # NUEVO: Actualizar tracker espacial
            spatial_state = self.spatial_tracker.update(
                ball_pos=ball_bbox,
                players=tracked_objects,
                frame=frame  # Para calibración
            )
            
            # Añadir info espacial al chunk_stats
            chunk_stats['field_position'] = spatial_state.get('field_position')
            chunk_stats['zone_id'] = spatial_state.get('zone_id')
            chunk_stats['calibration_valid'] = spatial_state.get('calibration_valid')
```

### 6.2 Modificaciones en `match_state.py`

**Extender estado para incluir info espacial**:

```python
class PossessionState:
    def __init__(self):
        # ... atributos existentes ...
        
        # NUEVO: Estadísticas espaciales
        self.possession_by_zone = {
            0: {},  # {zone_id: frames}
            1: {}
        }
        self.heatmaps = {
            0: None,
            1: None
        }
```

### 6.3 Actualización en `match_analyzer.py`

**Exportar estadísticas espaciales en el summary**:

```python
def get_summary(self) -> Dict[str, Any]:
    summary = {
        # ... campos existentes ...
        
        # NUEVO: Estadísticas espaciales
        'spatial': {
            'possession_by_zone': self.spatial_tracker.get_zone_statistics(),
            'heatmaps': self.spatial_tracker.get_spatial_statistics()['heatmaps'],
            'calibration_quality': 'good' if self.field_calibrator.has_valid_calibration() else 'poor'
        }
    }
    return summary
```

---

## 7. Datos Adicionales a Guardar

### 7.1 En `chunk_output`

```python
chunk_output.chunk_stats = {
    # ... stats existentes ...
    'field_position': (x_field, y_field) or None,
    'zone_id': zone_id,
    'calibration_valid': bool,
    'homography_confidence': float
}
```

### 7.2 En `match_summary.json`

```json
{
  "possession": {
    "percent_by_team": {...},
    "spatial": {
      "zones": [
        {
          "zone_id": 0,
          "zone_name": "defensive_left",
          "team_0_percent": 35.2,
          "team_1_percent": 15.8
        },
        ...
      ],
      "partition_type": "thirds_lanes"
    }
  }
}
```

### 7.3 En `heatmaps_XXX.npz`

```python
np.savez(
    'heatmaps.npz',
    team_0=heatmap_team0,  # Shape: (H, W)
    team_1=heatmap_team1,
    metadata={
        'resolution': (50, 34),
        'field_dims': (105, 68),
        'normalization': 'max'
    }
)
```

---

## 8. Comportamiento con Vistas Parciales

### 8.1 Escenario: Cámara Solo Ve Medio Campo

**¿Qué ocurre?**

1. **Detección de líneas**: Solo se detectan líneas visibles
2. **Homografía**: Mapea correctamente la porción visible
3. **Zonas**: 
   - Zonas visibles acumulan normalmente
   - Zonas no visibles tienen tiempo = 0
4. **Resultado**: Heatmap muestra concentración en área visible

**Ejemplo**:
- Vista típica de área: solo se calibra tercio defensivo
- Si Team 0 defiende: su heatmap se concentra ahí
- Team 1 ataca: su heatmap también en esa zona

### 8.2 Escenario: Múltiples Cámaras

**Si hay varias cámaras** (broadcast con cambios):

1. **Detección automática de cambio**:
   - Monitorear confidence de homografía
   - Si cae drásticamente → probable cambio de cámara

2. **Re-calibración**:
   ```python
   if homography_confidence < 0.3:
       calibrator.reset()  # Forzar nueva estimación
   ```

3. **Fusión espacial**:
   - Todas las cámaras calibran contra el mismo `FieldModel`
   - Posiciones se acumulan en el mismo sistema de coordenadas
   - Resultado: cobertura completa del campo

---

## 9. Limitaciones y Trabajo Futuro

### 9.1 Limitaciones Actuales

1. **Matching heurístico**: 
   - Funciona bien en broadcast estándar
   - Puede fallar en ángulos muy inusuales

2. **Sin modelo de aprendizaje profundo**:
   - Detector de líneas clásico (LSD) es rápido pero limitado
   - Podría mejorarse con CNN para keypoint detection

3. **Calibración por frame**:
   - Re-estima cada N frames (costoso)
   - Podría optimizarse con tracking de homografía

### 9.2 Mejoras Futuras

**Corto plazo**:
- ✅ Implementar matching geométrico robusto
- ✅ Añadir visualización de zonas en tiempo real
- ✅ Exportar heatmaps en formato imagen

**Medio plazo**:
- 🔄 Entrenar modelo DL para keypoint detection
  - Dataset: SoccerNet, WorldCup 2014
  - Arquitectura: HRNet o similar
  
- 🔄 Tracking de homografía con filtro de Kalman
  - Reducir re-cómputos
  - Predecir H en frames intermedios

**Largo plazo**:
- 📋 Calibración multi-cámara con fusión
- 📋 Estimación de profundidad (pseudo-3D)
- 📋 Tracking de patrones de movimiento colectivo

---

## 10. Ejemplo Completo de Uso

```python
from modules.field_calibration import FieldCalibrator
from modules.field_model import ZoneModel, FieldModel
from modules.spatial_possession_tracker import SpatialPossessionTracker

# 1. Inicializar componentes
field_model = FieldModel()
calibrator = FieldCalibrator(field_model=field_model)

zone_model = ZoneModel(
    field_model,
    partition_type='thirds_lanes'
)

tracker = SpatialPossessionTracker(
    calibrator=calibrator,
    zone_model=zone_model,
    enable_heatmaps=True
)

# 2. Procesar video
for frame_idx, frame in enumerate(video_frames):
    # Detecciones y tracking (ya existente)
    ball_pos, players = detect_and_track(frame)
    
    # Actualizar tracker espacial
    state = tracker.update(ball_pos, players, frame)
    
    # Opcional: visualizar calibración
    if frame_idx % 100 == 0:
        vis = calibrator.visualize_calibration(frame)
        cv2.imshow('Calibration', vis)

# 3. Obtener resultados
spatial_stats = tracker.get_spatial_statistics()
zone_stats = tracker.get_zone_statistics()

# 4. Exportar heatmaps
heatmap_0 = tracker.export_heatmap(team_id=0)
heatmap_1 = tracker.export_heatmap(team_id=1)

np.savez('heatmaps.npz', team_0=heatmap_0, team_1=heatmap_1)

# 5. Generar visualización final
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(heatmap_0, cmap='Reds', aspect='auto')
ax1.set_title('Team 0 Possession Heatmap')
ax2.imshow(heatmap_1, cmap='Blues', aspect='auto')
ax2.set_title('Team 1 Possession Heatmap')
plt.savefig('possession_heatmaps.png')
```

---

## 11. Testing y Validación

### 11.1 Tests Unitarios

```python
# test_field_calibration.py
def test_homography_estimation():
    calibrator = FieldCalibrator()
    
    # Frame de test con líneas visibles
    frame = cv2.imread('test_frame.jpg')
    H = calibrator.estimate_homography(frame)
    
    assert H is not None
    assert H.shape == (3, 3)

def test_reproyection_accuracy():
    # Usar imagen con keypoints conocidos
    known_correspondences = [...]
    
    for img_pt, field_pt in known_correspondences:
        field_pt_estimated = calibrator.image_to_field(*img_pt)
        error = np.linalg.norm(
            np.array(field_pt) - np.array(field_pt_estimated)
        )
        assert error < 1.0  # <1 metro de error
```

### 11.2 Validación Visual

```python
# validate_calibration.py
def visualize_calibration_on_dataset():
    for video_path in test_videos:
        calibrator = FieldCalibrator()
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        # Estimar y visualizar
        H = calibrator.estimate_homography(frame)
        vis = calibrator.visualize_calibration(frame)
        
        # Mostrar para inspección manual
        cv2.imshow(f'{video_path}', vis)
        cv2.waitKey(0)
```

---

## 12. Performance y Optimización

### 12.1 Métricas Actuales (Estimadas)

- **Detección de líneas**: ~10ms por frame (640×480)
- **Estimación de homografía**: ~5ms (con 10+ correspondencias)
- **Reproyección**: <1ms por punto
- **Actualización de heatmap**: <1ms

**Total overhead**: ~15-20ms por frame → ~50 FPS sostenible

### 12.2 Optimizaciones Posibles

1. **Calibración esporádica**:
   - Solo re-estimar cada 30 frames (1 segundo)
   - Usar filtro temporal entre estimaciones

2. **ROI para detección de líneas**:
   - Ignorar zonas sin líneas (cielo, gradas)
   - Reducir área de búsqueda en 50%

3. **Procesamiento paralelo**:
   - Calibración en thread separado
   - Pipeline de análisis principal no se bloquea

4. **Caching**:
   - Almacenar homografías por escena
   - Detectar cambios de cámara y re-usar

---

## Conclusión

Este sistema proporciona **calibración automática completa** sin intervención manual, permitiendo:

✅ Reproyección precisa de posiciones jugador → campo  
✅ Análisis espacial de posesión por zonas  
✅ Generación de heatmaps automáticos  
✅ Manejo robusto de vistas parciales  
✅ Integración mínima con pipeline existente  

El diseño es **modular y extensible**, permitiendo mejoras futuras sin cambios drásticos en la arquitectura base.
