# Sistema de Mapas de Calor para Análisis de Fútbol

Sistema completo para generar heatmaps de posición de jugadores proyectados a coordenadas de campo, resolviendo automáticamente la ambigüedad de flip horizontal cuando los keypoints detectados no distinguen entre izquierda y derecha.

## 📋 Características

- **Homografía por frame**: Estima matriz de transformación imagen→campo para cada frame
- **Resolución automática de flip**: Detecta y corrige orientación del campo (izq/der)
- **Acumulación espacial**: Genera heatmaps en coordenadas de campo normalizadas
- **Robusto a cámara móvil**: Funciona con pan/tilt/zoom variable
- **Visualización profesional**: Exporta heatmaps con matplotlib

## 🏗️ Arquitectura

```
modules/field_heatmap_system.py  # Sistema principal
├── FIELD_POINTS                 # Modelo teórico del campo (105x68m)
├── estimate_homography()        # Estima H de imagen→campo
├── flip_field_points()          # Transforma campo horizontalmente
├── homography_geom_error()      # Calcula error geométrico
├── estimate_homography_with_flip_resolution()  # H con detección de flip
├── project_points()             # Proyecta jugadores al campo
├── HeatmapAccumulator           # Acumula posiciones en cuadrícula
└── process_sequence()           # Pipeline completo
```

## 🚀 Uso Rápido

### 1. Importar el sistema

```python
from modules.field_heatmap_system import (
    FIELD_POINTS,
    HeatmapAccumulator,
    process_sequence
)
```

### 2. Preparar datos por frame

```python
# Formato de keypoints detectados
frames_keypoints = [
    [  # Frame 0
        {"cls_name": "midline_top_intersection", "xy": (960, 100), "conf": 0.95},
        {"cls_name": "bigarea_top_inner", "xy": (400, 300), "conf": 0.88},
        # ... más keypoints
    ],
    # ... más frames
]

# Formato de jugadores detectados
frames_players = [
    [  # Frame 0
        {"team_id": 0, "xy": (300, 400), "conf": 0.95},  # Team 0 (local)
        {"team_id": 1, "xy": (1200, 400), "conf": 0.94},  # Team 1 (visitante)
        # ... más jugadores
    ],
    # ... más frames
]
```

### 3. Procesar secuencia

```python
# Crear acumulador
accumulator = HeatmapAccumulator(
    field_length=105,  # Largo del campo (m)
    field_width=68,    # Ancho del campo (m)
    nx=105,            # Resolución X (celdas)
    ny=68              # Resolución Y (celdas)
)

# Procesar todos los frames
stats = process_sequence(
    frames_keypoints,
    frames_players,
    FIELD_POINTS,
    accumulator,
    verbose=True
)
```

### 4. Obtener heatmaps

```python
# Normalizado por valor máximo (0-1)
heatmap_team0 = accumulator.get_heatmap(0, normalize='max')
heatmap_team1 = accumulator.get_heatmap(1, normalize='max')

# Sin normalizar (conteos absolutos)
heatmap_raw = accumulator.get_heatmap(0, normalize=None)
```

### 5. Visualizar

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

im0 = axes[0].imshow(heatmap_team0, cmap='Reds', origin='lower', aspect='auto')
axes[0].set_title('Heatmap Team 0')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(heatmap_team1, cmap='Blues', origin='lower', aspect='auto')
axes[1].set_title('Heatmap Team 1')
plt.colorbar(im1, ax=axes[1])

plt.savefig('heatmaps.png', dpi=150)
```

## 📊 Keypoints Soportados

El sistema reconoce los siguientes keypoints (sin distinguir izq/der):

### Línea Central
- `midline_top_intersection` (52.5, 68)
- `midline_bottom_intersection` (52.5, 0)
- `center` (52.5, 34)

### Círculo Central
- `halfcircle_top` (52.5, 43.15)
- `halfcircle_bottom` (52.5, 24.85)

### Área Grande (Big Box)
- `bigarea_top_inner` (16.5, 54.15)
- `bigarea_bottom_inner` (16.5, 13.85)
- `bigarea_top_outter` (0, 54.15)
- `bigarea_bottom_outter` (0, 13.85)

### Área Pequeña (Small Box)
- `smallarea_top_inner` (5.5, 43.15)
- `smallarea_bottom_inner` (5.5, 24.85)
- `smallarea_top_outter` (0, 43.15)
- `smallarea_bottom_outter` (0, 24.85)

### Arcos de Penalti
- `top_arc_area_intersection` (11, 43.15)
- `bottom_arc_area_intersection` (11, 24.85)

### Esquinas
- `corner` (genérico, se mapea automáticamente)

## 🔬 Resolución de Flip Horizontal

El sistema resuelve automáticamente la ambigüedad de orientación del campo:

1. **Estima homografía normal**: Keypoints → coordenadas originales
2. **Estima homografía flipped**: Keypoints → coordenadas flipped (L-X, Y)
3. **Calcula error geométrico**: Compara distancias relativas proyectadas vs teóricas
4. **Selecciona mejor**: Usa la homografía con menor error

### Ejemplo de transformación flip:

```
Original:              Flipped:
(0, 0) ────> (105, 68) (105, 68) ────> (0, 0)
   │                      │
   │  bigarea_left        │  bigarea_right
   │                      │
```

## 🧪 Scripts de Prueba

### Test básico (datos sintéticos)

```bash
python modules/field_heatmap_system.py
```

Genera:
- `heatmap_example.png` - Visualización simple con 3 frames

### Test realista (30 frames simulados)

```bash
python test_heatmap_system.py
```

Genera:
- `test_heatmaps.png` - Visualización completa con 4 paneles:
  1. Heatmap Team 0 (rojo)
  2. Heatmap Team 1 (azul)
  3. Heatmap combinado
  4. Diferencia de presencia

## 📐 Matemáticas Clave

### Homografía (Imagen → Campo)

$$
\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = H \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

Donde:
- $(x, y)$ = coordenadas en imagen (píxeles)
- $(X, Y)$ = coordenadas en campo (metros)
- $H$ = matriz 3×3 estimada con RANSAC

### Error Geométrico

$$
E = \frac{1}{n} \sum_{i,j} \left| \frac{d_{proj}(i,j) - d_{theory}(i,j)}{d_{theory}(i,j)} \right|
$$

Donde:
- $d_{proj}$ = distancia entre keypoints proyectados
- $d_{theory}$ = distancia teórica en el modelo de campo
- Menor error → mejor orientación

## ⚙️ Parámetros de Configuración

### HeatmapAccumulator

```python
accumulator = HeatmapAccumulator(
    field_length=105,  # Largo del campo (m)
    field_width=68,    # Ancho del campo (m)
    nx=105,            # Resolución X: 1 celda = 1m
    ny=68              # Resolución Y: 1 celda = 1m
)
```

Recomendaciones:
- **Alta resolución** (nx=105, ny=68): 1 celda = 1m² → mapas detallados
- **Media resolución** (nx=42, ny=28): 1 celda = 2.5m² → balance
- **Baja resolución** (nx=21, ny=14): 1 celda = 5m² → visualización rápida

### estimate_homography_with_flip_resolution

```python
H, is_flipped = estimate_homography_with_flip_resolution(
    frame_keypoints,
    field_points,
    min_points=4,        # Mínimo de keypoints para H válida
    conf_threshold=0.4   # Confianza mínima de detecciones
)
```

## 📈 Estadísticas de Salida

```python
stats = process_sequence(...)
# {
#     'total_frames': 900,
#     'successful_frames': 837,
#     'flipped_frames': 421,
#     'success_rate': 0.93
# }
```

## 🎯 Casos de Uso

### 1. Análisis táctico

```python
# Comparar presencia territorial
diff = heatmap_team0 - heatmap_team1

# Identificar zonas calientes
hot_zones_team0 = np.where(heatmap_team0 > 0.7)
```

### 2. Informes automatizados

```python
# Calcular % de posesión territorial
total_team0 = heatmap_team0.sum()
total_team1 = heatmap_team1.sum()
possession_pct = total_team0 / (total_team0 + total_team1)
```

### 3. Exportar para herramientas externas

```python
# Guardar como NPZ
np.savez('heatmaps.npz',
         team0=heatmap_team0,
         team1=heatmap_team1,
         metadata={'nx': 105, 'ny': 68})
```

## 🐛 Troubleshooting

### Error: "Not enough keypoints"

**Causa**: Frame con menos de 4 keypoints detectados

**Solución**:
- Reducir `min_points=3` (menos robusto)
- Bajar `conf_threshold=0.3`
- Verificar que el modelo de keypoints funciona bien

### Error: "cv2.findHomography failed"

**Causa**: Keypoints colineales o muy cerca

**Solución**:
- Usar más tipos de keypoints (círculo + áreas)
- Aumentar RANSAC threshold en `estimate_homography()`

### Heatmaps vacíos

**Causa**: Homografías no se estiman correctamente

**Solución**:
```python
stats = process_sequence(..., verbose=True)
print(f"Success rate: {stats['success_rate']:.1%}")
# Si < 50%, revisar calidad de keypoints
```

## 📚 Referencias

- **Dimensiones FIFA**: [Laws of the Game](https://www.theifab.com/)
- **Homografía**: Hartley & Zisserman, "Multiple View Geometry"
- **RANSAC**: Fischler & Bolles, 1981

## 🤝 Contribuciones

Sistema desarrollado para TacticEYE2 - Análisis Táctico de Fútbol

**Autor**: TacticEYE2 Team  
**Fecha**: 2026-01-29  
**Versión**: 1.0.0

---

## 📝 Ejemplo Completo End-to-End

```python
from modules.field_heatmap_system import *

# 1. Preparar datos (simulación)
frames_keypoints = [...]  # Tu detector de keypoints
frames_players = [...]    # Tu detector de jugadores

# 2. Crear acumulador
accumulator = HeatmapAccumulator(nx=105, ny=68)

# 3. Procesar
stats = process_sequence(
    frames_keypoints,
    frames_players,
    FIELD_POINTS,
    accumulator
)

# 4. Obtener resultados
hm0 = accumulator.get_heatmap(0, normalize='max')
hm1 = accumulator.get_heatmap(1, normalize='max')

# 5. Visualizar
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.imshow(hm0, cmap='Reds')
plt.title('Team 0')
plt.subplot(122)
plt.imshow(hm1, cmap='Blues')
plt.title('Team 1')
plt.savefig('result.png')

print(f"✓ Procesados {stats['successful_frames']} frames")
print(f"✓ Tasa de éxito: {stats['success_rate']:.1%}")
```
