# Test de Calibración Automática y Tracking Espacial

## Instalación de Dependencias

Si aún no lo has hecho, instala las dependencias necesarias:

```bash
pip install opencv-python numpy matplotlib
```

## Uso Básico

### Prueba Simple

```bash
python test_spatial_tracking.py sample_match.mp4
```

Esto ejecutará el análisis con configuración por defecto:
- Zonas: tercios × carriles (9 zonas)
- Heatmaps: activados (resolución 50×34)
- Calibración automática: activada

### Ver Estadísticas Espaciales Detalladas

```bash
python test_spatial_tracking.py sample_match.mp4 --zones thirds_lanes
```

### Usar Rejilla Personalizada

```bash
python test_spatial_tracking.py sample_match.mp4 --zones grid --zone-nx 8 --zone-ny 6
```

Esto creará una rejilla de 8×6 = 48 zonas.

### Deshabilitar Heatmaps (Para Mayor Velocidad)

```bash
python test_spatial_tracking.py sample_match.mp4 --no-heatmaps
```

### Procesar Solo Primeros N Frames

```bash
python test_spatial_tracking.py sample_match.mp4 --max-frames 300
```

## Opciones Disponibles

```
Opciones principales:
  --model PATH              Modelo YOLO (default: weights/best.pt)
  --output-dir DIR          Directorio de salida (default: outputs_spatial_test)
  --batch-seconds FLOAT     Tamaño de batch en segundos (default: 3.0)

Configuración de zonas:
  --zones TYPE              Tipo de partición: grid, thirds, thirds_lanes
  --zone-nx INT             Divisiones en X para grid (default: 6)
  --zone-ny INT             Divisiones en Y para grid (default: 4)

Heatmaps:
  --no-heatmaps            Deshabilitar generación de heatmaps
  --heatmap-resolution WxH  Resolución como "width,height" (default: 50,34)

Otros:
  --disable-spatial        Deshabilitar tracking espacial completamente
  --max-frames N          Número máximo de frames a procesar
  --visualize             Mostrar visualización en tiempo real
```

## Outputs Generados

Después de ejecutar el análisis, encontrarás:

### 1. Directorio de Outputs (`outputs_spatial_test/`)

```
outputs_spatial_test/
└── sample_match/
    ├── detections_batch_0000.json
    ├── positions_batch_0000.json
    ├── events_batch_0000.json
    ├── stats_batch_0000.json
    ├── ... (más batches)
    └── sample_match_heatmaps.npz
```

### 2. Heatmaps (`*_heatmaps.npz`)

Archivo NumPy comprimido con:
- `team_0_heatmap`: Mapa de calor Team 0
- `team_1_heatmap`: Mapa de calor Team 1
- `possession_by_zone_team_0`: Tiempo por zona
- `possession_by_zone_team_1`: Tiempo por zona
- `zone_percentages_team_0`: Porcentajes por zona
- `zone_percentages_team_1`: Porcentajes por zona
- `metadata`: Info sobre resolución, partición, dimensiones

### 3. Estadísticas por Batch

Cada `stats_batch_XXXX.json` incluye:

```json
{
  "frames_processed": 90,
  "detections_count": 234,
  "possession_team": 0,
  "possession_player": 5,
  "spatial": {
    "calibration_valid": true,
    "possession_by_zone": {
      "0": [12, 5, 8, ...],
      "1": [3, 15, 10, ...]
    },
    "zone_percentages": {
      "0": [15.2, 6.3, 10.1, ...],
      "1": [3.8, 19.0, 12.7, ...]
    },
    "zone_partition_type": "thirds_lanes",
    "num_zones": 9
  }
}
```

## Interpretar Resultados

### Consola

Durante la ejecución verás:

```
============================================================
TEST DE CALIBRACIÓN AUTOMÁTICA Y TRACKING ESPACIAL
============================================================

Video: sample_match.mp4
Modelo: weights/best.pt
Output: outputs_spatial_test
Tracking espacial: HABILITADO
  - Tipo de zonas: thirds_lanes
  - Heatmaps: HABILITADOS
  - Resolución: 50×34

Iniciando análisis...

[sample_match] Iniciando nuevo análisis...
[sample_match] Abriendo fuente: uploaded_file
[sample_match] Video: 1280x720 @ 30.0 fps
✓ Inicializando calibración automática de campo...
  - Modelo de zonas: thirds_lanes
  - Número de zonas: 9
  - Heatmaps: Activados

[sample_match] Batch 0: frames 0-89 (90 frames)
  ✓ Procesado en 2.45s (36.7 fps, 1.22x realtime)
  ✓ Detecciones: 234
  ✓ Eventos: 2
  ✓ Posesión: Team 0

...

============================================================
ANÁLISIS COMPLETADO
============================================================

RESUMEN GENERAL:
  Frames procesados: 900
  Duración: 30.0 segundos
  Batches: 10

POSESIÓN GLOBAL:
  Team 0: 55.2% (16.6s)
  Team 1: 44.8% (13.4s)

PASES:
  Team 0: 12 pases
  Team 1: 8 pases

ESTADÍSTICAS ESPACIALES:
  ✓ Calibración de campo: VÁLIDA
  ✓ Heatmaps exportados: outputs_spatial_test/sample_match_heatmaps.npz
  Tipo de partición: thirds_lanes
  Número de zonas: 9

  Top 3 zonas Team 0:
    - mid_center: 35.2%
    - def_center: 22.1%
    - off_center: 15.8%

  Top 3 zonas Team 1:
    - mid_center: 28.5%
    - off_center: 25.3%
    - mid_right: 18.2%
```

### Nombres de Zonas

**thirds_lanes** (9 zonas):
- `def_left`, `def_center`, `def_right`: Tercio defensivo
- `mid_left`, `mid_center`, `mid_right`: Tercio medio
- `off_left`, `off_center`, `off_right`: Tercio ofensivo

**thirds** (3 zonas):
- `defensive`, `middle`, `offensive`

**grid** (nx × ny zonas):
- `zone_0`, `zone_1`, ..., `zone_N`

## Visualizar Heatmaps

Para visualizar los heatmaps generados:

```python
import numpy as np
import matplotlib.pyplot as plt

# Cargar heatmaps
data = np.load('outputs_spatial_test/sample_match_heatmaps.npz')

hm_0 = data['team_0_heatmap']
hm_1 = data['team_1_heatmap']

# Visualizar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.imshow(hm_0, cmap='Greens', aspect='auto', origin='lower')
ax1.set_title('Team 0 Possession Heatmap')

ax2.imshow(hm_1, cmap='Reds', aspect='auto', origin='lower')
ax2.set_title('Team 1 Possession Heatmap')

plt.show()
```

## Troubleshooting

### "Calibración de campo: NO DISPONIBLE"

Esto puede ocurrir si:
- El video no muestra suficientes líneas del campo
- La vista es muy inusual (cenital, desde portería, etc.)
- Hay mucho ruido visual (gradas, logos, etc.)

**Solución**: 
- El sistema sigue funcionando usando fallback (última posición conocida)
- Las estadísticas globales (posesión %) siguen siendo correctas
- Solo las estadísticas espaciales pueden estar incompletas

### Análisis muy lento

**Optimizaciones**:
1. Deshabilitar heatmaps: `--no-heatmaps`
2. Reducir resolución de heatmap: `--heatmap-resolution 25,17`
3. Aumentar batch size: `--batch-seconds 5.0`
4. Procesar menos frames: `--max-frames 600`

### Error "CUDA out of memory"

Cambiar a CPU:
```bash
# Editar en el código o forzar con:
CUDA_VISIBLE_DEVICES="" python test_spatial_tracking.py sample_match.mp4
```

## Próximos Pasos

Una vez verificado que funciona correctamente, el sistema se puede integrar en:
1. Interfaz web (ya integrado en `app.py`)
2. API REST para procesamiento batch
3. Dashboard de visualización en tiempo real

## Notas Técnicas

- La calibración se actualiza cada 30 frames para eficiencia
- Los heatmaps usan kernel gaussiano para suavizado
- Las zonas se calculan en coordenadas de campo (metros)
- El sistema es resiliente a pérdida temporal de calibración
