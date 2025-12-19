# Ejemplos de Uso - TacticEYE2
# ============================

## 📖 Ejemplos Prácticos

### 1. Análisis Básico
```bash
# Analizar un partido completo con configuración por defecto
python analyze_match.py --video partido.mp4
```

### 2. Demo Rápido (10 segundos)
```bash
# Análisis rápido para testing
python quick_demo.py
```

### 3. Análisis Personalizado
```bash
# Control total de parámetros
python analyze_match.py \
    --video partido.mp4 \
    --model weights/best.pt \
    --output ./resultados \
    --conf 0.4 \
    --iou 0.6 \
    --calibration-frame 200 \
    --max-frames 3000
```

### 4. Procesamiento en Batch (sin preview)
```bash
# Procesar varios vídeos sin interfaz gráfica
python analyze_match.py --video partido1.mp4 --no-preview
python analyze_match.py --video partido2.mp4 --no-preview
python analyze_match.py --video partido3.mp4 --no-preview
```

### 5. Visualización de Heatmaps
```bash
# Visualizar heatmaps generados
python visualize_heatmaps.py outputs/heatmaps_20250101_120000.npz

# Guardar imágenes de heatmaps
python visualize_heatmaps.py outputs/heatmaps_20250101_120000.npz --output ./heatmap_images
```

### 6. Utilidades - Exportar a Excel
```bash
# Convertir CSVs/JSONs a Excel
python utils.py excel ./outputs
```

### 7. Utilidades - Extraer Clip
```bash
# Extraer un segmento específico del vídeo
python utils.py clip partido.mp4 120.5 185.0 --output jugada_gol.mp4
# (desde segundo 120.5 hasta 185.0)
```

### 8. Utilidades - Vídeo Comparación
```bash
# Crear vídeo lado a lado: original vs analizado
python utils.py compare partido.mp4 outputs/analyzed_partido.mp4 --output comparison.mp4
```

### 9. Utilidades - Mostrar Estadísticas
```bash
# Ver resumen bonito de estadísticas en terminal
python utils.py stats ./outputs
```

---

## 🐍 Uso Programático

### Ejemplo 1: Integración en Script Python

```python
from analyze_match import TacticEYE2

# Inicializar sistema
analyzer = TacticEYE2(
    model_path='weights/best.pt',
    video_path='partido.mp4',
    output_dir='./mi_analisis',
    conf_threshold=0.35
)

# Analizar
analyzer.analyze_video(
    calibration_frame=100,
    show_preview=False,
    max_frames=1000
)
```

### Ejemplo 2: Procesamiento Frame a Frame

```python
import cv2
from analyze_match import TacticEYE2

analyzer = TacticEYE2('weights/best.pt', 'video.mp4')

cap = cv2.VideoCapture('video.mp4')

# Calibrar con primer frame
ret, frame = cap.read()
analyzer.calibrate_field(frame)

# Procesar frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Procesar
    result_frame = analyzer.process_frame(frame)
    
    # Tu código aquí...
    cv2.imshow('Resultado', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Ejemplo 3: Acceso a Estadísticas en Tiempo Real

```python
from analyze_match import TacticEYE2

analyzer = TacticEYE2('weights/best.pt', 'video.mp4')
# ... procesar frames ...

# Obtener estadísticas actuales
possession = analyzer.match_stats.get_possession_percentage()
print(f"Posesión: Local {possession[0]:.1f}% - Visitante {possession[1]:.1f}%")

# Top jugadores
top_runners = analyzer.match_stats.get_top_players('distance', top_n=5)
for track_id, team_id, distance_km in top_runners:
    print(f"Jugador #{track_id} (Equipo {team_id}): {distance_km:.2f} km")

# Estadísticas de un jugador específico
player_stats = analyzer.match_stats.get_player_stats(track_id=5)
if player_stats:
    print(f"Velocidad máxima: {player_stats.max_speed:.1f} km/h")
    print(f"Distancia total: {player_stats.total_distance:.1f} m")
```

### Ejemplo 4: Exportación Personalizada

```python
from modules.data_exporter import DataExporter
import time

exporter = DataExporter(output_dir='./custom_output')

# Exportar posiciones manualmente
exporter.add_position_record(
    frame=100,
    timestamp=time.time(),
    track_id=5,
    team_id=0,
    pos_pixels=(640, 480),
    pos_meters=(52.5, 34.0),
    velocity_kmh=18.5
)

# Exportar evento
exporter.add_event_record(
    timestamp=time.time(),
    frame=100,
    event_type='pass',
    team_id=0,
    player_id=5,
    position=(52.5, 34.0),
    success=True,
    metadata={'receiver_id': 7, 'distance': 15.2}
)

# Guardar
exporter.export_positions_csv('mi_partido_posiciones.csv')
exporter.export_events_json('mi_partido_eventos.json')
```

### Ejemplo 5: Calibración Manual del Campo

```python
import numpy as np
from analyze_match import TacticEYE2

analyzer = TacticEYE2('weights/best.pt', 'video.mp4')

# Definir esquinas manualmente (si detección automática falla)
# Orden: [top-left, top-right, bottom-right, bottom-left]
manual_corners = np.array([
    [100, 50],    # top-left
    [1800, 80],   # top-right
    [1850, 1000], # bottom-right
    [50, 980]     # bottom-left
], dtype=np.float32)

import cv2
frame = cv2.imread('frame_referencia.jpg')
analyzer.calibrate_field(frame, manual_corners=manual_corners)
```

---

## 🎯 Casos de Uso Avanzados

### 1. Análisis de Múltiples Partidos en Lote

```bash
#!/bin/bash
# batch_analyze.sh

for video in partidos/*.mp4; do
    echo "Analizando: $video"
    python analyze_match.py \
        --video "$video" \
        --output "./outputs/$(basename "$video" .mp4)" \
        --no-preview
done

echo "✅ Análisis completado para todos los partidos"
```

### 2. Comparación de Rendimiento Entre Partidos

```python
import pandas as pd
import glob

# Cargar todos los CSVs de posiciones
all_csvs = glob.glob('outputs/*/positions_*.csv')

for csv_file in all_csvs:
    df = pd.read_csv(csv_file)
    
    # Análisis por jugador
    for track_id in df['track_id'].unique():
        player_data = df[df['track_id'] == track_id]
        
        total_distance = player_data['velocity_kmh'].sum() / 3.6 * 0.033  # Aprox
        max_speed = player_data['velocity_kmh'].max()
        
        print(f"Jugador #{track_id}: {total_distance:.1f}m, Max: {max_speed:.1f} km/h")
```

### 3. Detección de Eventos Personalizados

```python
# Detectar sprints (velocidad > 25 km/h por > 3 segundos)
def detect_sprints(csv_file, speed_threshold=25.0, duration_threshold=3.0):
    df = pd.read_csv(csv_file)
    sprints = []
    
    for track_id in df['track_id'].unique():
        player_data = df[df['track_id'] == track_id].sort_values('frame')
        
        in_sprint = False
        sprint_start = None
        
        for idx, row in player_data.iterrows():
            if row['velocity_kmh'] > speed_threshold:
                if not in_sprint:
                    sprint_start = row['timestamp']
                    in_sprint = True
            else:
                if in_sprint:
                    sprint_duration = row['timestamp'] - sprint_start
                    if sprint_duration >= duration_threshold:
                        sprints.append({
                            'player': track_id,
                            'start': sprint_start,
                            'duration': sprint_duration
                        })
                    in_sprint = False
    
    return sprints
```

---

## 📊 Análisis de Datos Exportados

### Jupyter Notebook Example

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df_pos = pd.read_csv('outputs/positions_20250101_120000.csv')
df_pos['team_name'] = df_pos['team_id'].map({0: 'Local', 1: 'Visitante', 2: 'Árbitro'})

# 1. Distancia total por jugador
player_distances = df_pos.groupby(['track_id', 'team_name'])['velocity_kmh'].agg([
    ('total_distance_approx', lambda x: x.sum() / 3.6 * 0.033),
    ('max_speed', 'max'),
    ('avg_speed', 'mean')
]).reset_index()

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Distancias
player_distances.plot(x='track_id', y='total_distance_approx', 
                     kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Distancia Total por Jugador')
axes[0].set_ylabel('Metros')

# Velocidades
player_distances.plot(x='track_id', y='max_speed', 
                     kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Velocidad Máxima por Jugador')
axes[1].set_ylabel('km/h')

plt.tight_layout()
plt.savefig('player_performance.png', dpi=300)
plt.show()

# 2. Mapa de posiciones promedio
fig, ax = plt.subplots(figsize=(10, 7))

for team_id, color in [(0, 'blue'), (1, 'red')]:
    team_data = df_pos[df_pos['team_id'] == team_id]
    
    for track_id in team_data['track_id'].unique():
        player_data = team_data[team_data['track_id'] == track_id]
        avg_x = player_data['x_meters'].mean()
        avg_y = player_data['y_meters'].mean()
        
        ax.scatter(avg_x, avg_y, c=color, s=200, alpha=0.6)
        ax.text(avg_x, avg_y, f'#{track_id}', 
               ha='center', va='center', fontweight='bold')

ax.set_xlim(0, 105)
ax.set_ylim(0, 68)
ax.set_xlabel('Longitud (m)')
ax.set_ylabel('Ancho (m)')
ax.set_title('Posiciones Promedio de Jugadores')
ax.grid(True, alpha=0.3)
plt.savefig('average_positions.png', dpi=300)
plt.show()
```

---

## 🔧 Troubleshooting Tips

### Problema: Calibración falla
```python
# Probar con diferentes frames
for frame_num in [50, 100, 200, 500]:
    cap = cv2.VideoCapture('video.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if analyzer.calibrate_field(frame):
        print(f"✅ Calibrado en frame {frame_num}")
        break
```

### Problema: IDs inconsistentes
```python
# Ajustar parámetros del tracker
from modules.reid_tracker import ReIDTracker

tracker = ReIDTracker(
    max_age=120,  # Aumentar
    max_lost_time=90.0,  # Aumentar
    similarity_threshold=0.7  # Más estricto
)
```

### Problema: Equipos mal clasificados
```python
# Ajustar clasificador
from modules.team_classifier import TeamClassifier

classifier = TeamClassifier(n_teams=3)
# Procesar más frames antes de confiar en clasificación
# O implementar corrección manual post-procesamiento
```

---

¡Esto cubre los casos de uso más comunes! Para más ejemplos, revisa el código fuente de cada módulo.
