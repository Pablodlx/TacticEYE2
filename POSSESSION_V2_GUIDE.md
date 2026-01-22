# PossessionTrackerV2 - Especificación y Guía de Integración

## 📋 Especificación de la API

### Constructor

```python
PossessionTrackerV2(
    fps: float = 30.0,
    hysteresis_frames: int = 5,
    max_teams: int = 2
)
```

**Parámetros:**
- `fps`: Frames por segundo del vídeo (para conversión a segundos)
- `hysteresis_frames`: Frames consecutivos necesarios para confirmar cambio de posesión
- `max_teams`: Número máximo de equipos (típicamente 2)

---

### Método Principal: `update()`

```python
def update(frame_id: int, ball_owner_team: Optional[int]) -> None
```

**Entrada:**
- `frame_id` (int): ID del frame actual (debe ser monotónicamente creciente)
- `ball_owner_team` (int | None): 
  - `0` o `1`: Equipo del jugador que posee el balón
  - `None`: Balón en disputa, rebote, o sin poseedor claro

**Comportamiento:**
- Si `ball_owner_team is None`: el tiempo se asigna al equipo actual
- Si `ball_owner_team != current_possession_team`: se aplica histeresis
- El cambio se confirma solo tras `hysteresis_frames` consecutivos

**Regla fundamental:** TODO el tiempo siempre está asignado a un equipo (nunca hay tiempo "sin asignar")

---

### Métodos de Consulta

#### 1. `get_possession_stats()`

```python
def get_possession_stats() -> Dict[str, any]
```

**Retorna:**
```python
{
    'total_frames': int,              # Total de frames procesados
    'total_seconds': float,           # Tiempo total en segundos
    'possession_frames': {            # Frames por equipo
        0: int,
        1: int
    },
    'possession_seconds': {           # Segundos por equipo
        0: float,
        1: float
    },
    'possession_percent': {           # Porcentaje por equipo
        0: float,
        1: float
    }
}
```

**Ejemplo:**
```python
stats = tracker.get_possession_stats()
print(f"Team 0: {stats['possession_percent'][0]:.1f}%")
print(f"Team 1: {stats['possession_percent'][1]:.1f}%")
```

---

#### 2. `get_possession_timeline()`

```python
def get_possession_timeline() -> List[Tuple[int, int, int]]
```

**Retorna:**
Lista de tuplas `(start_frame, end_frame, team_id)` representando segmentos de posesión continua.

**Ejemplo:**
```python
timeline = tracker.get_possession_timeline()
for start, end, team in timeline:
    duration = end - start
    print(f"Frames {start}-{end}: Team {team} ({duration} frames)")
```

---

#### 3. `get_current_possession()`

```python
def get_current_possession() -> Dict[str, any]
```

**Retorna:**
```python
{
    'team': Optional[int],     # Equipo actual en posesión (None si no inicializado)
    'frame': int,              # Frame actual
    'initialized': bool        # Si el tracker está inicializado
}
```

---

#### 4. `reset()`

```python
def reset() -> None
```

Resetear el tracker a estado inicial (útil para procesar múltiples partidos).

---

## 🔗 Integración con Pipeline Existente

### Escenario 1: Integración con ReID Tracker

```python
from modules.possession_tracker_v2 import PossessionTrackerV2
from modules.reid_tracker import ReIDTracker

# Inicializar trackers
reid_tracker = ReIDTracker()
possession_tracker = PossessionTrackerV2(fps=30.0, hysteresis_frames=5)

# Variables para detección de poseedor
ball_pos = None
closest_player_team = None

# Loop principal
for frame_id, frame in enumerate(video_frames):
    # 1. Detección YOLO
    detections = yolo_model(frame)
    
    # 2. Tracking
    tracks = reid_tracker.update(frame, detections)
    
    # 3. Encontrar balón
    for det in detections:
        if det.class_id == 1:  # ball
            ball_pos = det.center
            break
    
    # 4. Encontrar jugador más cercano al balón
    if ball_pos is not None:
        min_dist = float('inf')
        closest_player_team = None
        
        for track_id, bbox, class_id in tracks:
            if class_id in [0, 3]:  # player or goalkeeper
                # Obtener team del track
                team_id = reid_tracker.active_tracks[track_id].team_id
                
                # Calcular distancia
                player_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                dist = math.hypot(ball_pos[0] - player_center[0], 
                                 ball_pos[1] - player_center[1])
                
                # Umbral de distancia máxima
                if dist < 150 and dist < min_dist:
                    min_dist = dist
                    closest_player_team = team_id
    
    # 5. Actualizar posesión
    possession_tracker.update(frame_id, closest_player_team)
    
    # 6. Visualizar
    stats = possession_tracker.get_possession_stats()
    cv2.putText(frame, 
                f"Possession: T0={stats['possession_percent'][0]:.1f}% T1={stats['possession_percent'][1]:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

# Resultados finales
final_stats = possession_tracker.get_possession_stats()
print(f"\n=== ESTADÍSTICAS FINALES ===")
print(f"Team 0: {final_stats['possession_seconds'][0]:.1f}s ({final_stats['possession_percent'][0]:.1f}%)")
print(f"Team 1: {final_stats['possession_seconds'][1]:.1f}s ({final_stats['possession_percent'][1]:.1f}%)")
```

---

### Escenario 2: Integración Simplificada (Sin ReID)

```python
from modules.possession_tracker_v2 import PossessionTrackerV2

possession_tracker = PossessionTrackerV2(fps=30.0, hysteresis_frames=5)

for frame_id, frame in enumerate(video_frames):
    # Detección YOLO
    detections = yolo_model(frame)
    
    # Simplificado: asignar equipo por posición en el campo
    ball_owner_team = None
    ball_pos = None
    
    # Encontrar balón
    for det in detections:
        if det.class_id == 1:
            ball_pos = (det.x_center, det.y_center)
            break
    
    # Heurística simple: equipo según mitad del campo
    if ball_pos is not None:
        # Buscar jugador más cercano
        min_dist = float('inf')
        for det in detections:
            if det.class_id in [0, 3]:  # player
                dist = math.hypot(ball_pos[0] - det.x_center, 
                                 ball_pos[1] - det.y_center)
                if dist < 150 and dist < min_dist:
                    min_dist = dist
                    # Asignar equipo por posición X
                    ball_owner_team = 0 if det.x_center < frame.shape[1]/2 else 1
    
    # Actualizar posesión
    possession_tracker.update(frame_id, ball_owner_team)
```

---

## 🎯 Características Clave

### ✅ Reglas Deterministas

1. **Nunca hay tiempo sin asignar**: Cada frame pertenece a un equipo
2. **Frames con `ball_owner=None`**: Se asignan al último equipo con posesión
3. **Histeresis**: Evita cambios erráticos por ruido en la detección

### ✅ Validaciones Garantizadas

```python
stats = tracker.get_possession_stats()

# La suma siempre es 100%
assert sum(stats['possession_percent'].values()) == 100.0

# Todo el tiempo está asignado
total_assigned = sum(stats['possession_frames'].values())
assert total_assigned == stats['total_frames']
```

---

## 📊 Casos de Uso

### Caso 1: Balón en disputa sostenida

```python
# Frames 1-10: Team 0 posee
tracker.update(1, 0)
tracker.update(2, 0)
# ...
tracker.update(10, 0)

# Frames 11-20: Disputa (ball_owner=None)
# → Se asigna a Team 0 (último poseedor)
tracker.update(11, None)
tracker.update(12, None)
# ...
tracker.update(20, None)

stats = tracker.get_possession_stats()
# Team 0 tiene 20 frames (100%)
```

---

### Caso 2: Cambio de posesión con histeresis

```python
tracker = PossessionTrackerV2(hysteresis_frames=5)

# Team 0 posee
tracker.update(1, 0)
tracker.update(2, 0)

# Team 1 intenta ganar posesión
tracker.update(3, 1)  # 1er frame
tracker.update(4, 1)  # 2do frame
tracker.update(5, 1)  # 3er frame
tracker.update(6, 1)  # 4to frame

# Aún no hay cambio (necesita 5 frames consecutivos)
assert tracker.current_possession_team == 0

tracker.update(7, 1)  # 5to frame → CAMBIO CONFIRMADO

assert tracker.current_possession_team == 1
```

---

### Caso 3: Cambio interrumpido por disputa

```python
tracker.update(1, 0)  # Team 0 posee

# Team 1 intenta ganar
tracker.update(2, 1)
tracker.update(3, 1)
tracker.update(4, 1)

# Disputa interrumpe el intento
tracker.update(5, None)

# El cambio NO se confirma, vuelve a Team 0
assert tracker.current_possession_team == 0
```

---

## 🧪 Tests Unitarios Sugeridos

```python
def test_all_time_assigned():
    """Todo el tiempo debe estar asignado a un equipo."""
    tracker = PossessionTrackerV2()
    
    tracker.update(1, 0)
    tracker.update(2, None)
    tracker.update(3, None)
    tracker.update(4, 1)
    
    stats = tracker.get_possession_stats()
    total_assigned = sum(stats['possession_frames'].values())
    
    assert total_assigned == stats['total_frames']


def test_percentages_sum_100():
    """Los porcentajes deben sumar 100%."""
    tracker = PossessionTrackerV2()
    
    for i in range(1, 101):
        tracker.update(i, i % 2)  # Alterna entre Team 0 y 1
    
    stats = tracker.get_possession_stats()
    total_percent = sum(stats['possession_percent'].values())
    
    assert abs(total_percent - 100.0) < 0.01


def test_hysteresis_prevents_rapid_changes():
    """Histeresis debe prevenir cambios rápidos."""
    tracker = PossessionTrackerV2(hysteresis_frames=5)
    
    tracker.update(1, 0)
    tracker.update(2, 1)  # Solo 1 frame de Team 1
    tracker.update(3, 0)
    
    # No debe haber cambiado
    assert tracker.current_possession_team == 0


def test_none_assigns_to_current():
    """Frames con None se asignan al equipo actual."""
    tracker = PossessionTrackerV2()
    
    tracker.update(1, 0)
    tracker.update(2, None)
    tracker.update(3, None)
    
    stats = tracker.get_possession_stats()
    
    # Los 3 frames deben ser de Team 0
    assert stats['possession_frames'][0] == 3
    assert stats['possession_frames'][1] == 0
```

---

## 🚀 Ventajas sobre PossessionTracker Original

| Característica | PossessionTracker (original) | PossessionTrackerV2 |
|----------------|------------------------------|---------------------|
| Estados | 5 estados (`contested`, `dead_ball`, etc.) | 2 estados (Team 0, Team 1) |
| Tiempo sin asignar | Posible | **Nunca** |
| Complejidad | Alta (sliding windows, oclusión) | Baja (reglas simples) |
| Porcentajes | Pueden no sumar 100% | **Siempre suman 100%** |
| Histeresis | Implícita en ventanas | **Explícita y configurable** |
| Depuración | Difícil (muchos estados) | **Fácil** (lógica determinista) |

---

## 📝 Notas de Implementación

### Limitaciones Conocidas

1. **Requiere `frame_id` monotónico**: No soporta frames fuera de orden
2. **Sin interpolación**: Saltos grandes en `frame_id` se asignan al equipo actual
3. **Histeresis fija**: Requiere N frames **consecutivos** (no acumulativos)

### Posibles Mejoras Futuras

1. **Histeresis acumulativa**: Contar votos en ventana deslizante en lugar de consecutivos
2. **Confianza por frame**: Ponderar frames según calidad de detección
3. **Exportar a JSON/CSV**: Método `export_timeline(filename)` para análisis posterior

---

## 📚 Referencias

- Módulo: `modules/possession_tracker_v2.py`
- Tests: Ejecutar `python modules/possession_tracker_v2.py`
- Integración: Ver ejemplos arriba

---

**Autor**: TacticEYE2 Team  
**Fecha**: 2026-01-21  
**Versión**: 2.0
