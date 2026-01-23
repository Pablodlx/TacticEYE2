# TacticEYE - Arquitectura de Micro-Batching

## 📋 Índice

1. [Visión General](#visión-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Módulos Principales](#módulos-principales)
4. [Flujo de Datos](#flujo-de-datos)
5. [Ejemplos de Uso](#ejemplos-de-uso)
6. [API Reference](#api-reference)
7. [Deployment](#deployment)

---

## 🎯 Visión General

TacticEYE implementa un sistema de análisis de partidos de fútbol basado en **micro-batching**, que permite:

### ✨ Características Principales

- **Análisis casi en tiempo real**: Procesa video en chunks de 2-5 segundos
- **Múltiples fuentes**: Archivos locales, YouTube (VOD/Live), HLS, RTMP, Veo
- **Estado persistente**: Recuperación automática ante fallos
- **Resultados parciales**: Consulta estadísticas durante el partido
- **Escalable**: Diseño preparado para workers distribuidos

### 🎪 Casos de Uso

1. **Análisis post-partido**: Video completo subido → Análisis batch completo
2. **Análisis en vivo**: YouTube Live / HLS → Análisis continuo con lag de segundos
3. **Análisis diferido de streams**: VOD de partido grabado → Procesamiento eficiente
4. **Monitoreo múltiple**: Varios partidos en paralelo (con workers)

---

## 🏗️ Arquitectura del Sistema

```
┌──────────────────────────────────────────────────────────────┐
│                      VIDEO SOURCES                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Local   │  │ YouTube  │  │   HLS    │  │   RTMP   │    │
│  │  Files   │  │ VOD/Live │  │  Stream  │  │  Stream  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       └─────────────┴─────────────┴──────────────┘           │
│                         │                                     │
│                    ┌────▼────┐                               │
│                    │  Frame  │                               │
│                    │Generator│                               │
│                    └────┬────┘                               │
└─────────────────────────┼──────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │      MICRO-BATCH GENERATOR         │
         │  ┌──────┐ ┌──────┐ ┌──────┐       │
         │  │Batch │ │Batch │ │Batch │ ...   │
         │  │  0   │ │  1   │ │  2   │       │
         │  └──┬───┘ └──┬───┘ └──┬───┘       │
         └─────┼────────┼────────┼────────────┘
               │        │        │
               ▼        ▼        ▼
         ┌──────────────────────────────────┐
         │      BATCH PROCESSOR              │
         │                                   │
         │  ┌─────────┐  ┌─────────────┐   │
         │  │  YOLO   │→ │   ReID      │   │
         │  │Detector │  │   Tracker   │   │
         │  └─────────┘  └─────┬───────┘   │
         │                      │            │
         │  ┌─────────────┐    │            │
         │  │    Team     │←───┘            │
         │  │ Classifier  │                 │
         │  └─────┬───────┘                 │
         │        │                          │
         │  ┌─────▼───────┐                 │
         │  │ Possession  │                 │
         │  │   Tracker   │                 │
         │  └─────────────┘                 │
         └───────┬──────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │        MATCH STATE             │
    │  ┌──────────┐  ┌────────────┐ │
    │  │ Tracker  │  │Team Class. │ │
    │  │  State   │  │   State    │ │
    │  └──────────┘  └────────────┘ │
    │  ┌──────────┐                 │
    │  │Possession│                 │
    │  │  State   │                 │
    │  └──────────┘                 │
    └────┬───────────────────────────┘
         │
         ├──► Storage (JSON/Redis)
         │
         └──► API / WebSocket
```

---

## 📦 Módulos Principales

### 1. **video_sources.py** - Capa de Ingesta

**Responsabilidad**: Abstraer diferentes fuentes de video en una interfaz única.

**Clases principales**:

```python
class VideoSource:
    """Interfaz base"""
    def get_frame_generator() -> Iterator[np.ndarray]
    def get_metadata() -> VideoMetadata
    def close()

class LocalFileSource(VideoSource):
    """Archivos locales con cv2.VideoCapture"""

class FFmpegStreamSource(VideoSource):
    """Streams genéricos con FFmpeg (HLS, RTMP, URLs)"""

class YouTubeSource(FFmpegStreamSource):
    """YouTube con yt-dlp para resolver URLs"""

class HLSSource(FFmpegStreamSource):
    """Streams HLS (.m3u8)"""
```

**Factory function**:

```python
def open_source(source_type: SourceType, source: str) -> VideoSource:
    """Crea la fuente apropiada según el tipo"""
```

**Utilidades**:

```python
def read_frame_batches(stream, batch_size_frames) -> Iterator[Tuple[int, list]]:
    """Agrupa frames en micro-batches"""

def calculate_batch_size(fps: float, seconds_per_batch: float) -> int:
    """Calcula frames por batch según duración deseada"""
```

---

### 2. **match_state.py** - Gestión de Estado

**Responsabilidad**: Mantener el estado persistente del análisis entre batches.

**Clases de estado**:

```python
@dataclass
class TrackerState:
    """Estado del ReID tracker"""
    active_tracks: Dict[int, Dict]
    next_id: int
    lost_tracks: Dict[int, Dict]
    last_frame_idx: int

@dataclass
class TeamClassifierState:
    """Estado del clasificador de equipos"""
    player_team_map: Dict[int, int]
    vote_history: Dict[int, List[int]]
    team_colors: Dict[int, List[float]]
    is_trained: bool

@dataclass
class PossessionState:
    """Estado del tracking de posesión"""
    current_team: int
    current_player: int
    frames_by_team: Dict[int, int]
    passes_by_team: Dict[int, int]
    possession_changes: List[Dict]

@dataclass
class MatchState:
    """Estado completo del partido"""
    match_id: str
    source_type: str
    fps: float
    status: str
    total_frames_processed: int
    last_batch_idx: int
    
    tracker_state: TrackerState
    team_classifier_state: TeamClassifierState
    possession_state: PossessionState
    
    def get_summary() -> Dict
    def save_to_file(path)
    def load_from_file(path)
```

**Storage backends**:

```python
class FileSystemStorage(StateStorage):
    """Almacenamiento en archivos JSON"""

class RedisStorage(StateStorage):
    """Almacenamiento en Redis (para múltiples workers)"""
```

---

### 3. **batch_processor.py** - Procesamiento de Chunks

**Responsabilidad**: Ejecutar el pipeline completo en un micro-batch.

**Clase principal**:

```python
class BatchProcessor:
    def __init__(self, model_path, device, conf_threshold, ...):
        """Inicializa YOLO y parámetros"""
    
    def initialize_modules(self, match_state: MatchState):
        """Restaura tracker, classifier, possession desde estado"""
    
    def process_chunk(
        self, 
        match_state: MatchState,
        frames: List[np.ndarray],
        start_frame_idx: int,
        fps: float
    ) -> Tuple[MatchState, ChunkOutput]:
        """
        Pipeline completo:
        1. Detección YOLO
        2. Tracking ReID
        3. Clasificación equipos
        4. Detección posesión
        5. Generación outputs
        """
    
    def save_modules_state(self, match_state: MatchState):
        """Guarda estado de tracker, classifier, possession"""
```

**Outputs**:

```python
@dataclass
class ChunkOutput:
    """Resultado del procesamiento de un chunk"""
    batch_idx: int
    start_frame: int
    end_frame: int
    detections_by_frame: Dict[int, Dict]
    player_positions: List[Dict]
    events: List[Dict]  # Pases, cambios de posesión
    chunk_stats: Dict
    processing_time_ms: float
```

---

### 4. **match_analyzer.py** - Loop Principal

**Responsabilidad**: Orquestar el análisis completo con micro-batching.

**Función principal**:

```python
def run_match_analysis(
    match_id: str,
    config: AnalysisConfig,
    resume: bool = True
) -> MatchState:
    """
    Loop principal:
    
    1. Cargar/crear MatchState
    2. Abrir VideoSource
    3. Generar micro-batches
    4. Para cada batch:
       a. Procesar chunk
       b. Guardar outputs
       c. Guardar estado (checkpoint)
       d. Notificar progreso
    5. Completar análisis
    """
```

**Configuración**:

```python
@dataclass
class AnalysisConfig:
    source_type: SourceType
    source_url: str
    batch_size_seconds: float = 3.0
    model_path: str = "weights/best.pt"
    
    # Callbacks
    on_progress: Optional[Callable]
    on_batch_complete: Optional[Callable]
    on_error: Optional[Callable]
```

**Shortcuts**:

```python
analyze_local_file(match_id, file_path)
analyze_youtube(match_id, youtube_url, is_live=False)
analyze_hls_stream(match_id, hls_url)
```

---

### 5. **app_streaming.py** - API Web

**Responsabilidad**: Exponer funcionalidad vía HTTP + WebSocket.

**Endpoints principales**:

```python
POST   /api/upload              # Subir video
POST   /api/analyze             # Iniciar análisis
GET    /api/match/{id}/summary  # Resumen del partido
GET    /api/match/{id}/events   # Eventos detectados
GET    /api/match/{id}/positions # Posiciones de jugadores
GET    /api/match/{id}/status   # Estado del análisis
DELETE /api/match/{id}          # Eliminar partido
GET    /api/matches             # Listar partidos
WS     /ws/{id}                 # WebSocket tiempo real
```

---

## 🔄 Flujo de Datos

### Análisis de Archivo Local

```python
# 1. Usuario sube video
POST /api/upload
→ file_id: "abc123"

# 2. Inicia análisis
POST /api/analyze
{
  "match_id": "match_001",
  "source_type": "uploaded_file",
  "file_id": "abc123",
  "batch_size_seconds": 3.0
}

# 3. Backend (thread separado):
with open_source(SourceType.UPLOADED_FILE, "uploads/abc123.mp4") as src:
    for batch_idx, frames in read_frame_batches(src, batch_size):
        state, output = processor.process_chunk(state, frames, ...)
        storage.save(match_id, state)
        save_chunk_output(match_id, output)
        notify_websocket(progress)

# 4. Cliente recibe actualizaciones vía WebSocket:
WS /ws/match_001
← {"type": "progress", "frame": 90, "progress": 10%}
← {"type": "batch_complete", "stats": {...}}
← {"type": "completed", "stats": {...}}

# 5. Consultar resultados:
GET /api/match/match_001/summary
→ {possession: {team_0: 60%, team_1: 40%}, passes: {...}}
```

### Análisis de YouTube Live

```python
# 1. Iniciar análisis directo
POST /api/analyze
{
  "match_id": "live_match_001",
  "source_type": "youtube_live",
  "source_url": "https://youtube.com/watch?v=...",
  "batch_size_seconds": 2.0  # Menor latencia
}

# 2. Backend:
with open_source(SourceType.YOUTUBE_LIVE, youtube_url) as src:
    # Stream infinito
    for batch_idx, frames in read_frame_batches(src, batch_size):
        # Procesar continuamente
        state, output = processor.process_chunk(state, frames, ...)
        # Los clientes ven stats actualizarse en tiempo real

# 3. Interrumpir análisis:
# Ctrl+C o DELETE /api/match/live_match_001
# Estado se guarda → puede reanudarse

# 4. Consultar durante el partido:
GET /api/match/live_match_001/summary
→ Estadísticas acumuladas hasta el momento
```

---

## 💡 Ejemplos de Uso

### Ejemplo 1: CLI - Análisis Simple

```python
from modules.match_analyzer import analyze_local_file

# Analizar archivo local
state = analyze_local_file(
    match_id="match_001",
    file_path="sample_match.mp4",
    batch_size_seconds=3.0
)

# Ver resumen
summary = state.get_summary()
print(f"Posesión Team 0: {summary['possession']['percent_by_team'][0]}%")
print(f"Pases Team 0: {summary['passes']['by_team'][0]}")
```

### Ejemplo 2: CLI - Análisis con Callbacks

```python
from modules.match_analyzer import run_match_analysis, AnalysisConfig
from modules.video_sources import SourceType

def on_progress(match_id, progress):
    print(f"Procesados {progress['frames_processed']} frames")

def on_batch_complete(match_id, output):
    print(f"Batch {output.batch_idx}: {len(output.events)} eventos")

config = AnalysisConfig(
    source_type=SourceType.UPLOADED_FILE,
    source_url="match.mp4",
    batch_size_seconds=3.0,
    on_progress=on_progress,
    on_batch_complete=on_batch_complete
)

state = run_match_analysis("match_002", config)
```

### Ejemplo 3: API - Análisis de YouTube

```bash
# Iniciar análisis
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "match_id": "youtube_match",
    "source_type": "youtube_vod",
    "source_url": "https://youtube.com/watch?v=...",
    "batch_size_seconds": 3.0
  }'

# Consultar progreso
curl http://localhost:8000/api/match/youtube_match/status

# Ver resumen
curl http://localhost:8000/api/match/youtube_match/summary

# Obtener eventos
curl http://localhost:8000/api/match/youtube_match/events

# Posiciones para heatmap
curl http://localhost:8000/api/match/youtube_match/positions?team_id=0
```

### Ejemplo 4: WebSocket Client (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/match_001');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'progress':
            updateProgressBar(data.progress);
            console.log(`Frame ${data.frame}/${data.total_frames}`);
            break;
        
        case 'batch_complete':
            updateStats(data.stats);
            break;
        
        case 'completed':
            showFinalResults(data.stats);
            break;
        
        case 'error':
            showError(data.message);
            break;
    }
};

// Mantener conexión
setInterval(() => ws.send('ping'), 30000);
```

### Ejemplo 5: Recuperación ante Fallos

```python
# Primera ejecución (falla en batch 50)
try:
    state = run_match_analysis("match_003", config)
except Exception as e:
    print(f"Error: {e}")
    # Estado guardado hasta batch 49

# Segunda ejecución (reanuda desde batch 50)
state = run_match_analysis("match_003", config, resume=True)
# ✓ Continúa desde donde se quedó
```

---

## 📚 API Reference

### POST /api/upload

Sube un archivo de video.

**Request**:
```
Content-Type: multipart/form-data
file: <binary>
```

**Response**:
```json
{
  "success": true,
  "file_id": "abc123",
  "filename": "match.mp4",
  "size": 125829120,
  "path": "uploads/abc123.mp4"
}
```

---

### POST /api/analyze

Inicia el análisis de un partido.

**Request**:
```json
{
  "match_id": "match_001",
  "source_type": "uploaded_file",  // o youtube_vod, youtube_live, hls, rtmp
  "file_id": "abc123",             // para uploaded_file
  "source_url": "https://...",     // para otros tipos
  "batch_size_seconds": 3.0,
  "model_path": "weights/best.pt",
  "conf_threshold": 0.3,
  "max_batches": null              // null = todos
}
```

**Response**:
```json
{
  "success": true,
  "match_id": "match_001",
  "status": "Analysis started",
  "source_type": "uploaded_file"
}
```

---

### GET /api/match/{match_id}/summary

Obtiene resumen del partido.

**Response**:
```json
{
  "match_id": "match_001",
  "status": "running",
  "progress": {
    "total_frames": 5400,
    "total_seconds": 180.0,
    "last_batch": 60
  },
  "possession": {
    "current_team": 0,
    "current_player": 5,
    "percent_by_team": {
      "0": 58.5,
      "1": 41.5
    },
    "seconds_by_team": {
      "0": 105.3,
      "1": 74.7
    }
  },
  "passes": {
    "by_team": {
      "0": 45,
      "1": 32
    },
    "total": 77
  },
  "teams": {
    "total_players": 22,
    "team_0_players": 11,
    "team_1_players": 11
  },
  "tracking": {
    "active_tracks": 18,
    "total_ids": 25
  }
}
```

---

### GET /api/match/{match_id}/events

Obtiene eventos detectados.

**Query params**:
- `batch_from`: int (default: 0)
- `batch_to`: int (default: último)

**Response**:
```json
{
  "match_id": "match_001",
  "batch_from": 0,
  "batch_to": 60,
  "total_events": 15,
  "events": [
    {
      "type": "possession_change",
      "frame": 450,
      "timestamp": 15.0,
      "from_team": 0,
      "to_team": 1,
      "player_id": 12
    },
    {
      "type": "pass",
      "frame": 480,
      "timestamp": 16.0,
      "team": 1,
      "from_player": 12,
      "to_player": 15
    }
  ]
}
```

---

### GET /api/match/{match_id}/positions

Obtiene posiciones de jugadores.

**Query params**:
- `frame_from`: int
- `frame_to`: int
- `player_id`: int
- `team_id`: int

**Response**:
```json
{
  "match_id": "match_001",
  "total_positions": 12500,
  "positions": [
    {
      "frame": 100,
      "timestamp": 3.33,
      "player_id": 5,
      "team_id": 0,
      "bbox": [450, 320, 520, 480],
      "position": [485, 400]
    }
  ]
}
```

---

### WebSocket /ws/{match_id}

Stream de actualizaciones en tiempo real.

**Mensajes del servidor**:

```javascript
// Progreso
{
  "type": "progress",
  "frame": 450,
  "total_frames": 5400,
  "progress": 8.3,
  "fps_processing": 45.2,
  "realtime_factor": 1.5,
  "message": "Processing batch 5..."
}

// Batch completado
{
  "type": "batch_complete",
  "batch_idx": 5,
  "stats": { /* resumen completo */ }
}

// Completado
{
  "type": "completed",
  "stats": { /* resumen final */ }
}

// Error
{
  "type": "error",
  "message": "Error message"
}
```

---

## 🚀 Deployment

### Opción 1: Servidor Simple

```bash
# Instalar dependencias
pip install -r requirements_streaming.txt

# Iniciar servidor
python app_streaming.py

# O con uvicorn
uvicorn app_streaming:app --host 0.0.0.0 --port 8000
```

### Opción 2: Con Workers (Celery)

```python
# tasks.py
from celery import Celery
from modules.match_analyzer import run_match_analysis

celery = Celery('tasks', broker='redis://localhost:6379/0')

@celery.task
def analyze_match_task(match_id, config_dict):
    config = AnalysisConfig(**config_dict)
    return run_match_analysis(match_id, config)
```

```bash
# Iniciar worker
celery -A tasks worker --loglevel=info

# En app_streaming.py, usar:
analyze_match_task.delay(match_id, config.dict())
```

### Opción 3: Docker

```dockerfile
FROM python:3.9

WORKDIR /app
COPY . .

RUN pip install -r requirements_streaming.txt

# Instalar ffmpeg y yt-dlp
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install yt-dlp

EXPOSE 8000

CMD ["uvicorn", "app_streaming:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t tacticeye-streaming .
docker run -p 8000:8000 -v $(pwd)/uploads:/app/uploads tacticeye-streaming
```

---

## ⚙️ Configuración Recomendada

### Para VOD (Post-análisis)

```python
config = AnalysisConfig(
    batch_size_seconds=5.0,    # Batches más grandes
    device="cuda",              # GPU si está disponible
    conf_threshold=0.3,
)
# Factor realtime esperado: 2-3x (más rápido que tiempo real)
```

### Para Live Streaming (Baja latencia)

```python
config = AnalysisConfig(
    batch_size_seconds=2.0,    # Batches pequeños
    device="cuda",              # GPU recomendada
    conf_threshold=0.35,        # Menos detecciones = más rápido
)
# Factor realtime esperado: 1-1.5x (casi tiempo real)
```

### Para Análisis Detallado (Calidad)

```python
config = AnalysisConfig(
    batch_size_seconds=3.0,
    conf_threshold=0.25,        # Más sensible
    imgsz=1280,                 # Mayor resolución
    device="cuda",
)
# Más lento pero más preciso
```

---

## 📊 Métricas de Performance

### Hardware recomendado:

- **CPU**: 8+ cores para análisis paralelo
- **GPU**: NVIDIA RTX 3060+ para tiempo real
- **RAM**: 16GB+ (8GB modelo + 8GB video buffering)
- **Disco**: SSD para I/O de checkpoints

### Benchmarks típicos (RTX 3070):

| Fuente       | FPS Video | FPS Procesamiento | Factor Realtime |
|--------------|-----------|-------------------|-----------------|
| Local 1080p  | 30        | 60                | 2.0x            |
| YouTube VOD  | 30        | 45                | 1.5x            |
| HLS Live     | 30        | 35                | 1.17x           |

---

## 🐛 Troubleshooting

### El análisis se queda trabado

```python
# Verificar estado guardado
from modules.match_state import get_default_storage

storage = get_default_storage()
state = storage.load("match_id")
print(state.status)
print(state.last_batch_idx)

# Reanudar desde último batch
run_match_analysis("match_id", config, resume=True)
```

### Error con YouTube

```bash
# Actualizar yt-dlp
pip install --upgrade yt-dlp

# Test manual
yt-dlp -f best[ext=mp4] -g "URL"
```

### Memoria insuficiente

```python
# Reducir batch size
config.batch_size_seconds = 1.0

# Procesar en CPU
config.device = "cpu"

# Reducir resolución
config.imgsz = 640
```

---

## 📝 Próximos Pasos

### Mejoras planificadas:

1. **Streaming de salida**: Generar video con anotaciones en tiempo real
2. **Heatmaps incrementales**: Generar heatmaps por batch
3. **Detección de eventos avanzados**: Tiros, corners, tarjetas
4. **Multi-worker**: Procesamiento distribuido con Celery/RQ
5. **Frontend mejorado**: Dashboard con visualización en vivo
6. **API de predicciones**: ML para predecir posesión futura

---

**Desarrollado por TacticEYE Team**  
*Análisis de Fútbol con AI - Micro-Batching Architecture*
