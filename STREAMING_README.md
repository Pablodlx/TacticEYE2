# 🚀 TacticEYE Streaming - Quick Start

Sistema de análisis de partidos de fútbol con **micro-batching** para procesamiento en tiempo real.

## ⚡ Inicio Rápido (5 minutos)

### 1. Instalar dependencias

```bash
# Dependencias base (si no las tienes)
pip install -r requirements.txt

# Dependencias de streaming
pip install -r requirements_streaming.txt

# Instalar ffmpeg (necesario para streams)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Opcional: yt-dlp para YouTube
pip install yt-dlp
```

### 2. Probar con archivo local

```bash
# Ejemplo CLI simple
python modules/match_analyzer.py match_001 sample_match.mp4

# Ejemplo interactivo
python demo_streaming.py 1
```

### 3. Iniciar servidor web

```bash
python app_streaming.py
```

Abre: http://localhost:8000

---

## 📚 Ejemplos de Uso

### CLI - Análisis Simple

```python
from modules.match_analyzer import analyze_local_file

state = analyze_local_file(
    match_id="match_001",
    file_path="sample_match.mp4",
    batch_size_seconds=3.0
)

summary = state.get_summary()
print(f"Posesión Team 0: {summary['possession']['percent_by_team'][0]}%")
```

### CLI - Con Callbacks

```python
from modules.match_analyzer import run_match_analysis, AnalysisConfig
from modules.video_sources import SourceType

def on_progress(match_id, progress):
    print(f"Procesados {progress['frames_processed']} frames")

config = AnalysisConfig(
    source_type=SourceType.UPLOADED_FILE,
    source_url="match.mp4",
    batch_size_seconds=3.0,
    on_progress=on_progress
)

state = run_match_analysis("match_002", config)
```

### API - Subir y Analizar

```bash
# 1. Subir video
curl -X POST http://localhost:8000/api/upload \
  -F "file=@match.mp4"

# Respuesta: {"file_id": "abc123", ...}

# 2. Iniciar análisis
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "match_id": "match_001",
    "source_type": "uploaded_file",
    "file_id": "abc123",
    "batch_size_seconds": 3.0
  }'

# 3. Ver progreso
curl http://localhost:8000/api/match/match_001/summary

# 4. Obtener eventos
curl http://localhost:8000/api/match/match_001/events
```

### API - Analizar YouTube

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "match_id": "youtube_match",
    "source_type": "youtube_vod",
    "source_url": "https://youtube.com/watch?v=YOUR_VIDEO_ID",
    "batch_size_seconds": 3.0
  }'
```

### WebSocket - Tiempo Real

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/match_001');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'progress') {
        console.log(`Progreso: ${data.progress}%`);
    }
    
    if (data.type === 'completed') {
        console.log('Análisis completado!', data.stats);
    }
};
```

---

## 🎯 Características Principales

### ✅ Fuentes de Video Soportadas

- **Archivos locales**: MP4, AVI, MOV, etc.
- **YouTube VOD**: Videos grabados
- **YouTube Live**: Streams en vivo
- **HLS**: Streams `.m3u8`
- **RTMP**: Streams RTMP
- **Veo**: Plataforma de análisis

### ✅ Micro-Batching

- Procesa video en chunks de 2-5 segundos
- Mantiene estado entre batches
- Recuperación automática ante fallos
- Resultados parciales en tiempo real

### ✅ Pipeline Completo

1. **Detección YOLO**: Jugadores, balón, árbitros
2. **Tracking ReID**: IDs persistentes
3. **Clasificación de Equipos**: KMeans LAB
4. **Posesión**: Detección de poseedor
5. **Eventos**: Pases, cambios de posesión

---

## 📊 API Endpoints

### Gestión de Análisis

- `POST /api/upload` - Subir video
- `POST /api/analyze` - Iniciar análisis
- `GET /api/match/{id}/status` - Estado actual
- `DELETE /api/match/{id}` - Eliminar partido
- `GET /api/matches` - Listar partidos

### Consultas de Datos

- `GET /api/match/{id}/summary` - Resumen (posesión, pases, etc.)
- `GET /api/match/{id}/events` - Eventos (pases, cambios)
- `GET /api/match/{id}/positions` - Posiciones de jugadores

### Tiempo Real

- `WS /ws/{id}` - WebSocket para updates en vivo

---

## 🔧 Configuración

### Para VOD (Post-análisis)

```python
config = AnalysisConfig(
    batch_size_seconds=5.0,    # Batches grandes
    device="cuda",              # GPU
    conf_threshold=0.3,
)
# Velocidad: 2-3x realtime
```

### Para Live (Baja latencia)

```python
config = AnalysisConfig(
    batch_size_seconds=2.0,    # Batches pequeños
    device="cuda",
    conf_threshold=0.35,
)
# Velocidad: 1-1.5x realtime
```

---

## 📖 Documentación Completa

Ver [MICROBATCHING_GUIDE.md](MICROBATCHING_GUIDE.md) para:

- Arquitectura detallada
- Diagramas de flujo
- Referencia completa de API
- Ejemplos avanzados
- Troubleshooting
- Deployment

---

## 🎬 Demo Interactivo

```bash
# Ver todos los ejemplos
python demo_streaming.py

# Ejecutar ejemplo específico
python demo_streaming.py 1  # Archivo local
python demo_streaming.py 2  # Con callbacks
python demo_streaming.py 3  # Recuperación ante fallos
python demo_streaming.py 4  # Consultar resultados
```

---

## 🗂️ Estructura de Archivos

```
TacticEYE2_github/
├── modules/
│   ├── video_sources.py        # Ingesta de video (📹)
│   ├── match_state.py          # Estado persistente (💾)
│   ├── batch_processor.py      # Procesamiento chunks (⚙️)
│   └── match_analyzer.py       # Loop principal (🔄)
├── app_streaming.py            # API FastAPI (🌐)
├── demo_streaming.py           # Ejemplos (🎬)
├── match_states/               # Estados guardados
├── outputs_streaming/          # Resultados por partido
└── uploads/                    # Videos subidos
```

---

## 💡 Casos de Uso

### 1. Post-Análisis de Partido

```python
# Analizar partido grabado
state = analyze_local_file("match_001", "partido_completo.mp4")
summary = state.get_summary()
# → Estadísticas completas en pocos minutos
```

### 2. Análisis en Vivo

```python
# Stream de YouTube Live
state = analyze_youtube(
    "live_match",
    "https://youtube.com/watch?v=LIVE_ID",
    is_live=True
)
# → Estadísticas actualizadas cada 2-3 segundos
```

### 3. Recuperación de Errores

```python
# Primera ejecución (falla en batch 50)
state = run_match_analysis("match_id", config)
# ... error ...

# Segunda ejecución (reanuda automáticamente)
state = run_match_analysis("match_id", config, resume=True)
# ✓ Continúa desde batch 50
```

### 4. Monitoreo Múltiple

```bash
# Iniciar análisis de varios partidos
curl -X POST .../api/analyze -d '{"match_id": "match_1", ...}'
curl -X POST .../api/analyze -d '{"match_id": "match_2", ...}'
curl -X POST .../api/analyze -d '{"match_id": "match_3", ...}'

# Cada uno procesa independientemente
# WebSocket independiente por partido
```

---

## 🚀 Performance

### Hardware Recomendado

- **CPU**: 8+ cores
- **GPU**: NVIDIA RTX 3060+ (para tiempo real)
- **RAM**: 16GB+
- **Disco**: SSD

### Benchmarks (RTX 3070)

| Fuente      | FPS Video | FPS Procesamiento | Factor Realtime |
|-------------|-----------|-------------------|-----------------|
| Local 1080p | 30        | 60                | 2.0x            |
| YouTube VOD | 30        | 45                | 1.5x            |
| Live Stream | 30        | 35                | 1.17x           |

---

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-fuente`
3. Commit: `git commit -am 'Añadir fuente Twitch'`
4. Push: `git push origin feature/nueva-fuente`
5. Pull Request

---

## 📄 Licencia

Ver [LICENSE](LICENSE)

---

## 🆘 Soporte

- **Documentación completa**: [MICROBATCHING_GUIDE.md](MICROBATCHING_GUIDE.md)
- **Issues**: GitHub Issues
- **Ejemplos**: `python demo_streaming.py`

---

**TacticEYE** - Análisis de Fútbol con AI  
*Micro-Batching Architecture para Tiempo Real*
