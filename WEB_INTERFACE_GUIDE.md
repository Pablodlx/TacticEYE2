# 🌐 TacticEYE Web Interface - Guía de Uso

## 🚀 Inicio Rápido

### 1. Instalar Dependencias

```bash
# Dependencias del sistema de micro-batching
pip install -r requirements_streaming.txt

# Dependencias originales
pip install -r requirements.txt

# FFmpeg (necesario para streams)
sudo apt-get install ffmpeg

# yt-dlp (para YouTube)
pip install yt-dlp
```

### 2. Iniciar Servidor

```bash
python app.py
```

Abre en tu navegador: **http://localhost:8000**

---

## 📺 Fuentes de Video Soportadas

La interfaz ahora soporta **4 tipos de fuentes**:

### 1️⃣ Local File (Archivo Local)
- **Uso**: Videos descargados en tu computadora
- **Formatos**: MP4, AVI, MOV, MKV, etc.
- **Proceso**:
  1. Clic en "Local File"
  2. Arrastra el archivo o haz clic en "Select Video"
  3. Clic en "Start Analysis"

### 2️⃣ YouTube
- **Uso**: Videos de YouTube (VOD o Live)
- **Ejemplos**:
  ```
  https://www.youtube.com/watch?v=dQw4w9WgXcQ
  https://youtu.be/dQw4w9WgXcQ
  https://www.youtube.com/live/xxxxx (Live streams)
  ```
- **Proceso**:
  1. Clic en "YouTube"
  2. Pega la URL del video
  3. Clic en "Start Analysis"

### 3️⃣ HLS Stream
- **Uso**: Streams HLS (.m3u8)
- **Ejemplos**:
  ```
  https://example.com/stream/playlist.m3u8
  https://broadcast.domain.com/live/stream.m3u8
  ```
- **Proceso**:
  1. Clic en "HLS Stream"
  2. Pega la URL del stream
  3. Clic en "Start Analysis"

### 4️⃣ RTMP
- **Uso**: Streams RTMP
- **Ejemplos**:
  ```
  rtmp://live.twitch.tv/app/stream_key
  rtmp://broadcast.domain.com/live/stream
  ```
- **Proceso**:
  1. Clic en "RTMP"
  2. Pega la URL del stream
  3. Clic en "Start Analysis"

---

## 🎯 Características de la Interfaz

### 📊 Análisis en Tiempo Real

La interfaz muestra:

- **Progreso del análisis** (barra de progreso)
- **Frame actual / Total frames**
- **Tiempo transcurrido**
- **Estadísticas en vivo**:
  - Posesión del balón (%)
  - Pases completados por equipo
  - Cambios de posesión

### 📈 Visualizaciones

Al finalizar el análisis, verás:

1. **Tarjetas de Estadísticas**:
   - Duración del partido
   - Total de frames procesados
   - Posesión de cada equipo

2. **Gráficos Interactivos**:
   - 🥧 **Ball Possession** (gráfico de pie)
   - 📊 **Completed Passes** (gráfico de barras)
   - 📈 **Possession Timeline** (timeline completo)

3. **Estadísticas Detalladas**:
   - Tiempo de posesión por equipo
   - Porcentaje de posesión
   - Pases completados

---

## 🔧 Sistema de Micro-Batching

La interfaz ahora usa el **sistema de micro-batching** implementado en background:

### Ventajas:

✅ **Procesamiento incremental**: El video se procesa en chunks de 2-5 segundos  
✅ **Recuperación ante fallos**: Si el análisis falla, puede reanudarse desde el último batch  
✅ **Estadísticas en vivo**: Actualización cada batch (cada ~3 segundos)  
✅ **Soporte multi-fuente**: Archivos locales, YouTube, HLS, RTMP  
✅ **WebSocket updates**: Actualización en tiempo real sin refrescar página  

### Arquitectura:

```
Frontend (HTML/JS)
       │
       ├─► HTTP POST /api/analyze/url (para URLs)
       ├─► HTTP POST /api/upload (para archivos)
       │
       ├─► WebSocket /ws/{session_id} (actualizaciones en vivo)
       │
Backend (FastAPI)
       │
       ├─► Video Sources Layer (modules/video_sources.py)
       │   ├─ LocalFileSource
       │   ├─ YouTubeSource
       │   ├─ HLSSource
       │   └─ RTMPSource
       │
       ├─► Match Analyzer (modules/match_analyzer.py)
       │   └─ Loop de micro-batching
       │
       ├─► Batch Processor (modules/batch_processor.py)
       │   └─ Pipeline: YOLO → ReID → TeamClassifier → Possession
       │
       └─► Match State (modules/match_state.py)
           └─ Estado persistente incremental
```

---

## 📝 Ejemplos de Uso

### Ejemplo 1: Analizar Archivo Local

1. Abre http://localhost:8000
2. Selecciona "Local File"
3. Arrastra `sample_match.mp4` al área de drop
4. Clic en "Start Analysis"
5. Observa el progreso en tiempo real
6. Visualiza los resultados al finalizar

### Ejemplo 2: Analizar YouTube Live

1. Abre http://localhost:8000
2. Selecciona "YouTube"
3. Pega URL de stream en vivo: `https://www.youtube.com/live/xxxxx`
4. Clic en "Start Analysis"
5. El análisis procesará el stream en tiempo real

### Ejemplo 3: Analizar HLS Stream

1. Abre http://localhost:8000
2. Selecciona "HLS Stream"
3. Pega URL del stream: `https://example.com/stream.m3u8`
4. Clic en "Start Analysis"
5. El sistema decodificará el stream con FFmpeg

---

## 🐛 Troubleshooting

### Error: "Failed to connect to stream"

**Causa**: URL inválida o stream no disponible  
**Solución**:
- Verifica que la URL sea correcta
- Para YouTube, asegúrate que el video sea público
- Para HLS/RTMP, verifica que el stream esté activo

### Error: "FFmpeg not found"

**Causa**: FFmpeg no instalado  
**Solución**:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Descargar desde: https://ffmpeg.org/download.html
```

### Error: "yt-dlp error"

**Causa**: yt-dlp no instalado o desactualizado  
**Solución**:
```bash
pip install --upgrade yt-dlp
```

### El análisis se detiene o falla

**Solución**:
1. Verifica los logs en la terminal donde ejecutas `python app.py`
2. El sistema guardará checkpoints automáticamente
3. Puedes reintentar y el análisis continuará desde el último batch

### WebSocket no conecta

**Solución**:
1. Verifica que el puerto 8000 no esté bloqueado por firewall
2. Intenta con `localhost` en vez de `127.0.0.1`
3. Revisa la consola del navegador (F12) para ver errores

---

## ⚙️ Configuración Avanzada

### Ajustar Batch Size

Edita en `app.py`:

```python
config = AnalysisConfig(
    batch_size_seconds=3.0,  # ← Cambiar aquí (2.0-5.0)
    ...
)
```

**Valores recomendados**:
- `2.0` segundos: Ultra-low latency (live streams)
- `3.0` segundos: Balance (default)
- `5.0` segundos: Máxima velocidad (VOD)

### Ajustar Umbral de Confianza

```python
config = AnalysisConfig(
    conf_threshold=0.30,  # ← Cambiar aquí (0.2-0.5)
    ...
)
```

**Valores**:
- `0.2`: Más detecciones (menos preciso)
- `0.3`: Balance (default)
- `0.5`: Menos detecciones (más preciso)

### Cambiar Dispositivo

```python
config = AnalysisConfig(
    device="cuda",  # ← "cuda" o "cpu"
    ...
)
```

---

## 📚 Documentación Adicional

- **Sistema de Micro-Batching**: Ver [MICROBATCHING_GUIDE.md](MICROBATCHING_GUIDE.md)
- **API REST Completa**: Ver [STREAMING_README.md](STREAMING_README.md)
- **Arquitectura**: Ver [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)
- **Resumen Ejecutivo**: Ver [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)

---

## 🎓 Conclusión

La interfaz web de TacticEYE ahora integra completamente el sistema de micro-batching, permitiendo:

✅ **Análisis de múltiples fuentes** (archivos, YouTube, streams)  
✅ **Actualizaciones en tiempo real** vía WebSocket  
✅ **Procesamiento incremental** con recuperación ante fallos  
✅ **Visualizaciones profesionales** tipo Wyscout/Opta  

**¡Disfruta del análisis táctico avanzado! ⚽📊**
