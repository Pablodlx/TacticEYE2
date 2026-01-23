# 🌐 TacticEYE2 - Aplicación Web

Interfaz web para análisis de partidos de fútbol con visualización en tiempo real.

## 🚀 Inicio Rápido

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
pip install -r requirements_web.txt
```

### 2. Iniciar servidor
```bash
python app.py
# O usar el script:
./start_web.sh
```

### 3. Abrir navegador
```
http://localhost:8000
```

## 📊 Características

### Funcionalidades
- ✅ Subida de videos (drag & drop)
- ✅ Análisis en tiempo real con WebSocket
- ✅ Barra de progreso en vivo
- ✅ Estadísticas actualizadas cada 100 frames

### Visualizaciones
- 📊 **Gráfico de Posesión** - Circular (doughnut)
- 📈 **Gráfico de Pases** - Barras comparativas
- ⏱️ **Tiempo de Posesión** - Barras animadas por equipo
- 📉 **Timeline** - Segmentos de posesión a lo largo del partido

### Estadísticas Mostradas
- Posesión por equipo (%)
- Tiempo de posesión (segundos)
- Pases completados por equipo
- Duración total del partido
- Timeline de cambios de posesión

## 🏗️ Arquitectura

### Backend (FastAPI)
```
app.py
├── /                    → Página principal
├── /api/upload         → Subir video
├── /api/analyze/{id}   → Iniciar análisis
├── /api/status/{id}    → Estado del análisis
└── /ws/{id}            → WebSocket para actualizaciones
```

### Frontend (HTML/CSS/JS)
```
templates/index.html     → Interfaz principal
static/
├── app.js              → Lógica (WebSocket, gráficos)
└── style.css           → Estilos Bootstrap + custom
```

### Tecnologías
- **Backend**: FastAPI + Uvicorn + WebSockets
- **Frontend**: Bootstrap 5 + Chart.js + Vanilla JS
- **Comunicación**: REST API + WebSocket
- **Gráficos**: Chart.js 4.4

## 🎨 Interfaz

### Secciones

1. **Upload**
   - Seleccionar archivo de video
   - Botón "Subir y Analizar"
   - Validación de formatos

2. **Progreso**
   - Barra de progreso animada
   - Frame actual / Total
   - Estado del análisis

3. **Resultados**
   - **Resumen**: Duración, total frames
   - **Posesión**: Gráfico circular + barras
   - **Pases**: Gráfico de barras
   - **Timeline**: Visualización de segmentos

### Colores
- **Equipo 0**: Verde (#00c851)
- **Equipo 1**: Rojo (#ff4444)

## 📡 Flujo de Datos

```
1. Usuario sube video → POST /api/upload
2. Backend guarda → uploads/{session_id}_{filename}
3. Usuario conecta WebSocket → /ws/{session_id}
4. Análisis inicia → POST /api/analyze/{session_id}
5. Backend procesa en background:
   - Cada 100 frames → WS update (progreso + stats)
   - Al finalizar → WS update (resultados finales)
6. Frontend actualiza gráficos en tiempo real
```

## 🔧 Configuración

### Puertos
- Por defecto: `8000`
- Cambiar en `app.py`:
  ```python
  uvicorn.run(app, host="0.0.0.0", port=8000)
  ```

### Directorios
- `uploads/` - Videos subidos por usuarios
- `outputs/` - Videos procesados (futuro)
- `static/` - CSS, JS, imágenes
- `templates/` - HTML templates

### Parámetros de Análisis
Actualmente fijos en `app.py`, línea ~156:
```python
tracker = ReIDTracker(max_age=30, max_lost_time=120.0)
possession = PossessionTrackerV2(fps=fps, hysteresis_frames=5)
team_classifier = TeamClassifierV2(
    kmeans_min_tracks=12,
    vote_history=4,
    use_L_channel=True,
    L_weight=0.5
)
```

## 🐛 Solución de Problemas

### Error: "Address already in use"
```bash
# Matar proceso en puerto 8000
lsof -ti:8000 | xargs kill -9
```

### WebSocket no conecta
- Verificar firewall
- Comprobar que el puerto 8000 está abierto
- Revisar consola del navegador (F12)

### Video no procesa
- Verificar que existe `weights/best.pt`
- Comprobar formato de video compatible
- Ver logs del servidor en terminal

### Gráficos no actualizan
- Verificar conexión WebSocket (F12 → Network → WS)
- Comprobar que Chart.js cargó correctamente
- Revisar consola de errores JavaScript

## 🚀 Producción

### Con Gunicorn (recomendado)
```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Con Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt requirements_web.txt ./
RUN pip install -r requirements.txt -r requirements_web.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Variables de Entorno
```bash
export TACTIC_MODEL_PATH="weights/best.pt"
export TACTIC_UPLOAD_DIR="uploads"
export TACTIC_MAX_VIDEO_SIZE="500MB"
```

## 📝 Próximas Mejoras

- [ ] Múltiples videos simultáneos
- [ ] Exportación de estadísticas (JSON, CSV)
- [ ] Descarga de video procesado con overlay
- [ ] Configuración de parámetros desde UI
- [ ] Autenticación de usuarios
- [ ] Base de datos para historial
- [ ] Comparación entre partidos
- [ ] Detección de eventos (goles, tarjetas)

## 🤝 Contribuir

Las mejoras a la interfaz web son bienvenidas:
- Nuevas visualizaciones
- Mejoras de UX/UI
- Optimizaciones de rendimiento
- Tests automatizados

---

**Versión Web v1.0** - Interfaz gráfica completa para TacticEYE2
