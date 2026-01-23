# 🎉 APLICACIÓN WEB LISTA

## ✅ La aplicación está corriendo en:
```
http://localhost:8000
```

## 🚀 Cómo usar:

### 1️⃣ Abrir en navegador
```bash
# En tu navegador favorito, abre:
http://localhost:8000
```

### 2️⃣ Subir video
- Haz clic en "Seleccionar archivo"
- Elige un video de fútbol (.mp4, .avi, etc.)
- Haz clic en "Subir y Analizar"

### 3️⃣ Ver análisis en tiempo real
La página mostrará:
- ✅ Barra de progreso en tiempo real
- ✅ Estadísticas actualizadas cada 100 frames
- ✅ Gráficos interactivos:
  - Posesión del balón (circular)
  - Pases completados (barras)
  - Timeline de posesión
  - Tiempo por equipo

### 4️⃣ Resultados finales
Al terminar el análisis verás:
- 📊 Posesión total por equipo (%)
- ⏱️ Tiempo de posesión (segundos)
- 🎯 Pases completados por equipo
- 📈 Timeline completo del partido

## 🎨 Características de la interfaz:

### Gráficos Interactivos
- **Posesión**: Gráfico circular (doughnut) con porcentajes
- **Pases**: Gráfico de barras comparativo
- **Timeline**: Visualización de segmentos de posesión
- **Barras animadas**: Tiempo de posesión por equipo

### Actualización en Tiempo Real
- Conexión WebSocket para updates instantáneos
- Progreso frame a frame
- Estadísticas actualizadas sin recargar página

### Diseño Responsive
- Compatible con escritorio, tablet y móvil
- Bootstrap 5 para diseño moderno
- Animaciones suaves

## 🔧 Controlar el servidor:

### Ver logs
```bash
# Los logs se muestran en la terminal donde ejecutaste python app.py
```

### Detener servidor
```bash
# Presiona Ctrl+C en la terminal
```

### Reiniciar servidor
```bash
cd /home/pablodlx/TacticEYE2_github
python app.py
```

## 📁 Archivos creados:

### Backend
- `app.py` - Servidor FastAPI con WebSocket
- `requirements_web.txt` - Dependencias web

### Frontend
- `templates/index.html` - Interfaz principal
- `static/app.js` - Lógica JavaScript
- `static/style.css` - Estilos CSS

### Directorios
- `uploads/` - Videos subidos se guardan aquí
- `outputs/` - Videos procesados (futuro)

### Documentación
- `WEB_README.md` - Guía completa de la app web
- `start_web.sh` - Script de inicio rápido

## 🎯 Próximos pasos:

1. **Probar con un video:**
   - Abre http://localhost:8000
   - Sube `sample_match.mp4` o `prueba3.mp4`
   - Observa el análisis en tiempo real

2. **Personalizar:**
   - Modificar colores en `static/style.css`
   - Ajustar parámetros en `app.py`
   - Añadir nuevas visualizaciones en `static/app.js`

3. **Producción:**
   - Ver `WEB_README.md` para deployment
   - Configurar Gunicorn para múltiples workers
   - Añadir autenticación si es necesario

## 📊 Datos mostrados:

### En tiempo real (cada 100 frames):
- Frame actual / Total frames
- Posesión acumulada (%)
- Pases acumulados por equipo

### Al finalizar:
- **Resumen general:**
  - Duración total (segundos)
  - Total de frames procesados

- **Posesión del balón:**
  - Team 0: X% (Y segundos)
  - Team 1: X% (Y segundos)

- **Pases completados:**
  - Team 0: N pases
  - Team 1: M pases

- **Timeline:**
  - Segmentos de posesión con inicio/fin
  - Visualización gráfica por equipo

## 🎨 Colores utilizados:

- **Equipo 0**: Verde (#00c851) 
- **Equipo 1**: Rojo (#ff4444)
- **Progreso**: Azul (Bootstrap)
- **Fondo**: Gris claro (#f8f9fa)

## ⚡ Rendimiento:

- Análisis en background (no bloquea UI)
- Updates cada 100 frames (evita saturar WebSocket)
- Gráficos optimizados con Chart.js
- Animaciones CSS suaves

---

## 🎉 ¡Listo para usar!

Abre http://localhost:8000 y empieza a analizar partidos 🚀
