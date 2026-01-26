# ✅ Integración Web - Sistema de Calibración Espacial

## 🎯 Resumen

El sistema de **calibración automática y tracking espacial** ha sido completamente integrado en la interfaz web de TacticEYE2. Ahora los usuarios pueden ver **heatmaps de posesión por zonas del campo** en tiempo real durante el análisis de partidos.

---

## 🚀 ¿Qué se ha integrado?

### Backend (app.py)

#### ✅ Habilitado en AnalysisConfig
```python
config = AnalysisConfig(
    # ... parámetros existentes ...
    
    # NUEVO: Spatial tracking
    enable_spatial_tracking=True,
    zone_partition_type='thirds_lanes',
    enable_heatmaps=True,
    heatmap_resolution=(50, 34)
)
```

#### ✅ Estadísticas espaciales por WebSocket
Cada vez que se completa un batch, el backend envía:
```javascript
{
  "type": "batch_complete",
  "stats": {
    // ... stats existentes ...
    "spatial": {
      "calibration_valid": true/false,
      "possession_by_zone": {
        "0": [frames por zona],
        "1": [frames por zona]
      },
      "zone_percentages": {
        "0": [% por zona],
        "1": [% por zona]
      },
      "partition_type": "thirds_lanes",
      "num_zones": 9
    }
  }
}
```

#### ✅ Nuevo endpoint para heatmaps
**GET** `/api/heatmap/{session_id}/{team_id}`

Retorna una imagen PNG del heatmap del equipo especificado:
- Usa matplotlib colormap (Greens para Team 0, Reds para Team 1)
- Redimensionado a 525×340px (aspect ratio del campo)
- Se actualiza automáticamente durante el análisis

---

### Frontend (templates/index.html)

#### ✅ Nueva sección de heatmaps
Se añadió una sección completa entre los gráficos de posesión y las estadísticas de equipo:

```html
<div id="spatial-heatmaps-section" class="row mb-4">
  <div class="card">
    <div class="card-header">
      <h5>Field Possession Heatmaps</h5>
      <span id="calibration-status">Calibrating...</span>
    </div>
    <div class="card-body">
      <!-- Heatmaps de ambos equipos -->
      <img id="heatmap-team-0" src="..." />
      <img id="heatmap-team-1" src="..." />
      
      <!-- Top 3 zonas por equipo -->
      <div id="top-zones-team-0"></div>
      <div id="top-zones-team-1"></div>
    </div>
  </div>
</div>
```

**Elementos visuales**:
- Badge de estado de calibración (verde si válida, amarillo si no)
- Heatmaps lado a lado (Team 0 / Team 1)
- Top 3 zonas con mayor posesión para cada equipo
- Info de tipo de partición y número de zonas

---

### JavaScript (static/app.js)

#### ✅ Función `updateSpatialStats(spatial)`
Se ejecuta cada vez que llega un batch_complete con datos espaciales:

```javascript
function updateSpatialStats(spatial) {
  // 1. Mostrar sección de heatmaps
  document.getElementById('spatial-heatmaps-section').style.display = 'block';
  
  // 2. Actualizar badge de calibración
  if (spatial.calibration_valid) {
    calibrationStatus.className = 'badge bg-success';
  }
  
  // 3. Actualizar imágenes de heatmaps
  updateHeatmapImages();
  
  // 4. Mostrar top 3 zonas
  updateTopZones(0, spatial.zone_percentages[0]);
  updateTopZones(1, spatial.zone_percentages[1]);
}
```

#### ✅ Función `updateHeatmapImages()`
Recarga las imágenes de heatmaps con un timestamp para evitar caché:

```javascript
function updateHeatmapImages() {
  const timestamp = new Date().getTime();
  
  heatmapTeam0.src = `/api/heatmap/${currentSessionId}/0?t=${timestamp}`;
  heatmapTeam1.src = `/api/heatmap/${currentSessionId}/1?t=${timestamp}`;
}
```

#### ✅ Función `updateTopZones(teamId, zonePercentages)`
Muestra las top 3 zonas ordenadas por porcentaje:

```javascript
function updateTopZones(teamId, zonePercentages) {
  const zones = [...].sort((a, b) => b.percent - a.percent);
  const top3 = zones.slice(0, 3);
  
  // Renderiza badges con nombres de zonas y porcentajes
  topZonesDiv.innerHTML = top3.map(zone => 
    `<span class="badge bg-success">1. ${zone.name} (${zone.percent}%)</span>`
  ).join('');
}
```

---

### CSS (static/style.css)

#### ✅ Estilos para heatmaps
```css
.heatmap-container {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 10px;
}

.heatmap-container img {
  max-width: 100%;
  border-radius: 4px;
}

#spatial-heatmaps-section .card-header {
  background: linear-gradient(135deg, var(--secondary-blue) 0%, var(--primary-blue) 100%);
  border-bottom: 2px solid var(--accent-green);
}
```

---

## 📊 Flujo de Datos

```
┌──────────────────────────────────────────────────────────────┐
│                  Usuario sube video                         │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Backend: AnalysisConfig con enable_spatial_tracking=True   │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  BatchProcessor procesa frames con:                         │
│  - FieldCalibrator (cada 30 frames)                         │
│  - SpatialPossessionTracker                                 │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  on_batch_complete callback                                 │
│  Envía por WebSocket:                                       │
│  - stats.spatial.calibration_valid                          │
│  - stats.spatial.possession_by_zone                         │
│  - stats.spatial.zone_percentages                           │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  JavaScript: handleWebSocketMessage()                       │
│  - Detecta type: 'batch_complete'                           │
│  - Llama updateSpatialStats(stats.spatial)                  │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  updateSpatialStats()                                       │
│  1. Muestra sección de heatmaps                             │
│  2. Actualiza badge de calibración                          │
│  3. Llama updateHeatmapImages()                             │
│  4. Llama updateTopZones() para cada equipo                 │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  updateHeatmapImages()                                      │
│  - GET /api/heatmap/{session_id}/0                          │
│  - GET /api/heatmap/{session_id}/1                          │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Backend: get_heatmap()                                     │
│  1. Carga NPZ file                                          │
│  2. Aplica colormap (Greens/Reds)                           │
│  3. Convierte a PNG con PIL                                 │
│  4. Retorna StreamingResponse                               │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Browser: Muestra heatmaps en tiempo real                   │
│  - Team 0 (verde) | Team 1 (rojo)                           │
│  - Top 3 zonas actualizadas                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 🖼️ Vista de la Interfaz

### Antes (sin spatial tracking)
```
┌─────────────────────────────────────────────┐
│  Posesión del Balón | Pases Completados    │
│  [Gráfico de torta] | [Gráfico de barras]  │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│  Team 0 Stats       | Team 1 Stats         │
│  Posesión: 52.3%    | Posesión: 47.7%      │
│  Pases: 45          | Pases: 38            │
└─────────────────────────────────────────────┘
```

### Ahora (con spatial tracking) ⭐
```
┌─────────────────────────────────────────────┐
│  Posesión del Balón | Pases Completados    │
│  [Gráfico de torta] | [Gráfico de barras]  │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐ ⬅️ NUEVO
│  Field Possession Heatmaps  [✓ Calibrated] │
│  ┌────────────────┐  ┌────────────────┐    │
│  │   Team 0       │  │   Team 1       │    │
│  │  [Heatmap      │  │  [Heatmap      │    │
│  │   verde]       │  │   rojo]        │    │
│  │                │  │                │    │
│  │ Top zonas:     │  │ Top zonas:     │    │
│  │ 1.Mid Center   │  │ 1.Off Center   │    │
│  │ 2.Def Center   │  │ 2.Mid Right    │    │
│  │ 3.Off Center   │  │ 3.Mid Left     │    │
│  └────────────────┘  └────────────────┘    │
│  Partition: thirds_lanes (9 zones)         │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│  Team 0 Stats       | Team 1 Stats         │
│  Posesión: 52.3%    | Posesión: 47.7%      │
│  Pases: 45          | Pases: 38            │
└─────────────────────────────────────────────┘
```

---

## 🎨 Nombres de Zonas (thirds_lanes)

| Zona ID | Nombre            | Descripción          |
|---------|-------------------|----------------------|
| 0       | Defensive Left    | Tercio def., carril izq. |
| 1       | Defensive Center  | Tercio def., carril central |
| 2       | Defensive Right   | Tercio def., carril der. |
| 3       | Midfield Left     | Tercio medio, carril izq. |
| 4       | Midfield Center   | Tercio medio, carril central |
| 5       | Midfield Right    | Tercio medio, carril der. |
| 6       | Offensive Left    | Tercio of., carril izq. |
| 7       | Offensive Center  | Tercio of., carril central |
| 8       | Offensive Right   | Tercio of., carril der. |

---

## 🔧 Cómo Usar

### 1. Iniciar el servidor
```bash
python app.py
```

### 2. Abrir en navegador
```
http://localhost:8000
```

### 3. Subir video o usar URL
- Opción 1: Subir archivo MP4 local
- Opción 2: URL de YouTube/stream

### 4. Ver análisis en tiempo real
Durante el procesamiento:
- ✅ Video con anotaciones en vivo
- ✅ Gráficos de posesión actualizados
- ✅ **Heatmaps espaciales en tiempo real** ⭐
- ✅ **Top 3 zonas por equipo** ⭐
- ✅ **Estado de calibración** ⭐

### 5. Resultados finales
Al completar:
- Todos los gráficos finalizados
- Heatmaps completos disponibles
- Estadísticas exportadas en outputs/

---

## 📁 Archivos Modificados

### Backend
- ✅ `app.py` (+120 líneas)
  - Habilitado spatial tracking en config
  - Añadido endpoint `/api/heatmap/{session_id}/{team_id}`
  - Envío de stats espaciales por WebSocket

### Frontend
- ✅ `templates/index.html` (+45 líneas)
  - Nueva sección `spatial-heatmaps-section`
  - Elementos para heatmaps y top zonas
  
- ✅ `static/app.js` (+110 líneas)
  - Función `updateSpatialStats()`
  - Función `updateHeatmapImages()`
  - Función `updateTopZones()`
  
- ✅ `static/style.css` (+45 líneas)
  - Estilos para heatmap containers
  - Estilos para badges de calibración
  - Estilos para top zones

---

## 🐛 Troubleshooting

### Problema: Heatmaps no se muestran

**Posibles causas**:
1. Video sin líneas de campo visibles → `calibration_valid: false`
2. No hay posesión detectada → heatmaps vacíos

**Solución**: El sistema muestra badge amarillo "No Calibration" pero continúa el análisis.

### Problema: Imágenes no se actualizan

**Causa**: Caché del navegador

**Solución**: El código ya incluye timestamp en la URL para evitar caché:
```javascript
/api/heatmap/${sessionId}/0?t=${timestamp}
```

### Problema: Error 404 en /api/heatmap

**Causa**: Archivo NPZ no encontrado

**Solución**: El endpoint busca en dos ubicaciones:
1. `outputs/{session_id}_heatmaps.npz`
2. `outputs/{session_id}/{session_id}_heatmaps.npz`

---

## ✨ Características Destacadas

### ✅ Actualización en Tiempo Real
Los heatmaps se actualizan **cada vez que se completa un batch** (~3 segundos), mostrando la evolución de la posesión espacial durante el partido.

### ✅ Calibración Automática
Sin necesidad de clicks manuales, el sistema detecta automáticamente las líneas del campo y calibra la perspectiva.

### ✅ Fallback Inteligente
Si no hay calibración válida (ej: vista muy parcial), el sistema:
- Muestra badge de advertencia
- Continúa con el análisis básico
- Mantiene la interfaz funcional

### ✅ Visualización Profesional
- Colormaps profesionales (matplotlib)
- Gradientes verdes/rojos por equipo
- Bordes y estilos coherentes con el diseño Wyscout

### ✅ Análisis Táctico
Los "Top 3 zonas" permiten identificar rápidamente:
- Zonas de dominio de cada equipo
- Patrones de juego (ancho banda, centro, etc.)
- Desequilibrios espaciales

---

## 📊 Ejemplo de Output Completo

Al finalizar el análisis, se generan:

1. **Archivos JSON** (detections, positions, events, stats)
2. **Archivo NPZ de heatmaps**:
   ```python
   data = np.load('outputs/{session_id}_heatmaps.npz')
   
   # Arrays disponibles:
   - team_0_heatmap: [50, 34]
   - team_1_heatmap: [50, 34]
   - possession_by_zone_team_0: [9]
   - possession_by_zone_team_1: [9]
   - zone_percentages_team_0: [9]
   - zone_percentages_team_1: [9]
   - metadata: dict
   ```

3. **Visualización web completa**:
   - Gráficos de posesión
   - Heatmaps interactivos
   - Estadísticas por equipo
   - Timeline de eventos

---

## 🚀 Próximos Pasos (Opcional)

### Mejoras Posibles:

1. **Overlay de zonas en el video**
   - Dibujar las 9 zonas sobre el video en vivo
   - Destacar zona actual del balón

2. **Gráfico de evolución temporal**
   - Line chart mostrando % de posesión por zona a lo largo del tiempo
   - Identificar momentos clave

3. **Exportar heatmaps como imagen**
   - Botón para descargar heatmaps en PNG/SVG
   - Incluir en PDF de reporte

4. **Comparación de partidos**
   - Comparar heatmaps de diferentes partidos
   - Análisis de tendencias tácticas

5. **Configuración de zonas en UI**
   - Selector para cambiar entre thirds/thirds_lanes/grid
   - Personalizar número de zonas en grid

---

## ✅ Resumen Final

### Integración Completa ✓

- ✅ Backend: Spatial tracking habilitado
- ✅ Backend: Endpoint de heatmaps funcional
- ✅ Backend: WebSocket enviando datos espaciales
- ✅ Frontend: Sección de heatmaps añadida
- ✅ Frontend: JavaScript procesando datos espaciales
- ✅ Frontend: CSS para estilos profesionales
- ✅ Sistema funcionando en tiempo real

### Resultado

**TacticEYE2 ahora incluye análisis espacial completo con calibración automática**, permitiendo a los usuarios visualizar **dónde domina cada equipo en el campo** de forma intuitiva y profesional.

🎉 **¡Sistema de calibración espacial completamente integrado en la interfaz web!** 🎉

---

## 📝 Documentación Relacionada

- [SPATIAL_POSSESSION_ARCHITECTURE.md](SPATIAL_POSSESSION_ARCHITECTURE.md) - Arquitectura técnica
- [SPATIAL_TRACKING_TEST.md](SPATIAL_TRACKING_TEST.md) - Guía de pruebas CLI
- [SPATIAL_SYSTEM_INTEGRATION.md](SPATIAL_SYSTEM_INTEGRATION.md) - Guía general
- [README.md](README.md) - Documentación principal

---

**Autor**: GitHub Copilot  
**Fecha**: Enero 2026  
**Versión**: 2.0 - Spatial Tracking Integrado
