"""
TacticEYE2 Web Application
FastAPI backend con WebSocket para análisis en tiempo real
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import cv2
import numpy as np
import json
import asyncio
from pathlib import Path
import uuid
from typing import Dict, Optional
import traceback
from datetime import datetime

try:
    from ultralytics import YOLO
    from modules.reid_tracker import ReIDTracker
    from modules.possession_tracker_v2 import PossessionTrackerV2
    from modules.team_classifier_v2 import TeamClassifierV2
    # Sistema de micro-batching
    from modules.video_sources import open_source, SourceType
    from modules.match_analyzer import run_match_analysis, AnalysisConfig
    from modules.match_state import FileSystemStorage
except Exception as e:
    print(f"Error importando módulos: {e}")

app = FastAPI(title="TacticEYE2 Web")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorios
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Estado global de análisis
analysis_state: Dict[str, dict] = {}


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_update(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
            except Exception as e:
                print(f"Error enviando update: {e}")


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Página principal"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Subir video para análisis"""
    try:
        # Generar ID único
        session_id = str(uuid.uuid4())
        
        # Guardar archivo
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Inicializar estado
        analysis_state[session_id] = {
            "status": "uploaded",
            "filename": file.filename,
            "file_path": str(file_path),
            "progress": 0,
            "stats": None
        }
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "filename": file.filename
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.post("/api/analyze/url")
async def analyze_from_url(request: Request):
    """Analizar video desde URL (YouTube, HLS, RTMP, etc.)"""
    try:
        data = await request.json()
        source_url = data.get('url')
        source_type_str = data.get('source_type', 'youtube')
        
        if not source_url:
            return JSONResponse({"success": False, "error": "URL requerida"}, status_code=400)
        
        # Generar ID único
        session_id = str(uuid.uuid4())
        
        # Mapear source_type
        source_type_map = {
            'youtube': SourceType.YOUTUBE_VOD,
            'hls': SourceType.HLS,
            'rtmp': SourceType.RTMP,
            'veo': SourceType.VEO
        }
        source_type = source_type_map.get(source_type_str, SourceType.YOUTUBE_VOD)
        
        # Inicializar estado
        analysis_state[session_id] = {
            "status": "initialized",
            "source_type": source_type_str,
            "source_url": source_url,
            "progress": 0,
            "stats": None
        }
        
        # Ejecutar análisis en background con nuevo sistema
        import threading
        thread = threading.Thread(target=process_video_streaming, args=(session_id, source_type, source_url))
        thread.daemon = True
        thread.start()
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "message": "Análisis iniciado"
        })
        
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/analyze/{session_id}")
async def start_analysis(session_id: str, background_tasks: BackgroundTasks):
    """Iniciar análisis de video subido"""
    if session_id not in analysis_state:
        return JSONResponse({"success": False, "error": "Sesión no encontrada"}, status_code=404)
    
    state = analysis_state[session_id]
    file_path = state.get("file_path")
    
    if not file_path:
        return JSONResponse({"success": False, "error": "Archivo no encontrado"}, status_code=404)
    
    # Ejecutar análisis en background con nuevo sistema
    import threading
    thread = threading.Thread(target=process_video_streaming, args=(session_id, SourceType.UPLOADED_FILE, file_path))
    thread.daemon = True
    thread.start()
    
    return JSONResponse({"success": True, "message": "Análisis iniciado"})


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket para actualizaciones en tiempo real"""
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Mantener conexión abierta
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    """Obtener estado del análisis"""
    if session_id not in analysis_state:
        return JSONResponse({"success": False, "error": "Sesión no encontrada"}, status_code=404)
    
    return JSONResponse({
        "success": True,
        "data": analysis_state[session_id]
    })


async def generate_keypoints_heatmap(data_file, team_id: int, session_id: str):
    """Genera heatmap desde datos de keypoints (nuevo sistema)"""
    import io
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, Arc
    from matplotlib import cm
    from scipy.ndimage import gaussian_filter
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Cargar datos JSON
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        team_data = data.get(f'team_{team_id}', [])
        
        if not team_data:
            logger.warning(f"No hay datos para team {team_id}")
            return JSONResponse({"success": False, "error": f"No hay datos para team {team_id}"}, status_code=404)
        
        # Extraer posiciones
        xs = [p['position'][0] for p in team_data]
        ys = [p['position'][1] for p in team_data]
        
        # Crear heatmap 2D
        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=50, 
                                                  range=[[0, 105], [0, 68]])
        
        # Suavizar
        heatmap_smooth = gaussian_filter(heatmap.T, sigma=2)
        
        # Normalizar
        if heatmap_smooth.max() > 0:
            heatmap_norm = heatmap_smooth / heatmap_smooth.max()
        else:
            heatmap_norm = heatmap_smooth
        
        # Crear figura
        fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.8), facecolor='#1a1a1a')
        ax.set_facecolor('#2d5016')
        
        # Dibujar campo de fútbol (105m x 68m)
        field_color = 'white'
        lw = 2
        
        # Bordes
        ax.add_patch(Rectangle((0, 0), 105, 68, fill=False, edgecolor=field_color, linewidth=lw))
        
        # Línea media
        ax.plot([52.5, 52.5], [0, 68], color=field_color, linewidth=lw)
        
        # Círculo central
        center_circle = Circle((52.5, 34), 9.15, fill=False, edgecolor=field_color, linewidth=lw)
        ax.add_patch(center_circle)
        ax.plot(52.5, 34, 'o', color=field_color, markersize=4)
        
        # Áreas grandes
        ax.add_patch(Rectangle((0, 34-20.16), 16.5, 40.32, fill=False, edgecolor=field_color, linewidth=lw))
        ax.add_patch(Rectangle((105-16.5, 34-20.16), 16.5, 40.32, fill=False, edgecolor=field_color, linewidth=lw))
        
        # Áreas pequeñas
        ax.add_patch(Rectangle((0, 34-9.16), 5.5, 18.32, fill=False, edgecolor=field_color, linewidth=lw))
        ax.add_patch(Rectangle((105-5.5, 34-9.16), 5.5, 18.32, fill=False, edgecolor=field_color, linewidth=lw))
        
        # Arcos de penalty
        left_arc = Arc((11, 34), 18.3, 18.3, angle=0, theta1=308, theta2=52, edgecolor=field_color, linewidth=lw)
        right_arc = Arc((94, 34), 18.3, 18.3, angle=0, theta1=128, theta2=232, edgecolor=field_color, linewidth=lw)
        ax.add_patch(left_arc)
        ax.add_patch(right_arc)
        
        # Puntos de penalty
        ax.plot(11, 34, 'o', color=field_color, markersize=4)
        ax.plot(94, 34, 'o', color=field_color, markersize=4)
        
        # Heatmap
        cmap = cm.Greens if team_id == 0 else cm.Reds
        title_color = '#4ade80' if team_id == 0 else '#f87171'
        
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax.imshow(heatmap_norm, extent=extent, origin='lower', 
                 cmap=cmap, alpha=0.6, interpolation='bilinear')
        
        # Título
        ax.set_title(f'Team {team_id} Heatmap - Basado en Keypoints\n({len(team_data)} posiciones)',
                    fontsize=16, fontweight='bold', color=title_color, pad=20)
        
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Guardar
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', dpi=80, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        logger.info(f"Heatmap de keypoints generado para team {team_id} ({len(team_data)} posiciones)")
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error generando heatmap de keypoints: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/heatmap/{session_id}/{team_id}")
async def get_heatmap(session_id: str, team_id: int):
    """Obtener heatmap de un equipo como imagen PNG (sistema clásico o keypoints)"""
    import io
    from PIL import Image
    from matplotlib import cm
    import logging
    import json
    
    logger = logging.getLogger(__name__)
    
    try:
        # OPCIÓN 1: Buscar heatmap basado en keypoints (nuevo sistema)
        keypoints_paths = [
            BASE_DIR / "outputs_heatmap" / f"heatmap_data_{session_id}.json",
            BASE_DIR / "outputs_heatmap" / f"{session_id}_keypoints.json",
        ]
        
        # Buscar todos los archivos JSON en outputs_heatmap
        heatmap_dir = BASE_DIR / "outputs_heatmap"
        if heatmap_dir.exists():
            json_files = sorted(heatmap_dir.glob("heatmap_data_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if json_files:
                keypoints_paths.insert(0, json_files[0])  # Usar el más reciente
        
        keypoints_heatmap_path = None
        for path in keypoints_paths:
            if path.exists():
                keypoints_heatmap_path = path
                logger.info(f"✓ Heatmap de keypoints encontrado: {path}")
                break
        
        if keypoints_heatmap_path:
            # Usar sistema de keypoints
            return await generate_keypoints_heatmap(keypoints_heatmap_path, team_id, session_id)
        
        # OPCIÓN 2: Buscar heatmap clásico (.npz)
        search_paths = [
            BASE_DIR / "outputs_streaming" / f"{session_id}_heatmaps.npz",
            OUTPUT_DIR / f"{session_id}_heatmaps.npz",
            OUTPUT_DIR / session_id / f"{session_id}_heatmaps.npz",
        ]
        
        heatmap_path = None
        for path in search_paths:
            logger.info(f"Buscando heatmap en: {path}")
            if path.exists():
                heatmap_path = path
                logger.info(f"✓ Heatmap encontrado en: {path}")
                break
        
        if heatmap_path is None:
            logger.warning(f"Heatmap no encontrado para session {session_id}, team {team_id}. Rutas buscadas: {search_paths}")
            return JSONResponse({"success": False, "error": "Heatmap no encontrado"}, status_code=404)
        
        # Cargar heatmap
        logger.info(f"Cargando heatmap desde: {heatmap_path}")
        data = np.load(str(heatmap_path), allow_pickle=True)
        
        # Verificar que la clave existe
        heatmap_key = f'team_{team_id}_heatmap'
        if heatmap_key not in data:
            logger.error(f"Clave {heatmap_key} no encontrada. Claves disponibles: {list(data.keys())}")
            return JSONResponse({"success": False, "error": f"Heatmap para team {team_id} no encontrado"}, status_code=404)
        
        heatmap = data[heatmap_key]
        logger.info(f"Heatmap cargado para team {team_id}, shape: {heatmap.shape}, sum={heatmap.sum():.2f}")
        
        # Validar que haya datos
        if heatmap.sum() == 0:
            logger.warning(f"Heatmap para team {team_id} está vacío (sum=0). Generando imagen de advertencia.")
        
        # Normalizar heatmap para mejor visualización
        if heatmap.max() > 0:
            heatmap_norm = heatmap / heatmap.max()
        else:
            heatmap_norm = heatmap
        
        # Crear figura con matplotlib para mejor visualización
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.8), facecolor='#1a1a1a')
        ax.set_facecolor('#2d5016')  # Verde campo
        
        # Dibujar campo de fútbol
        # Líneas blancas del campo
        field_color = 'white'
        field_lw = 2
        
        # Borde del campo
        ax.plot([0, 1], [0, 0], color=field_color, linewidth=field_lw)
        ax.plot([1, 1], [0, 1], color=field_color, linewidth=field_lw)
        ax.plot([1, 0], [1, 1], color=field_color, linewidth=field_lw)
        ax.plot([0, 0], [1, 0], color=field_color, linewidth=field_lw)
        
        # Línea central
        ax.plot([0.5, 0.5], [0, 1], color=field_color, linewidth=field_lw)
        
        # Círculo central
        circle = plt.Circle((0.5, 0.5), 0.087, color=field_color, fill=False, linewidth=field_lw)
        ax.add_patch(circle)
        ax.plot(0.5, 0.5, 'o', color=field_color, markersize=4)
        
        # Áreas
        # Área izquierda
        ax.plot([0, 0.157], [0.296, 0.296], color=field_color, linewidth=field_lw)
        ax.plot([0.157, 0.157], [0.296, 0.704], color=field_color, linewidth=field_lw)
        ax.plot([0.157, 0], [0.704, 0.704], color=field_color, linewidth=field_lw)
        
        # Área derecha
        ax.plot([1, 0.843], [0.296, 0.296], color=field_color, linewidth=field_lw)
        ax.plot([0.843, 0.843], [0.296, 0.704], color=field_color, linewidth=field_lw)
        ax.plot([0.843, 1], [0.704, 0.704], color=field_color, linewidth=field_lw)
        
        # Superponer heatmap con transparencia
        if team_id == 0:
            cmap = cm.Greens
            title_color = '#4ade80'
        else:
            cmap = cm.Reds
            title_color = '#f87171'
        
        # Flip verticalmente para que coincida con orientación del campo
        heatmap_display = np.flipud(heatmap_norm)
        
        im = ax.imshow(heatmap_display, extent=[0, 1, 0, 1], cmap=cmap, 
                      alpha=0.7, interpolation='bilinear', origin='lower')
        
        # Título con información clara
        team_name = f"Team {team_id}"
        if heatmap.sum() == 0:
            title_text = f'{team_name} Heatmap - SIN DATOS (clasificación pendiente)'
            title_color = 'gray'
        else:
            title_text = f'{team_name} Heatmap (intensidad: {heatmap.sum():.1f})'
        
        ax.set_title(title_text, fontsize=16, fontweight='bold', color=title_color, pad=20)
        
        # Quitar ejes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Guardar en buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', dpi=80, bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        logger.info(f"Heatmap generado exitosamente para session {session_id}, team {team_id}")
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error al generar heatmap: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/heatmap-summary/{session_id}")
async def get_heatmap_summary(session_id: str):
    """Obtener resumen con ambos heatmaps lado a lado (soporta keypoints y clásico)"""
    import io
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.patches import Rectangle, Circle, Arc
    from scipy.ndimage import gaussian_filter
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # OPCIÓN 1: Buscar heatmap basado en keypoints
        heatmap_dir = BASE_DIR / "outputs_heatmap"
        keypoints_heatmap_path = None
        
        if heatmap_dir.exists():
            json_files = sorted(heatmap_dir.glob("heatmap_data_*.json"), 
                              key=lambda p: p.stat().st_mtime, reverse=True)
            if json_files:
                keypoints_heatmap_path = json_files[0]
                logger.info(f"✓ Usando heatmap de keypoints: {keypoints_heatmap_path}")
        
        if keypoints_heatmap_path:
            # Cargar datos de keypoints
            with open(keypoints_heatmap_path, 'r') as f:
                data = json.load(f)
            
            team_0_data = data.get('team_0', [])
            team_1_data = data.get('team_1', [])
            
            # Generar heatmaps
            heatmaps = []
            for team_data in [team_0_data, team_1_data]:
                if team_data:
                    xs = [p['position'][0] for p in team_data]
                    ys = [p['position'][1] for p in team_data]
                    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=50, 
                                                              range=[[0, 105], [0, 68]])
                    heatmap_smooth = gaussian_filter(heatmap.T, sigma=2)
                    heatmaps.append((heatmap_smooth, xedges, yedges, len(team_data)))
                else:
                    heatmaps.append((np.zeros((50, 50)), None, None, 0))
            
            # Crear figura
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 6.8), facecolor='#1a1a1a')
            
            for ax, (heatmap, xedges, yedges, count), team_id in [
                (ax0, heatmaps[0], 0),
                (ax1, heatmaps[1], 1)
            ]:
                ax.set_facecolor('#2d5016')
                
                # Campo de fútbol (105m x 68m)
                field_color = 'white'
                lw = 2
                
                ax.add_patch(Rectangle((0, 0), 105, 68, fill=False, edgecolor=field_color, linewidth=lw))
                ax.plot([52.5, 52.5], [0, 68], color=field_color, linewidth=lw)
                
                center_circle = Circle((52.5, 34), 9.15, fill=False, edgecolor=field_color, linewidth=lw)
                ax.add_patch(center_circle)
                ax.plot(52.5, 34, 'o', color=field_color, markersize=4)
                
                ax.add_patch(Rectangle((0, 34-20.16), 16.5, 40.32, fill=False, edgecolor=field_color, linewidth=lw))
                ax.add_patch(Rectangle((105-16.5, 34-20.16), 16.5, 40.32, fill=False, edgecolor=field_color, linewidth=lw))
                ax.add_patch(Rectangle((0, 34-9.16), 5.5, 18.32, fill=False, edgecolor=field_color, linewidth=lw))
                ax.add_patch(Rectangle((105-5.5, 34-9.16), 5.5, 18.32, fill=False, edgecolor=field_color, linewidth=lw))
                
                left_arc = Arc((11, 34), 18.3, 18.3, angle=0, theta1=308, theta2=52, edgecolor=field_color, linewidth=lw)
                right_arc = Arc((94, 34), 18.3, 18.3, angle=0, theta1=128, theta2=232, edgecolor=field_color, linewidth=lw)
                ax.add_patch(left_arc)
                ax.add_patch(right_arc)
                
                ax.plot(11, 34, 'o', color=field_color, markersize=4)
                ax.plot(94, 34, 'o', color=field_color, markersize=4)
                
                # Heatmap
                if xedges is not None and count > 0:
                    heatmap_norm = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
                    cmap = cm.Greens if team_id == 0 else cm.Reds
                    title_color = '#4ade80' if team_id == 0 else '#f87171'
                    
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    ax.imshow(heatmap_norm, extent=extent, origin='lower',
                             cmap=cmap, alpha=0.6, interpolation='bilinear')
                    
                    ax.set_title(f'Team {team_id} Heatmap\n({count} posiciones)',
                                fontsize=18, fontweight='bold', color=title_color, pad=20)
                else:
                    ax.set_title(f'Team {team_id} - Sin datos',
                                fontsize=18, fontweight='bold', color='gray', pad=20)
                
                ax.set_xlim(0, 105)
                ax.set_ylim(0, 68)
                ax.set_aspect('equal')
                ax.axis('off')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='PNG', dpi=100, bbox_inches='tight',
                       facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
            buf.seek(0)
            
            logger.info(f"Heatmap summary (keypoints) generado para {session_id}")
            
            from fastapi.responses import StreamingResponse
            return StreamingResponse(buf, media_type="image/png")
        
        # OPCIÓN 2: Sistema clásico (.npz)
        search_paths = [
            BASE_DIR / "outputs_streaming" / f"{session_id}_heatmaps.npz",
            OUTPUT_DIR / f"{session_id}_heatmaps.npz",
            OUTPUT_DIR / session_id / f"{session_id}_heatmaps.npz",
        ]
        
        heatmap_path = None
        for path in search_paths:
            if path.exists():
                heatmap_path = path
                break
        
        if heatmap_path is None:
            logger.warning(f"Heatmap summary no encontrado para session {session_id}")
            return JSONResponse({"success": False, "error": "Heatmap no encontrado"}, status_code=404)
        
        # Cargar datos
        data = np.load(str(heatmap_path), allow_pickle=True)
        heatmap_0 = data['team_0_heatmap']
        heatmap_1 = data['team_1_heatmap']
        
        # Normalizar
        heatmap_0_norm = heatmap_0 / heatmap_0.max() if heatmap_0.max() > 0 else heatmap_0
        heatmap_1_norm = heatmap_1 / heatmap_1.max() if heatmap_1.max() > 0 else heatmap_1
        
        # Crear figura con 2 subplots
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 6.8), facecolor='#1a1a1a')
        
        for ax, heatmap_norm, heatmap_orig, team_id in [
            (ax0, heatmap_0_norm, heatmap_0, 0),
            (ax1, heatmap_1_norm, heatmap_1, 1)
        ]:
            ax.set_facecolor('#2d5016')
            
            # Campo de fútbol
            field_color = 'white'
            field_lw = 2
            
            ax.plot([0, 1], [0, 0], color=field_color, linewidth=field_lw)
            ax.plot([1, 1], [0, 1], color=field_color, linewidth=field_lw)
            ax.plot([1, 0], [1, 1], color=field_color, linewidth=field_lw)
            ax.plot([0, 0], [1, 0], color=field_color, linewidth=field_lw)
            ax.plot([0.5, 0.5], [0, 1], color=field_color, linewidth=field_lw)
            
            circle = plt.Circle((0.5, 0.5), 0.087, color=field_color, fill=False, linewidth=field_lw)
            ax.add_patch(circle)
            ax.plot(0.5, 0.5, 'o', color=field_color, markersize=4)
            
            # Áreas
            ax.plot([0, 0.157], [0.296, 0.296], color=field_color, linewidth=field_lw)
            ax.plot([0.157, 0.157], [0.296, 0.704], color=field_color, linewidth=field_lw)
            ax.plot([0.157, 0], [0.704, 0.704], color=field_color, linewidth=field_lw)
            ax.plot([1, 0.843], [0.296, 0.296], color=field_color, linewidth=field_lw)
            ax.plot([0.843, 0.843], [0.296, 0.704], color=field_color, linewidth=field_lw)
            ax.plot([0.843, 1], [0.704, 0.704], color=field_color, linewidth=field_lw)
            
            # Heatmap
            cmap = cm.Greens if team_id == 0 else cm.Reds
            title_color = '#4ade80' if team_id == 0 else '#f87171'
            
            heatmap_display = np.flipud(heatmap_norm)
            ax.imshow(heatmap_display, extent=[0, 1, 0, 1], cmap=cmap, 
                     alpha=0.7, interpolation='bilinear', origin='lower')
            
            ax.set_title(f'Team {team_id} Heatmap\n(intensidad: {heatmap_orig.sum():.1f})',
                        fontsize=18, fontweight='bold', color=title_color, pad=20)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Guardar en buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', dpi=100, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        logger.info(f"Heatmap summary generado para session {session_id}")
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error al generar heatmap summary: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


async def process_video(session_id: str):
    """Procesar video con tracking, clasificación y posesión (versión async wrapper)"""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, process_video_sync, session_id)


def process_video_streaming(session_id: str, source_type: SourceType, source: str):
    """Procesar video con sistema de micro-batching"""
    
    try:
        state = analysis_state[session_id]
        state["status"] = "processing"
        
        # Crear nuevo event loop para este thread
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Callbacks para progreso
        def on_progress(match_id, batch_idx, frames_processed, total_frames):
            try:
                progress_percent = int((frames_processed / total_frames * 100)) if total_frames > 0 else 0
                loop.run_until_complete(manager.send_update(session_id, {
                    "type": "progress",
                    "batch_idx": batch_idx,
                    "frame": frames_processed,
                    "total_frames": total_frames,
                    "progress": progress_percent,
                    "message": f"Procesando batch {batch_idx + 1}... ({frames_processed}/{total_frames} frames)"
                }))
                state["progress"] = progress_percent
            except Exception as e:
                print(f"Error en on_progress: {e}")
        
        def on_frame_visualized(match_id, frame, frame_idx):
            """Callback para enviar frames anotados por WebSocket"""
            try:
                import cv2
                import base64
                
                # Codificar frame a JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Enviar por WebSocket
                loop.run_until_complete(manager.send_update(session_id, {
                    "type": "frame",
                    "frame_idx": frame_idx,
                    "image": frame_base64
                }))
            except Exception as e:
                print(f"Error en on_frame_visualized: {e}")
        
        def on_batch_complete(match_id, batch_idx, chunk_output, match_state):
            # Actualizar stats en tiempo real
            try:
                # Obtener resumen de estadísticas acumuladas
                summary = match_state.get_summary()
                
                # Extraer datos para el frontend
                possession_data = summary['possession']
                passes_data = summary['passes']
                
                # Formatear para Chart.js
                possession_percent = [
                    possession_data['percent_by_team'].get(0, 0),
                    possession_data['percent_by_team'].get(1, 0)
                ]
                
                possession_seconds = [
                    possession_data['seconds_by_team'].get(0, 0),
                    possession_data['seconds_by_team'].get(1, 0)
                ]
                
                passes = [
                    passes_data['by_team'].get(0, 0),
                    passes_data['by_team'].get(1, 0)
                ]
                
                # Stats del chunk actual
                chunk_stats = chunk_output.chunk_stats if hasattr(chunk_output, 'chunk_stats') else {}
                total_detections = chunk_stats.get('detections_count', 0)
                events_count = chunk_stats.get('events_count', 0)
                
                # Estadísticas espaciales
                spatial_stats = chunk_stats.get('spatial', {})
                
                loop.run_until_complete(manager.send_update(session_id, {
                    "type": "batch_complete",
                    "batch_idx": batch_idx,
                    "start_frame": chunk_output.start_frame,
                    "end_frame": chunk_output.end_frame,
                    "processing_time_ms": chunk_output.processing_time_ms,
                    "stats": {
                        "possession_percent": possession_percent,
                        "possession_seconds": possession_seconds,
                        "passes": passes,
                        "detections": total_detections,
                        "events": events_count,
                        "current_team": possession_data['current_team'],
                        "current_player": possession_data['current_player'],
                        # Estadísticas espaciales
                        "spatial": {
                            "calibration_valid": spatial_stats.get('calibration_valid', False),
                            "possession_by_zone": spatial_stats.get('possession_by_zone', {}),
                            "zone_percentages": spatial_stats.get('zone_percentages', {}),
                            "partition_type": spatial_stats.get('zone_partition_type', 'thirds_lanes'),
                            "num_zones": spatial_stats.get('num_zones', 9)
                        }
                    },
                    "message": f"✓ Batch {batch_idx + 1} completado: Team 0: {possession_percent[0]:.1f}%, Team 1: {possession_percent[1]:.1f}%"
                }))
            except Exception as e:
                print(f"Error en on_batch_complete: {e}")
                import traceback
                traceback.print_exc()
        
        def on_error(match_id, batch_idx, error):
            try:
                loop.run_until_complete(manager.send_update(session_id, {
                    "type": "error",
                    "message": str(error),
                    "batch_idx": batch_idx
                }))
            except Exception as e:
                print(f"Error en on_error: {e}")
        
        # Configuración del análisis
        config = AnalysisConfig(
            source_type=source_type,
            source=source,
            batch_size_seconds=3.0,
            output_dir=str(BASE_DIR / "outputs_streaming"),
            device="cuda" if Path("weights/best.pt").exists() else "cpu",
            model_path="weights/best.pt",
            conf_threshold=0.30,
            on_progress=on_progress,
            on_batch_complete=on_batch_complete,
            on_error=on_error,
            on_frame_visualized=on_frame_visualized,
            # Spatial tracking habilitado
            enable_spatial_tracking=True,
            zone_partition_type='thirds_lanes',
            enable_heatmaps=True,
            heatmap_resolution=(50, 34)
        )
        
        # Ejecutar análisis con micro-batching
        try:
            loop.run_until_complete(manager.send_update(session_id, {
                "type": "status",
                "message": "Iniciando análisis con micro-batching..."
            }))
        except Exception as e:
            print(f"Error enviando mensaje inicial: {e}")
        
        final_state = run_match_analysis(
            match_id=session_id,
            config=config,
            resume=False
        )
        
        # Obtener estadísticas finales
        possession_stats = final_state.possession_state
        total_frames = final_state.total_frames_processed
        fps = final_state.fps or 30.0
        total_seconds = total_frames / fps
        
        # Calcular porcentajes
        frames_by_team = possession_stats.frames_by_team
        total_possession_frames = sum(frames_by_team.values())
        
        possession_percent = {}
        possession_seconds = {}
        for team_id, frames in frames_by_team.items():
            if team_id >= 0:  # Excluir -1 (sin posesión)
                pct = (frames / total_possession_frames * 100) if total_possession_frames > 0 else 0
                possession_percent[team_id] = pct
                possession_seconds[team_id] = frames / fps
        
        final_stats = {
            "total_frames": total_frames,
            "total_seconds": total_seconds,
            "possession_percent": possession_percent,
            "possession_seconds": possession_seconds,
            "passes": possession_stats.passes_by_team,
            "possession_changes": possession_stats.possession_changes
        }
        
        # Añadir estadísticas espaciales si están disponibles
        if config.enable_spatial_tracking:
            try:
                # Verificar si el archivo de heatmaps existe
                search_paths = [
                    BASE_DIR / "outputs_streaming" / f"{session_id}_heatmaps.npz",
                    OUTPUT_DIR / f"{session_id}_heatmaps.npz"
                ]
                
                spatial_available = False
                for path in search_paths:
                    if path.exists():
                        spatial_available = True
                        print(f"✓ Heatmaps disponibles para session {session_id} en {path}")
                        break
                
                final_stats["spatial"] = {
                    "calibration_valid": spatial_available,
                    "heatmaps_available": spatial_available,
                    "session_id": session_id
                }
                
                if not spatial_available:
                    print(f"⚠ Heatmaps no encontrados. Rutas buscadas: {search_paths}")
            except Exception as e:
                print(f"Error verificando heatmaps: {e}")
                final_stats["spatial"] = {"calibration_valid": False}
        
        state["status"] = "completed"
        state["progress"] = 100
        state["stats"] = final_stats
        
        try:
            loop.run_until_complete(manager.send_update(session_id, {
                "type": "completed",
                "stats": final_stats
            }))
        except Exception as e:
            print(f"Error enviando completed: {e}")
        
        # Cerrar el loop
        loop.close()
        
    except Exception as e:
        print(f"Error en análisis con micro-batching: {e}")
        traceback.print_exc()
        
        state["status"] = "error"
        state["error"] = str(e)
        
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.send_update(session_id, {
                "type": "error",
                "message": str(e)
            }))
            loop.close()
        except Exception as e2:
            print(f"Error enviando error message: {e2}")


def process_video_sync(session_id: str):
    """Procesar video con tracking, clasificación y posesión (versión sync - LEGACY)"""
    import asyncio
    
    try:
        state = analysis_state[session_id]
        state["status"] = "processing"
        
        video_path = state["file_path"]
        
        # Enviar update inicial
        asyncio.run(manager.send_update(session_id, {
            "type": "status",
            "message": "Cargando modelo YOLO..."
        }))
        
        model = YOLO("weights/best.pt")
        
        # Inicializar módulos
        tracker = ReIDTracker(max_age=30, max_lost_time=120.0)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        possession = PossessionTrackerV2(fps=fps, hysteresis_frames=5)
        team_classifier = TeamClassifierV2(
            kmeans_min_tracks=12,
            vote_history=4,
            use_L_channel=True,
            L_weight=0.5
        )
        tracker.team_classifier = team_classifier
        
        # Enviar update (sync)
        asyncio.run(manager.send_update(session_id, {
            "type": "status",
            "message": "Iniciando análisis..."
        }))
        
        # Procesar video
        cap = cv2.VideoCapture(video_path)
        frame_no = 0
        
        # Estadísticas acumuladas
        possession_history = []
        passes_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_no += 1
            
            # Detección YOLO
            results = model(frame, imgsz=640, conf=0.30, max_det=200, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                r = results[0]
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                
                # Tracking
                tracks = tracker.update(frame, boxes, scores, cls_ids)
                
                # Detección de posesión
                ball_pos = None
                for tid, bbox, cid in tracks:
                    if cid == 1:  # ball
                        x1, y1, x2, y2 = map(int, bbox)
                        ball_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        break
                
                ball_owner_team = None
                ball_owner_id = None
                min_distance = float('inf')
                
                if ball_pos is not None:
                    for tid, bbox, cid in tracks:
                        if cid not in (0, 3):  # Solo jugadores y porteros
                            continue
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        player_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        dist = np.hypot(player_pos[0] - ball_pos[0], player_pos[1] - ball_pos[1])
                        
                        if dist < min_distance:
                            min_distance = dist
                            ball_owner_id = tid
                            try:
                                ball_owner_team = tracker.active_tracks[tid].team_id
                            except Exception:
                                ball_owner_team = None
                
                # Validar distancia
                if min_distance > 60:
                    ball_owner_team = None
                    ball_owner_id = None
                
                # Filtrar team_id inválidos
                if ball_owner_team is not None and ball_owner_team < 0:
                    ball_owner_team = None
                
                # Actualizar posesión
                possession.update(frame_no, ball_owner_team, ball_owner_id)
            
            # Enviar actualización cada 100 frames
            if frame_no % 100 == 0:
                stats = possession.get_possession_stats()
                progress = int((frame_no / total_frames) * 100)
                
                # Enviar update (sync)
                asyncio.run(manager.send_update(session_id, {
                    "type": "progress",
                    "frame": frame_no,
                    "total_frames": total_frames,
                    "progress": progress,
                    "stats": {
                        "possession_percent": stats['possession_percent'],
                        "possession_seconds": stats['possession_seconds'],
                        "passes": stats['passes']
                    }
                }))
                
                state["progress"] = progress
        
        cap.release()
        
        # Estadísticas finales
        final_stats = possession.get_possession_stats()
        timeline = possession.get_possession_timeline()
        
        state["status"] = "completed"
        state["progress"] = 100
        state["stats"] = {
            "total_frames": final_stats['total_frames'],
            "total_seconds": final_stats['total_seconds'],
            "possession_percent": final_stats['possession_percent'],
            "possession_seconds": final_stats['possession_seconds'],
            "passes": final_stats['passes'],
            "timeline": [(int(s), int(e), int(t)) for s, e, t in timeline]
        }
        
        asyncio.run(manager.send_update(session_id, {
            "type": "completed",
            "stats": state["stats"]
        }))
        
    except Exception as e:
        print(f"Error procesando video: {e}")
        traceback.print_exc()
        
        state["status"] = "error"
        state["error"] = str(e)
        
        asyncio.run(manager.send_update(session_id, {
            "type": "error",
            "message": str(e)
        }))


# Montar archivos estáticos AL FINAL (después de todas las rutas)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 TacticEYE2 Web Application")
    print("="*60)
    print("\n📱 Abre en tu navegador:")
    print("   http://localhost:8000")
    print("   http://127.0.0.1:8000")
    print("\n💡 Presiona Ctrl+C para detener el servidor\n")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
