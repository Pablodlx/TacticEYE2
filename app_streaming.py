"""
TacticEYE Streaming API
=======================

API FastAPI con soporte para micro-batching y análisis en tiempo real.
Soporta múltiples fuentes de video: archivos, YouTube, HLS, RTMP.
"""

import os
import uuid
import asyncio
import threading
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from pydantic import BaseModel

# Imports de módulos de micro-batching
from modules.video_sources import SourceType
from modules.match_state import (
    MatchState, get_default_storage, StateStorage, FileSystemStorage
)
from modules.match_analyzer import (
    AnalysisConfig, run_match_analysis
)
from modules.batch_processor import ChunkOutput, load_match_outputs


# ============================================================================
# Configuración de la aplicación
# ============================================================================

app = FastAPI(
    title="TacticEYE Streaming API",
    description="API para análisis de partidos de fútbol en tiempo real con micro-batching",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates y archivos estáticos
templates = Jinja2Templates(directory="templates")

# Storage
storage = FileSystemStorage("match_states")

# Directorio de uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("outputs_streaming", exist_ok=True)

# Estado en memoria de análisis activos
active_analyses: Dict[str, Dict] = {}


# ============================================================================
# WebSocket Manager
# ============================================================================

class ConnectionManager:
    """Gestor de conexiones WebSocket por partido"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, match_id: str, websocket: WebSocket):
        await websocket.accept()
        if match_id not in self.active_connections:
            self.active_connections[match_id] = []
        self.active_connections[match_id].append(websocket)
    
    def disconnect(self, match_id: str, websocket: WebSocket):
        if match_id in self.active_connections:
            self.active_connections[match_id].remove(websocket)
            if not self.active_connections[match_id]:
                del self.active_connections[match_id]
    
    async def send_update(self, match_id: str, message: dict):
        """Envía actualización a todos los clientes conectados al partido"""
        if match_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[match_id]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Limpiar conexiones muertas
            for conn in disconnected:
                self.disconnect(match_id, conn)


manager = ConnectionManager()


# ============================================================================
# Modelos Pydantic
# ============================================================================

class AnalyzeRequest(BaseModel):
    """Request para iniciar análisis"""
    match_id: str
    source_type: str  # uploaded_file, youtube_vod, youtube_live, hls, rtmp
    source_url: Optional[str] = None  # URL para streams
    file_id: Optional[str] = None  # ID de archivo subido
    
    # Configuración de batching
    batch_size_seconds: float = 3.0
    
    # Configuración del modelo
    model_path: str = "weights/best.pt"
    conf_threshold: float = 0.3
    
    # Límites (opcional, para testing)
    max_batches: Optional[int] = None


class MatchSummaryResponse(BaseModel):
    """Respuesta con resumen del partido"""
    match_id: str
    status: str
    progress: Dict
    possession: Dict
    passes: Dict
    teams: Dict
    tracking: Dict


# ============================================================================
# Funciones de análisis en background
# ============================================================================

def run_analysis_background(match_id: str, config: AnalysisConfig):
    """
    Ejecuta el análisis en un thread separado con callbacks para WebSocket.
    """
    
    def on_progress(match_id: str, progress: dict):
        """Callback de progreso"""
        try:
            # Calcular porcentaje
            percent = 0
            if progress.get('total_frames') and progress['total_frames'] > 0:
                percent = (progress['frames_processed'] / progress['total_frames']) * 100
            
            message = {
                'type': 'progress',
                'frame': progress['frames_processed'],
                'total_frames': progress.get('total_frames', 0),
                'progress': round(percent, 1),
                'fps_processing': round(progress.get('fps_processing', 0), 1),
                'realtime_factor': round(progress.get('realtime_factor', 0), 2),
                'message': f"Processing batch {progress['batch_idx']}..."
            }
            
            # Enviar por WebSocket (sync)
            asyncio.run(manager.send_update(match_id, message))
            
        except Exception as e:
            print(f"Error en on_progress: {e}")
    
    def on_batch_complete(match_id: str, output: ChunkOutput):
        """Callback cuando se completa un batch"""
        try:
            # Cargar estado actual
            state = storage.load(match_id)
            if state:
                summary = state.get_summary()
                
                message = {
                    'type': 'batch_complete',
                    'batch_idx': output.batch_idx,
                    'stats': summary
                }
                
                asyncio.run(manager.send_update(match_id, message))
        
        except Exception as e:
            print(f"Error en on_batch_complete: {e}")
    
    def on_error(match_id: str, batch_idx: int, error: Exception):
        """Callback de error"""
        try:
            message = {
                'type': 'error',
                'batch_idx': batch_idx,
                'message': str(error)
            }
            
            asyncio.run(manager.send_update(match_id, message))
        
        except Exception as e:
            print(f"Error en on_error: {e}")
    
    # Configurar callbacks
    config.on_progress = on_progress
    config.on_batch_complete = on_batch_complete
    config.on_error = on_error
    
    try:
        # Notificar inicio
        asyncio.run(manager.send_update(match_id, {
            'type': 'status',
            'message': 'Starting analysis...'
        }))
        
        # Ejecutar análisis
        final_state = run_match_analysis(match_id, config, resume=True)
        
        # Notificar completado
        summary = final_state.get_summary()
        asyncio.run(manager.send_update(match_id, {
            'type': 'completed',
            'stats': summary
        }))
        
        # Actualizar estado en memoria
        if match_id in active_analyses:
            active_analyses[match_id]['status'] = 'completed'
    
    except Exception as e:
        print(f"Error en análisis de {match_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Notificar error
        asyncio.run(manager.send_update(match_id, {
            'type': 'error',
            'message': str(e)
        }))
        
        if match_id in active_analyses:
            active_analyses[match_id]['status'] = 'failed'
            active_analyses[match_id]['error'] = str(e)


# ============================================================================
# Endpoints de la API
# ============================================================================

@app.get("/")
async def read_root(request: Request):
    """Página principal"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Sube un archivo de video.
    
    Returns:
        file_id para usar en /api/analyze
    """
    try:
        # Generar ID único
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
        
        # Guardar archivo
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "path": file_path
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/analyze")
async def start_analysis(request: AnalyzeRequest):
    """
    Inicia el análisis de un partido.
    
    Soporta múltiples fuentes:
    - uploaded_file: archivo previamente subido con /api/upload
    - youtube_vod: video de YouTube (VOD)
    - youtube_live: stream de YouTube en vivo
    - hls: stream HLS (.m3u8)
    - rtmp: stream RTMP
    - veo: stream de Veo
    """
    
    match_id = request.match_id
    
    # Verificar si ya existe un análisis activo
    if match_id in active_analyses:
        if active_analyses[match_id]['status'] == 'running':
            return {
                "success": False,
                "error": "Analysis already running for this match_id"
            }
    
    # Determinar source_url
    source_url = None
    
    if request.source_type == "uploaded_file":
        if not request.file_id:
            raise HTTPException(400, "file_id required for uploaded_file")
        
        # Buscar el archivo
        import glob
        files = glob.glob(os.path.join(UPLOAD_DIR, f"{request.file_id}.*"))
        if not files:
            raise HTTPException(404, f"File not found: {request.file_id}")
        
        source_url = files[0]
    
    else:
        if not request.source_url:
            raise HTTPException(400, "source_url required for streaming sources")
        
        source_url = request.source_url
    
    # Mapear source_type a enum
    source_type_map = {
        "uploaded_file": SourceType.UPLOADED_FILE,
        "youtube_vod": SourceType.YOUTUBE_VOD,
        "youtube_live": SourceType.YOUTUBE_LIVE,
        "hls": SourceType.HLS,
        "rtmp": SourceType.RTMP,
        "veo": SourceType.VEO,
    }
    
    if request.source_type not in source_type_map:
        raise HTTPException(400, f"Invalid source_type: {request.source_type}")
    
    source_type = source_type_map[request.source_type]
    
    # Crear configuración
    config = AnalysisConfig(
        source_type=source_type,
        source_url=source_url,
        batch_size_seconds=request.batch_size_seconds,
        model_path=request.model_path,
        conf_threshold=request.conf_threshold,
        storage=storage,
        max_batches=request.max_batches
    )
    
    # Registrar análisis activo
    active_analyses[match_id] = {
        'match_id': match_id,
        'status': 'running',
        'source_type': request.source_type,
        'source_url': source_url,
        'started_at': datetime.utcnow().isoformat()
    }
    
    # Iniciar análisis en thread separado
    thread = threading.Thread(
        target=run_analysis_background,
        args=(match_id, config),
        daemon=True
    )
    thread.start()
    
    return {
        "success": True,
        "match_id": match_id,
        "status": "Analysis started",
        "source_type": request.source_type
    }


@app.get("/api/match/{match_id}/summary")
async def get_match_summary(match_id: str):
    """
    Obtiene el resumen actual del partido.
    
    Incluye:
    - Progreso
    - Posesión
    - Pases
    - Estadísticas de equipos
    """
    
    # Intentar cargar estado
    state = storage.load(match_id)
    
    if not state:
        raise HTTPException(404, f"Match not found: {match_id}")
    
    summary = state.get_summary()
    
    return summary


@app.get("/api/match/{match_id}/events")
async def get_match_events(match_id: str, batch_from: int = 0, batch_to: Optional[int] = None):
    """
    Obtiene los eventos detectados en el partido.
    
    Args:
        batch_from: Batch inicial (inclusive)
        batch_to: Batch final (inclusive), None = todos
    
    Returns:
        Lista de eventos (pases, cambios de posesión, etc.)
    """
    
    # Cargar outputs
    outputs = load_match_outputs(match_id)
    
    if not outputs:
        raise HTTPException(404, f"No outputs found for match: {match_id}")
    
    # Filtrar por rango de batches
    if batch_to is None:
        batch_to = max(o.batch_idx for o in outputs)
    
    outputs_filtered = [o for o in outputs if batch_from <= o.batch_idx <= batch_to]
    
    # Consolidar eventos
    all_events = []
    for output in outputs_filtered:
        all_events.extend(output.events)
    
    # Ordenar por frame
    all_events.sort(key=lambda e: e['frame'])
    
    return {
        "match_id": match_id,
        "batch_from": batch_from,
        "batch_to": batch_to,
        "total_events": len(all_events),
        "events": all_events
    }


@app.get("/api/match/{match_id}/positions")
async def get_player_positions(
    match_id: str,
    frame_from: Optional[int] = None,
    frame_to: Optional[int] = None,
    player_id: Optional[int] = None,
    team_id: Optional[int] = None
):
    """
    Obtiene las posiciones de jugadores.
    
    Útil para:
    - Generar heatmaps
    - Visualizar trayectorias
    - Análisis táctico
    
    Filters:
        frame_from/frame_to: Rango de frames
        player_id: Solo un jugador específico
        team_id: Solo jugadores de un equipo
    """
    
    # Cargar outputs
    outputs = load_match_outputs(match_id)
    
    if not outputs:
        raise HTTPException(404, f"No outputs found for match: {match_id}")
    
    # Consolidar posiciones
    all_positions = []
    for output in outputs:
        all_positions.extend(output.player_positions)
    
    # Aplicar filtros
    filtered = all_positions
    
    if frame_from is not None:
        filtered = [p for p in filtered if p['frame'] >= frame_from]
    
    if frame_to is not None:
        filtered = [p for p in filtered if p['frame'] <= frame_to]
    
    if player_id is not None:
        filtered = [p for p in filtered if p['player_id'] == player_id]
    
    if team_id is not None:
        filtered = [p for p in filtered if p['team_id'] == team_id]
    
    return {
        "match_id": match_id,
        "total_positions": len(filtered),
        "positions": filtered
    }


@app.get("/api/match/{match_id}/status")
async def get_match_status(match_id: str):
    """
    Obtiene el estado actual del análisis.
    
    Útil para polling sin WebSocket.
    """
    
    # Verificar si está en memoria (análisis activo)
    if match_id in active_analyses:
        return active_analyses[match_id]
    
    # Buscar en storage
    state = storage.load(match_id)
    
    if not state:
        raise HTTPException(404, f"Match not found: {match_id}")
    
    return {
        "match_id": match_id,
        "status": state.status,
        "total_frames": state.total_frames_processed,
        "last_batch": state.last_batch_idx,
        "last_update": state.last_update
    }


@app.delete("/api/match/{match_id}")
async def delete_match(match_id: str):
    """
    Elimina un partido y sus datos.
    
    WARNING: Esta operación es irreversible.
    """
    
    # Eliminar estado
    if storage.exists(match_id):
        storage.delete(match_id)
    
    # Eliminar outputs
    import shutil
    output_dir = os.path.join("outputs_streaming", match_id)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Eliminar de memoria
    if match_id in active_analyses:
        del active_analyses[match_id]
    
    return {
        "success": True,
        "match_id": match_id,
        "message": "Match deleted"
    }


@app.get("/api/matches")
async def list_matches():
    """Lista todos los partidos almacenados"""
    
    matches = []
    
    for match_id in storage.list_matches():
        state = storage.load(match_id)
        if state:
            matches.append({
                "match_id": match_id,
                "status": state.status,
                "source_type": state.source_type,
                "total_frames": state.total_frames_processed,
                "last_update": state.last_update
            })
    
    return {
        "total": len(matches),
        "matches": matches
    }


@app.websocket("/ws/{match_id}")
async def websocket_endpoint(websocket: WebSocket, match_id: str):
    """
    WebSocket para recibir actualizaciones en tiempo real.
    
    Mensajes:
    - type: 'status' -> Cambio de estado
    - type: 'progress' -> Progreso del análisis
    - type: 'batch_complete' -> Batch completado con stats
    - type: 'completed' -> Análisis completado
    - type: 'error' -> Error en el análisis
    """
    
    await manager.connect(match_id, websocket)
    
    try:
        # Enviar estado inicial si existe
        state = storage.load(match_id)
        if state:
            await websocket.send_json({
                'type': 'initial_state',
                'status': state.status,
                'summary': state.get_summary()
            })
        
        # Mantener conexión abierta
        while True:
            # Esperar mensajes del cliente (ping/pong)
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        manager.disconnect(match_id, websocket)
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(match_id, websocket)


# Montar archivos estáticos (al final)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup():
    print("\n" + "="*60)
    print("🚀 TacticEYE Streaming API")
    print("="*60)
    print("\n📱 API Documentation:")
    print("   http://localhost:8000/docs")
    print("\n🌐 Web Interface:")
    print("   http://localhost:8000")
    print("\n💡 Supported sources:")
    print("   - Uploaded files")
    print("   - YouTube (VOD & Live)")
    print("   - HLS streams")
    print("   - RTMP streams")
    print("   - Veo")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
