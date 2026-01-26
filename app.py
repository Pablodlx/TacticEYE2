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


@app.get("/api/heatmap/{session_id}/{team_id}")
async def get_heatmap(session_id: str, team_id: int):
    """Obtener heatmap de un equipo como imagen PNG"""
    import io
    from PIL import Image
    from matplotlib import cm
    
    try:
        # Buscar archivo de heatmaps
        heatmap_path = OUTPUT_DIR / f"{session_id}_heatmaps.npz"
        
        if not heatmap_path.exists():
            # Buscar en subdirectorio
            heatmap_path = OUTPUT_DIR / session_id / f"{session_id}_heatmaps.npz"
        
        if not heatmap_path.exists():
            return JSONResponse({"success": False, "error": "Heatmap no encontrado"}, status_code=404)
        
        # Cargar heatmap
        data = np.load(str(heatmap_path))
        heatmap = data[f'team_{team_id}_heatmap']
        
        # Aplicar colormap
        if team_id == 0:
            colored = cm.Greens(heatmap)
        else:
            colored = cm.Reds(heatmap)
        
        # Convertir a imagen
        img_array = (colored[:,:,:3] * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Redimensionar para visualización (mantener aspect ratio del campo)
        img = img.resize((525, 340), Image.BILINEAR)
        
        # Guardar en buffer
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        print(f"Error al generar heatmap: {e}")
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
