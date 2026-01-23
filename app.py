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


@app.post("/api/analyze/{session_id}")
async def start_analysis(session_id: str, background_tasks: BackgroundTasks):
    """Iniciar análisis de video"""
    if session_id not in analysis_state:
        return JSONResponse({"success": False, "error": "Sesión no encontrada"}, status_code=404)
    
    # Ejecutar análisis en background (sin async)
    import threading
    thread = threading.Thread(target=process_video_sync, args=(session_id,))
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


async def process_video(session_id: str):
    """Procesar video con tracking, clasificación y posesión (versión async wrapper)"""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, process_video_sync, session_id)


def process_video_sync(session_id: str):
    """Procesar video con tracking, clasificación y posesión (versión sync)"""
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
