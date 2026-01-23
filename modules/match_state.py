"""
Match State Management
======================

Estado persistente del análisis de un partido.
Permite análisis incremental y recuperación ante fallos.
"""

import json
import pickle
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np


@dataclass
class TrackerState:
    """
    Estado del tracker ReID.
    
    Contiene toda la información necesaria para continuar
    el tracking entre batches.
    """
    # IDs activos y sus features
    active_tracks: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Contador de IDs para asignar nuevos
    next_id: int = 0
    
    # Tracks perdidos (para reidentificación)
    lost_tracks: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Frame del último update
    last_frame_idx: int = -1
    
    def to_dict(self) -> dict:
        """Serializa a dict (para JSON/pickle)"""
        return {
            'active_tracks': self.active_tracks,
            'next_id': self.next_id,
            'lost_tracks': self.lost_tracks,
            'last_frame_idx': self.last_frame_idx
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrackerState':
        """Deserializa desde dict"""
        return cls(**data)


@dataclass
class TeamClassifierState:
    """
    Estado del clasificador de equipos.
    
    Mantiene las asignaciones de jugadores a equipos y
    los modelos de color de cada equipo.
    """
    # ID jugador -> ID equipo
    player_team_map: Dict[int, int] = field(default_factory=dict)
    
    # Historial de votaciones por jugador
    vote_history: Dict[int, List[int]] = field(default_factory=dict)
    
    # Centros de clusters (colores representativos)
    # team_id -> np.array de color LAB
    team_colors: Dict[int, List[float]] = field(default_factory=dict)
    
    # Número de frames usados para entrenar
    trained_frames: int = 0
    
    # Flag de si ya se hizo el entrenamiento inicial
    is_trained: bool = False
    
    def to_dict(self) -> dict:
        return {
            'player_team_map': self.player_team_map,
            'vote_history': self.vote_history,
            'team_colors': {k: [float(x) for x in v] for k, v in self.team_colors.items()},
            'trained_frames': self.trained_frames,
            'is_trained': self.is_trained
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TeamClassifierState':
        state = cls()
        state.player_team_map = data.get('player_team_map', {})
        state.vote_history = data.get('vote_history', {})
        state.team_colors = {int(k): v for k, v in data.get('team_colors', {}).items()}
        state.trained_frames = data.get('trained_frames', 0)
        state.is_trained = data.get('is_trained', False)
        return state


@dataclass
class PossessionState:
    """
    Estado del tracking de posesión.
    
    Mantiene estadísticas acumuladas de posesión del balón.
    """
    # Equipo actual con posesión (-1 = ninguno)
    current_team: int = -1
    
    # Jugador actual con posesión (-1 = ninguno)
    current_player: int = -1
    
    # Frames acumulados por equipo
    frames_by_team: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, -1: 0})
    
    # Pases completados por equipo
    passes_by_team: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    
    # Historial de cambios de posesión
    possession_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Último frame procesado
    last_frame_idx: int = -1
    
    def to_dict(self) -> dict:
        return {
            'current_team': self.current_team,
            'current_player': self.current_player,
            'frames_by_team': self.frames_by_team,
            'passes_by_team': self.passes_by_team,
            'possession_changes': self.possession_changes,
            'last_frame_idx': self.last_frame_idx
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PossessionState':
        state = cls()
        state.current_team = data.get('current_team', -1)
        state.current_player = data.get('current_player', -1)
        state.frames_by_team = data.get('frames_by_team', {0: 0, 1: 0, -1: 0})
        state.passes_by_team = data.get('passes_by_team', {0: 0, 1: 0})
        state.possession_changes = data.get('possession_changes', [])
        state.last_frame_idx = data.get('last_frame_idx', -1)
        return state


@dataclass
class MatchState:
    """
    Estado completo del análisis de un partido.
    
    Contiene todos los estados de los módulos del pipeline
    más metadata del partido.
    """
    # Identificador único del partido
    match_id: str = ""
    
    # Metadata de la fuente
    source_type: str = ""
    source_url: str = ""
    fps: float = 30.0
    
    # Estado del partido
    status: str = "initialized"  # initialized, running, paused, completed, failed
    
    # Progreso
    total_frames_processed: int = 0
    last_batch_idx: int = -1
    last_update: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Estados de los módulos
    tracker_state: TrackerState = field(default_factory=TrackerState)
    team_classifier_state: TeamClassifierState = field(default_factory=TeamClassifierState)
    possession_state: PossessionState = field(default_factory=PossessionState)
    
    # Resultados acumulados (para queries rápidas)
    cached_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata adicional
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_progress(self, batch_idx: int, frames_in_batch: int):
        """Actualiza el progreso del análisis"""
        self.last_batch_idx = batch_idx
        self.total_frames_processed += frames_in_batch
        self.last_update = datetime.utcnow().isoformat()
    
    def mark_running(self):
        """Marca el partido como en proceso"""
        self.status = "running"
        self.last_update = datetime.utcnow().isoformat()
    
    def mark_completed(self):
        """Marca el partido como completado"""
        self.status = "completed"
        self.last_update = datetime.utcnow().isoformat()
    
    def mark_failed(self, error: str):
        """Marca el partido como fallido"""
        self.status = "failed"
        self.metadata['error'] = error
        self.last_update = datetime.utcnow().isoformat()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Genera resumen de estadísticas actuales.
        
        Útil para consultas sin necesidad de cargar todo el estado.
        """
        # Calcular estadísticas de posesión
        total_frames = sum(self.possession_state.frames_by_team.values())
        
        if total_frames > 0:
            possession_percent = {
                team_id: (frames / total_frames) * 100
                for team_id, frames in self.possession_state.frames_by_team.items()
            }
        else:
            possession_percent = {0: 0, 1: 0, -1: 0}
        
        possession_seconds = {
            team_id: frames / self.fps if self.fps > 0 else 0
            for team_id, frames in self.possession_state.frames_by_team.items()
        }
        
        summary = {
            'match_id': self.match_id,
            'status': self.status,
            'progress': {
                'total_frames': self.total_frames_processed,
                'total_seconds': self.total_frames_processed / self.fps if self.fps > 0 else 0,
                'last_batch': self.last_batch_idx,
                'last_update': self.last_update
            },
            'possession': {
                'current_team': self.possession_state.current_team,
                'current_player': self.possession_state.current_player,
                'percent_by_team': possession_percent,
                'seconds_by_team': possession_seconds,
                'frames_by_team': self.possession_state.frames_by_team
            },
            'passes': {
                'by_team': self.possession_state.passes_by_team,
                'total': sum(self.possession_state.passes_by_team.values())
            },
            'teams': {
                'total_players': len(self.team_classifier_state.player_team_map),
                'team_0_players': sum(1 for t in self.team_classifier_state.player_team_map.values() if t == 0),
                'team_1_players': sum(1 for t in self.team_classifier_state.player_team_map.values() if t == 1),
            },
            'tracking': {
                'active_tracks': len(self.tracker_state.active_tracks),
                'total_ids': self.tracker_state.next_id
            }
        }
        
        # Cachear para queries rápidas
        self.cached_summary = summary
        
        return summary
    
    def to_dict(self) -> dict:
        """Serializa todo el estado a dict"""
        return {
            'match_id': self.match_id,
            'source_type': self.source_type,
            'source_url': self.source_url,
            'fps': self.fps,
            'status': self.status,
            'total_frames_processed': self.total_frames_processed,
            'last_batch_idx': self.last_batch_idx,
            'last_update': self.last_update,
            'tracker_state': self.tracker_state.to_dict(),
            'team_classifier_state': self.team_classifier_state.to_dict(),
            'possession_state': self.possession_state.to_dict(),
            'cached_summary': self.cached_summary,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MatchState':
        """Deserializa desde dict"""
        state = cls()
        state.match_id = data.get('match_id', '')
        state.source_type = data.get('source_type', '')
        state.source_url = data.get('source_url', '')
        state.fps = data.get('fps', 30.0)
        state.status = data.get('status', 'initialized')
        state.total_frames_processed = data.get('total_frames_processed', 0)
        state.last_batch_idx = data.get('last_batch_idx', -1)
        state.last_update = data.get('last_update', datetime.utcnow().isoformat())
        
        if 'tracker_state' in data:
            state.tracker_state = TrackerState.from_dict(data['tracker_state'])
        
        if 'team_classifier_state' in data:
            state.team_classifier_state = TeamClassifierState.from_dict(data['team_classifier_state'])
        
        if 'possession_state' in data:
            state.possession_state = PossessionState.from_dict(data['possession_state'])
        
        state.cached_summary = data.get('cached_summary', {})
        state.metadata = data.get('metadata', {})
        
        return state
    
    def save_to_file(self, file_path: str, format: str = 'json'):
        """
        Guarda el estado a archivo.
        
        Args:
            file_path: Ruta del archivo
            format: 'json' o 'pickle'
        """
        data = self.to_dict()
        
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    @classmethod
    def load_from_file(cls, file_path: str, format: str = 'json') -> 'MatchState':
        """
        Carga el estado desde archivo.
        
        Args:
            file_path: Ruta del archivo
            format: 'json' o 'pickle'
        """
        if format == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif format == 'pickle':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        return cls.from_dict(data)


# ============================================================================
# Persistencia - Adaptadores para diferentes backends
# ============================================================================

class StateStorage:
    """Interfaz base para almacenamiento de estados"""
    
    def save(self, match_id: str, state: MatchState):
        raise NotImplementedError
    
    def load(self, match_id: str) -> Optional[MatchState]:
        raise NotImplementedError
    
    def exists(self, match_id: str) -> bool:
        raise NotImplementedError
    
    def delete(self, match_id: str):
        raise NotImplementedError
    
    def list_matches(self) -> List[str]:
        raise NotImplementedError


class FileSystemStorage(StateStorage):
    """Almacenamiento en sistema de archivos"""
    
    def __init__(self, base_dir: str = "match_states"):
        self.base_dir = base_dir
        import os
        os.makedirs(base_dir, exist_ok=True)
    
    def _get_path(self, match_id: str) -> str:
        import os
        return os.path.join(self.base_dir, f"{match_id}.json")
    
    def save(self, match_id: str, state: MatchState):
        state.save_to_file(self._get_path(match_id), format='json')
    
    def load(self, match_id: str) -> Optional[MatchState]:
        import os
        path = self._get_path(match_id)
        if not os.path.exists(path):
            return None
        return MatchState.load_from_file(path, format='json')
    
    def exists(self, match_id: str) -> bool:
        import os
        return os.path.exists(self._get_path(match_id))
    
    def delete(self, match_id: str):
        import os
        path = self._get_path(match_id)
        if os.path.exists(path):
            os.remove(path)
    
    def list_matches(self) -> List[str]:
        import os
        files = os.listdir(self.base_dir)
        return [f.replace('.json', '') for f in files if f.endswith('.json')]


class RedisStorage(StateStorage):
    """
    Almacenamiento en Redis (ejemplo).
    
    Útil para:
    - Acceso rápido
    - Múltiples workers
    - TTL automático
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        # Requiere: pip install redis
        try:
            import redis
            self.redis = redis.from_url(redis_url)
        except ImportError:
            raise ImportError("Redis storage requiere: pip install redis")
    
    def _get_key(self, match_id: str) -> str:
        return f"match_state:{match_id}"
    
    def save(self, match_id: str, state: MatchState):
        key = self._get_key(match_id)
        data = json.dumps(state.to_dict())
        self.redis.set(key, data)
        # Opcional: TTL de 7 días
        self.redis.expire(key, 7 * 24 * 3600)
    
    def load(self, match_id: str) -> Optional[MatchState]:
        key = self._get_key(match_id)
        data = self.redis.get(key)
        if not data:
            return None
        return MatchState.from_dict(json.loads(data))
    
    def exists(self, match_id: str) -> bool:
        return self.redis.exists(self._get_key(match_id)) > 0
    
    def delete(self, match_id: str):
        self.redis.delete(self._get_key(match_id))
    
    def list_matches(self) -> List[str]:
        keys = self.redis.keys("match_state:*")
        return [k.decode().replace('match_state:', '') for k in keys]


# Factory por defecto
_default_storage = None

def get_default_storage() -> StateStorage:
    """Retorna el storage por defecto (FileSystem)"""
    global _default_storage
    if _default_storage is None:
        _default_storage = FileSystemStorage()
    return _default_storage


def set_default_storage(storage: StateStorage):
    """Configura el storage por defecto"""
    global _default_storage
    _default_storage = storage
