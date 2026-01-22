"""
PossessionTrackerV2 - Módulo simplificado de cálculo de posesión

Reglas fundamentales:
1. Siempre hay un equipo en posesión (nunca tiempo "sin asignar")
2. Cuando ball_owner_team es None, el tiempo se asigna al último equipo que tenía posesión
3. Histeresis configurable para evitar cambios rápidos de posesión

API:
    tracker = PossessionTrackerV2(fps=30, hysteresis_frames=5)
    tracker.update(frame_id, ball_owner_team)  # ball_owner_team: None, 0, 1, etc.
    
    # Consultas
    stats = tracker.get_possession_stats()
    timeline = tracker.get_possession_timeline()
    current = tracker.get_current_possession()

Author: TacticEYE2 Team
Date: 2026-01-21
"""

from typing import Optional, Dict, List, Tuple
from collections import deque


class PossessionTrackerV2:
    """
    Tracker de posesión simplificado con reglas deterministas.
    
    Características:
    - Todo el tiempo siempre asignado a un equipo
    - Frames con ball_owner=None se asignan al equipo actual
    - Histeresis para confirmar cambios de posesión
    - Timeline completo de posesiones
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        hysteresis_frames: int = 5,
        max_teams: int = 2
    ):
        """
        Args:
            fps: Frames por segundo del vídeo (para conversión a segundos)
            hysteresis_frames: Frames consecutivos necesarios para confirmar cambio de posesión
            max_teams: Número máximo de equipos (típicamente 2)
        """
        self.fps = float(fps)
        self.hysteresis_frames = hysteresis_frames
        self.max_teams = max_teams
        
        # Estado interno
        self.current_possession_team: Optional[int] = None
        self.last_frame_id: int = 0
        
        # Acumuladores de tiempo (en frames)
        self.total_frames_by_team: Dict[int, int] = {}
        for team_id in range(max_teams):
            self.total_frames_by_team[team_id] = 0
        
        # Timeline de posesiones: [(start_frame, end_frame, team_id), ...]
        self.possession_timeline: List[Tuple[int, int, int]] = []
        
        # Buffer para histeresis
        self._candidate_buffer: deque = deque(maxlen=hysteresis_frames)
        self._pending_change_team: Optional[int] = None
        
        # Inicio del segmento actual
        self._current_segment_start: Optional[int] = None
        
        # Flag de inicialización
        self._initialized = False
    
    def update(self, frame_id: int, ball_owner_team: Optional[int]) -> None:
        """
        Actualizar posesión con información de un nuevo frame.
        
        Args:
            frame_id: ID del frame actual (debe ser monotónicamente creciente)
            ball_owner_team: Equipo del poseedor del balón (None si no hay poseedor claro)
        
        Reglas:
        - Si ball_owner_team es None: el tiempo se asigna al equipo actual
        - Si ball_owner_team != current_possession_team: se aplica histeresis
        - El cambio se confirma solo tras hysteresis_frames consecutivos
        """
        # Validación básica
        if frame_id < self.last_frame_id:
            raise ValueError(f"frame_id debe ser monotónicamente creciente: {frame_id} < {self.last_frame_id}")
        
        # Calcular intervalo de tiempo desde el último frame
        if self._initialized:
            interval_frames = frame_id - self.last_frame_id
            
            # Asignar todo el intervalo al equipo actual
            if self.current_possession_team is not None:
                self.total_frames_by_team[self.current_possession_team] += interval_frames
        
        # === Lógica de actualización del equipo en posesión ===
        
        # Caso 1: Inicio del tracking (sin posesión previa)
        if not self._initialized:
            if ball_owner_team is not None:
                self._initialize_possession(frame_id, ball_owner_team)
            # Si ball_owner_team es None al inicio, esperamos
            self.last_frame_id = frame_id
            return
        
        # Caso 2: ball_owner_team es None
        if ball_owner_team is None:
            # No hay cambio: el tiempo ya se asignó al equipo actual
            # Resetear buffer de histeresis
            self._candidate_buffer.clear()
            self._pending_change_team = None
            self.last_frame_id = frame_id
            return
        
        # Caso 3: ball_owner_team coincide con el equipo actual
        if ball_owner_team == self.current_possession_team:
            # Continúa la posesión del mismo equipo
            self._candidate_buffer.clear()
            self._pending_change_team = None
            self.last_frame_id = frame_id
            return
        
        # Caso 4: ball_owner_team es diferente al equipo actual
        # Aplicar histeresis
        self._apply_hysteresis(frame_id, ball_owner_team)
        
        self.last_frame_id = frame_id
    
    def _initialize_possession(self, frame_id: int, team_id: int) -> None:
        """Inicializar la primera posesión del partido."""
        self.current_possession_team = int(team_id)
        self._current_segment_start = frame_id
        self._initialized = True
        # Asignar el frame de inicialización
        self.total_frames_by_team[int(team_id)] += 1
        print(f"[Possession] Inicializada posesión: Team {team_id} @ frame {frame_id}")
    
    def _apply_hysteresis(self, frame_id: int, new_team_id: int) -> None:
        """
        Aplicar lógica de histeresis para confirmar cambio de posesión.
        
        Requiere hysteresis_frames consecutivos del nuevo equipo para confirmar el cambio.
        """
        # Si es un nuevo candidato, iniciar buffer
        if self._pending_change_team != new_team_id:
            self._candidate_buffer.clear()
            self._pending_change_team = new_team_id
        
        # Añadir frame al buffer
        self._candidate_buffer.append(new_team_id)
        
        # Verificar si se cumple el umbral de histeresis
        if len(self._candidate_buffer) >= self.hysteresis_frames:
            # Confirmar cambio de posesión
            self._confirm_possession_change(frame_id, new_team_id)
    
    def _confirm_possession_change(self, frame_id: int, new_team_id: int) -> None:
        """
        Confirmar cambio de posesión de un equipo a otro.
        
        Actualiza:
        - Timeline de posesiones (cierra segmento anterior, abre nuevo)
        - current_possession_team
        - Resetea buffer de histeresis
        """
        old_team = self.current_possession_team
        
        # Cerrar segmento anterior en el timeline
        if self._current_segment_start is not None:
            # El segmento anterior termina en el frame donde se confirma el cambio
            # (los frames de histeresis ya se contaron para el equipo anterior)
            self.possession_timeline.append((
                self._current_segment_start,
                frame_id,
                old_team
            ))
        
        # Cambiar equipo en posesión
        self.current_possession_team = int(new_team_id)
        self._current_segment_start = frame_id
        
        # Resetear histeresis
        self._candidate_buffer.clear()
        self._pending_change_team = None
        
        print(f"[Possession] Cambio confirmado: Team {old_team} → Team {new_team_id} @ frame {frame_id}")
    
    def get_possession_stats(self) -> Dict[str, any]:
        """
        Obtener estadísticas de posesión.
        
        Returns:
            Dict con:
            - total_frames: Total de frames procesados
            - total_seconds: Total de tiempo en segundos
            - possession_frames: {team_id: frames} 
            - possession_seconds: {team_id: seconds}
            - possession_percent: {team_id: percentage}
        """
        total_frames = self.last_frame_id if self._initialized else 0
        total_seconds = total_frames / self.fps
        
        possession_seconds = {}
        possession_percent = {}
        
        for team_id in range(self.max_teams):
            frames = self.total_frames_by_team.get(team_id, 0)
            seconds = frames / self.fps
            percent = (frames / total_frames * 100.0) if total_frames > 0 else 0.0
            
            possession_seconds[team_id] = seconds
            possession_percent[team_id] = percent
        
        return {
            'total_frames': total_frames,
            'total_seconds': total_seconds,
            'possession_frames': self.total_frames_by_team.copy(),
            'possession_seconds': possession_seconds,
            'possession_percent': possession_percent
        }
    
    def get_possession_timeline(self) -> List[Tuple[int, int, int]]:
        """
        Obtener timeline completo de posesiones.
        
        Returns:
            Lista de tuplas (start_frame, end_frame, team_id)
            Cada tupla representa un segmento de posesión continua.
        """
        # Incluir segmento actual si existe
        timeline = self.possession_timeline.copy()
        
        if self._initialized and self._current_segment_start is not None:
            timeline.append((
                self._current_segment_start,
                self.last_frame_id,
                self.current_possession_team
            ))
        
        return timeline
    
    def get_current_possession(self) -> Dict[str, any]:
        """
        Obtener estado actual de posesión.
        
        Returns:
            Dict con:
            - team: Equipo actual en posesión (None si no inicializado)
            - frame: Frame actual
            - initialized: Si el tracker está inicializado
        """
        return {
            'team': self.current_possession_team,
            'frame': self.last_frame_id,
            'initialized': self._initialized
        }
    
    def reset(self) -> None:
        """Resetear el tracker a estado inicial."""
        self.current_possession_team = None
        self.last_frame_id = 0
        self.total_frames_by_team = {i: 0 for i in range(self.max_teams)}
        self.possession_timeline.clear()
        self._candidate_buffer.clear()
        self._pending_change_team = None
        self._current_segment_start = None
        self._initialized = False
        print("[Possession] Tracker reseteado")


# ============================================================================
# Ejemplo de uso
# ============================================================================

if __name__ == '__main__':
    """
    Ejemplo de uso con secuencia simulada de frames.
    
    Escenario:
    - Frames 0-10: Team 0 posee el balón
    - Frames 11-15: Disputa (ball_owner=None), se asigna a Team 0
    - Frames 16-25: Team 1 gana posesión (con histeresis de 5 frames)
    - Frames 26-30: Ball suelto (None), se asigna a Team 1
    - Frames 31-40: Team 0 recupera posesión
    """
    
    print("="*70)
    print("EJEMPLO: PossessionTrackerV2")
    print("="*70)
    
    # Crear tracker con histeresis de 5 frames
    tracker = PossessionTrackerV2(fps=30.0, hysteresis_frames=5)
    
    # Simular secuencia de frames
    sequence = [
        # (frame_id, ball_owner_team)
        # Inicio: Team 0 tiene el balón
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
        
        # Disputa: ball_owner=None (se asigna a Team 0)
        (11, None), (12, None), (13, None), (14, None), (15, None),
        
        # Team 1 intenta ganar posesión (histeresis de 5 frames)
        (16, 1), (17, 1), (18, 1), (19, 1),  # 4 frames, no suficiente
        (20, 1),  # 5to frame → cambio confirmado
        (21, 1), (22, 1), (23, 1), (24, 1), (25, 1),
        
        # Ball suelto (se asigna a Team 1)
        (26, None), (27, None), (28, None), (29, None), (30, None),
        
        # Team 0 recupera posesión
        (31, 0), (32, 0), (33, 0), (34, 0), (35, 0),  # cambio confirmado
        (36, 0), (37, 0), (38, 0), (39, 0), (40, 0),
    ]
    
    print("\nProcesando secuencia de frames...\n")
    
    for frame_id, ball_owner in sequence:
        tracker.update(frame_id, ball_owner)
        
        # Mostrar estado cada 10 frames
        if frame_id % 10 == 0:
            current = tracker.get_current_possession()
            print(f"Frame {frame_id:3d}: Posesión actual = Team {current['team']}")
    
    print("\n" + "="*70)
    print("ESTADÍSTICAS FINALES")
    print("="*70)
    
    stats = tracker.get_possession_stats()
    
    print(f"\nTotal de frames procesados: {stats['total_frames']}")
    print(f"Tiempo total: {stats['total_seconds']:.2f} segundos")
    
    print("\nPosesión por equipo:")
    for team_id in range(2):
        frames = stats['possession_frames'][team_id]
        seconds = stats['possession_seconds'][team_id]
        percent = stats['possession_percent'][team_id]
        print(f"  Team {team_id}: {frames} frames ({seconds:.2f}s) = {percent:.1f}%")
    
    print("\n" + "="*70)
    print("TIMELINE DE POSESIONES")
    print("="*70)
    
    timeline = tracker.get_possession_timeline()
    
    print("\nSegmentos de posesión continua:")
    for i, (start, end, team) in enumerate(timeline, 1):
        duration = end - start
        print(f"  {i}. Frames {start:3d}-{end:3d} (duración: {duration:2d}) → Team {team}")
    
    print("\n" + "="*70)
    print("VALIDACIÓN")
    print("="*70)
    
    # Validar que todo el tiempo está asignado
    total_assigned = sum(stats['possession_frames'].values())
    print(f"\nFrames totales: {stats['total_frames']}")
    print(f"Frames asignados: {total_assigned}")
    print(f"Sin asignar: {stats['total_frames'] - total_assigned}")
    
    assert total_assigned == stats['total_frames'], "ERROR: No todo el tiempo está asignado!"
    print("\n✅ VALIDACIÓN EXITOSA: Todo el tiempo está asignado a un equipo")
    
    # Validar suma de porcentajes
    total_percent = sum(stats['possession_percent'].values())
    print(f"\nSuma de porcentajes: {total_percent:.1f}%")
    assert abs(total_percent - 100.0) < 0.1, "ERROR: Los porcentajes no suman 100%!"
    print("✅ VALIDACIÓN EXITOSA: Los porcentajes suman 100%")
