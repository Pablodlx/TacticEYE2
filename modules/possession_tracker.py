"""
PossessionTracker

Objetivo: determinar posesión del balón por jugador y por equipo
sin usar homografía ni suposiciones métricas. Solo píxeles y ventanas
temporales.

Diseño:
- update(ball_pos, players): llamado por frame.
  - ball_pos: (x, y) o None
  - players: lista de dicts o tuples. Se aceptan formatos:
      - dict con keys: 'track_id', 'bbox' (x1,y1,x2,y2) o 'pos' (x,y), 'team_id' (0/1/None)
      - tuple (track_id, (x,y), team_id)
  - El tracker extrae la posición del jugador (centro del bbox si se da bbox), calcula
    distancias euclídeas al balón y decide el jugador candidato por frame.

- Buffer temporal (deque) para N frames consecutivos para confirmar dominancia de jugador.
- Buffer para M frames para confirmar cambio de posesión entre equipos.
- Estados: 'team_A_possession', 'team_B_possession', 'contested', 'dead_ball', 'no_possession'

Parámetros principales configurables en el constructor (con valores por defecto razonables):
- player_confirm_frames: N (min frames consecutivos para asignar posesión a jugador)
- team_confirm_frames: M (min frames para confirmar cambio de equipo)
- contested_gap_threshold: diferencia en píxeles entre 1ro y 2do más cercano para considerar contested
- max_control_distance: si la distancia mínima es mayor que esto, consideramos "dead/sin control"
- dead_ball_frames: frames consecutivos sin balón detectado para declarar dead_ball

Sin dependencias externas (solo collections, math, typing, numpy opcionalmente). Código modular y documentado.
"""

from collections import deque, Counter
from typing import List, Tuple, Optional, Dict, Any, Union
import math

PlayerInput = Union[Dict[str, Any], Tuple[int, Tuple[float, float], Optional[int]]]


class PossessionTracker:
    """Seguimiento de posesión del balón usando dominancia espacial + confirmación temporal.

    Ejemplo de uso:
        pt = PossessionTracker()
        pt.update(ball_pos, players)
        state = pt.get_current_possession()

    Notas de diseño:
    - No depende de unidades reales; todos los umbrales están en píxeles y son configurables.
    - No asigna posesión por un solo frame: usa ventanas `player_confirm_frames` y `team_confirm_frames`.
    - Maneja estados ambiguos: 'contested' cuando hay dos jugadores próximos, 'dead_ball' cuando
      no hay balón detectado consistentemente.
    """

    def __init__(
        self,
        # Window and thresholds: use a sliding window for robustness (not only consecutive frames)
        player_window: int = 5,
        dominance_threshold: float = 0.6,
        team_confirm_frames: int = 5,
        contested_gap_threshold: float = 18.0,
        max_control_distance: float = 140.0,
        # Stickiness: mantener poseedor si reaparece/ocluye brevemente
        hold_frames: int = 6,
        hold_distance_threshold: float = 60.0,
        dead_ball_frames: int = 7,
        history_len: int = 128,
    ):
        # Temporal buffers and sensitivity params
        self.player_window = player_window
        self.dominance_threshold = float(dominance_threshold)
        self.team_confirm_frames = team_confirm_frames
        self.contested_gap_threshold = contested_gap_threshold
        self.max_control_distance = max_control_distance
        self.hold_frames = hold_frames
        self.hold_distance_threshold = hold_distance_threshold
        self.dead_ball_frames = dead_ball_frames

        # Buffer que guarda el 'dominant' por frame: puede ser track_id, 'contested', None
        self._dominant_buffer = deque(maxlen=max(history_len, player_window))

        # Buffer que guarda últimos equipos dominantes (None o 0/1)
        self._team_buffer = deque(maxlen=max(history_len, team_confirm_frames))

        # Estado actual de posesión (string) y metadata
        self.current_state = 'no_possession'
        self.current_team = None  # 0 or 1
        self.current_player = None  # track_id
        self.frames_since_change = 0

        # Bookkeeping de frames sin balón
        self._consecutive_no_ball = 0

        # Occlusion pending buffer: si el balón desaparece, acumulamos frames pendientes
        # y resolvemos cuando reaparece. _occlusion_owner guarda el player_id que tenía
        # la posesión al inicio de la oclusión, y _pending_occlusion_frames cuenta frames.
        self._pending_occlusion_frames = 0
        self._occlusion_owner = None
        self._occlusion_owner_team = None
        # Si ya atribuimos los frames pendientes inmediatamente, evitar duplicado
        self._pending_attributed = False

        # Contadores acumulados de posesión por equipo (frames)
        self.total_possession_frames = {0: 0, 1: 0}
        # Contadores asignados por frame (cada frame tendrá un equipo asignado)
        self.total_assigned_frames = {0: 0, 1: 0}
        self.assigned_team = None
        self.last_assigned_team = None

        # Últimas posiciones y frames vistos por player (para lógica de hold)
        self._last_player_positions: Dict[int, Tuple[float, float]] = {}
        self._last_seen_frame: Dict[int, int] = {}

        # Historial simple
        self.frame_index = 0

    # ----------------------------- Helpers -----------------------------
    def _parse_player(self, p: PlayerInput) -> Tuple[int, Tuple[float, float], Optional[int]]:
        """Normaliza la entrada de jugador a (track_id, (x,y), team_id).

        Acepta dicts con 'track_id' y 'bbox' o 'pos', o tuplas (track_id, pos, team_id).
        """
        if isinstance(p, dict):
            tid = p.get('track_id')
            team = p.get('team_id') if 'team_id' in p else p.get('team')
            if 'pos' in p and p['pos'] is not None:
                pos = tuple(p['pos'])
            elif 'bbox' in p and p['bbox'] is not None:
                x1, y1, x2, y2 = p['bbox']
                pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            else:
                raise ValueError('Player dict debe contener "pos" o "bbox"')
            return int(tid), (float(pos[0]), float(pos[1])), (None if team is None else int(team))
        else:
            # tuple expected (track_id, (x,y), team_id)
            tid, pos, team = p
            return int(tid), (float(pos[0]), float(pos[1])), (None if team is None else int(team))

    @staticmethod
    def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ----------------------------- Core update -----------------------------
    def update(self, ball_pos: Optional[Tuple[float, float]], players: List[PlayerInput]) -> None:
        """Actualizar el estado con la lectura de un frame.

        - ball_pos: (x,y) o None
        - players: lista de `PlayerInput` (ver definición arriba)

        Efectos:
        - Actualiza buffers temporales
        - Recalcula current_state (pero respeta confirmaciones temporales)
        """
        self.frame_index += 1

        # Si no hay balón detectado
        if ball_pos is None:
            self._consecutive_no_ball += 1
            self._dominant_buffer.append(None)
            self._team_buffer.append(None)
            # comenzar/continuar oclusión pendiente
            self._pending_occlusion_frames += 1
            if self._pending_occlusion_frames == 1:
                # registrar propietario actual para posible atribución si se resuelve
                self._occlusion_owner = self.current_player
                self._occlusion_owner_team = self.current_team

            # Atribuir de forma optimista los frames de oclusión al equipo actual
            # si existe un equipo en posesión y estamos dentro del umbral de hold
            if self.current_team in (0, 1) and self._pending_occlusion_frames <= self.hold_frames:
                # contar este frame inmediatamente hacia el total de posesión
                self.total_possession_frames[self.current_team] += 1
                self._pending_attributed = True

            # Revisar dead_ball: si excede dead_ball_frames se marca dead_ball y se descarta oclusión
            if self._consecutive_no_ball >= self.dead_ball_frames:
                # descartar attributación pendiente
                self._pending_occlusion_frames = 0
                self._occlusion_owner = None
                self._occlusion_owner_team = None
                self._pending_attributed = False
                self._set_state('dead_ball')
            else:
                # no suficiente info, mantener o pasar a no_possession
                if self.current_state not in ('team_A_possession', 'team_B_possession'):
                    self._set_state('no_possession')
            return

        # Reset counter of missing ball
        self._consecutive_no_ball = 0
        # Si había oclusión pendiente, la resolveremos tras calcular el poseedor actual

        # Compute positions and distances
        parsed = []
        for p in players:
            try:
                tid, pos, team = self._parse_player(p)
            except Exception:
                continue
            parsed.append((tid, pos, team))

        # actualizar últimas posiciones y frames vistos
        for tid, pos, team in parsed:
            self._last_player_positions[tid] = pos
            self._last_seen_frame[tid] = self.frame_index

        if not parsed:
            # No players known
            self._dominant_buffer.append(None)
            self._team_buffer.append(None)
            self._set_state('no_possession')
            return

        # Compute distances
        distances = [(tid, self._euclidean(ball_pos, pos), team) for (tid, pos, team) in parsed]
        distances.sort(key=lambda x: x[1])  # orden por distancia asc

        # Nearest and second nearest for contested check
        nearest_tid, nearest_dist, nearest_team = distances[0]
        second_dist = distances[1][1] if len(distances) > 1 else float('inf')

        # Determine contest / candidate / out_of_range
        if nearest_dist > self.max_control_distance:
            candidate = None
            candidate_team = None
        else:
            # If second is close to nearest within threshold => contested
            if (second_dist - nearest_dist) <= self.contested_gap_threshold:
                candidate = 'contested'
                candidate_team = None
            else:
                candidate = nearest_tid
                candidate_team = nearest_team

        # Sticky hold: si ya había un poseedor reciente, y la nueva medición falla
        # por oclusión o transferencia breve, mantenemos al poseedor si cumple umbrales.
        if self.current_player is not None and self.current_state in ('team_A_possession', 'team_B_possession'):
            cp = self.current_player
            # comprobar último frame visto y distancia al balón
            last_seen = self._last_seen_frame.get(cp, None)
            last_pos = self._last_player_positions.get(cp, None)
            if last_pos is not None and last_seen is not None:
                frames_since_seen = self.frame_index - last_seen
                dist_to_ball = self._euclidean(ball_pos, last_pos)
                # Si estaba cerca y fue visto recientemente, mantener la posesión
                if frames_since_seen <= self.hold_frames and dist_to_ball <= self.hold_distance_threshold:
                    candidate = cp
                    candidate_team = self.current_team

        # Append to buffers
        self._dominant_buffer.append(candidate)
        self._team_buffer.append(candidate_team)

        # Decide player-level possession using a sliding window and dominance threshold
        player_possessed = None
        if len(self._dominant_buffer) >= self.player_window:
            last_w = list(self._dominant_buffer)[-self.player_window:]
            # Count occurrences (ignore None and 'contested' for player dominance)
            counts = Counter([x for x in last_w if x is not None and x != 'contested'])
            if counts:
                top_player, top_count = counts.most_common(1)[0]
                # If the top player appears in >= dominance_threshold fraction of the window -> possessed
                if top_count >= math.ceil(self.dominance_threshold * self.player_window):
                    player_possessed = top_player

        # Decide team-level candidate from player possession (lookup in current distances)
        team_candidate = None
        if player_possessed is not None:
            for tid, dist, team in distances:
                if tid == player_possessed:
                    team_candidate = team
                    break

        # If no single-player possession, check for contested
        if player_possessed is None:
            # If last window entries are 'contested' => contested
            if len(self._dominant_buffer) >= self.player_window:
                last_n = list(self._dominant_buffer)[-self.player_window:]
                if all(x == 'contested' for x in last_n):
                    self._set_state('contested')
                    return
            # Otherwise keep previous state or no_possession
            self._set_state('no_possession')
            return

        # At this point we have a player_possessed and team_candidate
        # Map team id to state string
        if team_candidate is None:
            # If player has no known team, treat as no_possession
            self._set_state('no_possession')
            return

        intended_team_state = 'team_A_possession' if int(team_candidate) == 0 else 'team_B_possession'

        # For team confirmation use last team_confirm_frames and require majority matching
        recent_teams = list(self._team_buffer)[-self.team_confirm_frames:]
        team_count = sum(1 for t in recent_teams if t is not None and t == team_candidate)

        if team_count >= math.ceil(0.6 * self.team_confirm_frames):
            # Confirm or switch possession
            if self.current_state != intended_team_state:
                self._set_state(intended_team_state, team=team_candidate, player=player_possessed)
            else:
                self._set_state(intended_team_state, team=team_candidate, player=player_possessed)
        else:
            # Not enough confirmation yet: keep metadata but avoid switching
            self.current_player = player_possessed

        # Contadores: sumar este frame a la posesión confirmada por equipo
        if self.current_team in (0, 1):
            self.total_possession_frames[self.current_team] += 1

        # ---------------- assign a team for this frame ----------------
        assigned = None
        if self.current_team in (0, 1):
            assigned = self.current_team
        else:
            # look at recent team buffer to infer assignment
            win = min(self.player_window, len(self._team_buffer))
            recent = list(self._team_buffer)[-win:] if win > 0 else []
            c0 = sum(1 for t in recent if t == 0)
            c1 = sum(1 for t in recent if t == 1)
            if c0 > c1:
                assigned = 0
            elif c1 > c0:
                assigned = 1
            else:
                # tie/unknown: fallback to last assigned team if any
                assigned = self.last_assigned_team
                # if still unknown, fallback to team with larger cumulative possession so far
                if assigned is None:
                    total0 = self.total_possession_frames.get(0, 0)
                    total1 = self.total_possession_frames.get(1, 0)
                    if total0 + total1 > 0:
                        assigned = 0 if total0 >= total1 else 1
                    else:
                        assigned = None

        # ensure assigned is stored and counted
        self.assigned_team = assigned
        if assigned in (0, 1):
            self.total_assigned_frames[assigned] += 1
            self.last_assigned_team = assigned
        # Si existían frames de oclusión pendiente, resolver atribución:
        # Si existían frames de oclusión pendiente, resolver atribución si no ya atribuida
        if self._pending_occlusion_frames > 0:
            if not self._pending_attributed:
                if self._occlusion_owner is not None and self.current_player is not None:
                    if self.current_player == self._occlusion_owner or self.current_team == self._occlusion_owner_team:
                        if self.current_team in (0, 1):
                            self.total_possession_frames[self.current_team] += self._pending_occlusion_frames
            # resetear estado de oclusión
            self._pending_occlusion_frames = 0
            self._occlusion_owner = None
            self._occlusion_owner_team = None
            self._pending_attributed = False

    # ----------------------------- State management -----------------------------
    def _set_state(self, state: str, team: Optional[int] = None, player: Optional[int] = None) -> None:
        """Internals: establece el estado actual y actualiza contadores.

        - state: uno de 'team_A_possession','team_B_possession','contested','dead_ball','no_possession'
        """
        prev_state = self.current_state
        prev_team = self.current_team

        # Update
        self.current_state = state
        if state == 'team_A_possession':
            self.current_team = 0
        elif state == 'team_B_possession':
            self.current_team = 1
        else:
            self.current_team = None

        self.current_player = player

        if prev_state != self.current_state or prev_team != self.current_team:
            # state change
            self.frames_since_change = 0
        else:
            self.frames_since_change += 1

    # ----------------------------- Public getters -----------------------------
    def get_current_possession(self) -> Dict[str, Any]:
        """Retorna la representación actual de la posesión.

        Devuelve dict con keys:
        - state: str
        - team: 0/1/None
        - player: track_id or None
        - frames_since_change: int
        - frame_index: int
        """
        # Also compute possession percentages over the player_window
        win = min(self.player_window, len(self._dominant_buffer))
        last_w = list(self._dominant_buffer)[-win:] if win > 0 else []
        team0 = team1 = contested = none = 0
        player_counts: Dict[Any, int] = {}
        for d in last_w:
            if d is None:
                none += 1
            elif d == 'contested':
                contested += 1
            else:
                # we don't know the team here; team buffer aligns by frame
                player_counts[d] = player_counts.get(d, 0) + 1

        # Build team percents by inspecting team buffer aligned to last_w
        last_team_w = list(self._team_buffer)[-win:] if win > 0 else []
        for t in last_team_w:
            if t is None:
                none += 0  # already counted above as None in dominant
            elif t == 0:
                team0 += 1
            elif t == 1:
                team1 += 1

        total = max(1, win)
        possession_percent = {
            'team_0': team0 / total,
            'team_1': team1 / total,
            'contested': contested / total,
            'none': none / total,
        }

        player_percent = None
        if self.current_player is not None and win > 0:
            player_percent = player_counts.get(self.current_player, 0) / total

        # Build raw totals dicts
        totals_possession = dict(self.total_possession_frames)
        totals_assigned = dict(self.total_assigned_frames)

        # Normalize assigned totals so team_0 + team_1 = 1.0 (if any assigned frames exist)
        total_assigned_sum = float(totals_assigned.get(0, 0) + totals_assigned.get(1, 0))
        if total_assigned_sum > 0.0:
            normalized_assigned = {
                0: totals_assigned.get(0, 0) / total_assigned_sum,
                1: totals_assigned.get(1, 0) / total_assigned_sum,
            }
        else:
            normalized_assigned = {0: 0.0, 1: 0.0}

        # Also compute assigned percentages relative to total frames processed
        total_frames = max(1, self.frame_index)
        assigned_pct_of_total = {
            0: totals_assigned.get(0, 0) / total_frames,
            1: totals_assigned.get(1, 0) / total_frames,
        }

        return {
            'state': self.current_state,
            'team': self.current_team,
            'player': self.current_player,
            'player_percent': player_percent,
            'possession_percent': possession_percent,
            'total_possession_frames': totals_possession,
            'total_assigned_frames': totals_assigned,
            # Normalized totals based on assigned-per-frame counts (sums to 1.0)
            'total_assigned_percent': normalized_assigned,
            # Percent of TOTAL FRAMES (frame_index) that each team was assigned possession
            'total_assigned_percent_of_total_frames': assigned_pct_of_total,
            # Backwards-compatible alias: total_possession_percent -> assigned percent
            'total_possession_percent': normalized_assigned,
            'assigned_team': self.assigned_team,
            'frames_since_change': self.frames_since_change,
            'frame_index': self.frame_index,
        }

    def reset(self) -> None:
        """Reset completo del tracker."""
        self._dominant_buffer.clear()
        self._team_buffer.clear()
        self.current_state = 'no_possession'
        self.current_team = None
        self.current_player = None
        self.frames_since_change = 0
        self._consecutive_no_ball = 0
        self.frame_index = 0


# ----------------------------- Minimal usage example -----------------------------
if __name__ == '__main__':
    # Simulación simple: dos jugadores A (team 0) y B (team 1) y balón que se mueve
    pt = PossessionTracker(player_window=3, team_confirm_frames=3,
                            contested_gap_threshold=15.0, max_control_distance=200.0,
                            dead_ball_frames=4)

    # Players: track_id, position, team
    player_A = (11, (100.0, 200.0), 0)
    player_B = (22, (300.0, 210.0), 1)

    # Simular secuencia de 12 frames donde el balón se acerca a A y luego a B
    ball_positions = [
        (150, 205),  # frame 1 - A closer
        (140, 203),  # frame 2
        (130, 202),  # frame 3 -> after 3 frames A should be dominant
        (200, 205),  # frame 4 - between
        (240, 208),  # frame 5 - B getting closer
        (260, 209),  # frame 6
        (280, 210),  # frame 7 -> after 3 frames B should be dominant
        None,        # frame 8 - ball lost
        None,        # frame 9
        None,        # frame 10 -> dead_ball after dead_ball_frames=4 (not reached yet)
        (110, 200),  # frame 11 - ball back near A
        (105, 198),  # frame 12
    ]

    for i, bpos in enumerate(ball_positions, 1):
        pt.update(bpos, [player_A, player_B])
        print(f'Frame {i}:', pt.get_current_possession())

    # Resultado: verás como tras 3 frames consecutivos A y luego B son confirmados,
    # estados ambiguous aparecen y dead_ball si el balón falta varios frames.
