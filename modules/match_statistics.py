"""
Match Statistics - Sistema de estadísticas en vivo
=================================================
Calcula posesión, pases, distancia, velocidad, presión
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass, field


@dataclass
class PlayerStats:
    """Estadísticas individuales de un jugador"""
    track_id: int
    team_id: int
    total_distance: float = 0.0  # metros
    max_speed: float = 0.0  # km/h
    avg_speed: float = 0.0  # km/h
    position_history: deque = field(default_factory=lambda: deque(maxlen=900))  # 30s @ 30fps
    last_position: Optional[np.ndarray] = None
    last_time: Optional[float] = None
    
    
@dataclass
class TeamStats:
    """Estadísticas de equipo"""
    team_id: int
    possession_time: float = 0.0  # segundos
    passes_completed: int = 0
    passes_attempted: int = 0
    total_distance: float = 0.0  # metros del equipo
    avg_speed: float = 0.0  # km/h promedio
    players: Dict[int, PlayerStats] = field(default_factory=dict)


class MatchStatistics:
    """
    Calcula y mantiene estadísticas del partido en tiempo real
    """
    
    def __init__(self, 
                 ball_possession_radius: float = 3.0,  # metros
                 pass_max_distance: float = 40.0,  # metros
                 fps: int = 30):
        """
        Args:
            ball_possession_radius: Radio en metros para considerar posesión
            pass_max_distance: Distancia máxima en metros para considerar pase
            fps: Frames por segundo del video
        """
        self.ball_possession_radius = ball_possession_radius
        self.pass_max_distance = pass_max_distance
        self.fps = fps
        self.frame_duration = 1.0 / fps
        
        # Estadísticas por equipo
        self.team_stats = {
            0: TeamStats(team_id=0),  # Local
            1: TeamStats(team_id=1),  # Visitante
        }
        
        # Estado del balón
        self.ball_position_history = deque(maxlen=300)  # 10s
        self.last_ball_possessor = None  # (team_id, track_id)
        self.possession_start_time = None
        
        # Detección de pases
        self.potential_pass_start = None  # (team_id, track_id, position, time)
        
        # Zonas de presión (dividir campo en tercios)
        self.pressure_zones = {
            'high': 0,    # Tercio ofensivo
            'medium': 0,  # Tercio medio
            'low': 0      # Tercio defensivo
        }
        
        self.match_start_time = time.time()
        
    def update(self,
              players_3d: Dict[int, Tuple[np.ndarray, int]],
              ball_3d: Optional[np.ndarray],
              field_length: float = 105.0):
        """
        Actualiza estadísticas con nuevo frame
        
        Args:
            players_3d: Dict {track_id: (position_3d, team_id)} en metros
            ball_3d: Posición del balón en metros [x, y] o None
            field_length: Longitud del campo en metros
        """
        current_time = time.time()
        
        # Actualizar estadísticas de jugadores
        for track_id, (position, team_id) in players_3d.items():
            if team_id not in [0, 1]:  # Ignorar árbitros
                continue
            
            team = self.team_stats[team_id]
            
            # Crear stats de jugador si no existe
            if track_id not in team.players:
                team.players[track_id] = PlayerStats(
                    track_id=track_id,
                    team_id=team_id
                )
            
            player = team.players[track_id]
            
            # Actualizar posición
            player.position_history.append((position, current_time))
            
            # Calcular distancia y velocidad
            if player.last_position is not None and player.last_time is not None:
                delta_pos = position - player.last_position
                distance = np.linalg.norm(delta_pos)  # metros
                delta_time = current_time - player.last_time  # segundos
                
                if delta_time > 0:
                    speed_ms = distance / delta_time  # m/s
                    speed_kmh = speed_ms * 3.6  # km/h
                    
                    # Filtrar outliers (velocidad máxima humana ~36 km/h)
                    if speed_kmh <= 40.0:
                        player.total_distance += distance
                        player.max_speed = max(player.max_speed, speed_kmh)
                        
                        # Velocidad promedio (últimos 30 frames)
                        if len(player.position_history) >= 2:
                            recent_positions = list(player.position_history)[-30:]
                            total_dist = 0.0
                            for i in range(1, len(recent_positions)):
                                total_dist += np.linalg.norm(
                                    recent_positions[i][0] - recent_positions[i-1][0]
                                )
                            time_span = recent_positions[-1][1] - recent_positions[0][1]
                            if time_span > 0:
                                player.avg_speed = (total_dist / time_span) * 3.6
            
            player.last_position = position
            player.last_time = current_time
        
        # Actualizar distancia total del equipo
        for team_id in [0, 1]:
            team = self.team_stats[team_id]
            team.total_distance = sum(p.total_distance for p in team.players.values())
            
            # Velocidad promedio del equipo
            if team.players:
                team.avg_speed = np.mean([p.avg_speed for p in team.players.values()])
        
        # Actualizar posesión del balón
        if ball_3d is not None:
            self.ball_position_history.append((ball_3d, current_time))
            self._update_possession(players_3d, ball_3d, current_time)
            self._detect_passes(players_3d, ball_3d, current_time)
            self._update_pressure_zones(players_3d, ball_3d, field_length)
    
    def _update_possession(self,
                          players_3d: Dict[int, Tuple[np.ndarray, int]],
                          ball_3d: np.ndarray,
                          current_time: float):
        """Actualiza estadísticas de posesión"""
        # Encontrar jugador más cercano al balón
        closest_player = None
        closest_distance = float('inf')
        
        for track_id, (position, team_id) in players_3d.items():
            if team_id not in [0, 1]:
                continue
            
            distance = np.linalg.norm(position - ball_3d)
            if distance < closest_distance:
                closest_distance = distance
                closest_player = (team_id, track_id)
        
        # Verificar si está dentro del radio de posesión
        if closest_player and closest_distance <= self.ball_possession_radius:
            team_id, track_id = closest_player
            
            # Si cambió la posesión
            if self.last_ball_possessor != closest_player:
                # Finalizar posesión anterior
                if (self.last_ball_possessor is not None and 
                    self.possession_start_time is not None):
                    prev_team = self.last_ball_possessor[0]
                    possession_duration = current_time - self.possession_start_time
                    self.team_stats[prev_team].possession_time += possession_duration
                
                # Iniciar nueva posesión
                self.last_ball_possessor = closest_player
                self.possession_start_time = current_time
            
        else:
            # Balón libre - finalizar posesión si había
            if self.last_ball_possessor is not None and self.possession_start_time is not None:
                team_id = self.last_ball_possessor[0]
                possession_duration = current_time - self.possession_start_time
                self.team_stats[team_id].possession_time += possession_duration
                
            self.last_ball_possessor = None
            self.possession_start_time = None
    
    def _detect_passes(self,
                      players_3d: Dict[int, Tuple[np.ndarray, int]],
                      ball_3d: np.ndarray,
                      current_time: float):
        """Detecta pases completados y fallidos"""
        if len(self.ball_position_history) < 10:
            return
        
        # Calcular velocidad del balón
        recent_ball = list(self.ball_position_history)[-10:]
        ball_speeds = []
        for i in range(1, len(recent_ball)):
            delta_pos = recent_ball[i][0] - recent_ball[i-1][0]
            delta_time = recent_ball[i][1] - recent_ball[i-1][1]
            if delta_time > 0:
                speed = np.linalg.norm(delta_pos) / delta_time
                ball_speeds.append(speed)
        
        avg_ball_speed = np.mean(ball_speeds) if ball_speeds else 0.0
        
        # Pase = balón moviéndose rápido (>5 m/s) entre jugadores del mismo equipo
        if avg_ball_speed > 5.0:
            if self.potential_pass_start is None:
                # Inicio de potencial pase
                if self.last_ball_possessor:
                    team_id, track_id = self.last_ball_possessor
                    if track_id in self.team_stats[team_id].players:
                        player_pos = self.team_stats[team_id].players[track_id].last_position
                        self.potential_pass_start = (team_id, track_id, player_pos, current_time)
        
        elif avg_ball_speed < 2.0 and self.potential_pass_start is not None:
            # Balón se detuvo - verificar si pase fue completado
            pass_team, pass_from, pass_start_pos, pass_start_time = self.potential_pass_start
            
            # Verificar si un jugador del mismo equipo tiene ahora el balón
            if self.last_ball_possessor:
                curr_team, curr_player = self.last_ball_possessor
                
                if curr_team == pass_team and curr_player != pass_from:
                    # Pase completado al mismo equipo
                    pass_distance = np.linalg.norm(ball_3d - pass_start_pos)
                    
                    if pass_distance <= self.pass_max_distance:
                        self.team_stats[pass_team].passes_completed += 1
                        self.team_stats[pass_team].passes_attempted += 1
                    
                elif curr_team != pass_team:
                    # Pase interceptado
                    self.team_stats[pass_team].passes_attempted += 1
            
            self.potential_pass_start = None
    
    def _update_pressure_zones(self,
                              players_3d: Dict[int, Tuple[np.ndarray, int]],
                              ball_3d: np.ndarray,
                              field_length: float):
        """Actualiza zonas de presión basadas en posición del balón"""
        # Dividir campo en tercios
        ball_x = ball_3d[0]
        
        if ball_x < field_length / 3:
            self.pressure_zones['low'] += 1
        elif ball_x < 2 * field_length / 3:
            self.pressure_zones['medium'] += 1
        else:
            self.pressure_zones['high'] += 1
    
    def get_possession_percentage(self) -> Dict[int, float]:
        """Retorna porcentaje de posesión de cada equipo"""
        total_possession = sum(team.possession_time for team in self.team_stats.values())
        
        if total_possession == 0:
            return {0: 50.0, 1: 50.0}
        
        return {
            team_id: (team.possession_time / total_possession) * 100
            for team_id, team in self.team_stats.items()
        }
    
    def get_pass_accuracy(self, team_id: int) -> float:
        """Retorna precisión de pases de un equipo (%)"""
        team = self.team_stats[team_id]
        if team.passes_attempted == 0:
            return 0.0
        return (team.passes_completed / team.passes_attempted) * 100
    
    def get_player_stats(self, track_id: int) -> Optional[PlayerStats]:
        """Obtiene estadísticas de un jugador específico"""
        for team in self.team_stats.values():
            if track_id in team.players:
                return team.players[track_id]
        return None
    
    def get_team_summary(self, team_id: int) -> Dict:
        """Obtiene resumen de estadísticas de un equipo"""
        if team_id not in self.team_stats:
            return {}
        
        team = self.team_stats[team_id]
        possession_pct = self.get_possession_percentage()
        
        return {
            'team_id': team_id,
            'possession_%': possession_pct[team_id],
            'passes_completed': team.passes_completed,
            'passes_attempted': team.passes_attempted,
            'pass_accuracy_%': self.get_pass_accuracy(team_id),
            'total_distance_km': team.total_distance / 1000.0,
            'avg_speed_kmh': team.avg_speed,
            'num_players': len(team.players)
        }
    
    def get_top_players(self, 
                       metric: str = 'distance',
                       top_n: int = 5) -> List[Tuple[int, int, float]]:
        """
        Obtiene top N jugadores por métrica
        
        Args:
            metric: 'distance', 'max_speed', 'avg_speed'
            top_n: Número de jugadores a retornar
            
        Returns:
            Lista de (track_id, team_id, value)
        """
        all_players = []
        
        for team in self.team_stats.values():
            for player in team.players.values():
                if metric == 'distance':
                    value = player.total_distance / 1000.0  # km
                elif metric == 'max_speed':
                    value = player.max_speed
                elif metric == 'avg_speed':
                    value = player.avg_speed
                else:
                    value = 0.0
                
                all_players.append((player.track_id, player.team_id, value))
        
        # Ordenar y retornar top N
        all_players.sort(key=lambda x: x[2], reverse=True)
        return all_players[:top_n]
    
    def get_pressure_stats(self) -> Dict[str, float]:
        """Retorna estadísticas de presión (% de tiempo en cada zona)"""
        total_frames = sum(self.pressure_zones.values())
        
        if total_frames == 0:
            return {'high': 0.0, 'medium': 0.0, 'low': 0.0}
        
        return {
            zone: (count / total_frames) * 100
            for zone, count in self.pressure_zones.items()
        }
    
    def get_match_duration(self) -> float:
        """Retorna duración del partido en segundos"""
        return time.time() - self.match_start_time
    
    def export_summary(self) -> Dict:
        """Exporta resumen completo de estadísticas"""
        return {
            'match_duration_seconds': self.get_match_duration(),
            'team_0': self.get_team_summary(0),
            'team_1': self.get_team_summary(1),
            'pressure_zones_%': self.get_pressure_stats(),
            'top_distance_runners': self.get_top_players('distance', 5),
            'fastest_players': self.get_top_players('max_speed', 5)
        }
