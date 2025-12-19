"""
Professional Overlay - Overlay profesional tipo Wyscout
========================================================
IDs, trayectorias, mini-mapa cenital, estadísticas en pantalla
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class ProfessionalOverlay:
    """
    Genera overlays profesionales para visualización del análisis
    """
    
    def __init__(self, 
                 show_ids: bool = True,
                 show_trajectories: bool = True,
                 show_minimap: bool = True,
                 show_stats: bool = True,
                 trajectory_length: int = 300):  # 10s @ 30fps
        """
        Args:
            show_ids: Mostrar IDs encima de jugadores
            show_trajectories: Mostrar trayectorias recientes
            show_minimap: Mostrar mini-mapa cenital
            show_stats: Mostrar estadísticas en pantalla
            trajectory_length: Longitud máxima de trayectorias
        """
        self.show_ids = show_ids
        self.show_trajectories = show_trajectories
        self.show_minimap = show_minimap
        self.show_stats = show_stats
        self.trajectory_length = trajectory_length
        
        # Colores por clase del modelo YOLO
        self.team_colors = {
            0: (0, 255, 0),      # Verde - Players
            2: (0, 255, 255),    # Amarillo - Referees
            3: (255, 0, 255),    # Magenta - Goalkeepers
            -1: (255, 255, 255)  # Blanco - Sin asignar
        }
        
        # Buffer de trayectorias
        self.trajectories: Dict[int, deque] = {}
    
    def draw_player_overlay(self,
                           frame: np.ndarray,
                           track_id: int,
                           bbox: np.ndarray,
                           team_id: int = -1,
                           velocity: Optional[float] = None) -> np.ndarray:
        """
        Dibuja overlay para un jugador individual
        
        Args:
            frame: Frame BGR
            track_id: ID del track
            bbox: Bounding box [x1, y1, x2, y2]
            team_id: ID del equipo (0, 1, 2)
            velocity: Velocidad en km/h (opcional)
            
        Returns:
            Frame con overlay dibujado
        """
        x1, y1, x2, y2 = map(int, bbox)
        color = self.team_colors.get(team_id, self.team_colors[-1])
        
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # ID encima de la bbox
        if self.show_ids:
            center_x = (x1 + x2) // 2
            label_y = y1 - 10
            
            # Texto del ID
            id_text = f"#{track_id}"
            
            # Fondo semi-transparente para el texto
            (text_w, text_h), baseline = cv2.getTextSize(
                id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                frame,
                (center_x - text_w // 2 - 5, label_y - text_h - 5),
                (center_x + text_w // 2 + 5, label_y + baseline),
                color,
                -1
            )
            
            # Texto
            cv2.putText(
                frame,
                id_text,
                (center_x - text_w // 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Velocidad debajo del ID (si disponible)
            if velocity is not None:
                speed_text = f"{velocity:.1f} km/h"
                cv2.putText(
                    frame,
                    speed_text,
                    (center_x - 30, label_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )
        
        return frame
    
    def draw_ball_overlay(self,
                         frame: np.ndarray,
                         bbox: np.ndarray) -> np.ndarray:
        """Dibuja overlay para el balón"""
        x1, y1, x2, y2 = map(int, bbox)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = max(10, int((x2 - x1) // 2))
        
        # Círculo amarillo brillante
        cv2.circle(frame, center, radius, (0, 255, 255), 2)
        cv2.circle(frame, center, 3, (0, 255, 255), -1)
        
        # Label
        cv2.putText(
            frame,
            "BALL",
            (center[0] - 20, center[1] - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )
        
        return frame
    
    def update_trajectory(self, track_id: int, position: Tuple[int, int]):
        """Actualiza trayectoria de un track"""
        if track_id not in self.trajectories:
            self.trajectories[track_id] = deque(maxlen=self.trajectory_length)
        
        self.trajectories[track_id].append(position)
    
    def draw_trajectories(self,
                         frame: np.ndarray,
                         team_id_map: Optional[Dict[int, int]] = None) -> np.ndarray:
        """
        Dibuja trayectorias de todos los tracks
        
        Args:
            frame: Frame BGR
            team_id_map: Dict {track_id: team_id} para colorear trayectorias
        """
        if not self.show_trajectories:
            return frame
        
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
            
            # Color basado en equipo
            if team_id_map and track_id in team_id_map:
                color = self.team_colors.get(team_id_map[track_id], (255, 255, 255))
            else:
                color = (255, 255, 255)
            
            # Dibujar línea con degradado de opacidad (más reciente = más visible)
            points = list(trajectory)
            for i in range(1, len(points)):
                # Opacidad basada en antigüedad
                alpha = i / len(points)
                
                # Grosor de línea (más grueso al final)
                thickness = max(1, int(3 * alpha))
                
                # Color con alpha
                color_alpha = tuple(int(c * alpha) for c in color)
                
                # Convertir puntos a tuplas de int
                pt1 = tuple(map(int, points[i-1]))
                pt2 = tuple(map(int, points[i]))
                
                cv2.line(
                    frame,
                    pt1,
                    pt2,
                    color_alpha,
                    thickness,
                    cv2.LINE_AA
                )
        
        return frame
    
    def draw_minimap(self,
                    frame: np.ndarray,
                    field_topdown: np.ndarray,
                    players_topdown: Dict[int, Tuple[np.ndarray, int]],
                    ball_topdown: Optional[np.ndarray] = None,
                    position: str = 'bottom-right',
                    size_ratio: float = 0.25) -> np.ndarray:
        """
        Dibuja mini-mapa cenital en esquina
        
        Args:
            frame: Frame principal
            field_topdown: Imagen del campo top-down
            players_topdown: Dict {track_id: (position, team_id)} en píxeles topdown
            ball_topdown: Posición del balón en píxeles topdown
            position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
            size_ratio: Tamaño del minimap relativo al frame
        """
        if not self.show_minimap:
            return frame
        
        h, w = frame.shape[:2]
        
        # Calcular tamaño del minimap
        minimap_w = int(w * size_ratio)
        minimap_h = int(h * size_ratio)
        
        # Redimensionar campo
        minimap = cv2.resize(field_topdown, (minimap_w, minimap_h))
        
        # Dibujar jugadores en minimap
        scale_x = minimap_w / field_topdown.shape[1]
        scale_y = minimap_h / field_topdown.shape[0]
        
        for track_id, (pos, team_id) in players_topdown.items():
            x = int(pos[0] * scale_x)
            y = int(pos[1] * scale_y)
            color = self.team_colors.get(team_id, (255, 255, 255))
            
            # Punto para jugador
            cv2.circle(minimap, (x, y), 4, color, -1)
            cv2.circle(minimap, (x, y), 5, (255, 255, 255), 1)
        
        # Dibujar balón
        if ball_topdown is not None:
            ball_x = int(ball_topdown[0] * scale_x)
            ball_y = int(ball_topdown[1] * scale_y)
            cv2.circle(minimap, (ball_x, ball_y), 5, (0, 255, 255), -1)
        
        # Borde del minimap
        cv2.rectangle(minimap, (0, 0), (minimap_w-1, minimap_h-1), (255, 255, 255), 2)
        
        # Determinar posición en frame
        margin = 20
        if position == 'top-left':
            y_start, y_end = margin, margin + minimap_h
            x_start, x_end = margin, margin + minimap_w
        elif position == 'top-right':
            y_start, y_end = margin, margin + minimap_h
            x_start, x_end = w - minimap_w - margin, w - margin
        elif position == 'bottom-left':
            y_start, y_end = h - minimap_h - margin, h - margin
            x_start, x_end = margin, margin + minimap_w
        else:  # bottom-right
            y_start, y_end = h - minimap_h - margin, h - margin
            x_start, x_end = w - minimap_w - margin, w - margin
        
        # Overlay con transparencia
        alpha = 0.8
        overlay = frame.copy()
        overlay[y_start:y_end, x_start:x_end] = minimap
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def draw_stats_panel(self,
                        frame: np.ndarray,
                        stats: Dict,
                        position: str = 'top-left') -> np.ndarray:
        """
        Dibuja panel de estadísticas en pantalla
        
        Args:
            frame: Frame BGR
            stats: Dict con estadísticas a mostrar
            position: 'top-left', 'top-right', etc.
        """
        if not self.show_stats:
            return frame
        
        h, w = frame.shape[:2]
        
        # Dimensiones del panel
        panel_w = 350
        panel_h = 200
        margin = 20
        
        # Posición
        if position == 'top-left':
            x_start = margin
            y_start = margin
        elif position == 'top-right':
            x_start = w - panel_w - margin
            y_start = margin
        else:  # top-left por defecto
            x_start = margin
            y_start = margin
        
        # Fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_start + panel_w, y_start + panel_h),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Borde
        cv2.rectangle(
            frame,
            (x_start, y_start),
            (x_start + panel_w, y_start + panel_h),
            (255, 255, 255),
            2
        )
        
        # Título
        cv2.putText(
            frame,
            "MATCH STATISTICS",
            (x_start + 10, y_start + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Línea divisoria
        cv2.line(
            frame,
            (x_start + 10, y_start + 35),
            (x_start + panel_w - 10, y_start + 35),
            (255, 255, 255),
            1
        )
        
        # Contenido
        y_offset = y_start + 55
        line_height = 25
        
        # Posesión
        if 'possession' in stats:
            poss_0 = stats['possession'].get(0, 0)
            poss_1 = stats['possession'].get(1, 0)
            
            cv2.putText(
                frame,
                f"Possession:",
                (x_start + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            
            # Barra de posesión
            bar_y = y_offset + 10
            bar_width = panel_w - 40
            bar_height = 15
            
            # Fondo
            cv2.rectangle(
                frame,
                (x_start + 20, bar_y),
                (x_start + 20 + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1
            )
            
            # Equipo 0 (verde)
            poss_0_width = int(bar_width * (poss_0 / 100))
            cv2.rectangle(
                frame,
                (x_start + 20, bar_y),
                (x_start + 20 + poss_0_width, bar_y + bar_height),
                self.team_colors[0],
                -1
            )
            
            # Porcentajes
            cv2.putText(
                frame,
                f"{poss_0:.1f}%",
                (x_start + 25, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
            
            cv2.putText(
                frame,
                f"{poss_1:.1f}%",
                (x_start + panel_w - 70, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
            
            y_offset += 40
        
        # Pases
        if 'passes' in stats:
            passes_0 = stats['passes'].get(0, {})
            passes_1 = stats['passes'].get(1, {})
            
            cv2.putText(
                frame,
                f"Passes:  {passes_0.get('completed', 0)}/{passes_0.get('attempted', 0)} " +
                f"({passes_0.get('accuracy', 0):.0f}%)",
                (x_start + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                self.team_colors[0],
                1
            )
            y_offset += line_height
            
            cv2.putText(
                frame,
                f"              {passes_1.get('completed', 0)}/{passes_1.get('attempted', 0)} " +
                f"({passes_1.get('accuracy', 0):.0f}%)",
                (x_start + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),  # Blanco
                1
            )
            y_offset += line_height
        
        # Distancia
        if 'distance' in stats:
            dist_0 = stats['distance'].get(0, 0)
            dist_1 = stats['distance'].get(1, 0)
            
            cv2.putText(
                frame,
                f"Distance: {dist_0:.1f} km",
                (x_start + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                self.team_colors[0],
                1
            )
            y_offset += line_height
            
            cv2.putText(
                frame,
                f"              {dist_1:.1f} km",
                (x_start + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),  # Blanco
                1
            )
        
        return frame
    
    def draw_complete_overlay(self,
                             frame: np.ndarray,
                             tracks: List[Tuple[int, np.ndarray, int]],
                             team_assignments: Dict[int, int],
                             field_topdown: np.ndarray,
                             players_topdown: Dict[int, Tuple[np.ndarray, int]],
                             ball_topdown: Optional[np.ndarray],
                             stats: Dict,
                             player_velocities: Optional[Dict[int, float]] = None) -> np.ndarray:
        """
        Dibuja overlay completo combinando todos los elementos
        """
        result = frame.copy()
        
        # 1. Trayectorias (primero, para que queden detrás)
        result = self.draw_trajectories(result, team_assignments)
        
        # 2. Overlays de jugadores y balón
        for track_id, bbox, class_id in tracks:
            if class_id == 0:  # Jugador
                team_id = team_assignments.get(track_id, -1)
                velocity = player_velocities.get(track_id) if player_velocities else None
                result = self.draw_player_overlay(result, track_id, bbox, team_id, velocity)
                
                # Actualizar trayectoria
                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                self.update_trajectory(track_id, center)
                
            elif class_id == 32:  # Balón
                result = self.draw_ball_overlay(result, bbox)
        
        # 3. Mini-mapa
        result = self.draw_minimap(
            result,
            field_topdown,
            players_topdown,
            ball_topdown,
            position='bottom-right'
        )
        
        # 4. Panel de estadísticas
        result = self.draw_stats_panel(result, stats, position='top-left')
        
        return result
    
    def reset_trajectories(self):
        """Limpia todas las trayectorias"""
        self.trajectories.clear()
