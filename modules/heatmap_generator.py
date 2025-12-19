"""
Heatmap Generator - Mapas de calor 3D en tiempo real
====================================================
Genera heatmaps de jugadores y balón con actualización continua
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, Tuple, List
import time


class HeatmapGenerator:
    """
    Genera y mantiene heatmaps 3D de posiciones de jugadores y balón
    """
    
    def __init__(self, 
                 field_size: Tuple[int, int] = (1050, 680),
                 update_interval: float = 5.0,
                 history_seconds: float = 60.0,
                 grid_resolution: int = 50):
        """
        Args:
            field_size: Tamaño en píxeles del campo top-down (ancho, alto)
            update_interval: Segundos entre actualizaciones de heatmap
            history_seconds: Segundos de histórico para mantener
            grid_resolution: Resolución de la grilla del heatmap
        """
        self.field_size = field_size
        self.update_interval = update_interval
        self.history_seconds = history_seconds
        self.grid_resolution = grid_resolution
        
        # Buffers de posiciones por equipo
        self.position_buffers = {
            'team_0': deque(maxlen=10000),  # Local
            'team_1': deque(maxlen=10000),  # Visitante
            'team_2': deque(maxlen=10000),  # Árbitros
            'ball': deque(maxlen=10000)     # Balón
        }
        
        # Heatmaps acumulados (grilla)
        self.heatmap_grids = {
            'team_0': np.zeros((grid_resolution, grid_resolution)),
            'team_1': np.zeros((grid_resolution, grid_resolution)),
            'team_2': np.zeros((grid_resolution, grid_resolution)),
            'ball': np.zeros((grid_resolution, grid_resolution))
        }
        
        self.last_update = time.time()
        
    def add_position(self, 
                    position: Tuple[float, float],
                    entity_type: str,
                    timestamp: float = None):
        """
        Añade una posición al buffer
        
        Args:
            position: (x, y) en píxeles de campo top-down
            entity_type: 'team_0', 'team_1', 'team_2', 'ball'
            timestamp: Tiempo de la observación (default: ahora)
        """
        if entity_type not in self.position_buffers:
            return
        
        if timestamp is None:
            timestamp = time.time()
        
        self.position_buffers[entity_type].append((position[0], position[1], timestamp))
    
    def update_heatmaps(self, force: bool = False):
        """
        Actualiza las grillas de heatmap con posiciones recientes
        
        Args:
            force: Forzar actualización aunque no haya pasado el intervalo
        """
        current_time = time.time()
        
        if not force and (current_time - self.last_update) < self.update_interval:
            return
        
        self.last_update = current_time
        cutoff_time = current_time - self.history_seconds
        
        # Limpiar y regenerar cada heatmap
        for entity_type in self.position_buffers:
            # Limpiar posiciones antiguas
            buffer = self.position_buffers[entity_type]
            while buffer and buffer[0][2] < cutoff_time:
                buffer.popleft()
            
            # Regenerar grilla
            grid = np.zeros((self.grid_resolution, self.grid_resolution))
            
            for x, y, t in buffer:
                # Convertir posición a índice de grilla
                grid_x = int(x / self.field_size[0] * self.grid_resolution)
                grid_y = int(y / self.field_size[1] * self.grid_resolution)
                
                # Clamp a límites de grilla
                grid_x = np.clip(grid_x, 0, self.grid_resolution - 1)
                grid_y = np.clip(grid_y, 0, self.grid_resolution - 1)
                
                # Peso basado en antigüedad (más reciente = mayor peso)
                age = current_time - t
                weight = np.exp(-age / (self.history_seconds / 3))
                
                grid[grid_y, grid_x] += weight
            
            self.heatmap_grids[entity_type] = grid
    
    def get_heatmap_image(self, 
                         entity_type: str,
                         colormap: int = cv2.COLORMAP_JET,
                         alpha: float = 0.6) -> np.ndarray:
        """
        Genera imagen de heatmap para visualización
        
        Args:
            entity_type: Tipo de entidad ('team_0', 'team_1', 'team_2', 'ball')
            colormap: Mapa de color de OpenCV
            alpha: Transparencia del heatmap
            
        Returns:
            Imagen BGR del heatmap con canal alpha
        """
        if entity_type not in self.heatmap_grids:
            return None
        
        grid = self.heatmap_grids[entity_type]
        
        # Normalizar a rango 0-255
        if grid.max() > 0:
            normalized = (grid / grid.max() * 255).astype(np.uint8)
        else:
            normalized = grid.astype(np.uint8)
        
        # Resize a tamaño de campo
        resized = cv2.resize(normalized, self.field_size, 
                            interpolation=cv2.INTER_LINEAR)
        
        # Aplicar blur para suavizar
        blurred = cv2.GaussianBlur(resized, (21, 21), 0)
        
        # Aplicar colormap
        heatmap_colored = cv2.applyColorMap(blurred, colormap)
        
        # Crear canal alpha
        alpha_channel = (blurred * alpha).astype(np.uint8)
        
        # Combinar BGR + Alpha
        heatmap_rgba = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2BGRA)
        heatmap_rgba[:, :, 3] = alpha_channel
        
        return heatmap_rgba
    
    def get_combined_heatmap(self, 
                            team_0_color: int = cv2.COLORMAP_AUTUMN,
                            team_1_color: int = cv2.COLORMAP_WINTER,
                            alpha: float = 0.5) -> np.ndarray:
        """
        Genera heatmap combinado de ambos equipos con colores diferentes
        
        Returns:
            Imagen BGR del heatmap combinado
        """
        # Heatmaps individuales
        hm_0 = self.get_heatmap_image('team_0', team_0_color, alpha)
        hm_1 = self.get_heatmap_image('team_1', team_1_color, alpha)
        
        if hm_0 is None or hm_1 is None:
            return np.zeros((self.field_size[1], self.field_size[0], 3), dtype=np.uint8)
        
        # Convertir a BGR
        hm_0_bgr = cv2.cvtColor(hm_0, cv2.COLOR_BGRA2BGR)
        hm_1_bgr = cv2.cvtColor(hm_1, cv2.COLOR_BGRA2BGR)
        
        # Combinar con blending
        combined = cv2.addWeighted(hm_0_bgr, 0.5, hm_1_bgr, 0.5, 0)
        
        return combined
    
    def overlay_on_field(self, 
                        field_image: np.ndarray,
                        entity_type: str,
                        colormap: int = cv2.COLORMAP_JET,
                        alpha: float = 0.6) -> np.ndarray:
        """
        Superpone heatmap sobre imagen del campo
        
        Args:
            field_image: Imagen BGR del campo top-down
            entity_type: Tipo de entidad
            colormap: Mapa de color
            alpha: Transparencia del heatmap
            
        Returns:
            Imagen con heatmap superpuesto
        """
        heatmap = self.get_heatmap_image(entity_type, colormap, alpha=1.0)
        
        if heatmap is None:
            return field_image.copy()
        
        # Convertir heatmap BGRA a BGR
        heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_BGRA2BGR)
        heatmap_mask = heatmap[:, :, 3]
        
        # Normalizar máscara
        mask_norm = heatmap_mask.astype(np.float32) / 255.0 * alpha
        
        # Blending manual
        result = field_image.copy().astype(np.float32)
        for c in range(3):
            result[:, :, c] = (1 - mask_norm) * result[:, :, c] + \
                             mask_norm * heatmap_bgr[:, :, c]
        
        return result.astype(np.uint8)
    
    def get_density_at_position(self, 
                                position: Tuple[float, float],
                                entity_type: str) -> float:
        """
        Obtiene densidad del heatmap en una posición específica
        
        Args:
            position: (x, y) en píxeles del campo
            entity_type: Tipo de entidad
            
        Returns:
            Valor de densidad normalizado 0-1
        """
        if entity_type not in self.heatmap_grids:
            return 0.0
        
        grid = self.heatmap_grids[entity_type]
        
        # Convertir posición a grilla
        grid_x = int(position[0] / self.field_size[0] * self.grid_resolution)
        grid_y = int(position[1] / self.field_size[1] * self.grid_resolution)
        
        # Clamp
        grid_x = np.clip(grid_x, 0, self.grid_resolution - 1)
        grid_y = np.clip(grid_y, 0, self.grid_resolution - 1)
        
        value = grid[grid_y, grid_x]
        
        # Normalizar
        if grid.max() > 0:
            return value / grid.max()
        return 0.0
    
    def get_hotspots(self, 
                    entity_type: str,
                    top_n: int = 5) -> List[Tuple[int, int, float]]:
        """
        Encuentra los N puntos más calientes del heatmap
        
        Returns:
            Lista de (x, y, density) en coordenadas de píxeles
        """
        if entity_type not in self.heatmap_grids:
            return []
        
        grid = self.heatmap_grids[entity_type]
        
        # Flatten y encontrar top N
        flat = grid.flatten()
        top_indices = np.argsort(flat)[-top_n:][::-1]
        
        hotspots = []
        for idx in top_indices:
            grid_y, grid_x = divmod(idx, self.grid_resolution)
            
            # Convertir a píxeles
            x = int(grid_x / self.grid_resolution * self.field_size[0])
            y = int(grid_y / self.grid_resolution * self.field_size[1])
            density = flat[idx]
            
            if density > 0:
                hotspots.append((x, y, density))
        
        return hotspots
    
    def clear_history(self, entity_type: str = None):
        """Limpia histórico de posiciones"""
        if entity_type is None:
            # Limpiar todo
            for key in self.position_buffers:
                self.position_buffers[key].clear()
                self.heatmap_grids[key] = np.zeros((self.grid_resolution, self.grid_resolution))
        elif entity_type in self.position_buffers:
            self.position_buffers[entity_type].clear()
            self.heatmap_grids[entity_type] = np.zeros((self.grid_resolution, self.grid_resolution))
    
    def get_statistics(self) -> Dict[str, Dict]:
        """Retorna estadísticas de los heatmaps"""
        stats = {}
        
        for entity_type, grid in self.heatmap_grids.items():
            buffer = self.position_buffers[entity_type]
            
            stats[entity_type] = {
                'total_positions': len(buffer),
                'max_density': float(grid.max()),
                'mean_density': float(grid.mean()),
                'coverage_area': float(np.sum(grid > 0) / grid.size)  # % del campo cubierto
            }
        
        return stats
