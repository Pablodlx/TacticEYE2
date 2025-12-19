"""
Data Exporter - Sistema de exportación de análisis
==================================================
Exporta vídeo con overlay, CSV con posiciones 3D, JSON con eventos
"""

import cv2
import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class PositionRecord:
    """Registro de posición para exportación"""
    frame: int
    timestamp: float
    track_id: int
    team_id: int
    x_pixels: float
    y_pixels: float
    x_meters: float
    y_meters: float
    velocity_kmh: float = 0.0


@dataclass
class EventRecord:
    """Registro de evento para exportación"""
    timestamp: float
    frame: int
    event_type: str  # 'pass', 'shot', 'tackle', 'possession_change'
    team_id: int
    player_id: Optional[int] = None
    x_meters: float = 0.0
    y_meters: float = 0.0
    success: bool = True
    metadata: Dict = None


class DataExporter:
    """
    Maneja exportación de todos los datos del análisis
    """
    
    def __init__(self, output_dir: str = './outputs'):
        """
        Args:
            output_dir: Directorio de salida para exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buffers de datos
        self.position_records: List[PositionRecord] = []
        self.event_records: List[EventRecord] = []
        
        # Video writer
        self.video_writer = None
        self.video_path = None
        
    def initialize_video_writer(self,
                                video_path: str,
                                fps: int,
                                frame_size: Tuple[int, int],
                                codec: str = 'mp4v'):
        """
        Inicializa el writer de vídeo
        
        Args:
            video_path: Ruta del vídeo de salida
            fps: Frames por segundo
            frame_size: (width, height)
            codec: Codec de vídeo (mp4v, H264, etc.)
        """
        self.video_path = self.output_dir / video_path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            fps,
            frame_size
        )
        
        if self.video_writer.isOpened():
            print(f"✓ Video writer inicializado: {self.video_path}")
        else:
            print(f"⚠ Error inicializando video writer")
    
    def write_frame(self, frame: np.ndarray):
        """Escribe un frame al vídeo de salida"""
        if self.video_writer is not None and self.video_writer.isOpened():
            self.video_writer.write(frame)
    
    def add_position_record(self,
                           frame: int,
                           timestamp: float,
                           track_id: int,
                           team_id: int,
                           pos_pixels: Tuple[float, float],
                           pos_meters: Tuple[float, float],
                           velocity_kmh: float = 0.0):
        """Añade registro de posición al buffer"""
        record = PositionRecord(
            frame=frame,
            timestamp=timestamp,
            track_id=track_id,
            team_id=team_id,
            x_pixels=pos_pixels[0],
            y_pixels=pos_pixels[1],
            x_meters=pos_meters[0],
            y_meters=pos_meters[1],
            velocity_kmh=velocity_kmh
        )
        self.position_records.append(record)
    
    def add_event_record(self,
                        timestamp: float,
                        frame: int,
                        event_type: str,
                        team_id: int,
                        player_id: Optional[int] = None,
                        position: Optional[Tuple[float, float]] = None,
                        success: bool = True,
                        metadata: Dict = None):
        """Añade registro de evento al buffer"""
        record = EventRecord(
            timestamp=timestamp,
            frame=frame,
            event_type=event_type,
            team_id=team_id,
            player_id=player_id,
            x_meters=position[0] if position else 0.0,
            y_meters=position[1] if position else 0.0,
            success=success,
            metadata=metadata or {}
        )
        self.event_records.append(record)
    
    def export_positions_csv(self, filename: str = None):
        """
        Exporta posiciones a CSV
        
        Args:
            filename: Nombre del archivo (auto-genera si None)
        """
        if not self.position_records:
            print("⚠ No hay registros de posición para exportar")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"positions_{timestamp}.csv"
        
        csv_path = self.output_dir / filename
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'frame', 'timestamp', 'track_id', 'team_id',
                'x_pixels', 'y_pixels', 'x_meters', 'y_meters', 'velocity_kmh'
            ])
            writer.writeheader()
            
            for record in self.position_records:
                writer.writerow(asdict(record))
        
        print(f"✓ Posiciones exportadas a {csv_path} ({len(self.position_records)} registros)")
    
    def export_events_json(self, filename: str = None):
        """
        Exporta eventos a JSON
        
        Args:
            filename: Nombre del archivo (auto-genera si None)
        """
        if not self.event_records:
            print("⚠ No hay eventos para exportar")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"events_{timestamp}.json"
        
        json_path = self.output_dir / filename
        
        # Convertir records a dicts
        events_data = [asdict(record) for record in self.event_records]
        
        with open(json_path, 'w') as f:
            json.dump({
                'events': events_data,
                'total_events': len(events_data),
                'export_date': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"✓ Eventos exportados a {json_path} ({len(self.event_records)} eventos)")
    
    def export_match_summary(self, 
                            stats: Dict,
                            metadata: Dict = None,
                            filename: str = None):
        """
        Exporta resumen completo del partido a JSON
        
        Args:
            stats: Diccionario con estadísticas del partido
            metadata: Metadata adicional (video source, model info, etc.)
            filename: Nombre del archivo
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"match_summary_{timestamp}.json"
        
        json_path = self.output_dir / filename
        
        # Convertir tipos numpy a Python nativos
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        summary = {
            'export_info': {
                'date': datetime.now().isoformat(),
                'version': '2.0.0',
                'system': 'TacticEYE2'
            },
            'match_statistics': convert_numpy(stats),
            'total_frames_processed': len(set(r.frame for r in self.position_records)),
            'total_tracks': len(set(r.track_id for r in self.position_records)),
            'total_events': len(self.event_records)
        }
        
        if metadata:
            summary['metadata'] = convert_numpy(metadata)
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Resumen del partido exportado a {json_path}")
    
    def export_heatmap_data(self,
                           heatmap_grids: Dict[str, np.ndarray],
                           field_size: Tuple[int, int],
                           filename: str = None):
        """
        Exporta datos de heatmaps a formato numpy comprimido
        
        Args:
            heatmap_grids: Dict con grillas de heatmaps
            field_size: Tamaño del campo en píxeles
            filename: Nombre del archivo
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"heatmaps_{timestamp}.npz"
        
        npz_path = self.output_dir / filename
        
        # Guardar todas las grillas
        np.savez_compressed(
            npz_path,
            field_size=field_size,
            **heatmap_grids
        )
        
        print(f"✓ Heatmaps exportados a {npz_path}")
    
    def export_trajectory_data(self,
                              trajectories: Dict[int, List[Tuple[int, int]]],
                              filename: str = None):
        """
        Exporta trayectorias de jugadores a JSON
        
        Args:
            trajectories: Dict {track_id: [(x, y), ...]}
            filename: Nombre del archivo
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectories_{timestamp}.json"
        
        json_path = self.output_dir / filename
        
        # Convertir a formato serializable
        traj_data = {
            str(track_id): [{'x': int(x), 'y': int(y)} for x, y in traj]
            for track_id, traj in trajectories.items()
        }
        
        with open(json_path, 'w') as f:
            json.dump({
                'trajectories': traj_data,
                'total_tracks': len(traj_data),
                'export_date': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"✓ Trayectorias exportadas a {json_path}")
    
    def finalize_video(self):
        """Finaliza y cierra el vídeo de salida"""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"✓ Vídeo finalizado: {self.video_path}")
            self.video_writer = None
    
    def export_all(self,
                  stats: Dict,
                  heatmap_grids: Dict[str, np.ndarray],
                  trajectories: Dict[int, List[Tuple[int, int]]],
                  field_size: Tuple[int, int],
                  metadata: Dict = None):
        """
        Exporta todos los datos de una vez
        
        Args:
            stats: Estadísticas del partido
            heatmap_grids: Grillas de heatmaps
            trajectories: Trayectorias de jugadores
            field_size: Tamaño del campo
            metadata: Metadata adicional
        """
        print("\n🔄 Exportando todos los datos...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV de posiciones
        self.export_positions_csv(f"positions_{timestamp}.csv")
        
        # JSON de eventos
        self.export_events_json(f"events_{timestamp}.json")
        
        # Resumen del partido
        self.export_match_summary(stats, metadata, f"match_summary_{timestamp}.json")
        
        # Heatmaps
        self.export_heatmap_data(heatmap_grids, field_size, f"heatmaps_{timestamp}.npz")
        
        # Trayectorias
        self.export_trajectory_data(trajectories, f"trajectories_{timestamp}.json")
        
        print(f"\n✅ Exportación completa en: {self.output_dir}")
        print(f"   - Vídeo: {self.video_path.name if self.video_path else 'N/A'}")
        print(f"   - Posiciones CSV: positions_{timestamp}.csv")
        print(f"   - Eventos JSON: events_{timestamp}.json")
        print(f"   - Resumen: match_summary_{timestamp}.json")
        print(f"   - Heatmaps: heatmaps_{timestamp}.npz")
        print(f"   - Trayectorias: trajectories_{timestamp}.json")
    
    def clear_buffers(self):
        """Limpia todos los buffers de datos"""
        self.position_records.clear()
        self.event_records.clear()
        print("✓ Buffers limpiados")
