"""
TacticEYE2 - Sistema Completo de Análisis Táctico
==================================================
Script principal que integra todos los módulos para análisis profesional
"""

import os
# Desactivar completamente Qt para evitar crashes
if 'QT_QPA_PLATFORM' in os.environ:
    del os.environ['QT_QPA_PLATFORM']
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']

import cv2
# DESHABILITAR COMPLETAMENTE TODAS LAS FUNCIONES DE VISUALIZACIÓN
cv2.imshow = lambda *args, **kwargs: None
cv2.waitKey = lambda *args, **kwargs: -1
cv2.destroyAllWindows = lambda *args, **kwargs: None
cv2.namedWindow = lambda *args, **kwargs: None

import numpy as np
import torch
from pathlib import Path
import time
import argparse
from ultralytics import YOLO
import sys

# Importar módulos propios
sys.path.append(str(Path(__file__).parent))
from modules.reid_tracker import ReIDTracker
# from modules.team_classifier import TeamClassifier  # DESACTIVADO - usar solo modelo
from modules.field_calibration import FieldCalibration, FieldDimensions
from modules.heatmap_generator import HeatmapGenerator
from modules.match_statistics import MatchStatistics
from modules.professional_overlay import ProfessionalOverlay
from modules.data_exporter import DataExporter


class TacticEYE2:
    """
    Sistema completo de análisis táctico de fútbol
    """
    
    def __init__(self, 
                 model_path: str,
                 video_path: str,
                 output_dir: str = './outputs',
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.5):
        """
        Args:
            model_path: Ruta al modelo YOLO entrenado
            video_path: Ruta al vídeo a analizar
            output_dir: Directorio de salida
            conf_threshold: Umbral de confianza para detecciones
            iou_threshold: Umbral de IoU para NMS
        """
        self.model_path = Path(model_path)
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Inicializar componentes
        print("🚀 Inicializando TacticEYE2...")
        
        # 1. Detector YOLO
        print(f"📦 Cargando modelo: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # 2. ReID Tracker
        print("🔍 Inicializando ReID Tracker...")
        self.tracker = ReIDTracker(
            max_age=150,
            max_lost_time=120.0,
            feature_buffer_size=20,
            similarity_threshold=0.5
        )
        
        # 3. Team Classifier (DESACTIVADO - solo usar clases del modelo)
        # print("👕 Inicializando clasificador de equipos...")
        # self.team_classifier = TeamClassifier(n_teams=3)
        self.team_classifier = None
        
        # 4. Field Calibration
        print("🏟️  Inicializando calibración de campo...")
        self.field_calibration = FieldCalibration(
            field_dims=FieldDimensions()
        )
        
        # 5. Heatmap Generator
        print("🔥 Inicializando generador de heatmaps...")
        self.heatmap_generator = HeatmapGenerator(
            field_size=(1050, 680),
            update_interval=5.0,
            history_seconds=60.0
        )
        
        # 6. Match Statistics
        print("📊 Inicializando sistema de estadísticas...")
        self.match_stats = MatchStatistics(
            ball_possession_radius=3.0,
            pass_max_distance=40.0,
            fps=30
        )
        
        # 7. Professional Overlay
        print("🎨 Inicializando overlay profesional...")
        self.overlay = ProfessionalOverlay(
            show_ids=True,
            show_trajectories=True,
            show_minimap=True,
            show_stats=True
        )
        
        # 8. Data Exporter
        print("💾 Inicializando exportador...")
        self.exporter = DataExporter(output_dir=str(self.output_dir))
        
        # Estado
        self.calibrated = False
        self.field_topdown = None
        self.frame_count = 0
        self.fps = 30
        
        print("✅ TacticEYE2 inicializado correctamente\n")
    
    def calibrate_field(self, frame: np.ndarray, manual_corners: np.ndarray = None, frame_number: int = 0) -> bool:
        """
        Calibra el campo automáticamente
        
        Args:
            frame: Frame de referencia para calibración
            manual_corners: Opcional - esquinas manuales si auto-detección falla
            frame_number: Número de frame actual
            
        Returns:
            True si calibración exitosa
        """
        print("🔧 Calibrando campo...")
        
        # Detectar líneas
        lines_img = self.field_calibration.detect_field_lines(frame)
        
        # Calcular homografía
        success = self.field_calibration.compute_homography(frame, manual_corners, frame_number)
        
        if success:
            # Crear vista top-down
            self.field_topdown = self.field_calibration.create_topdown_view()
            self.calibrated = True
            print("✅ Campo calibrado correctamente\n")
            return True
        else:
            print("⚠️  Error en calibración del campo")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un frame completo con todos los módulos
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            Frame con overlay completo
        """
        self.frame_count += 1
        current_time = time.time()
        
        # 1. DETECCIÓN YOLO
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # Extraer detecciones
        if len(results.boxes) == 0:
            return frame
        
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        
        # 2. TRACKING CON ReID
        tracks = self.tracker.update(frame, boxes, scores, classes)
        
        if not tracks:
            return frame
        
        # 3. CLASIFICACIÓN DIRECTA DEL MODELO YOLO
        # Usar directamente las 4 clases del modelo:
        # 0 (player) -> Verde
        # 1 (ball) -> Detección automática
        # 2 (referee) -> Amarillo
        # 3 (goalkeeper) -> Magenta
        team_assignments = {}
        for track_id, bbox, class_id in tracks:
            if class_id == 0:  # player
                team_assignments[track_id] = 0
            elif class_id == 2:  # referee
                team_assignments[track_id] = 2
            elif class_id == 3:  # goalkeeper
                team_assignments[track_id] = 3
            # ball (1) no necesita team
        
        # 4. CONVERSIÓN A COORDENADAS 3D (si calibrado)
        players_3d = {}
        ball_3d = None
        players_topdown = {}
        ball_topdown = None
        
        if self.calibrated:
            for track_id, bbox, class_id in tracks:
                # Centro inferior de bbox (punto de contacto con suelo)
                foot_pos = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                
                # Convertir a metros
                pos_3d = self.field_calibration.pixel_to_meters(foot_pos)
                
                if pos_3d is not None:
                    if class_id in [0, 2, 3]:  # Player, Referee, Goalkeeper (3 como player)
                        team_id = team_assignments.get(track_id, -1)
                        players_3d[track_id] = (pos_3d, team_id)
                        
                        # Posición en top-down
                        pos_topdown = self.field_calibration.project_positions_to_topdown(
                            foot_pos.reshape(1, 2)
                        )
                        if pos_topdown is not None:
                            players_topdown[track_id] = (pos_topdown[0], team_id)
                            
                            # Añadir a heatmap
                            entity_type = f'team_{team_id}' if team_id in [0, 1, 2] else 'team_0'
                            self.heatmap_generator.add_position(
                                tuple(pos_topdown[0]),
                                entity_type,
                                current_time
                            )
                    
                    elif class_id == 1:  # Balón
                        ball_3d = pos_3d
                        
                        # Posición en top-down
                        pos_topdown = self.field_calibration.project_positions_to_topdown(
                            foot_pos.reshape(1, 2)
                        )
                        if pos_topdown is not None:
                            ball_topdown = pos_topdown[0]
                            self.heatmap_generator.add_position(
                                tuple(pos_topdown[0]),
                                'ball',
                                current_time
                            )
        
        # 5. ACTUALIZAR ESTADÍSTICAS
        if players_3d:
            self.match_stats.update(
                players_3d,
                ball_3d,
                field_length=self.field_calibration.field_dims.length
            )
        
        # 6. ACTUALIZAR HEATMAPS (cada 5s)
        self.heatmap_generator.update_heatmaps(force=False)
        
        # 7. PREPARAR DATOS PARA OVERLAY
        possession_pct = self.match_stats.get_possession_percentage()
        
        stats_data = {
            'possession': possession_pct,
            'passes': {
                0: {
                    'completed': self.match_stats.team_stats[0].passes_completed,
                    'attempted': self.match_stats.team_stats[0].passes_attempted,
                    'accuracy': self.match_stats.get_pass_accuracy(0)
                },
                1: {
                    'completed': self.match_stats.team_stats[1].passes_completed,
                    'attempted': self.match_stats.team_stats[1].passes_attempted,
                    'accuracy': self.match_stats.get_pass_accuracy(1)
                }
            },
            'distance': {
                0: self.match_stats.team_stats[0].total_distance / 1000.0,
                1: self.match_stats.team_stats[1].total_distance / 1000.0
            }
        }
        
        # Obtener velocidades de jugadores
        player_velocities = {}
        for track_id in team_assignments.keys():
            player_stats = self.match_stats.get_player_stats(track_id)
            if player_stats:
                player_velocities[track_id] = player_stats.avg_speed
        
        # 8. DIBUJAR OVERLAY COMPLETO
        if self.calibrated and self.field_topdown is not None:
            result_frame = self.overlay.draw_complete_overlay(
                frame,
                tracks,
                team_assignments,
                self.field_topdown,
                players_topdown,
                ball_topdown,
                stats_data,
                player_velocities
            )
        else:
            # Overlay básico sin minimap
            result_frame = frame.copy()
            for track_id, bbox, class_id in tracks:
                if class_id in [0, 2, 3]:  # Player, Referee, Goalkeeper
                    team_id = team_assignments.get(track_id, -1)
                    velocity = player_velocities.get(track_id)
                    result_frame = self.overlay.draw_player_overlay(
                        result_frame, track_id, bbox, team_id, velocity
                    )
                elif class_id == 1:
                    result_frame = self.overlay.draw_ball_overlay(result_frame, bbox)
        
        # 9. EXPORTAR DATOS (cada frame)
        for track_id, bbox, class_id in tracks:
            if class_id in [0, 2, 3] and track_id in players_3d:  # Player, Referee, Goalkeeper
                pos_3d, team_id = players_3d[track_id]
                foot_pos = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                velocity = player_velocities.get(track_id, 0.0)
                
                self.exporter.add_position_record(
                    frame=self.frame_count,
                    timestamp=current_time,
                    track_id=track_id,
                    team_id=team_id,
                    pos_pixels=tuple(foot_pos),
                    pos_meters=tuple(pos_3d),
                    velocity_kmh=velocity
                )
        
        return result_frame
    
    def analyze_video(self, 
                     calibration_frame: int = 100,
                     show_preview: bool = False,
                     max_frames: int = None):
        # FORZAR: Nunca mostrar preview para evitar crashes
        show_preview = False
        """
        Analiza un vídeo completo
        
        Args:
            calibration_frame: Frame a usar para calibración
            show_preview: Mostrar preview en tiempo real
            max_frames: Número máximo de frames a procesar (None = todos)
        """
        print(f"🎬 Analizando vídeo: {self.video_path}\n")
        
        # Abrir vídeo
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            print(f"❌ Error: No se pudo abrir el vídeo {self.video_path}")
            return
        
        # Obtener propiedades del vídeo
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Propiedades del vídeo:")
        print(f"   - Resolución: {frame_width}x{frame_height}")
        print(f"   - FPS: {self.fps}")
        print(f"   - Total frames: {total_frames}")
        print(f"   - Duración: {total_frames / self.fps:.1f}s\n")
        
        # Inicializar exportador de vídeo
        output_video_name = f"analyzed_{self.video_path.stem}.mp4"
        self.exporter.initialize_video_writer(
            output_video_name,
            self.fps,
            (frame_width, frame_height),
            codec='mp4v'
        )
        
        # Procesar frames
        start_time = time.time()
        last_fps_time = start_time
        frame_count_for_fps = 0
        current_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Límite de frames
                if max_frames and self.frame_count >= max_frames:
                    print(f"\n⏸️  Límite de {max_frames} frames alcanzado")
                    break
                
                # Calibración automática multi-frame
                if self.frame_count == calibration_frame and not self.calibrated:
                    print(f"\n🔧 Calibrando en frame {calibration_frame}...")
                    self.calibrate_field(frame, frame_number=self.frame_count)
                elif self.calibrated and self.field_calibration.should_recalibrate(self.frame_count):
                    # Recalibrar periódicamente para acumular mejores homografías
                    if self.field_calibration.compute_homography(frame, frame_number=self.frame_count):
                        pass  # Silencioso, solo acumula
                
                # Procesar frame
                processed_frame = self.process_frame(frame)
                
                # Escribir al vídeo de salida
                self.exporter.write_frame(processed_frame)
                
                # Calcular FPS de procesamiento
                frame_count_for_fps += 1
                if time.time() - last_fps_time >= 1.0:
                    current_fps = frame_count_for_fps / (time.time() - last_fps_time)
                    last_fps_time = time.time()
                    frame_count_for_fps = 0
                
                # Añadir FPS al frame
                cv2.putText(
                    processed_frame,
                    f"Processing FPS: {current_fps:.1f}",
                    (frame_width - 250, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                # Preview
                if show_preview:
                    # Redimensionar para preview si es muy grande
                    if frame_width > 1920:
                        scale = 1920 / frame_width
                        preview_frame = cv2.resize(
                            processed_frame,
                            (int(frame_width * scale), int(frame_height * scale))
                        )
                    else:
                        preview_frame = processed_frame
                    
                    cv2.imshow('TacticEYE2 - Análisis en vivo', preview_frame)
                    
                    # Presionar 'q' para salir, 'p' para pausar
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⏹️  Análisis detenido por usuario")
                        break
                    elif key == ord('p'):
                        print("\n⏸️  Pausado (presiona cualquier tecla para continuar)")
                        cv2.waitKey(0)
                
                # Progreso
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100 if total_frames > 0 else 0
                    elapsed = time.time() - start_time
                    estimated_total = (elapsed / self.frame_count) * total_frames if self.frame_count > 0 else 0
                    remaining = estimated_total - elapsed
                    
                    print(f"📊 Frame {self.frame_count}/{total_frames} ({progress:.1f}%) | "
                          f"FPS: {current_fps:.1f} | "
                          f"Tiempo restante: {remaining/60:.1f}min")
        
        except KeyboardInterrupt:
            print("\n⚠️  Análisis interrumpido por usuario")
        
        finally:
            # Refinar calibración con todas las homografías acumuladas
            if self.calibrated and len(self.field_calibration.homography_candidates) > 1:
                print("\n🔍 Refinando calibración con múltiples frames...")
                self.field_calibration.refine_homography(min_quality=0.4)
                
                # Mostrar estadísticas
                info = self.field_calibration.get_calibration_info()
                print(f"  Total calibraciones: {info['num_calibrations']}")
                if 'best_quality' in info:
                    print(f"  Mejor calidad: {info['best_quality']:.3f} (frame {info['best_frame']})")
            
            # Limpieza
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            
            # Finalizar vídeo
            self.exporter.finalize_video()
            
            # Exportar todos los datos
            print("\n📦 Exportando datos finales...")
            
            # Preparar datos para exportación
            match_summary = self.match_stats.export_summary()
            
            heatmap_grids = {
                key: self.heatmap_generator.heatmap_grids[key]
                for key in self.heatmap_generator.heatmap_grids
            }
            
            trajectories = {
                track_id: list(traj)
                for track_id, traj in self.overlay.trajectories.items()
            }
            
            metadata = {
                'video_source': str(self.video_path),
                'model': str(self.model_path),
                'total_frames_processed': self.frame_count,
                'fps': self.fps,
                'resolution': f"{frame_width}x{frame_height}",
                'processing_time_seconds': time.time() - start_time
            }
            
            self.exporter.export_all(
                stats=match_summary,
                heatmap_grids=heatmap_grids,
                trajectories=trajectories,
                field_size=self.heatmap_generator.field_size,
                metadata=metadata
            )
            
            # Resumen final
            elapsed_total = time.time() - start_time
            possession_pct = self.match_stats.get_possession_percentage()
            print(f"\n{'='*60}")
            print(f"✅ ANÁLISIS COMPLETADO")
            print(f"{'='*60}")
            print(f"⏱️  Tiempo total: {elapsed_total/60:.1f} minutos")
            print(f"🎞️  Frames procesados: {self.frame_count}")
            print(f"⚡ FPS promedio: {self.frame_count / elapsed_total:.1f}")
            print(f"👥 Tracks detectados: {self.tracker.next_track_id - 1}")
            print(f"📊 Posesión: {possession_pct[0]:.1f}% vs {possession_pct[1]:.1f}%")
            print(f"📁 Salida: {self.output_dir}")
            print(f"{'='*60}\n")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='TacticEYE2 - Sistema Completo de Análisis Táctico de Fútbol'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='weights/best.pt',
        help='Ruta al modelo YOLO'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Ruta al vídeo a analizar'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./outputs',
        help='Directorio de salida'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.3,
        help='Umbral de confianza'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help='Umbral de IoU'
    )
    parser.add_argument(
        '--calibration-frame',
        type=int,
        default=100,
        help='Frame para calibración automática'
    )
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Desactivar preview en tiempo real'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Número máximo de frames a procesar'
    )
    
    args = parser.parse_args()
    
    # Inicializar sistema
    system = TacticEYE2(
        model_path=args.model,
        video_path=args.video,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Analizar vídeo
    system.analyze_video(
        calibration_frame=args.calibration_frame,
        show_preview=not args.no_preview,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()
