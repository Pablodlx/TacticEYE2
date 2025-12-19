"""
Utilities - Herramientas adicionales
====================================
Scripts de utilidad para análisis y debugging
"""

import cv2
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def export_analysis_to_excel(output_dir: str):
    """
    Convierte CSVs y JSONs a un Excel multi-hoja para análisis fácil
    
    Args:
        output_dir: Directorio con archivos exportados
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Error: Directorio no encontrado: {output_dir}")
        return
    
    # Buscar archivos
    position_files = list(output_path.glob("positions_*.csv"))
    summary_files = list(output_path.glob("match_summary_*.json"))
    
    if not position_files and not summary_files:
        print("❌ No se encontraron archivos de análisis")
        return
    
    # Crear Excel writer
    excel_path = output_path / "analysis_complete.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Sheet 1: Posiciones
        if position_files:
            print("📊 Exportando posiciones...")
            df_positions = pd.read_csv(position_files[0])
            df_positions.to_excel(writer, sheet_name='Posiciones', index=False)
        
        # Sheet 2: Resumen de partido
        if summary_files:
            print("📊 Exportando resumen del partido...")
            with open(summary_files[0], 'r') as f:
                summary = json.load(f)
            
            # Estadísticas de equipos
            if 'match_statistics' in summary:
                stats = summary['match_statistics']
                
                # Team stats
                team_data = []
                for team_key in ['team_0', 'team_1']:
                    if team_key in stats:
                        team_stats = stats[team_key]
                        team_stats['team'] = 'Local' if team_key == 'team_0' else 'Visitante'
                        team_data.append(team_stats)
                
                if team_data:
                    df_teams = pd.DataFrame(team_data)
                    df_teams.to_excel(writer, sheet_name='Equipos', index=False)
        
        # Sheet 3: Top Jugadores (distancia)
        if summary_files:
            with open(summary_files[0], 'r') as f:
                summary = json.load(f)
            
            if 'match_statistics' in summary:
                stats = summary['match_statistics']
                
                if 'top_distance_runners' in stats:
                    top_dist = stats['top_distance_runners']
                    df_dist = pd.DataFrame(top_dist, columns=['ID', 'Equipo', 'Distancia (km)'])
                    df_dist['Equipo'] = df_dist['Equipo'].map({0: 'Local', 1: 'Visitante'})
                    df_dist.to_excel(writer, sheet_name='Top Distancia', index=False)
                
                if 'fastest_players' in stats:
                    top_speed = stats['fastest_players']
                    df_speed = pd.DataFrame(top_speed, columns=['ID', 'Equipo', 'Velocidad (km/h)'])
                    df_speed['Equipo'] = df_speed['Equipo'].map({0: 'Local', 1: 'Visitante'})
                    df_speed.to_excel(writer, sheet_name='Top Velocidad', index=False)
    
    print(f"\n✅ Excel generado: {excel_path}")


def extract_video_clip(video_path: str, 
                       start_time: float, 
                       end_time: float,
                       output_path: str = None):
    """
    Extrae un clip de vídeo entre dos timestamps
    
    Args:
        video_path: Ruta al vídeo
        start_time: Tiempo inicial en segundos
        end_time: Tiempo final en segundos
        output_path: Ruta del clip de salida
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if output_path is None:
        output_path = f"clip_{start_time:.1f}_{end_time:.1f}.mp4"
    
    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Ir al frame inicial
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"✂️  Extrayendo clip: {start_time:.1f}s - {end_time:.1f}s")
    
    frame_count = 0
    total_frames = end_frame - start_frame
    
    while True:
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progreso: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    print(f"✅ Clip guardado: {output_path}")


def create_comparison_video(original_path: str,
                           analyzed_path: str,
                           output_path: str = "comparison.mp4"):
    """
    Crea vídeo lado a lado: original vs analizado
    
    Args:
        original_path: Vídeo original
        analyzed_path: Vídeo analizado
        output_path: Vídeo de salida
    """
    cap1 = cv2.VideoCapture(original_path)
    cap2 = cv2.VideoCapture(analyzed_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Error: No se pudieron abrir los vídeos")
        return
    
    fps = cap1.get(cv2.CAP_PROP_FPS)
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Vídeo de salida: doble ancho
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w1*2, h1))
    
    print("🎬 Creando vídeo de comparación...")
    
    frame_count = 0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Combinar lado a lado
        combined = np.hstack([frame1, frame2])
        
        # Añadir labels
        cv2.putText(combined, "ORIGINAL", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(combined, "ANALYZED", (w1 + 50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        out.write(combined)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"   Procesados {frame_count} frames")
    
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"✅ Vídeo de comparación guardado: {output_path}")


def analyze_statistics(output_dir: str):
    """
    Analiza y muestra estadísticas resumidas de forma bonita
    
    Args:
        output_dir: Directorio con exports
    """
    output_path = Path(output_dir)
    summary_files = list(output_path.glob("match_summary_*.json"))
    
    if not summary_files:
        print("❌ No se encontró archivo de resumen")
        return
    
    with open(summary_files[0], 'r') as f:
        data = json.load(f)
    
    stats = data.get('match_statistics', {})
    
    print("\n" + "="*60)
    print("📊 RESUMEN DEL PARTIDO".center(60))
    print("="*60)
    
    # Duración
    if 'match_duration_seconds' in stats:
        duration = stats['match_duration_seconds']
        print(f"\n⏱️  Duración: {duration/60:.1f} minutos")
    
    # Equipos
    for team_key in ['team_0', 'team_1']:
        if team_key in stats:
            team_name = "🟢 EQUIPO LOCAL" if team_key == 'team_0' else "🔵 EQUIPO VISITANTE"
            team = stats[team_key]
            
            print(f"\n{team_name}")
            print("-" * 60)
            print(f"  Posesión:       {team.get('possession_%', 0):.1f}%")
            print(f"  Pases:          {team.get('passes_completed', 0)}/{team.get('passes_attempted', 0)} "
                  f"({team.get('pass_accuracy_%', 0):.1f}%)")
            print(f"  Distancia:      {team.get('total_distance_km', 0):.2f} km")
            print(f"  Vel. promedio:  {team.get('avg_speed_kmh', 0):.1f} km/h")
            print(f"  Jugadores:      {team.get('num_players', 0)}")
    
    # Zonas de presión
    if 'pressure_zones_%' in stats:
        zones = stats['pressure_zones_%']
        print(f"\n🎯 PRESIÓN POR ZONAS")
        print("-" * 60)
        print(f"  Alta:    {zones.get('high', 0):.1f}%")
        print(f"  Media:   {zones.get('medium', 0):.1f}%")
        print(f"  Baja:    {zones.get('low', 0):.1f}%")
    
    # Top jugadores
    if 'top_distance_runners' in stats:
        print(f"\n🏃 TOP 5 DISTANCIA")
        print("-" * 60)
        for i, (tid, team, dist) in enumerate(stats['top_distance_runners'][:5], 1):
            team_name = "Local" if team == 0 else "Visitante"
            print(f"  {i}. ID #{tid} ({team_name}): {dist:.2f} km")
    
    if 'fastest_players' in stats:
        print(f"\n⚡ TOP 5 VELOCIDAD")
        print("-" * 60)
        for i, (tid, team, speed) in enumerate(stats['fastest_players'][:5], 1):
            team_name = "Local" if team == 0 else "Visitante"
            print(f"  {i}. ID #{tid} ({team_name}): {speed:.1f} km/h")
    
    print("\n" + "="*60 + "\n")


def main():
    """CLI de utilidades"""
    parser = argparse.ArgumentParser(
        description='Utilidades adicionales de TacticEYE2'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Excel export
    excel_parser = subparsers.add_parser('excel', help='Exportar a Excel')
    excel_parser.add_argument('output_dir', help='Directorio con exports')
    
    # Video clip
    clip_parser = subparsers.add_parser('clip', help='Extraer clip de vídeo')
    clip_parser.add_argument('video', help='Vídeo de entrada')
    clip_parser.add_argument('start', type=float, help='Tiempo inicial (s)')
    clip_parser.add_argument('end', type=float, help='Tiempo final (s)')
    clip_parser.add_argument('--output', help='Archivo de salida')
    
    # Comparison
    comp_parser = subparsers.add_parser('compare', help='Vídeo comparación')
    comp_parser.add_argument('original', help='Vídeo original')
    comp_parser.add_argument('analyzed', help='Vídeo analizado')
    comp_parser.add_argument('--output', default='comparison.mp4', help='Salida')
    
    # Stats
    stats_parser = subparsers.add_parser('stats', help='Mostrar estadísticas')
    stats_parser.add_argument('output_dir', help='Directorio con exports')
    
    args = parser.parse_args()
    
    if args.command == 'excel':
        export_analysis_to_excel(args.output_dir)
    elif args.command == 'clip':
        extract_video_clip(args.video, args.start, args.end, args.output)
    elif args.command == 'compare':
        create_comparison_video(args.original, args.analyzed, args.output)
    elif args.command == 'stats':
        analyze_statistics(args.output_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
