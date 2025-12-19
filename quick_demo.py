"""
Quick Demo - Análisis rápido de muestra
========================================
Script simplificado para testing rápido
"""

import os
# Desactivar DISPLAY completamente
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']
if 'QT_QPA_PLATFORM' in os.environ:
    del os.environ['QT_QPA_PLATFORM']

import sys
from pathlib import Path

# Agregar módulos al path
sys.path.append(str(Path(__file__).parent))

from analyze_match import TacticEYE2


def main():
    """Demo rápido con configuración por defecto"""
    
    print("🎯 TacticEYE2 - Demo Rápido\n")
    
    # Verificar si existe sample_match.mp4
    video_path = Path("sample_match.mp4")
    if not video_path.exists():
        print("❌ Error: No se encuentra 'sample_match.mp4'")
        print("   Por favor, coloca un vídeo de ejemplo en el directorio raíz")
        return
    
    # Verificar modelo
    model_path = Path("weights/best.pt")
    if not model_path.exists():
        print("❌ Error: No se encuentra el modelo en 'weights/best.pt'")
        return
    
    print("✅ Archivos encontrados")
    print(f"   - Vídeo: {video_path}")
    print(f"   - Modelo: {model_path}\n")
    
    # Inicializar sistema
    system = TacticEYE2(
        model_path=str(model_path),
        video_path=str(video_path),
        output_dir='./outputs_demo',
        conf_threshold=0.3,
        iou_threshold=0.5
    )
    
    # Analizar solo los primeros 300 frames (10 segundos @ 30fps)
    print("🎬 Analizando primeros 10 segundos del vídeo...\n")
    
    system.analyze_video(
        calibration_frame=50,
        show_preview=False,  # Sin ventanas
        max_frames=300
    )
    
    print("\n✨ Demo completado!")
    print("   Revisa la carpeta './outputs_demo' para ver los resultados")


if __name__ == '__main__':
    main()
