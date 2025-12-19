#!/usr/bin/env python3
"""
Setup & Installation Checker for TacticEYE2
===========================================
Verifica la instalación y dependencias
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Verifica versión de Python"""
    print("🐍 Verificando versión de Python...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (se requiere 3.8+)")
        return False


def check_dependencies():
    """Verifica dependencias principales"""
    print("\n📦 Verificando dependencias...")
    
    dependencies = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'ultralytics': 'Ultralytics YOLO',
        'sklearn': 'scikit-learn',
        'numpy': 'NumPy',
        'pandas': 'Pandas'
    }
    
    all_ok = True
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NO INSTALADO")
            all_ok = False
    
    return all_ok


def check_cuda():
    """Verifica disponibilidad de CUDA"""
    print("\n🚀 Verificando CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA disponible")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("   ⚠️  CUDA no disponible (se usará CPU)")
            return False
    except:
        print("   ❌ Error verificando CUDA")
        return False


def check_project_structure():
    """Verifica estructura del proyecto"""
    print("\n📁 Verificando estructura del proyecto...")
    
    required_files = [
        'analyze_match.py',
        'config.yaml',
        'requirements.txt',
        'README.md',
        'modules/__init__.py',
        'modules/reid_tracker.py',
        'modules/team_classifier.py',
        'modules/field_calibration.py',
        'modules/heatmap_generator.py',
        'modules/match_statistics.py',
        'modules/professional_overlay.py',
        'modules/data_exporter.py'
    ]
    
    all_ok = True
    base_path = Path(__file__).parent
    
    for file in required_files:
        file_path = base_path / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - NO ENCONTRADO")
            all_ok = False
    
    return all_ok


def check_model_weights():
    """Verifica modelo YOLO"""
    print("\n🏋️  Verificando modelo...")
    
    weights_path = Path('weights/best.pt')
    
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Modelo encontrado ({size_mb:.1f} MB)")
        return True
    else:
        print("   ⚠️  Modelo no encontrado en weights/best.pt")
        print("      Asegúrate de tener tu modelo YOLO entrenado")
        return False


def install_dependencies():
    """Instala dependencias faltantes"""
    print("\n🔧 ¿Deseas instalar las dependencias automáticamente? (s/n): ", end='')
    response = input().lower().strip()
    
    if response == 's':
        print("\n📥 Instalando dependencias...")
        try:
            subprocess.check_call([
                sys.executable, 
                '-m', 
                'pip', 
                'install', 
                '-r', 
                'requirements.txt'
            ])
            print("✅ Dependencias instaladas correctamente")
            return True
        except subprocess.CalledProcessError:
            print("❌ Error instalando dependencias")
            return False
    else:
        print("⏭️  Saltando instalación automática")
        return False


def create_directories():
    """Crea directorios necesarios"""
    print("\n📂 Creando directorios...")
    
    directories = ['outputs', 'data']
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        print(f"   ✅ {directory}/")
    
    return True


def run_quick_test():
    """Ejecuta test rápido del sistema"""
    print("\n🧪 ¿Deseas ejecutar un test rápido del sistema? (s/n): ", end='')
    response = input().lower().strip()
    
    if response == 's':
        print("\n🔬 Ejecutando test...")
        try:
            # Test de importación de módulos
            from modules.reid_tracker import ReIDTracker
            from modules.team_classifier import TeamClassifier
            from modules.field_calibration import FieldCalibration
            from modules.heatmap_generator import HeatmapGenerator
            from modules.match_statistics import MatchStatistics
            from modules.professional_overlay import ProfessionalOverlay
            from modules.data_exporter import DataExporter
            
            print("   ✅ Todos los módulos se importan correctamente")
            
            # Test básico de inicialización
            tracker = ReIDTracker()
            print("   ✅ ReID Tracker inicializado")
            
            classifier = TeamClassifier()
            print("   ✅ Team Classifier inicializado")
            
            calibration = FieldCalibration()
            print("   ✅ Field Calibration inicializado")
            
            print("\n✅ Test completado exitosamente")
            return True
            
        except Exception as e:
            print(f"\n❌ Error en test: {str(e)}")
            return False
    else:
        print("⏭️  Saltando test")
        return True


def main():
    """Función principal"""
    print("="*60)
    print("🎯 TacticEYE2 - Setup & Installation Checker")
    print("="*60)
    
    results = {
        'Python': check_python_version(),
        'Dependencies': check_dependencies(),
        'CUDA': check_cuda(),
        'Structure': check_project_structure(),
        'Model': check_model_weights()
    }
    
    # Si faltan dependencias, ofrecer instalar
    if not results['Dependencies']:
        if install_dependencies():
            results['Dependencies'] = check_dependencies()
    
    # Crear directorios
    create_directories()
    
    # Test opcional
    test_ok = run_quick_test()
    
    # Resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("="*60)
    
    for check, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check}")
    
    all_ok = all(results.values())
    
    if all_ok and test_ok:
        print("\n🎉 ¡Todo listo! TacticEYE2 está correctamente instalado")
        print("\n📖 Comandos de inicio rápido:")
        print("   python quick_demo.py")
        print("   python analyze_match.py --video sample_match.mp4")
    else:
        print("\n⚠️  Hay algunos problemas que necesitan atención")
        print("   Revisa los mensajes arriba para más detalles")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
