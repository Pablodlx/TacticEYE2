#!/usr/bin/env python3
"""
Setup Check para TacticEYE Web Interface
Verifica que todas las dependencias estén instaladas correctamente
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando versión de Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (se requiere 3.8+)")
        return False

def check_package(package_name, import_name=None):
    """Verificar instalación de paquete Python"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"   ✅ {package_name}")
        return True
    except ImportError:
        print(f"   ❌ {package_name} (no instalado)")
        return False

def check_ffmpeg():
    """Verificar instalación de FFmpeg"""
    print("\n🎬 Verificando FFmpeg...")
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"   ✅ {version_line}")
            return True
        else:
            print("   ❌ FFmpeg no responde correctamente")
            return False
    except FileNotFoundError:
        print("   ❌ FFmpeg no encontrado")
        print("      Instalar: sudo apt-get install ffmpeg")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_yt_dlp():
    """Verificar instalación de yt-dlp"""
    print("\n📺 Verificando yt-dlp...")
    try:
        result = subprocess.run(
            ['yt-dlp', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"   ✅ yt-dlp {version}")
            return True
        else:
            print("   ❌ yt-dlp no responde correctamente")
            return False
    except FileNotFoundError:
        print("   ❌ yt-dlp no encontrado")
        print("      Instalar: pip install yt-dlp")
        return False
    except Exception as e:
        print(f"   ⚠️  yt-dlp instalado vía pip (no en PATH)")
        # Intentar importar
        try:
            import yt_dlp
            print(f"   ✅ yt-dlp disponible como módulo Python")
            return True
        except:
            return False

def check_weights():
    """Verificar modelo YOLO"""
    print("\n🏋️  Verificando modelo YOLO...")
    weights_path = Path("weights/best.pt")
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ weights/best.pt ({size_mb:.1f} MB)")
        return True
    else:
        print("   ❌ weights/best.pt no encontrado")
        print("      El modelo YOLO es necesario para el análisis")
        return False

def check_directories():
    """Verificar/crear directorios necesarios"""
    print("\n📁 Verificando directorios...")
    dirs = [
        "uploads",
        "outputs",
        "outputs_streaming",
        "static",
        "templates"
    ]
    
    all_ok = True
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ⚠️  {dir_name}/ (creando...)")
            dir_path.mkdir(exist_ok=True, parents=True)
    
    return all_ok

def main():
    """Ejecutar todas las verificaciones"""
    print("=" * 60)
    print("🔍 TacticEYE Web Interface - Setup Check")
    print("=" * 60)
    
    results = []
    
    # Python
    results.append(("Python 3.8+", check_python_version()))
    
    # Paquetes básicos
    print("\n📦 Verificando paquetes Python básicos...")
    results.append(("FastAPI", check_package("fastapi")))
    results.append(("Uvicorn", check_package("uvicorn")))
    results.append(("OpenCV", check_package("opencv-python", "cv2")))
    results.append(("NumPy", check_package("numpy")))
    results.append(("PyTorch", check_package("torch")))
    
    # Paquetes YOLO/Tracking
    print("\n🎯 Verificando paquetes de detección...")
    results.append(("Ultralytics", check_package("ultralytics")))
    
    # Paquetes de streaming
    print("\n🌐 Verificando paquetes de streaming...")
    results.append(("FFmpeg-Python", check_package("ffmpeg-python", "ffmpeg")))
    results.append(("WebSockets", check_package("websockets")))
    
    # Herramientas del sistema
    results.append(("FFmpeg", check_ffmpeg()))
    results.append(("yt-dlp", check_yt_dlp()))
    
    # Archivos y directorios
    results.append(("Modelo YOLO", check_weights()))
    results.append(("Directorios", check_directories()))
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    print(f"\n✅ Pasadas: {passed}/{total}")
    print(f"❌ Fallidas: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ¡Todo listo! Puedes ejecutar:")
        print("   python app.py")
        print("\n   Luego abre: http://localhost:8000")
    else:
        print("\n⚠️  Algunas dependencias faltan. Instalar con:")
        print("   pip install -r requirements.txt")
        print("   pip install -r requirements_streaming.txt")
        print("   sudo apt-get install ffmpeg")
        print("   pip install yt-dlp")
    
    print("\n" + "=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
