# 🚀 Guía de Instalación - TacticEYE2

Esta guía te llevará paso a paso para configurar TacticEYE2 en tu sistema.

## 📋 Requisitos Previos

### Hardware Mínimo
- **CPU**: Intel i5 / AMD Ryzen 5 o superior
- **RAM**: 8GB (16GB recomendado)
- **GPU**: NVIDIA con 6GB VRAM (opcional pero muy recomendado)
- **Almacenamiento**: 5GB libres

### Sistema Operativo
- ✅ Linux (Ubuntu 20.04+, Debian, etc.)
- ✅ Windows 10/11
- ✅ macOS 11+ (sin aceleración GPU)

---

## 🔧 Instalación Paso a Paso

### Paso 1: Instalar Python 3.8+

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

#### Windows
1. Descargar desde [python.org](https://www.python.org/downloads/)
2. Ejecutar instalador
3. ✅ Marcar "Add Python to PATH"

#### macOS
```bash
brew install python@3.10
```

### Paso 2: Clonar el Repositorio

```bash
git clone https://github.com/Pablodlx/TacticEYE2.git
cd TacticEYE2
```

### Paso 3: Crear Entorno Virtual

#### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows (CMD)
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Windows (PowerShell)
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> **Nota**: Si hay error en PowerShell, ejecutar primero:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Paso 4: Instalar Dependencias

#### Opción A: Con GPU (NVIDIA)

Primero, verificar versión de CUDA:
```bash
nvidia-smi
```

Instalar PyTorch con CUDA:
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Luego, instalar el resto:
```bash
pip install -r requirements.txt
```

#### Opción B: Solo CPU (sin GPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Paso 5: Verificar Instalación

```bash
python setup_check.py
```

Este script verificará:
- ✅ Versión de Python
- ✅ Dependencias instaladas
- ✅ CUDA disponible (si hay GPU)
- ✅ Estructura del proyecto
- ✅ Modelo YOLO

---

## 🎯 Configuración del Modelo

### Si ya tienes `weights/best.pt`
✅ ¡Listo! El modelo está incluido.

### Si NO tienes el modelo

Opciones:

1. **Usar modelo pre-entrenado genérico**:
```bash
# Descargar YOLO11l
from ultralytics import YOLO
model = YOLO('yolo11l.pt')
# Moverlo a weights/best.pt
```

2. **Entrenar tu propio modelo**:
```bash
python train_fast.py --data soccernet.yaml --epochs 50 --img 1280
```

---

## 🧪 Prueba Rápida

### 1. Demo de 10 segundos

```bash
python quick_demo.py
```

### 2. Análisis completo

```bash
python analyze_match.py --video sample_match.mp4
```

---

## 🐛 Solución de Problemas Comunes

### Error: "No module named 'cv2'"
```bash
pip install opencv-python opencv-contrib-python
```

### Error: "CUDA out of memory"
Reducir resolución:
```bash
python analyze_match.py --video video.mp4 --conf 0.4
```

O usar CPU:
```bash
# Editar analyze_match.py, línea de device
device = 'cpu'
```

### Error: "torch not found"
```bash
pip uninstall torch torchvision
# Reinstalar según tu sistema (GPU/CPU)
```

### Error: "calibration failed"
Probar con otro frame:
```bash
python analyze_match.py --video video.mp4 --calibration-frame 500
```

### Windows: "Script activation error"
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 🔄 Actualización

Para actualizar a la última versión:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

## 📦 Instalación con Docker (Avanzado)

Si prefieres usar Docker:

```bash
# Construir imagen
docker build -t tacticeye2 .

# Ejecutar
docker run --gpus all -v $(pwd)/outputs:/app/outputs tacticeye2 \
    python analyze_match.py --video sample_match.mp4
```

---

## 🌐 Instalación en Servidor (Sin Display)

Para servidores sin interfaz gráfica:

```bash
# Instalar dependencias sin GUI
pip install opencv-python-headless

# Ejecutar sin preview
python analyze_match.py --video video.mp4 --no-preview
```

---

## 📊 Verificación Final

Ejecutar todos los tests:

```bash
# 1. Verificar instalación
python setup_check.py

# 2. Test de importación
python -c "from analyze_match import TacticEYE2; print('OK')"

# 3. Demo rápido
python quick_demo.py
```

Si los 3 tests pasan: **¡Estás listo! 🎉**

---

## 🆘 Soporte

Si encuentras problemas:

1. 📖 Revisa [EXAMPLES.md](EXAMPLES.md) para más ejemplos
2. 🔍 Busca en [Issues](https://github.com/Pablodlx/TacticEYE2/issues)
3. 💬 Abre un nuevo Issue con:
   - Sistema operativo
   - Versión de Python
   - Mensaje de error completo
   - Output de `python setup_check.py`

---

## 🎓 Siguientes Pasos

Después de la instalación:

1. 📖 Lee el [README.md](README.md) completo
2. 🎯 Prueba el [quick_demo.py](quick_demo.py)
3. 📚 Explora [EXAMPLES.md](EXAMPLES.md)
4. ⚙️ Personaliza [config.yaml](config.yaml)
5. 🚀 ¡Analiza tus propios partidos!

---

**¡Disfruta analizando fútbol con IA! ⚽🤖**
