# 🤝 Guía de Contribución - TacticEYE2

¡Gracias por tu interés en contribuir a TacticEYE2! Este documento te guiará para hacer contribuciones efectivas.

---

## 📋 Formas de Contribuir

### 🐛 Reportar Bugs
- Usa la plantilla de Issues en GitHub
- Incluye pasos para reproducir el error
- Especifica tu sistema operativo y versión de Python
- Adjunta logs o screenshots si es posible

### ✨ Proponer Features
- Describe el problema que resuelve
- Explica la solución propuesta
- Considera el impacto en la arquitectura existente
- Discute en Issues antes de implementar

### 📚 Mejorar Documentación
- Corrige typos o errores
- Añade ejemplos prácticos
- Traduce a otros idiomas
- Mejora explicaciones técnicas

### 🔧 Contribuir Código
- Fork el repositorio
- Crea un branch para tu feature
- Mantén el estilo de código existente
- Añade tests si es posible
- Actualiza documentación relacionada

---

## 🏗️ Configuración del Entorno de Desarrollo

### 1. Fork y Clone
```bash
# Fork en GitHub, luego:
git clone https://github.com/TU_USUARIO/TacticEYE2.git
cd TacticEYE2
```

### 2. Instalar en Modo Desarrollo
```bash
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
pip install -r requirements.txt
pip install -e .  # Instalación editable
```

### 3. Crear Branch
```bash
git checkout -b feature/mi-nueva-feature
# o
git checkout -b fix/mi-bug-fix
```

---

## 📝 Estándares de Código

### Estilo Python
- Sigue PEP 8
- Usa docstrings para todas las clases y funciones
- Nombres descriptivos de variables
- Máximo 100 caracteres por línea

### Estructura de Docstrings
```python
def funcion_ejemplo(parametro1: int, parametro2: str) -> bool:
    """
    Breve descripción de la función
    
    Args:
        parametro1: Descripción del parámetro 1
        parametro2: Descripción del parámetro 2
        
    Returns:
        Descripción del valor de retorno
        
    Raises:
        ValueError: Cuando ocurre X
    """
    pass
```

### Imports
```python
# Standard library
import os
import sys

# Third-party
import numpy as np
import cv2
import torch

# Local
from modules.reid_tracker import ReIDTracker
```

---

## 🧪 Testing

### Ejecutar Tests
```bash
# Verificación completa
python setup_check.py

# Test de módulo específico
python -c "from modules.reid_tracker import ReIDTracker; t = ReIDTracker(); print('OK')"
```

### Añadir Tests
- Crea archivos test_*.py en directorio tests/
- Usa pytest o unittest
- Cubre casos normales y edge cases

---

## 📦 Commits

### Formato de Mensaje
```
tipo(scope): breve descripción

Descripción más detallada si es necesaria.

Fixes #123
```

### Tipos de Commit
- `feat`: Nueva feature
- `fix`: Bug fix
- `docs`: Cambios en documentación
- `style`: Formato, sin cambios de código
- `refactor`: Refactorización de código
- `perf`: Mejora de rendimiento
- `test`: Añadir o modificar tests
- `chore`: Tareas de mantenimiento

### Ejemplos
```bash
git commit -m "feat(tracking): añadir support para multi-cámara"
git commit -m "fix(calibration): corregir detección de líneas en campos oscuros"
git commit -m "docs(readme): añadir sección de troubleshooting"
```

---

## 🔄 Pull Requests

### Antes de Crear PR
- [ ] Código sigue los estándares del proyecto
- [ ] Tests pasan correctamente
- [ ] Documentación actualizada
- [ ] Commits son claros y descriptivos
- [ ] Branch está actualizado con main

### Crear Pull Request
1. Push tu branch a tu fork
2. Abre PR en GitHub
3. Usa la plantilla de PR
4. Describe cambios claramente
5. Enlaza Issues relacionados
6. Espera review

### Durante el Review
- Responde a comentarios
- Realiza cambios solicitados
- Mantén comunicación activa
- Sé receptivo al feedback

---

## 🎯 Áreas Prioritarias

### Alto Impacto
- 🔥 Optimización de rendimiento
- 🐛 Fixes de bugs críticos
- 📚 Documentación de funciones complejas
- ✨ Features del roadmap

### Medio Impacto
- 🔧 Refactorización de código
- 📊 Mejoras de visualización
- 🧪 Tests adicionales
- 🌐 Traducciones

### Bajo Impacto (pero bienvenidos!)
- 📝 Typos en documentación
- 🎨 Mejoras de estilo
- 💬 Comentarios adicionales
- 📖 Ejemplos adicionales

---

## 🚀 Roadmap de Features

### v2.1 (Próximo)
- [ ] Dashboard web con Streamlit
- [ ] Integración con torchreid oficial
- [ ] Tests unitarios completos
- [ ] CI/CD con GitHub Actions

### v2.2 (Futuro)
- [ ] Detección automática de eventos
- [ ] Análisis de formaciones tácticas
- [ ] Soporte multi-cámara
- [ ] API REST

### v3.0 (Largo plazo)
- [ ] Pose estimation para acciones
- [ ] Sistema de recomendación táctica
- [ ] Base de datos de partidos
- [ ] Realidad aumentada

---

## 📞 Comunicación

### Canales
- **Issues**: Para bugs y features
- **Discussions**: Para preguntas generales
- **Pull Requests**: Para contribuciones de código
- **Email**: Para contacto directo (ver README)

### Etiqueta
- Sé respetuoso y constructivo
- Proporciona contexto suficiente
- Sé paciente con las respuestas
- Agradece el tiempo de los revisores

---

## 🎓 Recursos para Nuevos Contribuidores

### Documentación Esencial
1. README.md - Vista general del proyecto
2. INSTALL.md - Configuración del entorno
3. EXAMPLES.md - Ejemplos de uso
4. PROJECT_SUMMARY.md - Arquitectura técnica

### Issues para Empezar
Busca etiquetas:
- `good first issue` - Ideal para principiantes
- `help wanted` - Necesitamos ayuda
- `documentation` - Mejoras de docs

### Aprende el Código
1. Empieza con módulos simples (team_classifier.py)
2. Lee los docstrings y comentarios
3. Ejecuta quick_demo.py para entender el flujo
4. Prueba modificaciones pequeñas primero

---

## 🏆 Reconocimientos

Los contribuidores serán:
- Listados en README.md
- Mencionados en release notes
- Parte de la comunidad TacticEYE2

---

## ⚖️ Licencia

Al contribuir, aceptas que tus contribuciones serán licenciadas bajo MIT License (igual que el proyecto).

---

## 🙏 Agradecimientos

Gracias por ayudar a hacer de TacticEYE2 el mejor sistema de análisis táctico open-source del mundo!

---

**¿Preguntas?** Abre un Issue o contacta a [@Pablodlx](https://github.com/Pablodlx)

**¡Happy coding! 🚀⚽**
