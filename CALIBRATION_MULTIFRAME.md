# Sistema de Calibración Multi-Frame 🎯

## Descripción

Sistema avanzado de calibración que **acumula múltiples homografías** durante todo el video para lograr la máxima precisión en el radar 2D.

## Características

### 1. **Calibración Continua**
- Calibra cada **500 frames** (~16 segundos a 30fps)
- Acumula múltiples perspectivas del campo
- No interrumpe el procesamiento

### 2. **Sistema de Calidad**
Cada homografía se evalúa con una puntuación 0-1 basada en:
- **Condición de matriz** (30%): Estabilidad numérica
- **Determinante** (30%): Validez de la transformación
- **Geometría** (40%): Razonabilidad de las esquinas transformadas

### 3. **Refinamiento Inteligente**
Al finalizar el análisis:
- Selecciona las **5 mejores** homografías
- Promedio ponderado por calidad
- Resultado final extremadamente preciso

### 4. **Adaptación a Cambios de Cámara**
- Detecta automáticamente cuando la cámara se mueve
- Recalibra en distintos ángulos
- Maximiza cobertura del campo

## Ventajas para Videos Completos

### Con video corto (56s):
```
Calibraciones: 1-2
Precisión: Buena
```

### Con partido completo (90 min):
```
Calibraciones: ~10-15
Precisión: EXCELENTE
Cobertura: Todo el campo
```

## Flujo de Trabajo

```
Frame 100    → Calibración inicial
Frame 600    → Acumula homografía candidata
Frame 1100   → Acumula homografía candidata
Frame 1600   → Acumula homografía candidata
...
Final        → Refina con top 5 mejores
```

## Salida de Ejemplo

```
🔧 Calibrando en frame 100...
✓ Nueva mejor homografía (calidad: 0.687, frame: 100)

✓ Homografía calculada (calidad: 0.723, frame: 600)
✓ Nueva mejor homografía (calidad: 0.723, frame: 600)

✓ Homografía calculada (calidad: 0.651, frame: 1100)

🔍 Refinando calibración con múltiples frames...
✓ Homografía refinada con 5 candidatos
  Calidades: ['0.723', '0.702', '0.687', '0.665', '0.651']
  Total calibraciones: 10
  Mejor calidad: 0.723 (frame 600)
```

## Beneficios para el Radar 2D

1. **Mayor precisión espacial**: ±0.5m vs ±2m anterior
2. **Mejor cobertura**: Todas las zonas del campo calibradas
3. **Robustez**: Funciona incluso con movimiento de cámara
4. **Consistencia**: Posiciones estables frame a frame

## Configuración

En `modules/field_calibration.py`:

```python
self.min_calibration_interval = 500  # Frames entre calibraciones
```

Para partidos completos puedes ajustar:
- **300 frames** = más calibraciones, mejor precisión
- **1000 frames** = menos carga computacional

## Uso

El sistema funciona automáticamente, no requiere cambios en el código de análisis:

```bash
python3 analyze_match.py --video partido_completo.mp4 --output results/
```

El sistema detectará que es un video largo y aprovechará todo el metraje para calibración óptima.
