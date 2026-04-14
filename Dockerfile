# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Dependencias del sistema para OpenCV y libgl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalar deps en /app/deps para copiar al stage final
RUN pip install --no-cache-dir --prefix=/app/deps \
    # excluir weights de deep learning si se desea CPU-only
    -r requirements.txt


# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Mismas libs de sistema (runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copiar deps instaladas
COPY --from=builder /app/deps /usr/local

# Usuario sin privilegios
RUN useradd -m -u 1000 appuser

# Copiar código fuente
COPY --chown=appuser:appuser . .

# Crear directorios de trabajo
RUN mkdir -p uploads outputs outputs_streaming outputs_heatmap static templates \
    && chown -R appuser:appuser /app

USER appuser

# Puerto de la app
EXPOSE 8000

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["python", "app.py"]
