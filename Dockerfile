# Dockerfile
ARG PYTHON_VERSION=3.13

# ---- Builder Stage ----
# Esta etapa instala poetry y todas las dependencias necesarias.
FROM python:${PYTHON_VERSION}-slim AS builder

# Configura el entorno de Poetry
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Instala poetry
RUN pip install --no-cache-dir poetry

# Copia los archivos de definición del proyecto e instala las dependencias
# Usamos --mount para mejorar el caching de la capa de dependencias de git
# Requiere BuildKit (que ya estás usando)
COPY pyproject.toml poetry.lock* ./
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --no-root --without dev


# ---- Final Stage ----
# Esta etapa crea la imagen final y limpia. No contiene poetry.
FROM python:${PYTHON_VERSION}-slim AS final

WORKDIR /app

# Copia las dependencias instaladas desde la etapa 'builder'
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

# Copia el código fuente de la aplicación
COPY src/ .
COPY config/ config/

# Copia el directorio de tests para los health checks
COPY tests/ tests/

# Corre la aplicación directamente con Python, ya que las dependencias están en el path global
CMD ["python", "-m", "invoice_processor.app.main", "--mode", "slack"]
