# ──────────────────────────────────────────────────────────────
# Dockerfile — Hugging Face Space (SDK = docker)
# ──────────────────────────────────────────────────────────────
# • Copia todos os ficheiros (webapp.py, app.py, CSV, templates…)
# • Instala libs de sistema GDAL / GEOS / PROJ
# • Instala dependências Python do requirements.txt
# • Arranca FastAPI (Flask montado) em :7860
# ----------------------------------------------------------------

FROM python:3.11-slim

# Instalar dependências de sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev \
        libproj-dev proj-data proj-bin \
        libgeos-dev \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copiar arquivos da app
COPY . /app
WORKDIR /app

# Definir comando de execução
CMD ["uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "7860"]