# --- Stage 1: build dependencies (needed because obspy may compile parts) ---
FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System packages required for scientific Python + ObsPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching (create one if missing)
# If you already have a requirements.txt in repo it will be used; otherwise we create a fallback.
WORKDIR /tmp/build
COPY requirements.txt ./requirements.txt
# Fallback (if empty or missing) – append core deps
RUN if [ ! -s requirements.txt ]; then \
      echo "numpy" >> requirements.txt && \
      echo "scipy" >> requirements.txt && \
      echo "pandas" >> requirements.txt && \
      echo "matplotlib" >> requirements.txt && \
      echo "obspy" >> requirements.txt && \
      echo "kagglehub" >> requirements.txt ; \
    fi

RUN pip wheel --wheel-dir /tmp/wheels -r requirements.txt

# --- Stage 2: runtime image ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    # Set a writable cache location for kagglehub inside container
    KAGGLEHUB_CACHE=/root/.cache/kagglehub

# Minimal runtime libs (OpenBLAS etc.) for numpy/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    liblapack-dev \
    libxml2 \
    libxslt1.1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=build /tmp/wheels /tmp/wheels
COPY --from=build /tmp/build/requirements.txt /requirements.txt
RUN pip install --no-index --find-links /tmp/wheels -r /requirements.txt

# Create app directory
WORKDIR /app

# Copy project source
COPY . /app

# (Optional) non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Default arguments (can be overridden)
# APP_TARGET can be mars.py, moon.py, or planet_unified.py
ENV APP_TARGET=mars.py

# Example: additional default STA/LTA params (override at runtime)
ENV STA_SEC=120 \
    LTA_SEC=600 \
    THR_ON=4.0 \
    THR_OFF=1.5

# Entry script allowing parameter override
ENTRYPOINT ["bash", "-c", "python $APP_TARGET \"$@\"", "--"]
CMD ["--list", "--limit", "5"]