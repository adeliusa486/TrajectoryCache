# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

LABEL maintainer="trajectorycache-authors"
LABEL description="Spatial-urgency-aware vehicular edge cache"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Source ────────────────────────────────────────────────────────────────────
FROM deps AS app
COPY src/ ./src/
COPY configs/ ./configs/
COPY setup.py pyproject.toml ./
RUN pip install -e . --no-deps

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM app AS runtime
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "trajectorycache.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
