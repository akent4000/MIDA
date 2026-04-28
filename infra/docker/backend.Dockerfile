# syntax=docker/dockerfile:1.7
# Backend image for MIDA. Two targets, single source:
#   prod — minimal runtime: numpy + ONNX + FastAPI + Celery (NO torch)
#   dev  — adds CPU torch wheels for local full-stack testing
#
# Build:
#   docker build -f infra/docker/backend.Dockerfile --target prod -t mida-backend:prod .
#   docker build -f infra/docker/backend.Dockerfile --target dev  -t mida-backend:dev  .
#
# The prod image must NOT contain torch — the inference module's PEP 562 lazy
# __getattr__ guard keeps it that way at runtime, and we verify at build time.

FROM python:3.11-slim AS prod

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    INFERENCE_BACKEND=onnx

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1001 mida

WORKDIR /app

# ── Layer 1: dependencies (cached until pyproject.toml changes) ──────────────
COPY pyproject.toml ./
# Stub lets pip resolve and install all deps without the real source.
# The editable link already points to /app, so the COPY below just fills it in.
RUN mkdir -p backend && touch backend/__init__.py
# --mount=type=cache keeps downloaded wheels in the BuildKit layer cache
# (not included in the final image) so rebuilds after dep changes are fast.
RUN --mount=type=cache,target=/root/.cache/pip pip install -e .

# ── Layer 2: application source (invalidated on every commit, but cheap) ─────
COPY alembic.ini ./
COPY backend ./backend

# Smoke check: importing the inference package must not pull torch.
# Catches accidental top-level `import torch` in code paths that prod runs.
RUN python -c "import backend.app.modules.inference; import sys; \
    assert 'torch' not in sys.modules, 'torch leaked into the prod image'; \
    print('prod inference import OK, no torch')"

# Named volume weights_cache is mounted here; mkdir pre-sets ownership so
# Docker copies mida:mida into the volume on first creation.
RUN mkdir -p /tmp/mida-weights && chown mida:mida /tmp/mida-weights

USER mida
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ---------------------------------------------------------------------------
# dev — adds CPU torch wheels on top of prod
# ---------------------------------------------------------------------------
FROM prod AS dev

USER root
ENV INFERENCE_BACKEND=pytorch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.6,<2.8" "torchvision>=0.21,<0.23" \
    && pip install -e ".[dev,train]"

USER mida
