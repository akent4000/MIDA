"""MIRA FastAPI application.

Research prototype — not a medical device.

Start development server:
    uvicorn backend.app.main:app --reload

Start Celery worker (in a second terminal):
    celery -A backend.app.workers.celery_app worker --loglevel=info
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import get_settings
from backend.app.modules.ml_tools.registry import build_registry
from backend.app.modules.storage.service import StorageService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()

    # Storage
    try:
        app.state.storage = StorageService(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            bucket=settings.MINIO_BUCKET,
            secure=settings.MINIO_SECURE,
        )
    except Exception:
        # MinIO not available — tests override app.state.storage directly
        app.state.storage = None

    # Registry
    registry = build_registry()
    if settings.PNEUMONIA_WEIGHTS_PATH:
        weights = Path(settings.PNEUMONIA_WEIGHTS_PATH)
        if weights.exists():
            registry.load("pneumonia_classifier_v1", weights)
    app.state.registry = registry

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="MIRA — Medical Imaging Recognition Assistant",
        description=(
            "Headless, API-first medical imaging analysis. "
            "**Research prototype — not a medical device.**"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from backend.app.api.v1 import inference, models, studies, tasks
    from backend.app.api.ws import tasks as ws_tasks

    app.include_router(studies.router)
    app.include_router(inference.router)
    app.include_router(models.router)
    app.include_router(tasks.router)
    app.include_router(ws_tasks.router)

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
