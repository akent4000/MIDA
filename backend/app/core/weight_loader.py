"""Resolve model weight paths for the registry.

Called at app startup (main.py lifespan) and Celery worker init (tasks.py).
Handles two sources:
  1. PNEUMONIA_WEIGHTS_MINIO_KEY / PNEUMONIA_ENSEMBLE_MINIO_KEYS (prod)
     — downloaded from MinIO on first use, cached locally.
  2. PNEUMONIA_WEIGHTS_PATH (dev, PyTorch backend)
     — local file path, used as-is.

Returns the local path to the single checkpoint and sets
os.environ["PNEUMONIA_ENSEMBLE_PATHS"] so PneumoniaTool picks it up.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.core.config import Settings
    from backend.app.modules.model_store.service import ModelStoreService

logger = logging.getLogger(__name__)


def build_model_store(settings: Settings) -> ModelStoreService:
    from backend.app.modules.model_store.service import ModelStoreService

    return ModelStoreService(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        bucket=settings.MINIO_MODELS_BUCKET,
        cache_dir=Path(settings.WEIGHTS_CACHE_DIR),
        secure=settings.MINIO_SECURE,
    )


def resolve_weights(settings: Settings, model_store: ModelStoreService) -> Path | None:
    """Download / locate single-model weights and set up ensemble paths.

    Side-effect: sets os.environ["PNEUMONIA_ENSEMBLE_PATHS"] when ensemble
    keys are configured so PneumoniaTool reads the resolved local paths.

    Returns the local path to the single checkpoint, or None if no weights
    are configured (inference will be unavailable until weights are added).
    """
    single_path: Path | None = None

    if settings.PNEUMONIA_WEIGHTS_MINIO_KEY:
        try:
            single_path = model_store.resolve(settings.PNEUMONIA_WEIGHTS_MINIO_KEY)
        except Exception as exc:
            logger.error("model_store: failed to resolve single weight: %s", exc)

    elif settings.PNEUMONIA_WEIGHTS_PATH:
        p = Path(settings.PNEUMONIA_WEIGHTS_PATH)
        if p.exists():
            single_path = p
        else:
            logger.warning("PNEUMONIA_WEIGHTS_PATH %s does not exist — skipping", p)

    if settings.PNEUMONIA_ENSEMBLE_MINIO_KEYS:
        keys = [k.strip() for k in settings.PNEUMONIA_ENSEMBLE_MINIO_KEYS.split(",") if k.strip()]
        resolved: list[str] = []
        for key in keys:
            try:
                resolved.append(str(model_store.resolve(key)))
            except Exception as exc:
                logger.error("model_store: failed to resolve ensemble weight %r: %s", key, exc)
        if resolved:
            os.environ["PNEUMONIA_ENSEMBLE_PATHS"] = ",".join(resolved)

    return single_path
