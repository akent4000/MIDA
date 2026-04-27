from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql+psycopg2://mida:mida@localhost:5432/mida"

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"

    # MinIO — studies bucket (DICOM uploads)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "mida-studies"
    MINIO_SECURE: bool = False

    # MinIO — models bucket (weight artefacts)
    MINIO_MODELS_BUCKET: str = "mida-models"

    # Local directory where downloaded weights are cached.
    # Use a named Docker volume so weights survive container restarts.
    WEIGHTS_CACHE_DIR: str = "/tmp/mida-weights"

    # ML
    INFERENCE_BACKEND: str = "pytorch"

    # Dev: local .pt path (PyTorch backend). Leave empty to skip auto-load.
    PNEUMONIA_WEIGHTS_PATH: str = ""

    # Prod: MinIO object keys inside MINIO_MODELS_BUCKET.
    # When set, weights are downloaded lazily on first use and cached locally.
    # Example: "onnx/single-int8.onnx"
    PNEUMONIA_WEIGHTS_MINIO_KEY: str = ""
    # Example: "onnx/fold1-int8.onnx,onnx/fold2-int8.onnx,..."
    PNEUMONIA_ENSEMBLE_MINIO_KEYS: str = ""

    ENVIRONMENT: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
