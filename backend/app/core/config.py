from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql+psycopg2://mida:mida@localhost:5432/mida"

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "mida-studies"
    MINIO_SECURE: bool = False

    # ML
    INFERENCE_BACKEND: str = "pytorch"
    PNEUMONIA_WEIGHTS_PATH: str = ""

    ENVIRONMENT: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
