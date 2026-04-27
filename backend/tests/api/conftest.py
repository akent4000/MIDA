"""Shared fixtures for API tests.

Uses SQLite in-memory DB + in-memory storage (no MinIO/Celery required).
Celery tasks are mocked with patch so no broker is needed.
"""

from __future__ import annotations

import io
import uuid
from typing import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from backend.app.core.database import get_session
from backend.app.core.dependencies import get_registry, get_storage
from backend.app.main import app
from backend.app.modules.ml_tools.registry import build_registry

# ---------------------------------------------------------------------------
# In-memory SQLite
# ---------------------------------------------------------------------------

_TEST_DB_URL = "sqlite://"

_engine = create_engine(
    _TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


def _override_session() -> Generator[Session, None, None]:
    with Session(_engine) as session:
        yield session


# ---------------------------------------------------------------------------
# In-memory storage
# ---------------------------------------------------------------------------


class _MemoryStorage:
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def upload(self, key: str, data: bytes, content_type: str = "") -> str:
        self._store[key] = data
        return key

    def download(self, key: str) -> bytes:
        return self._store[key]

    def exists(self, key: str) -> bool:
        return key in self._store

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def presigned_url(self, key: str, expires_seconds: int = 3600) -> str:
        return f"http://localhost/minio/{key}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    SQLModel.metadata.create_all(_engine)
    _storage = _MemoryStorage()
    _registry = build_registry()

    # Override dependencies so lifespan MinIO setup doesn't interfere
    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_storage] = lambda: _storage
    app.dependency_overrides[get_registry] = lambda: _registry

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    SQLModel.metadata.drop_all(_engine)
    app.dependency_overrides.clear()


@pytest.fixture()
def png_bytes() -> bytes:
    """Minimal 32×32 grayscale PNG."""
    arr = (np.random.rand(32, 32) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
