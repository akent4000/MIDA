"""Tests for ModelStoreService and the weight_loader helpers.

All tests use a local tmp_path — no real MinIO connection required.
MinIO client methods are patched via unittest.mock.
"""

from __future__ import annotations

import datetime
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from minio.error import S3Error

from backend.app.modules.model_store.service import (
    ModelStoreService,
    WeightNotFoundError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path, bucket: str = "mida-models") -> ModelStoreService:
    store = ModelStoreService(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        bucket=bucket,
        cache_dir=tmp_path / "cache",
    )
    return store


def _fake_fget(src_bytes: bytes) -> object:
    """Return a side_effect for fget_object that writes *src_bytes* to the dest path."""

    def _side_effect(_bucket: str, _key: str, dest: str) -> None:
        Path(dest).write_bytes(src_bytes)

    return _side_effect


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------


class TestResolve:
    def test_downloads_on_cache_miss(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        with patch.object(store._client, "fget_object", side_effect=_fake_fget(b"weights")):
            path = store.resolve("onnx/single-int8.onnx")

        assert path.exists()
        assert path.read_bytes() == b"weights"

    def test_cache_hit_skips_download(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        cached = tmp_path / "cache" / "onnx" / "single-int8.onnx"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"cached")

        # stat_object returns a timestamp older than the local file → not stale
        stat_mock = MagicMock()
        stat_mock.last_modified = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
        with (
            patch.object(store._client, "stat_object", return_value=stat_mock),
            patch.object(store._client, "fget_object") as mock_fget,
        ):
            path = store.resolve("onnx/single-int8.onnx")
            mock_fget.assert_not_called()

        assert path.read_bytes() == b"cached"

    def test_stale_cache_triggers_redownload(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        cached = tmp_path / "cache" / "onnx" / "single-int8.onnx"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"old")

        # stat_object returns a timestamp newer than the local file → stale
        stat_mock = MagicMock()
        stat_mock.last_modified = datetime.datetime(2099, 1, 1, tzinfo=datetime.timezone.utc)
        with (
            patch.object(store._client, "stat_object", return_value=stat_mock),
            patch.object(store._client, "fget_object", side_effect=_fake_fget(b"new")),
        ):
            path = store.resolve("onnx/single-int8.onnx")

        assert path.read_bytes() == b"new"

    def test_stat_error_treats_cache_as_fresh(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        cached = tmp_path / "cache" / "onnx" / "single-int8.onnx"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"cached")

        stat_err = S3Error("ServiceUnavailable", "", "", "", "", "")
        with (
            patch.object(store._client, "stat_object", side_effect=stat_err),
            patch.object(store._client, "fget_object") as mock_fget,
        ):
            path = store.resolve("onnx/single-int8.onnx")
            mock_fget.assert_not_called()

        assert path.read_bytes() == b"cached"

    def test_local_path_mirrors_key_structure(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        with patch.object(store._client, "fget_object", side_effect=_fake_fget(b"x")):
            path = store.resolve("onnx/fold1-int8.onnx")

        assert path == tmp_path / "cache" / "onnx" / "fold1-int8.onnx"

    def test_raises_weight_not_found_for_missing_key(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        err = S3Error(
            code="NoSuchKey",
            message="not found",
            resource="/bucket/key",
            request_id="r",
            host_id="h",
            response=MagicMock(),
        )
        with patch.object(store._client, "fget_object", side_effect=err), pytest.raises(WeightNotFoundError, match="not found"):
            store.resolve("onnx/missing.onnx")

    def test_partial_file_cleaned_up_on_error(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        tmp_file = tmp_path / "cache" / "onnx" / "bad.onnx.tmp"

        def _fail(_bucket: str, _key: str, dest: str) -> None:
            Path(dest).write_bytes(b"partial")
            raise RuntimeError("network error")

        with patch.object(store._client, "fget_object", side_effect=_fail), pytest.raises(RuntimeError, match="network error"):
            store.resolve("onnx/bad.onnx")

        assert not tmp_file.exists()

    def test_concurrent_resolve_downloads_once(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        call_count = 0

        def _counted(_bucket: str, _key: str, dest: str) -> None:
            nonlocal call_count
            call_count += 1
            Path(dest).write_bytes(b"ok")

        errors: list[Exception] = []

        def _worker() -> None:
            try:
                with patch.object(store._client, "fget_object", side_effect=_counted):
                    store.resolve("onnx/single-int8.onnx")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # At most 1 real download — subsequent threads hit the cached file
        assert call_count <= 1


# ---------------------------------------------------------------------------
# weight_loader helpers
# ---------------------------------------------------------------------------


class TestWeightLoader:
    def test_resolve_weights_uses_minio_key_when_set(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        from backend.app.core.weight_loader import resolve_weights

        settings = MagicMock()
        settings.PNEUMONIA_WEIGHTS_MINIO_KEY = "onnx/single-int8.onnx"
        settings.PNEUMONIA_ENSEMBLE_MINIO_KEYS = ""

        model_store = MagicMock()
        model_store.resolve.return_value = tmp_path / "single-int8.onnx"

        result = resolve_weights(settings, model_store)

        model_store.resolve.assert_called_once_with("onnx/single-int8.onnx")
        assert result == tmp_path / "single-int8.onnx"

    def test_resolve_weights_falls_back_to_local_path(self, tmp_path: Path) -> None:
        from backend.app.core.weight_loader import resolve_weights

        local = tmp_path / "best.pt"
        local.write_bytes(b"weights")

        settings = MagicMock()
        settings.PNEUMONIA_WEIGHTS_MINIO_KEY = ""
        settings.PNEUMONIA_WEIGHTS_PATH = str(local)
        settings.PNEUMONIA_ENSEMBLE_MINIO_KEYS = ""

        model_store = MagicMock()
        result = resolve_weights(settings, model_store)

        model_store.resolve.assert_not_called()
        assert result == local

    def test_resolve_weights_returns_none_when_nothing_configured(
        self, tmp_path: Path
    ) -> None:
        from backend.app.core.weight_loader import resolve_weights

        settings = MagicMock()
        settings.PNEUMONIA_WEIGHTS_MINIO_KEY = ""
        settings.PNEUMONIA_WEIGHTS_PATH = ""
        settings.PNEUMONIA_ENSEMBLE_MINIO_KEYS = ""

        result = resolve_weights(settings, MagicMock())
        assert result is None

    def test_ensemble_keys_set_env_var(self, tmp_path: Path) -> None:
        from backend.app.core.weight_loader import resolve_weights

        settings = MagicMock()
        settings.PNEUMONIA_WEIGHTS_MINIO_KEY = "onnx/single-int8.onnx"
        settings.PNEUMONIA_ENSEMBLE_MINIO_KEYS = "onnx/fold1-int8.onnx,onnx/fold2-int8.onnx"

        fold1 = tmp_path / "fold1-int8.onnx"
        fold2 = tmp_path / "fold2-int8.onnx"

        model_store = MagicMock()
        model_store.resolve.side_effect = [
            tmp_path / "single-int8.onnx",
            fold1,
            fold2,
        ]

        resolve_weights(settings, model_store)

        assert os.environ.get("PNEUMONIA_ENSEMBLE_PATHS") == f"{fold1},{fold2}"

    def test_download_error_on_single_returns_none(self, tmp_path: Path) -> None:
        from backend.app.core.weight_loader import resolve_weights

        settings = MagicMock()
        settings.PNEUMONIA_WEIGHTS_MINIO_KEY = "onnx/missing.onnx"
        settings.PNEUMONIA_ENSEMBLE_MINIO_KEYS = ""

        model_store = MagicMock()
        model_store.resolve.side_effect = WeightNotFoundError("not found")

        result = resolve_weights(settings, model_store)
        assert result is None
