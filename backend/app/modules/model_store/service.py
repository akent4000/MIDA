"""Lazy weight resolver — downloads model artefacts from MinIO on first use.

The local cache survives container restarts if the cache directory is on a
named Docker volume.  On each resolve() call the S3 LastModified timestamp
is compared against the local file mtime; if the remote object is newer the
file is re-downloaded automatically, so uploading a new model to MinIO and
restarting the worker is enough to deploy it — no manual cache clearing.

Download is atomic: the bytes are written to a .tmp sibling first, then
renamed, so a killed process never leaves a partial file that looks valid.

Thread safety: a per-key threading.Lock prevents duplicate concurrent
downloads.  On prod (Celery --pool=solo) there is only one thread, but the
locking is cheap and keeps the service correct if that ever changes.
"""

from __future__ import annotations

import datetime
import logging
import threading
from pathlib import Path

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


class WeightNotFoundError(FileNotFoundError):
    """Raised when the requested key does not exist in the models bucket."""


class ModelStoreService:
    """Resolves MinIO object keys to locally cached file paths."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        cache_dir: Path,
        secure: bool = False,
    ) -> None:
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._bucket = bucket
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_mu = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, key: str) -> Path:
        """Return the local path for *key*, downloading from MinIO if needed.

        If a cached copy already exists its modification time is compared
        against the S3 LastModified timestamp.  When the remote object is
        newer (i.e. a model was re-uploaded to MinIO) the file is
        re-downloaded automatically, so uploading a new model + restarting
        the worker is sufficient to deploy it without manual cache clearing.

        Raises WeightNotFoundError if the key does not exist in the bucket.
        """
        local = self._local_path(key)

        lock = self._key_lock(key)
        with lock:
            if local.exists() and not self._is_stale(key, local):
                logger.debug("model_store cache hit: %s", key)
                return local
            self._download(key, local)

        return local

    def _is_stale(self, key: str, local: Path) -> bool:
        """Return True if the S3 object is newer than the local cached file."""
        try:
            stat = self._client.stat_object(self._bucket, key)
            local_mtime = datetime.datetime.fromtimestamp(
                local.stat().st_mtime, tz=datetime.UTC
            )
            stale = stat.last_modified > local_mtime
            if stale:
                logger.info(
                    "model_store: %s is stale (local %s < s3 %s), will re-download",
                    key,
                    local_mtime.isoformat(),
                    stat.last_modified.isoformat(),
                )
            return stale
        except S3Error:
            return False

    def exists_remote(self, key: str) -> bool:
        """Return True if *key* exists in the MinIO bucket."""
        try:
            self._client.stat_object(self._bucket, key)
            return True
        except S3Error:
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _local_path(self, key: str) -> Path:
        # Preserve directory structure from the key so fold1-int8.onnx and
        # fold2-int8.onnx (same filename, different prefix) don't collide.
        return self._cache_dir / key

    def _key_lock(self, key: str) -> threading.Lock:
        with self._locks_mu:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            logger.info("model_store: downloading %s/%s → %s", self._bucket, key, dest)
            self._client.fget_object(self._bucket, key, str(tmp))
            tmp.rename(dest)
            logger.info(
                "model_store: saved %s (%.1f MiB)", dest.name, dest.stat().st_size / 1024**2
            )
        except S3Error as exc:
            tmp.unlink(missing_ok=True)
            if exc.code == "NoSuchKey":
                raise WeightNotFoundError(
                    f"Model weight not found in MinIO bucket {self._bucket!r}: {key!r}"
                ) from exc
            raise
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
