from __future__ import annotations

import dataclasses
import io
import json
import uuid

import numpy as np
from PIL import Image
from sqlmodel import Session, select

from backend.app.models.study import Study
from backend.app.modules.dicom.service import DicomService
from backend.app.modules.storage.service import StorageService


class StudyService:
    def __init__(self, db: Session, storage: StorageService) -> None:
        self._db = db
        self._storage = storage
        self._dicom = DicomService()

    def upload(self, raw: bytes, filename: str) -> Study:
        loaded = self._dicom.load(raw)
        anon = self._dicom.anonymize(loaded)

        study_id = uuid.uuid4()
        ext = _detect_ext(filename, anon.file_format)
        key = f"studies/{study_id}/original{ext}"

        self._storage.upload(key, raw, content_type="application/octet-stream")

        meta_dict = {
            k: v
            for k, v in dataclasses.asdict(anon.metadata).items()
            if v is not None and k != "extra"
        }

        record = Study(
            id=study_id,
            file_key=key,
            file_format=anon.file_format,
            file_size=len(raw),
            metadata_json=json.dumps(meta_dict),
            anonymized=True,
        )
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)
        return record

    def get(self, study_id: uuid.UUID) -> Study | None:
        return self._db.get(Study, study_id)

    def list(self, limit: int = 100, offset: int = 0) -> list[Study]:
        stmt = (
            select(Study).order_by(Study.created_at.desc()).limit(limit).offset(offset)  # type: ignore[attr-defined]
        )
        return list(self._db.exec(stmt).all())

    def get_preview_png(self, study: Study) -> bytes:
        raw = self._storage.download(study.file_key)
        loaded = self._dicom.load(raw)
        arr = (loaded.pixel_data * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def _detect_ext(filename: str, file_format: str) -> str:
    if "." in filename:
        suffix = filename.rsplit(".", 1)[-1].lower()
        if suffix in ("dcm", "png", "jpg", "jpeg"):
            return f".{suffix}"
        if suffix == "gz" and filename.endswith(".nii.gz"):
            return ".nii.gz"
    return {"dicom": ".dcm", "nifti": ".nii.gz", "png": ".png", "jpeg": ".jpg"}.get(
        file_format, ".bin"
    )
