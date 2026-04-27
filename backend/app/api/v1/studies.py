from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import Response

from backend.app.core.dependencies import SessionDep, StorageDep
from backend.app.models.study import StudyPublic
from backend.app.services.study_service import StudyService

router = APIRouter(prefix="/api/v1/studies", tags=["studies"])


@router.post("", response_model=StudyPublic, status_code=201)
def upload_study(
    file: UploadFile,
    db: SessionDep,
    storage: StorageDep,
) -> StudyPublic:
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    try:
        svc = StudyService(db, storage)  # type: ignore[arg-type]
        study = svc.upload(raw, file.filename or "upload.bin")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {exc}") from exc
    return StudyPublic.from_db(study)


@router.get("/{study_id}", response_model=StudyPublic)
def get_study(study_id: uuid.UUID, db: SessionDep, storage: StorageDep) -> StudyPublic:
    svc = StudyService(db, storage)  # type: ignore[arg-type]
    study = svc.get(study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found.")
    return StudyPublic.from_db(study)


@router.get("/{study_id}/image")
def get_study_image(study_id: uuid.UUID, db: SessionDep, storage: StorageDep) -> Response:
    svc = StudyService(db, storage)  # type: ignore[arg-type]
    study = svc.get(study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found.")
    try:
        png = svc.get_preview_png(study)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not render image: {exc}") from exc
    return Response(content=png, media_type="image/png")
