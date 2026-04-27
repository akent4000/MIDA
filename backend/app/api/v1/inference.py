from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from backend.app.core.dependencies import RegistryDep, SessionDep, StorageDep
from backend.app.models.inference_result import (
    InferenceResultPublic,
    InferenceSubmitResponse,
)
from backend.app.services.inference_service import InferenceService
from backend.app.services.study_service import StudyService

router = APIRouter(prefix="/api/v1", tags=["inference"])


class InferenceRequest(BaseModel):
    tool_id: str = "pneumonia_classifier_v1"


@router.post(
    "/studies/{study_id}/inference",
    response_model=InferenceSubmitResponse,
    status_code=202,
)
def submit_inference(
    study_id: uuid.UUID,
    body: InferenceRequest,
    db: SessionDep,
    storage: StorageDep,
    registry: RegistryDep,
) -> InferenceSubmitResponse:
    svc = StudyService(db, storage)  # type: ignore[arg-type]
    study = svc.get(study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found.")

    if not registry.is_registered(body.tool_id):
        raise HTTPException(
            status_code=400, detail=f"Unknown tool_id: {body.tool_id!r}."
        )

    # Pre-assign task ID so DB record and Celery task share the same UUID.
    task_id = str(uuid.uuid4())

    inf_svc = InferenceService(db)
    ir = inf_svc.create(study_id=study_id, tool_id=body.tool_id, task_id=task_id)

    from backend.app.workers.tasks import run_inference

    run_inference.apply_async(
        kwargs={
            "study_id": str(study_id),
            "inference_result_id": str(ir.id),
            "tool_id": body.tool_id,
        },
        task_id=task_id,
    )

    return InferenceSubmitResponse(
        inference_id=ir.id,
        task_id=task_id,
        status=ir.status,
        tool_id=ir.tool_id,
    )


@router.get("/inference/{inference_id}/result", response_model=InferenceResultPublic)
def get_inference_result(
    inference_id: uuid.UUID,
    db: SessionDep,
    storage: StorageDep,
) -> InferenceResultPublic:
    inf_svc = InferenceService(db)
    ir = inf_svc.get(inference_id)
    if ir is None:
        raise HTTPException(status_code=404, detail="Inference result not found.")
    return InferenceResultPublic.from_db(ir)


@router.get("/inference/{inference_id}/explanation")
def get_explanation(
    inference_id: uuid.UUID,
    db: SessionDep,
    storage: StorageDep,
) -> Response:
    inf_svc = InferenceService(db)
    ir = inf_svc.get(inference_id)
    if ir is None:
        raise HTTPException(status_code=404, detail="Inference result not found.")
    if ir.gradcam_key is None:
        raise HTTPException(
            status_code=404,
            detail="No Grad-CAM available for this result.",
        )
    try:
        png = storage.download(ir.gradcam_key)  # type: ignore[union-attr]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load heatmap: {exc}") from exc
    return Response(content=png, media_type="image/png")
