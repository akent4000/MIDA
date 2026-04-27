from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.core.dependencies import SessionDep
from backend.app.models.inference_result import TaskStatusResponse
from backend.app.services.inference_service import InferenceService

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str, db: SessionDep) -> TaskStatusResponse:
    svc = InferenceService(db)
    ir = svc.get_by_task_id(task_id)
    if ir is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskStatusResponse(
        task_id=task_id,
        status=ir.status,
        inference_result_id=ir.id,
    )
