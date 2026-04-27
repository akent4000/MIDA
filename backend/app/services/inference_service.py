from __future__ import annotations

import uuid

from sqlmodel import Session, select

from backend.app.models.inference_result import InferenceResult


class InferenceService:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, study_id: uuid.UUID, tool_id: str, task_id: str) -> InferenceResult:
        result = InferenceResult(
            study_id=study_id,
            tool_id=tool_id,
            task_id=task_id,
            status="pending",
        )
        self._db.add(result)
        self._db.commit()
        self._db.refresh(result)
        return result

    def get(self, inference_id: uuid.UUID) -> InferenceResult | None:
        return self._db.get(InferenceResult, inference_id)

    def get_by_task_id(self, task_id: str) -> InferenceResult | None:
        return self._db.exec(
            select(InferenceResult).where(InferenceResult.task_id == task_id)
        ).first()
