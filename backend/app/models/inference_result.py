import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from backend.app.models.study import Study


def _utcnow() -> datetime:
    return datetime.now(UTC)


class InferenceResultBase(SQLModel):
    tool_id: str
    status: str = "pending"  # pending | running | done | failed


class InferenceResult(InferenceResultBase, table=True):
    __tablename__ = "inference_results"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    study_id: uuid.UUID = Field(foreign_key="studies.id")
    task_id: str
    result_json: str | None = Field(default=None)
    gradcam_key: str | None = Field(default=None)
    error_message: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = Field(default=None)

    study: Optional["Study"] = Relationship(back_populates="results")


class InferenceResultPublic(SQLModel):
    id: uuid.UUID
    study_id: uuid.UUID
    tool_id: str
    task_id: str
    status: str
    result: dict[str, Any] | None = None
    gradcam_key: str | None = None
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None

    @classmethod
    def from_db(cls, ir: "InferenceResult") -> "InferenceResultPublic":
        return cls(
            id=ir.id,
            study_id=ir.study_id,
            tool_id=ir.tool_id,
            task_id=ir.task_id,
            status=ir.status,
            result=json.loads(ir.result_json) if ir.result_json else None,
            gradcam_key=ir.gradcam_key,
            error_message=ir.error_message,
            created_at=ir.created_at,
            completed_at=ir.completed_at,
        )


class TaskStatusResponse(SQLModel):
    task_id: str
    status: str
    inference_result_id: uuid.UUID


class InferenceSubmitResponse(SQLModel):
    inference_id: uuid.UUID
    task_id: str
    status: str
    tool_id: str
