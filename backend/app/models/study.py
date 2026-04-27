import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from backend.app.models.inference_result import InferenceResult


def _utcnow() -> datetime:
    return datetime.now(UTC)


class StudyBase(SQLModel):
    file_format: str
    file_size: int
    anonymized: bool = False


class Study(StudyBase, table=True):
    __tablename__ = "studies"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    file_key: str
    metadata_json: str = Field(default="{}")
    created_at: datetime = Field(default_factory=_utcnow)

    results: list["InferenceResult"] = Relationship(back_populates="study")


class StudyPublic(BaseModel):
    id: uuid.UUID
    file_format: str
    file_size: int
    anonymized: bool
    file_key: str
    created_at: datetime
    dicom_metadata: dict[str, Any]

    @classmethod
    def from_db(cls, study: Study) -> "StudyPublic":
        return cls(
            id=study.id,
            file_format=study.file_format,
            file_size=study.file_size,
            anonymized=study.anonymized,
            file_key=study.file_key,
            created_at=study.created_at,
            dicom_metadata=json.loads(study.metadata_json),
        )
