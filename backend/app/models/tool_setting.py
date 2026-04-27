from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ToolSetting(SQLModel, table=True):
    __tablename__ = "tool_settings"

    tool_id: str = Field(primary_key=True)
    key: str = Field(primary_key=True)
    value_json: str = Field(default="null")
    updated_at: datetime = Field(default_factory=_utcnow)
