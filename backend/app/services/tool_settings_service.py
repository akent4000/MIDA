from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session, select

from backend.app.models.tool_setting import ToolSetting
from backend.app.modules.ml_tools.registry import ToolRegistry
from backend.app.modules.ml_tools.settings import (
    SettingField,
    schema_to_dict,
    validate_values,
)


class ToolSettingsService:
    """Read/write per-tool runtime settings persisted in the DB.

    The DB is the source of truth.  The Celery worker re-reads on every
    inference task, so changes take effect without process restarts.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    def get_values(self, tool_id: str) -> dict[str, Any]:
        rows = self._db.exec(select(ToolSetting).where(ToolSetting.tool_id == tool_id)).all()
        return {r.key: json.loads(r.value_json) for r in rows}

    def get_with_defaults(self, tool_id: str, schema: list[SettingField]) -> dict[str, Any]:
        stored = self.get_values(tool_id)
        return {f.key: stored.get(f.key, f.default) for f in schema}

    def set_values(self, tool_id: str, values: dict[str, Any]) -> None:
        now = datetime.now(UTC)
        for key, value in values.items():
            existing = self._db.get(ToolSetting, (tool_id, key))
            if existing is None:
                self._db.add(
                    ToolSetting(
                        tool_id=tool_id,
                        key=key,
                        value_json=json.dumps(value),
                        updated_at=now,
                    )
                )
            else:
                existing.value_json = json.dumps(value)
                existing.updated_at = now
                self._db.add(existing)
        self._db.commit()

    def patch(
        self,
        tool_id: str,
        schema: list[SettingField],
        partial: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate `partial` against schema, persist, return full effective values."""
        validated = validate_values(schema, partial)
        # validate_values fills in defaults for missing keys; persist only the
        # subset the caller actually supplied so user intent is preserved.
        to_persist = {k: validated[k] for k in partial}
        self.set_values(tool_id, to_persist)
        return self.get_with_defaults(tool_id, schema)


def apply_settings_from_db(db: Session, registry: ToolRegistry, tool_id: str) -> None:
    """Read stored settings and push them into the live tool instance."""
    if not registry.is_loaded(tool_id):
        return
    tool = registry.get(tool_id)
    schema = tool.get_settings_schema()
    if not schema:
        return
    svc = ToolSettingsService(db)
    values = svc.get_with_defaults(tool_id, schema)
    tool.apply_settings(values)


def serialize_config(schema: list[SettingField], values: dict[str, Any]) -> dict[str, Any]:
    """Build the API response shape: schema + current values."""
    return {
        "schema": schema_to_dict(schema),
        "values": values,
    }
