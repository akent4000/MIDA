from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.dependencies import RegistryDep, SessionDep
from backend.app.modules.ml_tools.registry import ToolNotFoundError
from backend.app.modules.ml_tools.settings import SettingsValidationError
from backend.app.services.tool_settings_service import (
    ToolSettingsService,
    apply_settings_from_db,
    serialize_config,
)

router = APIRouter(prefix="/api/v1/tools", tags=["tools"])


class ConfigResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    tool_id: str
    schema_: list[dict[str, Any]] = Field(alias="schema")
    values: dict[str, Any]


class ConfigPatch(BaseModel):
    values: dict[str, Any]


def _resolve_schema(registry: Any, tool_id: str) -> list[Any]:
    if not registry.is_registered(tool_id):
        raise HTTPException(status_code=404, detail=f"Tool {tool_id!r} not found")
    # Use loaded instance if available, otherwise instantiate a fresh one to
    # introspect schema without weights.
    tool = (
        registry.get(tool_id)
        if registry.is_loaded(tool_id)
        else registry._classes[tool_id]()
    )
    schema: list[Any] = tool.get_settings_schema()
    return schema


@router.get("/{tool_id}/config", response_model=ConfigResponse)
def get_config(
    tool_id: str,
    session: SessionDep,
    registry: RegistryDep,
) -> ConfigResponse:
    try:
        schema = _resolve_schema(registry, tool_id)
    except ToolNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    svc = ToolSettingsService(session)
    values = svc.get_with_defaults(tool_id, schema)
    payload = serialize_config(schema, values)
    return ConfigResponse.model_validate({"tool_id": tool_id, **payload})


@router.patch("/{tool_id}/config", response_model=ConfigResponse)
def patch_config(
    tool_id: str,
    body: ConfigPatch,
    session: SessionDep,
    registry: RegistryDep,
) -> ConfigResponse:
    try:
        schema = _resolve_schema(registry, tool_id)
    except ToolNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    if not schema:
        raise HTTPException(status_code=400, detail=f"Tool {tool_id!r} declares no settings")
    svc = ToolSettingsService(session)
    try:
        values = svc.patch(tool_id, schema, body.values)
    except SettingsValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Apply to the FastAPI-process registry instance immediately.  The Celery
    # worker re-reads on each task, so its registry will pick the change up
    # on the next inference run.
    apply_settings_from_db(session, registry, tool_id)

    payload = serialize_config(schema, values)
    return ConfigResponse.model_validate({"tool_id": tool_id, **payload})
