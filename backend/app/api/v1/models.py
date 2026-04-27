from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.core.dependencies import RegistryDep

router = APIRouter(prefix="/api/v1/models", tags=["models"])


class ModelListItem(BaseModel):
    tool_id: str
    name: str
    version: str
    description: str
    modality: str
    task_type: str
    input_shape: list[int]
    class_names: list[str]
    loaded: bool


@router.get("", response_model=list[ModelListItem])
def list_models(registry: RegistryDep) -> list[ModelListItem]:
    items: list[ModelListItem] = []
    for tool_id in registry.list_available():
        loaded = registry.is_loaded(tool_id)
        # Instantiate without weights to read static metadata when not loaded
        info = (
            registry.get(tool_id).info
            if loaded
            else registry._classes[tool_id]().info  # type: ignore[attr-defined]
        )
        items.append(
            ModelListItem(
                tool_id=tool_id,
                name=info.name,
                version=info.version,
                description=info.description,
                modality=info.modality.value,
                task_type=info.task_type.value,
                input_shape=list(info.input_shape),
                class_names=list(info.class_names),
                loaded=loaded,
            )
        )
    return items
