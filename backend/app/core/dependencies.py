from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request
from sqlmodel import Session

from backend.app.core.database import get_session

if TYPE_CHECKING:
    from backend.app.modules.ml_tools.registry import ToolRegistry
    from backend.app.modules.storage.service import StorageService


def get_storage(request: Request) -> StorageService:
    return request.app.state.storage  # type: ignore[no-any-return]


def get_registry(request: Request) -> ToolRegistry:
    return request.app.state.registry  # type: ignore[no-any-return]


SessionDep = Annotated[Session, Depends(get_session)]
StorageDep = Annotated["StorageService", Depends(get_storage)]
RegistryDep = Annotated["ToolRegistry", Depends(get_registry)]
