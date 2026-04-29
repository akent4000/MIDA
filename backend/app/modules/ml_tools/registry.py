"""Tool registry — discovers, manages, and dispatches MLTool instances.

The registry is deliberately non-singleton so tests can create isolated instances
and FastAPI can expose it as a Depends() dependency.

Typical usage:
    registry = build_registry()
    registry.load("pneumonia_classifier_v1", Path("weights/best.pt"))
    result = registry.get("pneumonia_classifier_v1").predict(image)

To add a new tool: import its class in build_registry() and call register_class().
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .base import MLTool, ModelInfo, ToolResult

if TYPE_CHECKING:
    pass


class ToolNotFoundError(KeyError):
    """Raised when a tool_id has never been registered."""


class ToolNotLoadedError(RuntimeError):
    """Raised when predict/get is called before load()."""


class ToolRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[MLTool]] = {}
        self._instances: dict[str, MLTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_class(self, tool_id: str, cls: type[MLTool]) -> None:
        """Register a tool class (no weights loaded yet)."""
        self._classes[tool_id] = cls

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, tool_id: str, weights_path: Path) -> None:
        """Instantiate the tool class and load weights from *weights_path*."""
        if tool_id not in self._classes:
            raise ToolNotFoundError(
                f"Unknown tool: {tool_id!r}. " f"Available: {list(self._classes)}"
            )
        instance = self._classes[tool_id]()
        instance.load(weights_path)
        self._instances[tool_id] = instance

    def unload(self, tool_id: str) -> None:
        """Remove a loaded tool (frees GPU/CPU memory on next GC cycle)."""
        self._instances.pop(tool_id, None)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, tool_id: str) -> MLTool:
        """Return a loaded tool instance, or raise ToolNotLoadedError."""
        if tool_id not in self._instances:
            raise ToolNotLoadedError(
                f"Tool {tool_id!r} is registered but not loaded. "
                "Call registry.load(tool_id, weights_path) first."
            )
        return self._instances[tool_id]

    def predict(self, tool_id: str, image: np.ndarray) -> ToolResult:
        """Convenience: get(tool_id).predict(image)."""
        return self.get(tool_id).predict(image)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_available(self) -> list[str]:
        """All registered tool IDs (loaded or not)."""
        return list(self._classes)

    def list_loaded(self) -> list[ModelInfo]:
        """ModelInfo for every currently loaded tool."""
        return [t.info for t in self._instances.values()]

    def is_loaded(self, tool_id: str) -> bool:
        return tool_id in self._instances

    def is_registered(self, tool_id: str) -> bool:
        return tool_id in self._classes


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------


def build_registry() -> ToolRegistry:
    """Return a ToolRegistry with all built-in tools pre-registered.

    Weights are NOT loaded here — call registry.load(tool_id, weights_path)
    before running inference.

    To add a new tool:
        from backend.app.modules.ml_tools.my_tool.tool import MyTool
        registry.register_class(MyTool.TOOL_ID, MyTool)
    """
    from backend.app.modules.ml_tools.chexpert.tool import CheXpertTool
    from backend.app.modules.ml_tools.pneumonia.tool import PneumoniaTool

    registry = ToolRegistry()
    registry.register_class(PneumoniaTool.TOOL_ID, PneumoniaTool)
    registry.register_class(CheXpertTool.TOOL_ID, CheXpertTool)
    return registry
