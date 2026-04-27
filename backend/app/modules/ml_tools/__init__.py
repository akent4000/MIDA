"""ML Tools — pluggable clinical analysis tools.

Public API:
    from backend.app.modules.ml_tools import (
        MLTool, ModelInfo, TaskType, Modality,
        ClassificationResult, SegmentationResult, DetectionResult, ToolResult,
        ToolRegistry, ToolNotFoundError, ToolNotLoadedError,
        build_registry,
    )
"""

from .base import (
    ClassificationResult,
    DetectionResult,
    MLTool,
    Modality,
    ModelInfo,
    SegmentationResult,
    TaskType,
    ToolResult,
)
from .registry import ToolNotFoundError, ToolNotLoadedError, ToolRegistry, build_registry

__all__ = [
    "ClassificationResult",
    "DetectionResult",
    "MLTool",
    "Modality",
    "ModelInfo",
    "SegmentationResult",
    "TaskType",
    "ToolNotFoundError",
    "ToolNotLoadedError",
    "ToolRegistry",
    "ToolResult",
    "build_registry",
]
