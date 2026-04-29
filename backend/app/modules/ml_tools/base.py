"""Pluggable ML tool abstraction — the extensibility seam of MIDA.

Each clinical task (pneumonia classification, tumour segmentation, ECG analysis, …)
is a self-contained MLTool that implements this contract.  The ToolRegistry manages
instances; the service/API layer calls them by tool_id without knowing internals.

Adding a new tool:
    1. Subclass MLTool and set TOOL_ID.
    2. Implement info / load / predict / get_preprocessing_config.
    3. Call registry.register_class(MyTool.TOOL_ID, MyTool) in build_registry().
    That is all — no changes to DICOM, preprocessing, or API code required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"


class Modality(StrEnum):
    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    FUNDUS = "fundus"
    DERMOSCOPY = "dermoscopy"
    ECG = "ecg"
    PATHOLOGY = "pathology"


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """Static description of a tool — returned by the /api/v1/models endpoint."""

    tool_id: str
    name: str
    version: str
    description: str
    modality: Modality
    task_type: TaskType
    input_shape: tuple[int, ...]  # expected (C, H, W) after preprocessing
    class_names: list[str]


# ---------------------------------------------------------------------------
# Result hierarchy
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Base result returned by any MLTool.predict()."""

    tool_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationResult(ToolResult):
    prob: float = 0.0
    label: int = 0
    label_name: str = ""
    threshold: float = 0.5
    class_names: list[str] = field(default_factory=list)
    # CAM heatmap (H, W) float32 in [0, 1]; None when unavailable (ONNX without
    # features output, or ensemble mode where per-fold CAMs are averaged).
    cam: np.ndarray | None = None


@dataclass
class MultiLabelClassificationResult(ToolResult):
    """Result for multi-label classification tasks (e.g. CheXpert 14-class).

    Each class is an independent binary classifier — a single image may have
    multiple active labels simultaneously.
    """

    probs: list[float] = field(default_factory=list)  # one probability per class
    labels: list[int] = field(default_factory=list)  # 1 if prob >= threshold
    class_names: list[str] = field(default_factory=list)
    threshold: float = 0.5
    # Grad-CAM per active class (class_name → (H, W) float32 in [0, 1]).
    # Only classes with label=1 are included; empty dict when CAMs are unavailable.
    cams: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SegmentationResult(ToolResult):
    """Pixel-level class mask."""

    mask: np.ndarray | None = None  # (H, W) or (num_classes, H, W), float32
    class_names: list[str] = field(default_factory=list)


@dataclass
class DetectionResult(ToolResult):
    """Bounding-box detections."""

    # Each item: {"box": [x, y, w, h], "label": int, "label_name": str, "score": float}
    boxes: list[dict[str, Any]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class MLTool(ABC):
    """Pluggable ML analysis tool.

    Lifecycle:
        tool = MyTool()
        tool.load(weights_path)          # idempotent, must be called first
        result = tool.predict(image)     # (C, H, W) float32, preprocessed

    To add a new clinical tool:
        1. Subclass MLTool, set TOOL_ID.
        2. Implement all @abstractmethod methods.
        3. Register in registry.build_registry().
    """

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Return static metadata for this tool."""

    @abstractmethod
    def load(self, weights_path: Path) -> None:
        """Load model weights from *weights_path*. Must be called before predict()."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> ToolResult:
        """Run inference on a preprocessed (C, H, W) float32 image."""

    @abstractmethod
    def get_preprocessing_config(self) -> dict[str, Any]:
        """Return a config dict compatible with PreprocessingPipeline.from_config()."""

    def is_loaded(self) -> bool:
        return bool(getattr(self, "_loaded", False))

    def get_gradcam_target_layer(self) -> str | None:
        """Return the PyTorch named-module path for Grad-CAM hooks, or None."""
        return None

    # ------------------------------------------------------------------
    # Runtime-tunable settings (optional — default is no settings)
    # ------------------------------------------------------------------

    def get_settings_schema(self) -> list[SettingField]:
        """Declare runtime-tunable settings exposed to the API/UI."""
        return []

    def apply_settings(self, values: dict[str, Any]) -> None:
        """Apply validated settings to the live tool instance.

        Called by the worker before each predict() so settings changes take
        effect without restarting the process.  Tools that override this must
        be idempotent — applying the same values twice is a no-op.
        """
        return None


# Forward-reference resolution for SettingField (avoid circular import at module top)
from backend.app.modules.ml_tools.settings import SettingField  # noqa: E402, F401
