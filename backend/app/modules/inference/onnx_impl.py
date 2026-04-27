"""ONNX Runtime inference backend — prod server (CPU-only, INT8).

Implementation deferred to Phase 5 (ONNX export + quantization pipeline).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import ModelInference, Prediction

_PHASE5_MSG = (
    "OnnxInference is not yet implemented. "
    "It will be wired up in Phase 5 (ONNX export + INT8 quantization)."
)


class OnnxInference(ModelInference):
    """ONNX Runtime backend for CPU-only prod inference (stub)."""

    def load(self, checkpoint_path: Path) -> None:
        raise NotImplementedError(_PHASE5_MSG)

    def predict(self, image: np.ndarray) -> Prediction:
        raise NotImplementedError(_PHASE5_MSG)
