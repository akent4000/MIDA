"""PyTorch inference backend — dev/training environment only.

Do NOT import this module on the prod server (ONNX-only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from backend.ml.training.model import build_model as _build_model

from .base import ModelInference, Prediction


class PyTorchInference(ModelInference):
    """Runs a trained checkpoint on the PyTorch runtime (ResNet50, EfficientNet-B0, …).

    Two construction paths:
      * ``PyTorchInference(model=my_model)``  — inject model directly (tests, interactive).
      * ``PyTorchInference.from_checkpoint(path)`` — build arch from checkpoint config.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        device: torch.device | None = None,
        threshold: float = 0.5,
    ) -> None:
        self._model = model
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device | None = None,
    ) -> PyTorchInference:
        """Build and return a ready-to-use instance from a .pt checkpoint."""
        d = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance = cls(device=d)
        instance.load(checkpoint_path)
        return instance

    # ------------------------------------------------------------------
    # ModelInference interface
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: Path) -> None:
        """Load weights + threshold from a train_baseline .pt checkpoint."""
        ckpt: dict[str, Any] = torch.load(
            checkpoint_path, map_location=self._device, weights_only=False
        )
        self._model = _build_model(ckpt["config"])  # delegates to ml/training/model.py
        self._model.load_state_dict(ckpt["model_state"])
        self._model.to(self._device)
        self._model.eval()
        self._threshold = float(ckpt.get("threshold_youden", 0.5))

    def predict(self, image: np.ndarray) -> Prediction:
        """Return binary Prediction for one pre-processed image."""
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() or use from_checkpoint() first.")

        tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # (C, H, W) → (1, C, H, W)
        fmt = torch.channels_last if self._device.type == "cuda" else torch.contiguous_format
        tensor = tensor.to(self._device, memory_format=fmt)

        with torch.no_grad():
            logit = self._model(tensor)
        prob = float(torch.sigmoid(logit).item())
        label = int(prob >= self._threshold)
        return Prediction(prob=prob, label=label, threshold=self._threshold)
