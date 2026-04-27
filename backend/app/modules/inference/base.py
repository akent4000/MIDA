"""Inference module public contract.

The ABC uses np.ndarray (CHW float32) instead of torch.Tensor so the interface
is torch-free — OnnxInference runs on prod without a torch install.
PyTorchInference converts internally before the forward pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """Result of a single-image binary classification inference."""

    prob: float  # sigmoid probability in [0, 1]
    label: int  # 0 or 1 after applying threshold
    threshold: float  # decision threshold used


class ModelInference(ABC):
    """Strategy interface for model backends (PyTorch dev / ONNX prod)."""

    @abstractmethod
    def load(self, checkpoint_path: Path) -> None:
        """Load model weights and metadata from *checkpoint_path*."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> Prediction:
        """Run inference on a pre-processed image.

        Args:
            image: float32 ndarray, shape (C, H, W) or (1, C, H, W), values
                   normalised to ImageNet stats (same preprocessing as training).

        Returns:
            Prediction with prob, binary label, and the threshold used.
        """
