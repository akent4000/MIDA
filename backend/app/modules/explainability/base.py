"""Explainability module — abstract interface for saliency methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Explainer(ABC):
    """Interface for model explanation methods (Grad-CAM, attention maps, SHAP, …).

    All implementations accept the same preprocessed image tensor that was fed
    to the model and return a spatial heatmap of equal resolution.
    """

    @abstractmethod
    def explain(self, image: np.ndarray) -> np.ndarray:
        """Compute a saliency / attention map.

        Args:
            image: Pre-processed (C, H, W) float32 input — same as model input.

        Returns:
            (H, W) float32 heatmap normalised to [0, 1].
        """
