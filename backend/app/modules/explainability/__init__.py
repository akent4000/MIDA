"""Explainability module — saliency / attention maps for model decisions.

GradCAMExplainer is lazy-imported so the prod container (ONNX-only, no torch)
never loads PyTorch just by importing this package.

Usage:
    from backend.app.modules.explainability import GradCAMExplainer
    cam = GradCAMExplainer.from_layer_name(model, "features.denseblock4")
    heatmap = cam.explain(image)   # (H, W) float32 in [0, 1]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Explainer

if TYPE_CHECKING:
    from .gradcam import GradCAMExplainer

__all__ = ["Explainer", "GradCAMExplainer"]


def __getattr__(name: str) -> object:
    if name == "GradCAMExplainer":
        from .gradcam import GradCAMExplainer

        return GradCAMExplainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
