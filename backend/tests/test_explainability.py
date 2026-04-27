"""Unit tests for the explainability module.

GradCAMExplainer tests require PyTorch and use a tiny toy model.
The Explainer ABC and lazy-import tests run without PyTorch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Explainer ABC
# ---------------------------------------------------------------------------


class TestExplainerABC:
    def test_cannot_instantiate_directly(self) -> None:
        from backend.app.modules.explainability.base import Explainer

        with pytest.raises(TypeError):
            Explainer()  # type: ignore[abstract]

    def test_lazy_import_does_not_pull_gradcam(self) -> None:
        # Importing the package should not eagerly import gradcam.py
        import importlib
        import sys

        # Remove any cached modules
        for key in list(sys.modules.keys()):
            if "gradcam" in key:
                del sys.modules[key]

        import backend.app.modules.explainability as exp_mod  # noqa: F401

        assert "backend.app.modules.explainability.gradcam" not in sys.modules

    def test_lazy_import_gradcam_on_access(self) -> None:
        from backend.app.modules.explainability import GradCAMExplainer

        assert GradCAMExplainer is not None


# ---------------------------------------------------------------------------
# GradCAMExplainer with a toy model
# ---------------------------------------------------------------------------


class _TinyConvNet(nn.Module):
    """Minimal CNN: one conv layer followed by global average pool + linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


class TestGradCAMExplainer:
    @pytest.fixture()
    def model_and_layer(self):
        model = _TinyConvNet()
        model.eval()
        return model, model.conv

    def test_explain_returns_hwf32_heatmap(self, model_and_layer) -> None:
        from backend.app.modules.explainability.gradcam import GradCAMExplainer

        model, layer = model_and_layer
        with GradCAMExplainer(model, layer) as cam:
            image = np.random.rand(3, 32, 32).astype(np.float32)
            heatmap = cam.explain(image)

        assert heatmap.dtype == np.float32
        assert heatmap.ndim == 2
        assert heatmap.shape == (32, 32)

    def test_heatmap_in_unit_range(self, model_and_layer) -> None:
        from backend.app.modules.explainability.gradcam import GradCAMExplainer

        model, layer = model_and_layer
        with GradCAMExplainer(model, layer) as cam:
            heatmap = cam.explain(np.random.rand(3, 32, 32).astype(np.float32))

        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_heatmap_spatial_matches_input(self, model_and_layer) -> None:
        from backend.app.modules.explainability.gradcam import GradCAMExplainer

        model, layer = model_and_layer
        h, w = 48, 64
        with GradCAMExplainer(model, layer) as cam:
            heatmap = cam.explain(np.random.rand(3, h, w).astype(np.float32))

        assert heatmap.shape == (h, w)

    def test_from_layer_name_factory(self) -> None:
        from backend.app.modules.explainability.gradcam import GradCAMExplainer

        model = _TinyConvNet()
        cam = GradCAMExplainer.from_layer_name(model, "conv")
        assert cam is not None
        cam.remove_hooks()

    def test_from_layer_name_unknown_raises(self) -> None:
        from backend.app.modules.explainability.gradcam import GradCAMExplainer

        model = _TinyConvNet()
        with pytest.raises(ValueError, match="nonexistent"):
            GradCAMExplainer.from_layer_name(model, "nonexistent")

    def test_hooks_removed_after_context_exit(self, model_and_layer) -> None:
        from backend.app.modules.explainability.gradcam import GradCAMExplainer

        model, layer = model_and_layer
        with GradCAMExplainer(model, layer) as cam:
            pass
        assert len(cam._handles) == 0

    def test_remove_hooks_idempotent(self, model_and_layer) -> None:
        from backend.app.modules.explainability.gradcam import GradCAMExplainer

        model, layer = model_and_layer
        cam = GradCAMExplainer(model, layer)
        cam.remove_hooks()
        cam.remove_hooks()  # must not raise
