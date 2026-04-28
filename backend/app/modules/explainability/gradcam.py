"""Grad-CAM explainability — PyTorch only (dev environment).

Do NOT import this module on the prod server (ONNX-only, no torch).
Use the lazy __getattr__ in explainability/__init__.py instead.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization."
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch import Tensor

from .base import Explainer


class GradCAMExplainer(Explainer):
    """Gradient-weighted Class Activation Mapping.

    Registers forward + backward hooks on *target_layer* and derives a spatial
    attention map weighted by the gradient of the output logit.

    Usage:
        # Resolve the target layer by name (e.g. "features.denseblock4")
        layer = dict(model.named_modules())["features.denseblock4"]
        cam = GradCAMExplainer(model, layer)
        heatmap = cam.explain(preprocessed_image)  # (H, W) float32 in [0, 1]
        cam.remove_hooks()                          # release references

    Or use as a context manager:
        with GradCAMExplainer(model, layer) as cam:
            heatmap = cam.explain(image)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self._model = model
        self._target_layer = target_layer
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._handles: list[Any] = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        def _fwd(_mod: Any, _inp: Any, output: Tensor) -> None:
            self._activations = output.detach()

        def _bwd(_mod: Any, _grad_in: Any, grad_out: Any) -> None:
            # grad_out is tuple[Tensor, ...] in practice, but the typed hook
            # signature in torch's stubs allows tuple[Tensor, ...] | Tensor.
            self._gradients = grad_out[0].detach()

        self._handles = [
            self._target_layer.register_forward_hook(_fwd),
            self._target_layer.register_full_backward_hook(_bwd),
        ]

    def remove_hooks(self) -> None:
        """Detach all hooks and release captured tensors."""
        for h in self._handles:
            h.remove()
        self._handles = []
        self._activations = None
        self._gradients = None

    def __enter__(self) -> GradCAMExplainer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.remove_hooks()

    def __del__(self) -> None:
        self.remove_hooks()

    # ------------------------------------------------------------------
    # Explainer interface
    # ------------------------------------------------------------------

    def explain(self, image: np.ndarray) -> np.ndarray:
        """Return (H, W) float32 Grad-CAM heatmap in [0, 1].

        Args:
            image: (C, H, W) float32, preprocessed — same input as model.predict().
        """
        device = next(self._model.parameters()).device
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # (1, C, H, W)
        tensor = tensor.to(device)

        self._model.eval()
        self._model.zero_grad()

        # Forward + backward so hooks fire
        logit: Tensor = self._model(tensor)
        logit.backward()  # type: ignore[no-untyped-call]

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "Verify that target_layer is on the forward path."
            )

        # Global average pool the gradients → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C', 1, 1)
        cam_t = (weights * self._activations).sum(dim=1).squeeze(0)  # (H', W')
        cam_arr: np.ndarray = torch.clamp(cam_t, min=0).cpu().numpy()

        # Upsample to input spatial dimensions
        h_in, w_in = image.shape[-2], image.shape[-1]
        # GaussianBlur requires an integer-mode image ("L"), not float32 ("F").
        cam_max = cam_arr.max()
        cam_u8 = ((cam_arr / (cam_max + 1e-8)) * 255).astype(np.uint8)
        cam_img = Image.fromarray(cam_u8).resize((w_in, h_in), Image.Resampling.BICUBIC)

        # Smooth: Gaussian blur with radius proportional to image size
        blur_radius = max(1, h_in // 48)
        cam_img = cam_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        cam_out = np.asarray(cam_img, dtype=np.float32)

        # Percentile stretch: clip outliers then normalize to [0, 1]
        lo, hi = np.percentile(cam_out, [2, 98])
        if hi > lo:
            return np.asarray(np.clip((cam_out - lo) / (hi - lo), 0.0, 1.0), dtype=np.float32)
        return np.zeros_like(cam_out)

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_layer_name(
        cls,
        model: nn.Module,
        layer_name: str,
    ) -> GradCAMExplainer:
        """Build a GradCAMExplainer from a named module path.

        Example:
            cam = GradCAMExplainer.from_layer_name(model, "features.denseblock4")
        """
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise ValueError(
                f"Layer {layer_name!r} not found in model. "
                f"Available: {list(modules)[:10]} …"
            )
        return cls(model, modules[layer_name])
