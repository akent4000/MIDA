"""CheXpert 14-class multi-label chest X-ray classifier.

TOOL_ID: "chexpert_14"

DenseNet-121 with 14 sigmoid outputs, trained on CheXpert with U-Ignore loss.
Each output is an independent binary classifier for one of the 14 pathologies.
Grad-CAM is computed per active class (prob >= threshold) on the denseblock4 layer.

Research prototype — not for clinical use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.modules.ml_tools.base import (
    MLTool,
    Modality,
    ModelInfo,
    MultiLabelClassificationResult,
    TaskType,
    ToolResult,
)

CHEXPERT_LABELS: list[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
NUM_CLASSES = len(CHEXPERT_LABELS)

TOOL_ID = "chexpert_14"
_GRADCAM_LAYER = "features.denseblock4"
_DEFAULT_THRESHOLD = 0.5

logger = logging.getLogger(__name__)


class CheXpertTool(MLTool):
    """CheXpert 14-class multi-label chest X-ray classifier."""

    TOOL_ID = TOOL_ID

    def __init__(self) -> None:
        self._model: Any = None
        self._device: Any = None  # set in load()
        self._loaded = False

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            tool_id=TOOL_ID,
            name="CheXpert 14-Class Multi-Label Classifier",
            version="1.0",
            description=(
                "DenseNet-121 trained on CheXpert detecting 14 chest pathologies. "
                "Research prototype — not for clinical use."
            ),
            modality=Modality.XRAY,
            task_type=TaskType.CLASSIFICATION,
            input_shape=(3, 320, 320),
            class_names=list(CHEXPERT_LABELS),
        )

    def load(self, weights_path: Path) -> None:
        if self._loaded:
            return
        import os

        if os.environ.get("INFERENCE_BACKEND", "pytorch").lower() == "onnx":
            raise NotImplementedError(
                "CheXpertTool: ONNX backend not implemented yet — "
                "this tool requires PyTorch (dev environment only)."
            )
        import torch

        from backend.ml.training.model import build_model

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = {
            "model": {
                "arch": "densenet121",
                "weights": None,
                "num_classes": NUM_CLASSES,
                "backbone_checkpoint": None,
            }
        }
        model = build_model(cfg).to(self._device)
        ckpt = torch.load(weights_path, map_location=self._device, weights_only=False)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()
        if self._device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        self._model = model
        self._loaded = True
        logger.info("CheXpertTool loaded from %s on %s", weights_path.name, self._device)

    def predict(self, image: np.ndarray) -> ToolResult:
        """Predict 14-class probabilities from a preprocessed (C, H, W) float32 image."""
        import torch

        assert self._model is not None, "call load() first"
        fmt = torch.channels_last if self._device.type == "cuda" else torch.contiguous_format
        tensor = torch.from_numpy(image).unsqueeze(0).to(self._device, memory_format=fmt)
        with torch.no_grad():
            logits = self._model(tensor)  # (1, 14)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (14,)

        labels = [int(p >= _DEFAULT_THRESHOLD) for p in probs]
        cams = self._compute_cams(image, probs)

        return MultiLabelClassificationResult(
            tool_id=TOOL_ID,
            probs=probs.tolist(),
            labels=labels,
            class_names=list(CHEXPERT_LABELS),
            threshold=_DEFAULT_THRESHOLD,
            cams=cams,
        )

    def _compute_cams(self, image: np.ndarray, probs: np.ndarray) -> dict[str, np.ndarray]:
        """Grad-CAM for each active class (prob >= threshold)."""
        import torch
        import torch.nn as nn

        assert self._model is not None
        active = [i for i, p in enumerate(probs) if p >= _DEFAULT_THRESHOLD]
        if not active:
            return {}

        target_layer = dict(self._model.named_modules()).get(_GRADCAM_LAYER)
        if target_layer is None:
            return {}

        cams: dict[str, np.ndarray] = {}
        activations: list[Any] = []
        gradients: list[Any] = []

        def fwd_hook(_m: nn.Module, _inp: Any, out: Any) -> None:
            activations[:] = [out.detach()]

        def bwd_hook(_m: nn.Module, _gi: Any, go: tuple[Any, ...]) -> None:
            gradients[:] = [go[0].detach()]

        fwd_h = target_layer.register_forward_hook(fwd_hook)
        bwd_h = target_layer.register_full_backward_hook(bwd_hook)

        fmt = torch.channels_last if self._device.type == "cuda" else torch.contiguous_format
        try:
            for idx in active:
                inp = (
                    torch.from_numpy(image)
                    .unsqueeze(0)
                    .to(self._device, memory_format=fmt)
                    .requires_grad_(True)
                )
                logits = self._model(inp)
                self._model.zero_grad()
                logits[0, idx].backward()

                if not activations or not gradients:
                    continue
                act = activations[0].squeeze(0)  # (C, H, W)
                grad = gradients[0].squeeze(0)  # (C, H, W)
                weights = grad.mean(dim=(1, 2))  # (C,)
                cam = torch.relu((weights[:, None, None] * act).sum(0)).cpu().numpy()
                lo, hi = float(cam.min()), float(cam.max())
                cam = (cam - lo) / (hi - lo + 1e-8)
                cams[CHEXPERT_LABELS[idx]] = cam
        finally:
            fwd_h.remove()
            bwd_h.remove()

        return cams

    def get_preprocessing_config(self) -> dict[str, Any]:
        return {
            "clahe": {"clip_limit": 0.01},
            "resize": {"size": 320},
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }

    def get_gradcam_target_layer(self) -> str | None:
        return _GRADCAM_LAYER
