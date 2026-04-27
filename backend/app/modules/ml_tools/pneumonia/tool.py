"""PneumoniaTool — RSNA pneumonia binary classifier.

Wraps the existing ModelInference backends (PyTorch dev / ONNX prod) behind
the MLTool contract so the service layer is backend-agnostic.

INFERENCE_BACKEND env var selects PyTorch (dev) or ONNX (prod), identical
to the standalone inference module.

Training metrics (DenseNet-121, 5-fold CV, held-out 4,003-patient test set):
  AUC: 0.8927  |  Sensitivity@Youden: 0.788  |  Specificity@Youden: 0.826
Target from plan §8: AUC ≥ 0.90 (not yet reached).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.app.modules.inference.base import ModelInference
from backend.app.modules.ml_tools.base import (
    ClassificationResult,
    MLTool,
    Modality,
    ModelInfo,
    TaskType,
)


class PneumoniaTool(MLTool):
    """DenseNet-121 binary classifier for RSNA Pneumonia Detection."""

    TOOL_ID: str = "pneumonia_classifier_v1"
    _CLASS_NAMES: list[str] = ["Normal", "Pneumonia"]

    _MODEL_INFO: ModelInfo = ModelInfo(
        tool_id=TOOL_ID,
        name="RSNA Pneumonia Classifier",
        version="1.0.0",
        description=(
            "DenseNet-121 5-fold CV ensemble trained on the RSNA Pneumonia "
            "Detection Challenge (26,684 chest X-rays). "
            "Research prototype — not a medical device."
        ),
        modality=Modality.XRAY,
        task_type=TaskType.CLASSIFICATION,
        input_shape=(3, 384, 384),
        class_names=_CLASS_NAMES,
    )

    def __init__(self) -> None:
        self._inference: ModelInference | None = None
        self._loaded: bool = False

    @property
    def info(self) -> ModelInfo:
        return self._MODEL_INFO

    def load(self, weights_path: Path) -> None:
        from backend.app.modules.inference import get_inference_backend

        self._inference = get_inference_backend(checkpoint_path=weights_path)
        self._loaded = True

    def predict(self, image: np.ndarray) -> ClassificationResult:
        if self._inference is None:
            raise RuntimeError(f"{self.TOOL_ID}: not loaded — call load() first.")
        pred = self._inference.predict(image)
        return ClassificationResult(
            tool_id=self.TOOL_ID,
            prob=pred.prob,
            label=pred.label,
            label_name=self._CLASS_NAMES[pred.label],
            threshold=pred.threshold,
            class_names=list(self._CLASS_NAMES),
        )

    def get_preprocessing_config(self) -> dict[str, Any]:
        # Must match the val/test transform pipeline in backend/ml/training/datasets.py:
        # EnsureChannelFirst → CLAHETransform(0.01) → Resize(384) → RepeatChannel(3)
        # → NormalizeIntensity(ImageNet)
        return {
            "resize": [384, 384],
            "channels": 3,
            "clahe": {"clip_limit": 0.01},
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }

    def get_gradcam_target_layer(self) -> str | None:
        return "features.denseblock4"
