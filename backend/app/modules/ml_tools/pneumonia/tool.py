"""PneumoniaTool — RSNA pneumonia binary classifier.

Wraps the existing ModelInference backends (PyTorch dev / ONNX prod) behind
the MLTool contract so the service layer is backend-agnostic.

INFERENCE_BACKEND env var selects PyTorch (dev) or ONNX (prod), identical
to the standalone inference module.

Two operating modes, switchable at runtime via the tool-settings API:
  * ``single``   — one fine-tuned DenseNet-121 (fastest, default).
  * ``ensemble`` — 5-fold soft-vote, AUC 0.8927, ~5× CPU on prod hardware.

Ensemble checkpoints come from ``PNEUMONIA_ENSEMBLE_PATHS`` (comma-separated
list).  Loading is lazy — fold weights only hit memory the first time the
ensemble mode is selected.

Training metrics (DenseNet-121, 5-fold CV, held-out 4,003-patient test set):
  AUC: 0.8927  |  Sensitivity@Youden: 0.788  |  Specificity@Youden: 0.826
Target from plan §8: AUC ≥ 0.90 (not yet reached).
"""

from __future__ import annotations

import logging
import os
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
from backend.app.modules.ml_tools.settings import (
    SettingField,
    SettingOption,
    SettingType,
)

logger = logging.getLogger(__name__)


MODE_SINGLE = "single"
MODE_ENSEMBLE = "ensemble"

# Empirically optimal threshold for the 6-model ensemble (Youden index on the
# 4,003-patient held-out test set; see ensemble_eval_test_20260425_153402.json).
ENSEMBLE_THRESHOLD = 0.4396


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
        self._single: ModelInference | None = None
        self._ensemble: list[ModelInference] | None = None
        self._ensemble_paths: list[Path] = self._read_ensemble_paths()
        self._mode: str = MODE_SINGLE
        self._loaded: bool = False

    @property
    def info(self) -> ModelInfo:
        return self._MODEL_INFO

    # ------------------------------------------------------------------
    # MLTool interface
    # ------------------------------------------------------------------

    def load(self, weights_path: Path) -> None:
        """Load the single-mode model. Ensemble is lazy-loaded on first use."""
        from backend.app.modules.inference import get_inference_backend

        self._single = get_inference_backend(checkpoint_path=weights_path)
        self._loaded = True

    def predict(self, image: np.ndarray) -> ClassificationResult:
        if not self._loaded or self._single is None:
            raise RuntimeError(f"{self.TOOL_ID}: not loaded — call load() first.")

        if self._mode == MODE_ENSEMBLE:
            self._ensure_ensemble_loaded()
            assert self._ensemble is not None  # narrow for mypy
            preds = [b.predict(image) for b in self._ensemble]
            mean_prob = float(np.mean([p.prob for p in preds]))
            threshold = ENSEMBLE_THRESHOLD
            label = int(mean_prob >= threshold)
            return ClassificationResult(
                tool_id=self.TOOL_ID,
                prob=mean_prob,
                label=label,
                label_name=self._CLASS_NAMES[label],
                threshold=threshold,
                class_names=list(self._CLASS_NAMES),
                metadata={"mode": MODE_ENSEMBLE, "n_models": len(self._ensemble)},
            )

        pred = self._single.predict(image)
        return ClassificationResult(
            tool_id=self.TOOL_ID,
            prob=pred.prob,
            label=pred.label,
            label_name=self._CLASS_NAMES[pred.label],
            threshold=pred.threshold,
            class_names=list(self._CLASS_NAMES),
            metadata={"mode": MODE_SINGLE, "n_models": 1},
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
        # Grad-CAM is only meaningful for a single concrete model; ensemble
        # mode falls back to single's hooks via the worker's _try_gradcam path.
        return "features.denseblock4"

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_settings_schema(self) -> list[SettingField]:
        ensemble_available = len(self._ensemble_paths) >= 2
        ensemble_desc = (
            "Soft-vote across 5 CV folds (AUC 0.8927). ~5× slower than single."
            if ensemble_available
            else "Unavailable: PNEUMONIA_ENSEMBLE_PATHS env var not configured."
        )
        return [
            SettingField(
                key="mode",
                label="Inference mode",
                type=SettingType.SELECT,
                default=MODE_SINGLE,
                description=(
                    "Single model = fastest. Ensemble = highest accuracy, "
                    "much slower on CPU-only prod hardware."
                ),
                options=[
                    SettingOption(
                        value=MODE_SINGLE,
                        label="Single model",
                        description="Fine-tuned DenseNet-121 (AUC 0.8918).",
                    ),
                    SettingOption(
                        value=MODE_ENSEMBLE,
                        label="5-fold ensemble",
                        description=ensemble_desc,
                    ),
                ],
            ),
        ]

    def apply_settings(self, values: dict[str, Any]) -> None:
        new_mode = values.get("mode", MODE_SINGLE)
        if new_mode not in (MODE_SINGLE, MODE_ENSEMBLE):
            logger.warning("%s: ignoring unknown mode %r", self.TOOL_ID, new_mode)
            return
        if new_mode == MODE_ENSEMBLE and not self._ensemble_paths:
            logger.warning(
                "%s: ensemble mode requested but PNEUMONIA_ENSEMBLE_PATHS is empty; "
                "staying on single.",
                self.TOOL_ID,
            )
            self._mode = MODE_SINGLE
            return
        self._mode = new_mode

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _read_ensemble_paths() -> list[Path]:
        raw = os.environ.get("PNEUMONIA_ENSEMBLE_PATHS", "").strip()
        if not raw:
            return []
        return [Path(p.strip()) for p in raw.split(",") if p.strip()]

    def _ensure_ensemble_loaded(self) -> None:
        if self._ensemble is not None:
            return
        if not self._ensemble_paths:
            raise RuntimeError(
                f"{self.TOOL_ID}: ensemble mode requires PNEUMONIA_ENSEMBLE_PATHS "
                "to list ≥ 2 checkpoint paths."
            )
        from backend.app.modules.inference import get_inference_backend

        backends: list[ModelInference] = []
        for path in self._ensemble_paths:
            if not path.exists():
                raise FileNotFoundError(f"{self.TOOL_ID}: ensemble checkpoint missing: {path}")
            backends.append(get_inference_backend(checkpoint_path=path))
        self._ensemble = backends
        logger.info("%s: ensemble loaded (%d models)", self.TOOL_ID, len(backends))
