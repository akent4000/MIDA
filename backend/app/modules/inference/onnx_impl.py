"""ONNX Runtime inference backend — prod server (CPU-only).

Ships in the prod container without a torch dependency. Reads the Youden
threshold from the model's metadata_props (set by export_onnx.py).

Thread/intra-op counts are tuned for the i3-4030U (2 physical cores, no
hyperthreading): intra=2, inter=1 keeps a single inference busy on both cores
without thrashing the OS scheduler.

CAM support: when export_onnx.py exports a DenseNet-121 with a ``features``
output, this backend computes exact CAM (Class Activation Map) in numpy:

    CAM = relu(features @ classifier_weights.T)  →  normalize  →  resize

This is mathematically equivalent to GradCAM for architectures with
GlobalAveragePooling + Linear (DenseNet-121, ResNet, etc.).
"""

from __future__ import annotations

import base64
import contextlib
from pathlib import Path

import numpy as np

from .base import ModelInference, Prediction

DEFAULT_INTRA_OP_THREADS = 2
DEFAULT_INTER_OP_THREADS = 1


class OnnxInference(ModelInference):
    """Stateless ONNX Runtime backend wrapping an exported .onnx model."""

    def __init__(
        self,
        threshold: float = 0.5,
        intra_op_threads: int = DEFAULT_INTRA_OP_THREADS,
        inter_op_threads: int = DEFAULT_INTER_OP_THREADS,
    ) -> None:
        self._session: object | None = None
        self._threshold = threshold
        self._input_name: str = "input"
        self._intra = intra_op_threads
        self._inter = inter_op_threads
        # Set during load() when the model has a features output and embedded
        # classifier weights — shape (1, C) float32.
        self._classifier_weights: np.ndarray | None = None
        self._has_features_output: bool = False

    def load(self, checkpoint_path: Path) -> None:
        """Load *.onnx and pull threshold + classifier weights from metadata_props."""
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = self._intra
        sess_opts.inter_op_num_threads = self._inter
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(checkpoint_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._session = session
        self._input_name = session.get_inputs()[0].name

        meta = session.get_modelmeta().custom_metadata_map

        with contextlib.suppress(ValueError):
            if "threshold_youden" in meta:
                self._threshold = float(meta["threshold_youden"])

        # CAM: check if the model has a features output and classifier weights.
        output_names = [o.name for o in session.get_outputs()]
        self._has_features_output = "features" in output_names

        if self._has_features_output and "classifier_weights" in meta:
            with contextlib.suppress(Exception):
                raw = base64.b64decode(meta["classifier_weights"])
                # Shape is (out_features, in_features) — for binary (1, 1024)
                w = np.frombuffer(raw, dtype=np.float32).copy()
                # Reshape to (out, in); for binary classification out=1
                n_in = w.shape[0] // 1  # assume out=1 for binary
                self._classifier_weights = w.reshape(1, n_in)

    def predict(self, image: np.ndarray) -> Prediction:
        if self._session is None:
            raise RuntimeError("OnnxInference not loaded — call load() first.")

        arr = np.ascontiguousarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]  # (C, H, W) → (1, C, H, W)

        if self._has_features_output:
            outputs = self._session.run(  # type: ignore[attr-defined]
                ["logit", "features"], {self._input_name: arr}
            )
            logit = float(np.asarray(outputs[0]).reshape(-1)[0])
            features = np.asarray(outputs[1])  # (1, C, H, W)
            cam = _compute_cam(features, self._classifier_weights)
        else:
            outputs = self._session.run(None, {self._input_name: arr})  # type: ignore[attr-defined]
            logit = float(np.asarray(outputs[0]).reshape(-1)[0])
            cam = None

        prob = float(_sigmoid(logit))
        label = int(prob >= self._threshold)
        return Prediction(prob=prob, label=label, threshold=self._threshold, cam=cam)


def _compute_cam(
    features: np.ndarray,
    classifier_weights: np.ndarray | None,
) -> np.ndarray | None:
    """Compute CAM heatmap from feature maps and classifier weights.

    Args:
        features: (1, C, H, W) float32 feature maps from the last conv block.
        classifier_weights: (1, C) float32 — the linear classifier's weight row.

    Returns:
        (H_in, W_in) float32 heatmap in [0, 1] resized to the model input size,
        or None if classifier_weights are unavailable.
    """
    if classifier_weights is None:
        return None

    # cam[h,w] = sum_c( w[c] * features[c, h, w] ) — equivalent to GradCAM
    # for GAP + Linear architectures.
    feat = features[0]  # (C, H, W)
    cam = np.einsum("c,chw->hw", classifier_weights[0], feat)  # (H, W)
    cam = np.maximum(cam, 0)  # ReLU

    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min) if cam_max > cam_min else np.zeros_like(cam)

    result: np.ndarray = cam.astype(np.float32)
    return result


def _sigmoid(x: float) -> float:
    # Avoid overflow on very negative logits.
    if x >= 0:
        z = float(np.exp(-x))
        return 1.0 / (1.0 + z)
    z = float(np.exp(x))
    return z / (1.0 + z)
