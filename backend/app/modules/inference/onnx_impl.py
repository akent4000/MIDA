"""ONNX Runtime inference backend — prod server (CPU-only).

Ships in the prod container without a torch dependency. Reads the Youden
threshold from the model's metadata_props (set by export_onnx.py).

Thread/intra-op counts are tuned for the i3-4030U (2 physical cores, no
hyperthreading): intra=2, inter=1 keeps a single inference busy on both cores
without thrashing the OS scheduler.
"""

from __future__ import annotations

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

    def load(self, checkpoint_path: Path) -> None:
        """Load *.onnx and pull the Youden threshold from metadata_props."""
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

        # Read threshold from metadata_props (export_onnx.py writes it).
        import contextlib

        meta = session.get_modelmeta().custom_metadata_map
        if "threshold_youden" in meta:
            with contextlib.suppress(ValueError):
                self._threshold = float(meta["threshold_youden"])

    def predict(self, image: np.ndarray) -> Prediction:
        if self._session is None:
            raise RuntimeError("OnnxInference not loaded — call load() first.")

        arr = np.ascontiguousarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]  # (C, H, W) → (1, C, H, W)

        outputs = self._session.run(None, {self._input_name: arr})  # type: ignore[attr-defined]
        logit = float(np.asarray(outputs[0]).reshape(-1)[0])
        prob = float(_sigmoid(logit))
        label = int(prob >= self._threshold)
        return Prediction(prob=prob, label=label, threshold=self._threshold)


def _sigmoid(x: float) -> float:
    # Avoid overflow on very negative logits.
    if x >= 0:
        z = float(np.exp(-x))
        return 1.0 / (1.0 + z)
    z = float(np.exp(x))
    return z / (1.0 + z)
