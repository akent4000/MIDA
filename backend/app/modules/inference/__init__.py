"""Inference module — backend selection via INFERENCE_BACKEND env var.

Usage:
    backend = get_inference_backend(Path("weights/best.pt"))
    result  = backend.predict(image_array)   # Prediction(prob, label, threshold)

Set INFERENCE_BACKEND=pytorch (default, dev) or INFERENCE_BACKEND=onnx (prod).

Re-exports (PyTorchInference, OnnxInference) are lazy via __getattr__ so that
the prod container (INFERENCE_BACKEND=onnx, no torch installed) never loads
pytorch_impl just by importing this package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .base import ModelInference, Prediction

if TYPE_CHECKING:
    from .onnx_impl import OnnxInference
    from .pytorch_impl import PyTorchInference

__all__ = [
    "ModelInference",
    "OnnxInference",
    "Prediction",
    "PyTorchInference",
    "get_inference_backend",
]


def __getattr__(name: str) -> object:
    """Lazy re-exports — only load the submodule when the name is actually used."""
    if name == "PyTorchInference":
        from .pytorch_impl import PyTorchInference

        return PyTorchInference
    if name == "OnnxInference":
        from .onnx_impl import OnnxInference

        return OnnxInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_inference_backend(checkpoint_path: Path | None = None) -> ModelInference:
    """Return the configured inference backend, optionally loading a checkpoint.

    Reads ``INFERENCE_BACKEND`` from the environment:
      * ``pytorch`` (default) — :class:`PyTorchInference`
      * ``onnx`` — :class:`OnnxInference` (Phase 5 stub)

    Args:
        checkpoint_path: If provided, ``load()`` is called before returning.
    """
    backend = os.environ.get("INFERENCE_BACKEND", "pytorch").lower()
    instance: ModelInference
    if backend == "pytorch":
        from .pytorch_impl import PyTorchInference

        instance = PyTorchInference()
    elif backend == "onnx":
        from .onnx_impl import OnnxInference

        instance = OnnxInference()
    else:
        raise ValueError(f"Unknown INFERENCE_BACKEND={backend!r}. Expected 'pytorch' or 'onnx'.")
    if checkpoint_path is not None:
        instance.load(checkpoint_path)
    return instance
