"""Export a trained .pt checkpoint to FP32 ONNX.

Usage:
    python -m backend.ml.export.export_onnx \\
        --checkpoint backend/ml/weights/cv_rsna_densenet121_384_fold4_*/best.pt \\
        --out backend/ml/weights/onnx/fold4.onnx

The exported model embeds:
  * The Youden-optimal threshold from the checkpoint as ONNX metadata_props.
  * Dynamic batch dim so inference can batch (server uses bs=1, but ensemble
    soft-vote could batch the 5 folds).
  * For DenseNet-121: a second output ``features`` (B, 1024, H, W) from
    denseblock4 and the classifier weights in metadata_props["classifier_weights"].
    This enables exact CAM on prod (ONNX) without gradient computation.

Numerical equivalence is verified: torch FP32 vs ort FP32 on a random tensor,
absolute tolerance 1e-4. Non-equivalence aborts the export — silent drift
between dev (torch) and prod (onnx) is the worst possible failure mode.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from backend.ml.training.model import build_model

OPSET = 17
ATOL = 1e-4
RTOL = 1e-3


# ---------------------------------------------------------------------------
# DenseNet-121 CAM wrapper
# ---------------------------------------------------------------------------


class _DenseNetCamWrapper(nn.Module):
    """Wraps a DenseNet-121 to emit (logit, features) instead of logit only.

    ``features`` is the output of denseblock4 before global average pooling,
    shape (B, 1024, H/32, W/32).  Combined with the stored classifier weights
    it enables exact CAM on prod without gradient computation.
    """

    def __init__(self, densenet: nn.Module) -> None:
        super().__init__()
        # DenseNet stores everything under .features (a Sequential) plus .classifier
        self._features_extractor = densenet.features
        self._pool = nn.AdaptiveAvgPool2d((1, 1))
        self._classifier = densenet.classifier

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self._features_extractor(x)  # (B, 1024, H', W')
        feat = torch.relu(feat)  # DenseNet applies relu after the last norm
        pooled = self._pool(feat).flatten(1)  # (B, 1024)
        logit = self._classifier(pooled)  # (B, 1)
        return logit, feat


def _is_densenet121(model: nn.Module) -> bool:
    return hasattr(model, "features") and hasattr(model, "classifier") and isinstance(
        getattr(model, "classifier", None), nn.Linear
    ) and getattr(model.classifier, "in_features", 0) == 1024  # type: ignore[union-attr]


def _encode_weights(w: np.ndarray) -> str:
    """Base64-encode a float32 numpy array for storage in ONNX metadata."""
    return base64.b64encode(w.astype(np.float32).tobytes()).decode()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_model(
    model: nn.Module,
    out_path: Path,
    input_shape: tuple[int, int, int] = (3, 384, 384),
    threshold: float = 0.5,
    extra_metadata: dict[str, str] | None = None,
    opset: int = OPSET,
) -> dict[str, Any]:
    """Export an in-memory torch model to ONNX with parity check.

    For DenseNet-121 models a second ONNX output ``features`` is added and
    the classifier weights are embedded in metadata_props["classifier_weights"]
    so OnnxInference can compute CAM without gradient computation.

    Use this when you already have a model instance (tests, custom loaders).
    For training-checkpoint exports use :func:`export` instead.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    use_cam_wrapper = _is_densenet121(model)
    export_model_: nn.Module
    output_names: list[str]

    if use_cam_wrapper:
        export_model_ = _DenseNetCamWrapper(model)
        output_names = ["logit", "features"]
        dynamic_axes = {
            "input": {0: "batch"},
            "logit": {0: "batch"},
            "features": {0: "batch"},
        }
    else:
        export_model_ = model
        output_names = ["logit"]
        dynamic_axes = {"input": {0: "batch"}, "logit": {0: "batch"}}

    export_model_.eval()
    dummy = torch.randn(1, *input_shape, dtype=torch.float32)

    torch.onnx.export(
        export_model_,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(out_path))
    _set_metadata(onnx_model, "threshold_youden", str(threshold))
    _set_metadata(onnx_model, "input_shape", json.dumps(list(input_shape)))
    if use_cam_wrapper:
        # Store classifier weight vector (shape [1, 1024]) so OnnxInference can
        # compute CAM = features @ weights without loading the torch model.
        w = model.classifier.weight.detach().cpu().numpy()  # type: ignore[union-attr]
        _set_metadata(onnx_model, "classifier_weights", _encode_weights(w))
        _set_metadata(onnx_model, "cam_output", "features")
    for k, v in (extra_metadata or {}).items():
        _set_metadata(onnx_model, k, v)
    onnx.save(onnx_model, str(out_path))

    onnx.checker.check_model(onnx_model)

    # Numerical parity check — torch logit vs ort logit on the dummy input
    with torch.no_grad():
        torch_logit = model(dummy).cpu().numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_logit = sess.run(["logit"], {"input": dummy.numpy()})[0]
    max_drift = float(np.max(np.abs(torch_logit - ort_logit)))
    if not np.allclose(torch_logit, ort_logit, atol=ATOL, rtol=RTOL):
        raise RuntimeError(
            f"ONNX export drift: max |torch - ort| = {max_drift:.6f} > atol={ATOL}. "
            "Aborting — refusing to ship a model that disagrees with training."
        )

    return {
        "out_path": str(out_path),
        "threshold": threshold,
        "input_shape": list(input_shape),
        "max_drift": max_drift,
        "size_bytes": out_path.stat().st_size,
        "cam_enabled": use_cam_wrapper,
    }


def export(
    checkpoint_path: Path,
    out_path: Path,
    input_shape: tuple[int, int, int] = (3, 384, 384),
    opset: int = OPSET,
) -> dict[str, Any]:
    """Load a training checkpoint and export it to *out_path*."""
    ckpt: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    threshold = float(ckpt.get("threshold_youden", 0.5))
    return export_model(
        model,
        out_path,
        input_shape,
        threshold=threshold,
        extra_metadata={"source_checkpoint": checkpoint_path.name},
        opset=opset,
    )


def _set_metadata(model: onnx.ModelProto, key: str, value: str) -> None:
    """Idempotent set/replace of metadata_props[key]."""
    for prop in model.metadata_props:
        if prop.key == key:
            prop.value = value
            return
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value


def _cli() -> None:
    p = argparse.ArgumentParser(description="Export .pt → ONNX (FP32) with parity check.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--input-shape",
        type=int,
        nargs=3,
        default=(3, 384, 384),
        metavar=("C", "H", "W"),
    )
    p.add_argument("--opset", type=int, default=OPSET)
    args = p.parse_args()

    info = export(args.checkpoint, args.out, tuple(args.input_shape), args.opset)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    _cli()
