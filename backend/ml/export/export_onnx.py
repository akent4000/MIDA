"""Export a trained .pt checkpoint to FP32 ONNX.

Usage:
    python -m backend.ml.export.export_onnx \\
        --checkpoint backend/ml/weights/cv_rsna_densenet121_384_fold4_*/best.pt \\
        --out backend/ml/weights/onnx/fold4.onnx

The exported model embeds:
  * The Youden-optimal threshold from the checkpoint as ONNX metadata_props.
  * Dynamic batch dim so inference can batch (server uses bs=1, but ensemble
    soft-vote could batch the 5 folds).

Numerical equivalence is verified: torch FP32 vs ort FP32 on a random tensor,
absolute tolerance 1e-4. Non-equivalence aborts the export — silent drift
between dev (torch) and prod (onnx) is the worst possible failure mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from backend.ml.training.model import build_model

OPSET = 17
ATOL = 1e-4
RTOL = 1e-3


def export_model(
    model: torch.nn.Module,
    out_path: Path,
    input_shape: tuple[int, int, int] = (3, 384, 384),
    threshold: float = 0.5,
    extra_metadata: dict[str, str] | None = None,
    opset: int = OPSET,
) -> dict[str, Any]:
    """Export an in-memory torch model to ONNX with parity check.

    Use this when you already have a model instance (tests, custom loaders).
    For training-checkpoint exports use :func:`export` instead.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy = torch.randn(1, *input_shape, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logit"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(out_path))
    _set_metadata(onnx_model, "threshold_youden", str(threshold))
    _set_metadata(onnx_model, "input_shape", json.dumps(list(input_shape)))
    for k, v in (extra_metadata or {}).items():
        _set_metadata(onnx_model, k, v)
    onnx.save(onnx_model, str(out_path))

    onnx.checker.check_model(onnx_model)

    # Numerical parity check — torch vs ort on the dummy input
    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["logit"], {"input": dummy.numpy()})[0]
    max_drift = float(np.max(np.abs(torch_out - ort_out)))
    if not np.allclose(torch_out, ort_out, atol=ATOL, rtol=RTOL):
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
