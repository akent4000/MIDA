"""Patch existing INT8 ONNX models with missing metadata and graph outputs.

quantize_static (onnxruntime) strips:
  1. custom metadata_props (threshold_youden, classifier_weights, cam_output)
  2. secondary graph outputs (``features``) — only ``logit`` survives

Run this once to fix all already-quantized INT8 models without re-quantizing.

Usage:
    python -m backend.ml.export.patch_int8_metadata \\
        --onnx-dir backend/ml/weights/onnx_export

For each <name>-int8.onnx, it finds <name>.onnx and:
  - copies missing metadata_props from FP32 → INT8
  - re-adds the ``features`` graph output by inserting an Identity node on the
    tensor that feeds ReduceMean (GAP), which is denseblock4 relu output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _find_features_tensor(model) -> str | None:
    """Return the name of the denseblock4-relu tensor (input to ReduceMean/GAP)."""
    import onnx.helper  # noqa: F401 — ensure onnx is imported

    for node in model.graph.node:
        if node.op_type == "ReduceMean" and len(node.input) >= 1:
            return node.input[0]
    return None


def _add_features_output(model, tensor_name: str) -> bool:
    """Insert Identity(tensor_name) → features and expose it as a graph output.

    Returns True if the output was added, False if it was already present.
    """
    import onnx
    import onnx.helper

    output_names = [o.name for o in model.graph.output]
    if "features" in output_names:
        return False

    # Add Identity node: features = Identity(tensor_name)
    identity = onnx.helper.make_node("Identity", inputs=[tensor_name], outputs=["features"])
    model.graph.node.append(identity)

    # Add features as a graph output (shape unknown → empty type info is fine for ORT)
    features_out = onnx.helper.make_tensor_value_info("features", onnx.TensorProto.FLOAT, None)
    model.graph.output.append(features_out)
    return True


def patch_dir(onnx_dir: Path, dry_run: bool = False) -> list[dict]:
    import onnx

    results = []
    for int8_path in sorted(onnx_dir.glob("*-int8.onnx")):
        stem = int8_path.stem[: -len("-int8")]
        fp32_path = int8_path.with_name(f"{stem}.onnx")
        if not fp32_path.exists():
            results.append({"int8": str(int8_path), "status": "skipped", "reason": "no fp32 found"})
            continue

        fp32_model = onnx.load(str(fp32_path), load_external_data=False)
        int8_model = onnx.load(str(int8_path))

        # 1. Copy missing metadata_props from FP32 → INT8
        existing_keys = {p.key for p in int8_model.metadata_props}
        fp32_meta = {p.key: p.value for p in fp32_model.metadata_props}
        added_meta = []
        for k, v in fp32_meta.items():
            if k not in existing_keys:
                if not dry_run:
                    entry = int8_model.metadata_props.add()
                    entry.key = k
                    entry.value = v
                added_meta.append(k)

        # 2. Re-add ``features`` graph output if missing
        features_tensor = _find_features_tensor(int8_model)
        added_features = False
        if features_tensor:
            if not dry_run:
                added_features = _add_features_output(int8_model, features_tensor)
            else:
                output_names = [o.name for o in int8_model.graph.output]
                added_features = "features" not in output_names

        if not dry_run and (added_meta or added_features):
            onnx.save(int8_model, str(int8_path))

        results.append({
            "int8": str(int8_path),
            "status": "dry_run" if dry_run else ("patched" if (added_meta or added_features) else "ok"),
            "added_metadata": added_meta,
            "added_features_output": added_features,
            "features_tensor": features_tensor,
        })

    return results


def _cli() -> None:
    p = argparse.ArgumentParser(description="Patch INT8 ONNX models with FP32 metadata + features output.")
    p.add_argument("--onnx-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    results = patch_dir(args.onnx_dir, dry_run=args.dry_run)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    _cli()
