"""Static INT8 quantization of an exported ONNX model.

Usage:
    python -m backend.ml.export.quantize_int8 \\
        --fp32 backend/ml/weights/onnx/fold4.onnx \\
        --int8 backend/ml/weights/onnx/fold4-int8.onnx \\
        --calibration-dir backend/ml/data/rsna/stage_2_train_images \\
        --num-samples 200

We use *static* (not dynamic) INT8 because dynamic quantization on CNNs
typically loses 2-4 AUC points; static calibration over a representative
sample of val/train images is required to keep the quality drop under 1%.

Calibration data must be raw DICOM files; the same val-time preprocessing
pipeline (CLAHE → resize → ImageNet normalize) is applied to feed the
calibration reader. This mirrors prod inference exactly.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from backend.app.modules.dicom.service import DicomService
from backend.app.modules.preprocessing.pipeline import PreprocessingPipeline


class _RsnaCalibrationReader(CalibrationDataReader):
    """Iterates pre-processed RSNA calibration samples for static quantization."""

    def __init__(
        self,
        sample_paths: list[Path],
        preprocess_config: dict[str, Any],
        input_name: str,
    ) -> None:
        self._paths = iter(sample_paths)
        self._dicom = DicomService()
        self._preproc = PreprocessingPipeline.from_config(preprocess_config)
        self._input_name = input_name

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            path = next(self._paths)
        except StopIteration:
            return None
        with path.open("rb") as f:
            study = self._dicom.load(f.read())
        image = self._preproc.apply(study.pixel_data)
        # ONNX expects a leading batch dim
        return {self._input_name: image[np.newaxis, ...].astype(np.float32)}

    def rewind(self) -> None:  # called between epochs by the quantizer
        # We materialise the list above; a one-shot iterator is fine because
        # quantize_static only iterates once.
        pass


# Default preprocessing config — must match PneumoniaTool.get_preprocessing_config().
# Duplicated here to keep this script importable without the full app.
_DEFAULT_PREPROC = {
    "resize": [384, 384],
    "channels": 3,
    "clahe": {"clip_limit": 0.01},
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


def quantize(
    fp32_path: Path,
    int8_path: Path,
    calibration_dir: Path,
    num_samples: int = 200,
    glob: str = "*.dcm",
    seed: int = 42,
    preprocess_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run static INT8 quantization. Returns a metadata dict."""
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = preprocess_config or _DEFAULT_PREPROC
    all_samples = sorted(calibration_dir.rglob(glob))
    if len(all_samples) < num_samples:
        raise RuntimeError(
            f"Need {num_samples} calibration samples; found only {len(all_samples)} "
            f"matching {glob!r} under {calibration_dir}."
        )
    rng = random.Random(seed)
    sample_paths = rng.sample(all_samples, num_samples)

    pre_path = int8_path.with_suffix(".pre.onnx")
    # skip_symbolic_shape: DenseNet's dense-concat graph has dynamic shapes that
    # SymbolicShapeInference can't resolve at opset 18; static ONNX shape inference
    # and graph optimization still run and are sufficient for correct quantization.
    quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=True)

    fp32 = onnx.load(str(pre_path))
    input_name = fp32.graph.input[0].name

    reader = _RsnaCalibrationReader(sample_paths, cfg, input_name=input_name)

    quantize_static(
        model_input=str(pre_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )

    pre_path.unlink(missing_ok=True)

    # quantize_static strips custom metadata_props (threshold, classifier_weights,
    # cam_output). Copy them from the original FP32 model so OnnxInference can
    # read the threshold and compute CAM on prod.
    _copy_metadata(fp32_path, int8_path)

    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size

    return {
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "num_calibration_samples": num_samples,
        "fp32_size_bytes": fp32_size,
        "int8_size_bytes": int8_size,
        "compression_ratio": round(fp32_size / int8_size, 2),
    }


def _copy_metadata(fp32_path: Path, int8_path: Path) -> None:
    """Copy custom metadata_props from the FP32 model into the INT8 model.

    quantize_static strips metadata_props (threshold_youden, classifier_weights,
    cam_output, input_shape).  Restoring them from the original FP32 file lets
    OnnxInference read the threshold and compute CAM on prod without re-quantizing.
    """
    fp32_model = onnx.load(str(fp32_path), load_external_data=False)
    int8_model = onnx.load(str(int8_path))

    existing_keys = {p.key for p in int8_model.metadata_props}
    for prop in fp32_model.metadata_props:
        if prop.key not in existing_keys:
            entry = int8_model.metadata_props.add()
            entry.key = prop.key
            entry.value = prop.value

    onnx.save(int8_model, str(int8_path))


def _cli() -> None:
    p = argparse.ArgumentParser(description="Static INT8 quantize an ONNX model.")
    p.add_argument("--fp32", type=Path, required=True, help="Input FP32 .onnx")
    p.add_argument("--int8", type=Path, required=True, help="Output INT8 .onnx")
    p.add_argument(
        "--calibration-dir",
        type=Path,
        required=True,
        help="Directory of DICOM samples used for calibration.",
    )
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--glob", type=str, default="*.dcm")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    info = quantize(
        args.fp32,
        args.int8,
        args.calibration_dir,
        num_samples=args.num_samples,
        glob=args.glob,
        seed=args.seed,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    _cli()
