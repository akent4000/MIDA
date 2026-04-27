"""Preprocessing pipeline — torch-free, numpy/PIL/skimage only.

Converts raw pixel data (H, W) float32 in [0, 1]  →  model-ready (C, H, W) float32.
Config is a plain dict, typically obtained from MLTool.get_preprocessing_config().

Steps (applied in this order when present in config):
    clahe     — contrast-limited adaptive histogram equalisation (skimage)
    resize    — bilinear resize to [H, W]
    channels  — repeat single-channel image to N channels
    normalize — per-channel mean/std normalisation (e.g. ImageNet stats)

The pipeline replicates the val/test transforms in ml/training/datasets.py without
touching PyTorch or MONAI, so it runs identically on both the dev box and the
prod server (ONNX-only, no torch installed).

CLI usage (headless test):
    python -m backend.app.modules.preprocessing.pipeline path/to/image.dcm
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist

if TYPE_CHECKING:
    from backend.app.modules.ml_tools.base import MLTool


class PreprocessingPipeline:
    """Configurable per-tool preprocessing pipeline.

    Config keys (all optional):
        clahe      dict  {"clip_limit": float}       — CLAHE equalisation
        resize     list  [H, W]                       — bilinear resize
        channels   int                                — repeat to N channels
        normalize  dict  {"mean": [...], "std": [...]} — subtract mean / div std
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PreprocessingPipeline:
        return cls(config)

    @classmethod
    def for_tool(cls, tool: MLTool) -> PreprocessingPipeline:
        return cls(tool.get_preprocessing_config())

    # ------------------------------------------------------------------
    # Main transform
    # ------------------------------------------------------------------

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Transform a (H, W) or (H, W, C) float32 image → (C, H, W) float32.

        Input must be float32 in [0, 1] (as produced by DicomService.load()).
        """
        arr = np.asarray(image, dtype=np.float32)

        if arr.ndim == 3:
            # Collapse to (H, W) — take first channel regardless of layout
            arr = arr[..., 0] if arr.shape[-1] in (1, 3, 4) else arr[0]
        elif arr.ndim != 2:
            raise ValueError(f"Expected 2-D (H,W) or 3-D input, got shape {arr.shape}")

        cfg = self._config

        # Step 1: CLAHE
        if "clahe" in cfg:
            clip = float(cfg["clahe"].get("clip_limit", 0.01))
            arr = equalize_adapthist(arr, clip_limit=clip).astype(np.float32)

        # Step 2: resize
        if "resize" in cfg:
            th, tw = int(cfg["resize"][0]), int(cfg["resize"][1])
            arr = self._resize(arr, th, tw)

        # Step 3: expand to N channels  →  (C, H, W)
        n_channels = int(cfg.get("channels", 1))
        out = np.stack([arr] * n_channels, axis=0)  # (C, H, W)

        # Step 4: normalise
        if "normalize" in cfg:
            mean = np.array(cfg["normalize"]["mean"], dtype=np.float32)
            std = np.array(cfg["normalize"]["std"], dtype=np.float32)
            if mean.shape[0] != n_channels or std.shape[0] != n_channels:
                raise ValueError(
                    f"normalize mean/std have {mean.shape[0]} values "
                    f"but channels={n_channels}"
                )
            out = (out - mean[:, None, None]) / std[:, None, None]

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
        img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), mode="L")
        img = img.resize((w, h), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# CLI — headless smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m backend.app.modules.preprocessing.pipeline <image> [config.json]")
        sys.exit(1)

    from backend.app.modules.dicom.service import DicomService

    study = DicomService().load(sys.argv[1])
    cfg: dict[str, Any] = {}
    if len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            cfg = json.load(f)
    else:
        # Default: pneumonia-style config
        cfg = {
            "resize": [384, 384],
            "channels": 3,
            "clahe": {"clip_limit": 0.01},
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        }

    out = PreprocessingPipeline.from_config(cfg).apply(study.pixel_data)
    print(f"Output shape : {out.shape}  dtype={out.dtype}")
    print(f"Value range  : [{out.min():.3f}, {out.max():.3f}]")
