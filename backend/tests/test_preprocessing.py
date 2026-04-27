"""Unit tests for PreprocessingPipeline.

All tests use synthetic numpy arrays — no DICOM files or model weights needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from backend.app.modules.preprocessing.pipeline import PreprocessingPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_image(h: int = 128, w: int = 128) -> np.ndarray:
    """Return a random (H, W) float32 array in [0, 1]."""
    return np.random.rand(h, w).astype(np.float32)


def _basic_cfg() -> dict[str, Any]:
    return {
        "resize": [64, 64],
        "channels": 3,
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_basic_output_shape(self) -> None:
        out = PreprocessingPipeline.from_config(_basic_cfg()).apply(_rand_image())
        assert out.shape == (3, 64, 64)

    def test_output_dtype_float32(self) -> None:
        out = PreprocessingPipeline.from_config(_basic_cfg()).apply(_rand_image())
        assert out.dtype == np.float32

    def test_single_channel_output(self) -> None:
        cfg: dict[str, Any] = {"resize": [32, 32], "channels": 1}
        out = PreprocessingPipeline.from_config(cfg).apply(_rand_image())
        assert out.shape == (1, 32, 32)

    def test_resize_to_arbitrary_shape(self) -> None:
        cfg: dict[str, Any] = {"resize": [50, 100], "channels": 1}
        out = PreprocessingPipeline.from_config(cfg).apply(_rand_image(200, 200))
        assert out.shape == (1, 50, 100)

    def test_no_resize_preserves_input_size(self) -> None:
        cfg: dict[str, Any] = {"channels": 1}
        out = PreprocessingPipeline.from_config(cfg).apply(_rand_image(40, 60))
        assert out.shape == (1, 40, 60)


# ---------------------------------------------------------------------------
# CLAHE step
# ---------------------------------------------------------------------------


class TestCLAHEStep:
    def test_clahe_runs_without_error(self) -> None:
        cfg: dict[str, Any] = {"clahe": {"clip_limit": 0.01}, "channels": 1}
        out = PreprocessingPipeline.from_config(cfg).apply(_rand_image())
        assert out.shape[0] == 1

    def test_clahe_output_in_unit_range_before_normalize(self) -> None:
        cfg: dict[str, Any] = {"clahe": {"clip_limit": 0.01}, "channels": 1}
        out = PreprocessingPipeline.from_config(cfg).apply(_rand_image())
        # After CLAHE (no normalize), values should stay near [0, 1]
        # (skimage equalize_adapthist always returns [0, 1])
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# Normalisation step
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_after_normalize_values_not_in_unit_range(self) -> None:
        out = PreprocessingPipeline.from_config(_basic_cfg()).apply(_rand_image())
        # ImageNet normalisation shifts values well outside [0, 1]
        assert out.min() < 0.0

    def test_wrong_mean_length_raises(self) -> None:
        cfg: dict[str, Any] = {
            "channels": 3,
            "normalize": {"mean": [0.5], "std": [0.5]},  # length 1 but channels=3
        }
        with pytest.raises(ValueError, match="channels"):
            PreprocessingPipeline.from_config(cfg).apply(_rand_image())


# ---------------------------------------------------------------------------
# Input format handling
# ---------------------------------------------------------------------------


class TestInputFormats:
    def test_3d_hwc_input(self) -> None:
        hwc = np.random.rand(64, 64, 3).astype(np.float32)
        out = PreprocessingPipeline.from_config({"channels": 1}).apply(hwc)
        assert out.ndim == 3

    def test_invalid_ndim_raises(self) -> None:
        bad = np.random.rand(4, 64, 64, 1).astype(np.float32)
        with pytest.raises(ValueError):
            PreprocessingPipeline.from_config({"channels": 1}).apply(bad)


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


class TestFactories:
    def test_from_config(self) -> None:
        pipe = PreprocessingPipeline.from_config({"channels": 1})
        assert isinstance(pipe, PreprocessingPipeline)

    def test_for_tool_uses_tool_config(self) -> None:
        from backend.app.modules.ml_tools.pneumonia.tool import PneumoniaTool

        pipe = PreprocessingPipeline.for_tool(PneumoniaTool())
        out = pipe.apply(_rand_image())
        # PneumoniaTool expects (3, 384, 384)
        assert out.shape == (3, 384, 384)

    def test_for_tool_pneumonia_full_pipeline(self) -> None:
        from backend.app.modules.ml_tools.pneumonia.tool import PneumoniaTool

        pipe = PreprocessingPipeline.for_tool(PneumoniaTool())
        out = pipe.apply(_rand_image(256, 256))
        assert out.shape == (3, 384, 384)
        assert out.dtype == np.float32
