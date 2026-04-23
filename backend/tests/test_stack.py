"""Smoke test: the ML stack imports cleanly.

Keeps Phase 0 honest — if someone breaks the env, `pytest` fails before any
real training test runs.
"""

from __future__ import annotations


def test_torch_imports() -> None:
    import torch

    assert torch.__version__


def test_ml_libs_import() -> None:
    import monai  # noqa: F401
    import pydicom  # noqa: F401
    import skimage  # noqa: F401
