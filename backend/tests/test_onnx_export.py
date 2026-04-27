"""Tests for the ONNX export + OnnxInference round-trip.

Uses a tiny hand-built torch model so tests don't depend on torchvision
weights download or RSNA data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

from backend.app.modules.inference.onnx_impl import OnnxInference
from backend.ml.export.export_onnx import export_model


class _TinyClassifier(nn.Module):
    """Single-conv binary classifier — small enough to round-trip in ms."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


@pytest.fixture()
def exported_onnx(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = _TinyClassifier()
    out = tmp_path / "tiny.onnx"
    export_model(
        model,
        out,
        input_shape=(3, 32, 32),
        threshold=0.4396,
        extra_metadata={"source_checkpoint": "fake_test_ckpt.pt"},
    )
    return out


# ---------------------------------------------------------------------------
# Export round-trip
# ---------------------------------------------------------------------------


class TestExport:
    def test_round_trip_creates_valid_onnx(self, exported_onnx: Path) -> None:
        m = onnx.load(str(exported_onnx))
        onnx.checker.check_model(m)

    def test_metadata_props_set(self, exported_onnx: Path) -> None:
        m = onnx.load(str(exported_onnx))
        meta = {p.key: p.value for p in m.metadata_props}
        assert meta["threshold_youden"] == "0.4396"
        assert meta["input_shape"] == "[3, 32, 32]"
        assert meta["source_checkpoint"] == "fake_test_ckpt.pt"

    def test_dynamic_batch_dim(self, exported_onnx: Path) -> None:
        m = onnx.load(str(exported_onnx))
        input_dim = m.graph.input[0].type.tensor_type.shape.dim[0]
        # dim_param means dynamic; dim_value would mean fixed
        assert input_dim.dim_param == "batch"

    def test_export_aborts_on_numerical_drift(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If torch and ort disagree, we'd rather crash than ship a bad model."""
        import backend.ml.export.export_onnx as mod

        # Fake an ort session whose output never matches torch
        class _Drift:
            def run(self, _names: object, _inputs: object) -> list[np.ndarray]:
                return [np.array([[1e6]], dtype=np.float32)]

        class _Stub:
            def __init__(self, *_a: object, **_kw: object) -> None: ...

            def InferenceSession(  # noqa: N802 — mimics onnxruntime API
                self, *_a: object, **_kw: object
            ) -> _Drift:
                return _Drift()

        monkeypatch.setattr(mod, "ort", _Stub())
        with pytest.raises(RuntimeError, match="drift"):
            export_model(_TinyClassifier(), tmp_path / "x.onnx", input_shape=(3, 32, 32))


# ---------------------------------------------------------------------------
# OnnxInference load + predict
# ---------------------------------------------------------------------------


class TestOnnxInference:
    def test_load_pulls_threshold_from_metadata(self, exported_onnx: Path) -> None:
        backend = OnnxInference(threshold=0.5)  # constructor default
        backend.load(exported_onnx)
        assert backend._threshold == 0.4396

    def test_predict_returns_prediction(self, exported_onnx: Path) -> None:
        backend = OnnxInference()
        backend.load(exported_onnx)
        img = np.random.rand(3, 32, 32).astype(np.float32)
        pred = backend.predict(img)
        assert 0.0 <= pred.prob <= 1.0
        assert pred.label in (0, 1)
        assert pred.threshold == 0.4396

    def test_predict_handles_4d_input(self, exported_onnx: Path) -> None:
        backend = OnnxInference()
        backend.load(exported_onnx)
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        pred = backend.predict(img)
        assert pred.label in (0, 1)

    def test_predict_before_load_raises(self) -> None:
        backend = OnnxInference()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.predict(np.zeros((3, 32, 32), dtype=np.float32))

    def test_torch_onnx_outputs_match(self, tmp_path: Path) -> None:
        """End-to-end: same input → torch and OnnxInference produce same prob."""
        torch.manual_seed(42)
        model = _TinyClassifier()
        out = tmp_path / "tiny.onnx"
        export_model(model, out, input_shape=(3, 32, 32), threshold=0.5)

        np.random.seed(0)
        img = np.random.randn(3, 32, 32).astype(np.float32)

        # torch reference
        with torch.no_grad():
            torch_logit = float(model(torch.from_numpy(img).unsqueeze(0)).item())
        torch_prob = 1.0 / (1.0 + float(np.exp(-torch_logit)))

        backend = OnnxInference()
        backend.load(out)
        onnx_prob = backend.predict(img).prob

        assert abs(torch_prob - onnx_prob) < 1e-4
