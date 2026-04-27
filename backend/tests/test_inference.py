"""Unit tests for the inference ABC, PyTorchInference, OnnxInference, and factory.

All tests run without RSNA data and without downloading pretrained weights.
The load() / from_checkpoint() tests use monkeypatch to substitute the real
_build_model (ResNet50, ~100 MB state dict) with a tiny model that fits the
same checkpoint format but runs instantly on CPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """(1, C, H, W) → (1, 1) linear head after global avg pool.
    Minimal stand-in for ResNet50 in load() tests.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


def _make_checkpoint(tmp_path: Path, model: nn.Module, threshold: float = 0.42) -> Path:
    cfg = {"model": {"arch": "resnet50", "weights": None, "num_classes": 1}}
    path = tmp_path / "tiny.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state": model.state_dict(),
            "val_auc": 0.75,
            "threshold_youden": threshold,
            "config": cfg,
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Prediction dataclass
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_fields_stored(self) -> None:
        from backend.app.modules.inference.base import Prediction

        p = Prediction(prob=0.7, label=1, threshold=0.5)
        assert p.prob == pytest.approx(0.7)
        assert p.label == 1
        assert p.threshold == pytest.approx(0.5)

    def test_frozen(self) -> None:
        from backend.app.modules.inference.base import Prediction

        p = Prediction(prob=0.7, label=1, threshold=0.5)
        with pytest.raises((AttributeError, TypeError)):
            p.prob = 0.8  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelInference ABC
# ---------------------------------------------------------------------------


class TestModelInferenceABC:
    def test_cannot_instantiate_abc(self) -> None:
        from backend.app.modules.inference.base import ModelInference

        with pytest.raises(TypeError):
            ModelInference()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# PyTorchInference — behaviour tests with mock model
# ---------------------------------------------------------------------------


class TestPyTorchInferenceBehaviour:
    def _make(self, logit: float, threshold: float = 0.5):  # -> PyTorchInference
        from backend.app.modules.inference.pytorch_impl import PyTorchInference

        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[logit]])
        return PyTorchInference(model=mock_model, device=torch.device("cpu"), threshold=threshold)

    def test_predict_returns_prediction(self) -> None:
        from backend.app.modules.inference.base import Prediction

        inf = self._make(logit=0.0)  # sigmoid(0) = 0.5
        result = inf.predict(np.zeros((3, 224, 224), dtype=np.float32))
        assert isinstance(result, Prediction)

    def test_logit_zero_prob_half(self) -> None:
        inf = self._make(logit=0.0, threshold=0.5)
        result = inf.predict(np.zeros((3, 224, 224), dtype=np.float32))
        assert result.prob == pytest.approx(0.5, abs=1e-5)
        assert result.label == 1  # 0.5 >= 0.5

    def test_high_logit_gives_label_1(self) -> None:
        inf = self._make(logit=5.0)
        result = inf.predict(np.zeros((3, 224, 224), dtype=np.float32))
        assert result.label == 1
        assert result.prob > 0.99

    def test_low_logit_gives_label_0(self) -> None:
        inf = self._make(logit=-5.0)
        result = inf.predict(np.zeros((3, 224, 224), dtype=np.float32))
        assert result.label == 0
        assert result.prob < 0.01

    def test_threshold_applied_correctly(self) -> None:
        # logit=0 → prob≈0.5; threshold=0.9 → label=0
        inf = self._make(logit=0.0, threshold=0.9)
        result = inf.predict(np.zeros((3, 224, 224), dtype=np.float32))
        assert result.label == 0
        assert result.threshold == pytest.approx(0.9)

    def test_chw_input_auto_batched(self) -> None:
        inf = self._make(logit=0.0)
        # (3, H, W) → should be unsqueezed to (1, 3, H, W) internally
        result = inf.predict(np.zeros((3, 224, 224), dtype=np.float32))
        assert 0.0 <= result.prob <= 1.0

    def test_nchw_input_passthrough(self) -> None:
        inf = self._make(logit=0.0)
        # already (1, 3, H, W) — should NOT be unsqueezed again
        result = inf.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))
        assert 0.0 <= result.prob <= 1.0

    def test_predict_without_load_raises(self) -> None:
        from backend.app.modules.inference.pytorch_impl import PyTorchInference

        inf = PyTorchInference()
        with pytest.raises(RuntimeError, match="load"):
            inf.predict(np.zeros((3, 224, 224), dtype=np.float32))


# ---------------------------------------------------------------------------
# PyTorchInference — load() + from_checkpoint() (monkeypatched _build_model)
# ---------------------------------------------------------------------------


class TestPyTorchInferenceLoad:
    def test_load_reads_threshold(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.app.modules.inference import pytorch_impl
        from backend.app.modules.inference.pytorch_impl import PyTorchInference

        tiny = _TinyModel()
        ckpt_path = _make_checkpoint(tmp_path, tiny, threshold=0.37)
        monkeypatch.setattr(pytorch_impl, "_build_model", lambda _cfg: _TinyModel())

        inf = PyTorchInference(device=torch.device("cpu"))
        inf.load(ckpt_path)

        assert inf._threshold == pytest.approx(0.37)

    def test_load_then_predict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.app.modules.inference import pytorch_impl
        from backend.app.modules.inference.base import Prediction
        from backend.app.modules.inference.pytorch_impl import PyTorchInference

        tiny = _TinyModel()
        ckpt_path = _make_checkpoint(tmp_path, tiny)
        monkeypatch.setattr(pytorch_impl, "_build_model", lambda _cfg: _TinyModel())

        inf = PyTorchInference(device=torch.device("cpu"))
        inf.load(ckpt_path)
        result = inf.predict(np.random.rand(3, 224, 224).astype(np.float32))

        assert isinstance(result, Prediction)
        assert 0.0 <= result.prob <= 1.0
        assert result.label in (0, 1)

    def test_from_checkpoint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.app.modules.inference import pytorch_impl
        from backend.app.modules.inference.pytorch_impl import PyTorchInference

        tiny = _TinyModel()
        ckpt_path = _make_checkpoint(tmp_path, tiny, threshold=0.6)
        monkeypatch.setattr(pytorch_impl, "_build_model", lambda _cfg: _TinyModel())

        inf = PyTorchInference.from_checkpoint(ckpt_path, device=torch.device("cpu"))
        assert inf._threshold == pytest.approx(0.6)
        result = inf.predict(np.zeros((3, 224, 224), dtype=np.float32))
        assert 0.0 <= result.prob <= 1.0

    def test_checkpoint_missing_threshold_defaults_to_half(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from backend.app.modules.inference import pytorch_impl
        from backend.app.modules.inference.pytorch_impl import PyTorchInference

        tiny = _TinyModel()
        cfg = {"model": {"arch": "resnet50", "weights": None, "num_classes": 1}}
        ckpt_path = tmp_path / "no_threshold.pt"
        torch.save({"epoch": 1, "model_state": tiny.state_dict(), "config": cfg}, ckpt_path)
        monkeypatch.setattr(pytorch_impl, "_build_model", lambda _cfg: _TinyModel())

        inf = PyTorchInference(device=torch.device("cpu"))
        inf.load(ckpt_path)
        assert inf._threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# OnnxInference — basic shape, full round-trip lives in test_onnx_export.py
# ---------------------------------------------------------------------------


class TestOnnxInference:
    def test_predict_before_load_raises(self) -> None:
        from backend.app.modules.inference.onnx_impl import OnnxInference

        inf = OnnxInference()
        with pytest.raises(RuntimeError, match="not loaded"):
            inf.predict(np.zeros((3, 224, 224), dtype=np.float32))

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        from backend.app.modules.inference.onnx_impl import OnnxInference

        inf = OnnxInference()
        import onnxruntime

        # ort raises NoSuchFile (subclass of Fail / RuntimeError) for missing files
        with pytest.raises(onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile):
            inf.load(tmp_path / "does_not_exist.onnx")


# ---------------------------------------------------------------------------
# Factory — get_inference_backend
# ---------------------------------------------------------------------------


class TestFactory:
    def test_default_is_pytorch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
        from backend.app.modules.inference import PyTorchInference, get_inference_backend

        result = get_inference_backend()
        assert isinstance(result, PyTorchInference)

    def test_pytorch_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INFERENCE_BACKEND", "pytorch")
        from backend.app.modules.inference import PyTorchInference, get_inference_backend

        result = get_inference_backend()
        assert isinstance(result, PyTorchInference)

    def test_onnx_backend_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INFERENCE_BACKEND", "onnx")
        from backend.app.modules.inference import OnnxInference, get_inference_backend

        result = get_inference_backend()
        assert isinstance(result, OnnxInference)

    def test_unknown_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INFERENCE_BACKEND", "tensorflow")
        from backend.app.modules.inference import get_inference_backend

        with pytest.raises(ValueError, match="INFERENCE_BACKEND"):
            get_inference_backend()

    def test_checkpoint_path_triggers_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When checkpoint_path is passed, load() is called on the returned backend."""
        from backend.app.modules.inference import get_inference_backend, pytorch_impl

        monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
        tiny = _TinyModel()
        ckpt_path = _make_checkpoint(tmp_path, tiny)
        monkeypatch.setattr(pytorch_impl, "_build_model", lambda _cfg: _TinyModel())

        backend = get_inference_backend(checkpoint_path=ckpt_path)
        assert backend._model is not None  # type: ignore[union-attr]
