"""Unit tests for the ml_tools registry and pluggable tool contracts.

All tests run without model weights or DICOM data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.app.modules.ml_tools.base import (
    ClassificationResult,
    DetectionResult,
    MLTool,
    Modality,
    ModelInfo,
    SegmentationResult,
    TaskType,
    ToolResult,
)
from backend.app.modules.ml_tools.registry import (
    ToolNotFoundError,
    ToolNotLoadedError,
    ToolRegistry,
    build_registry,
)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TestEnums:
    def test_task_types(self) -> None:
        assert TaskType.CLASSIFICATION == "classification"
        assert TaskType.SEGMENTATION == "segmentation"
        assert TaskType.DETECTION == "detection"

    def test_modalities(self) -> None:
        assert Modality.XRAY == "xray"
        assert Modality.CT == "ct"
        assert Modality.MRI == "mri"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


class TestResultTypes:
    def test_classification_result_fields(self) -> None:
        r = ClassificationResult(
            tool_id="t1",
            prob=0.72,
            label=1,
            label_name="Pneumonia",
            threshold=0.44,
            class_names=["Normal", "Pneumonia"],
        )
        assert r.prob == pytest.approx(0.72)
        assert r.label == 1
        assert r.label_name == "Pneumonia"
        assert r.class_names == ["Normal", "Pneumonia"]
        assert isinstance(r, ToolResult)

    def test_segmentation_result_has_mask(self) -> None:
        mask = np.zeros((128, 128), dtype=np.float32)
        r = SegmentationResult(tool_id="t2", mask=mask, class_names=["bg", "tumour"])
        assert r.mask is not None
        assert r.mask.shape == (128, 128)

    def test_detection_result_has_boxes(self) -> None:
        r = DetectionResult(
            tool_id="t3",
            boxes=[{"box": [0, 0, 50, 50], "label": 1, "score": 0.9}],
            class_names=["bg", "nodule"],
        )
        assert len(r.boxes) == 1
        assert r.boxes[0]["score"] == pytest.approx(0.9)

    def test_tool_result_metadata_defaults_empty(self) -> None:
        r = ToolResult(tool_id="t4")
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# MLTool ABC
# ---------------------------------------------------------------------------


class _FakeTool(MLTool):
    TOOL_ID = "fake_tool_v1"
    _info = ModelInfo(
        tool_id=TOOL_ID,
        name="Fake",
        version="0.0",
        description="Test stub",
        modality=Modality.CT,
        task_type=TaskType.CLASSIFICATION,
        input_shape=(1, 64, 64),
        class_names=["a", "b"],
    )

    def __init__(self) -> None:
        self._loaded = False

    @property
    def info(self) -> ModelInfo:
        return self._info

    def load(self, weights_path: Path) -> None:
        self._loaded = True

    def predict(self, image: np.ndarray) -> ToolResult:
        return ClassificationResult(
            tool_id=self.TOOL_ID,
            prob=0.5,
            label=0,
            label_name="a",
            threshold=0.5,
            class_names=["a", "b"],
        )

    def get_preprocessing_config(self) -> dict[str, Any]:
        return {"resize": [64, 64], "channels": 1}


class TestMLToolABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            MLTool()  # type: ignore[abstract]

    def test_is_loaded_false_before_load(self) -> None:
        t = _FakeTool()
        assert not t.is_loaded()

    def test_is_loaded_true_after_load(self, tmp_path: Path) -> None:
        t = _FakeTool()
        t.load(tmp_path / "dummy.pt")
        assert t.is_loaded()

    def test_gradcam_target_layer_default_none(self) -> None:
        t = _FakeTool()
        assert t.get_gradcam_target_layer() is None


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_list_available(self) -> None:
        reg = ToolRegistry()
        reg.register_class("fake_tool_v1", _FakeTool)
        assert "fake_tool_v1" in reg.list_available()

    def test_get_before_load_raises(self) -> None:
        reg = ToolRegistry()
        reg.register_class("fake_tool_v1", _FakeTool)
        with pytest.raises(ToolNotLoadedError):
            reg.get("fake_tool_v1")

    def test_load_unknown_raises(self, tmp_path: Path) -> None:
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            reg.load("nonexistent_tool", tmp_path / "x.pt")

    def test_load_and_get(self, tmp_path: Path) -> None:
        reg = ToolRegistry()
        reg.register_class("fake_tool_v1", _FakeTool)
        reg.load("fake_tool_v1", tmp_path / "dummy.pt")
        tool = reg.get("fake_tool_v1")
        assert isinstance(tool, _FakeTool)
        assert tool.is_loaded()

    def test_is_loaded_flag(self, tmp_path: Path) -> None:
        reg = ToolRegistry()
        reg.register_class("fake_tool_v1", _FakeTool)
        assert not reg.is_loaded("fake_tool_v1")
        reg.load("fake_tool_v1", tmp_path / "dummy.pt")
        assert reg.is_loaded("fake_tool_v1")

    def test_unload_removes_instance(self, tmp_path: Path) -> None:
        reg = ToolRegistry()
        reg.register_class("fake_tool_v1", _FakeTool)
        reg.load("fake_tool_v1", tmp_path / "dummy.pt")
        reg.unload("fake_tool_v1")
        assert not reg.is_loaded("fake_tool_v1")

    def test_list_loaded_returns_model_infos(self, tmp_path: Path) -> None:
        reg = ToolRegistry()
        reg.register_class("fake_tool_v1", _FakeTool)
        reg.load("fake_tool_v1", tmp_path / "dummy.pt")
        infos = reg.list_loaded()
        assert len(infos) == 1
        assert infos[0].tool_id == "fake_tool_v1"

    def test_predict_dispatches_to_tool(self, tmp_path: Path) -> None:
        reg = ToolRegistry()
        reg.register_class("fake_tool_v1", _FakeTool)
        reg.load("fake_tool_v1", tmp_path / "dummy.pt")
        image = np.zeros((1, 64, 64), dtype=np.float32)
        result = reg.predict("fake_tool_v1", image)
        assert isinstance(result, ClassificationResult)


# ---------------------------------------------------------------------------
# build_registry factory
# ---------------------------------------------------------------------------


class TestBuildRegistry:
    def test_pneumonia_tool_registered(self) -> None:
        reg = build_registry()
        assert "pneumonia_classifier_v1" in reg.list_available()

    def test_is_registered_not_loaded_by_default(self) -> None:
        reg = build_registry()
        assert not reg.is_loaded("pneumonia_classifier_v1")


# ---------------------------------------------------------------------------
# PneumoniaTool (without loading weights)
# ---------------------------------------------------------------------------


class TestPneumoniaTool:
    def test_tool_id(self) -> None:
        from backend.app.modules.ml_tools.pneumonia.tool import PneumoniaTool

        assert PneumoniaTool.TOOL_ID == "pneumonia_classifier_v1"

    def test_info_fields(self) -> None:
        from backend.app.modules.ml_tools.pneumonia.tool import PneumoniaTool

        t = PneumoniaTool()
        info = t.info
        assert info.modality == Modality.XRAY
        assert info.task_type == TaskType.CLASSIFICATION
        assert info.input_shape == (3, 384, 384)
        assert "Normal" in info.class_names
        assert "Pneumonia" in info.class_names

    def test_preprocessing_config_shape(self) -> None:
        from backend.app.modules.ml_tools.pneumonia.tool import PneumoniaTool

        cfg = PneumoniaTool().get_preprocessing_config()
        assert cfg["resize"] == [384, 384]
        assert cfg["channels"] == 3
        assert "clahe" in cfg
        assert "normalize" in cfg
        assert len(cfg["normalize"]["mean"]) == 3

    def test_gradcam_target_layer(self) -> None:
        from backend.app.modules.ml_tools.pneumonia.tool import PneumoniaTool

        t = PneumoniaTool()
        assert t.get_gradcam_target_layer() == "features.denseblock4"

    def test_predict_without_load_raises(self) -> None:
        from backend.app.modules.ml_tools.pneumonia.tool import PneumoniaTool

        t = PneumoniaTool()
        with pytest.raises(RuntimeError, match="not loaded"):
            t.predict(np.zeros((3, 384, 384), dtype=np.float32))

    def test_predict_with_mocked_inference(self) -> None:
        from backend.app.modules.inference.base import Prediction
        from backend.app.modules.ml_tools.pneumonia.tool import PneumoniaTool

        t = PneumoniaTool()
        mock_inf = MagicMock()
        mock_inf.predict.return_value = Prediction(prob=0.82, label=1, threshold=0.44)
        t._single = mock_inf
        t._loaded = True

        result = t.predict(np.zeros((3, 384, 384), dtype=np.float32))

        assert isinstance(result, ClassificationResult)
        assert result.prob == pytest.approx(0.82)
        assert result.label == 1
        assert result.label_name == "Pneumonia"
        assert result.tool_id == "pneumonia_classifier_v1"
