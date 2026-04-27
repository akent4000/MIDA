"""Unit tests for PostprocessingPipeline."""

from __future__ import annotations

import pytest

from backend.app.modules.ml_tools.base import (
    ClassificationResult,
    DetectionResult,
    SegmentationResult,
    ToolResult,
)
from backend.app.modules.postprocessing.pipeline import (
    PostprocessedResult,
    PostprocessingPipeline,
)


def _clf(prob: float, label: int) -> ClassificationResult:
    name = "Pneumonia" if label else "Normal"
    return ClassificationResult(
        tool_id="pneumonia_classifier_v1",
        prob=prob,
        label=label,
        label_name=name,
        threshold=0.44,
        class_names=["Normal", "Pneumonia"],
    )


class TestPostprocessingClassification:
    def test_returns_postprocessed_result(self) -> None:
        result = PostprocessingPipeline().apply(_clf(0.82, 1))
        assert isinstance(result, PostprocessedResult)

    def test_raw_result_preserved(self) -> None:
        clf = _clf(0.82, 1)
        out = PostprocessingPipeline().apply(clf)
        assert out.raw is clf

    def test_positive_result_interpretation(self) -> None:
        out = PostprocessingPipeline().apply(_clf(0.82, 1))
        assert "Pneumonia" in out.interpretation
        assert "not a medical device" in out.interpretation.lower()

    def test_negative_result_interpretation(self) -> None:
        out = PostprocessingPipeline().apply(_clf(0.05, 0))
        assert "No findings" in out.interpretation

    def test_high_confidence_band_high_prob(self) -> None:
        out = PostprocessingPipeline().apply(_clf(0.92, 1))
        assert out.confidence_band == "high"

    def test_high_confidence_band_low_prob(self) -> None:
        out = PostprocessingPipeline().apply(_clf(0.08, 0))
        assert out.confidence_band == "high"

    def test_low_confidence_band_near_boundary(self) -> None:
        out = PostprocessingPipeline().apply(_clf(0.50, 1))
        assert out.confidence_band == "low"

    def test_medium_confidence_band(self) -> None:
        out = PostprocessingPipeline().apply(_clf(0.70, 1))
        assert out.confidence_band == "medium"

    def test_extra_contains_prob_and_threshold(self) -> None:
        out = PostprocessingPipeline().apply(_clf(0.82, 1))
        assert "prob" in out.extra
        assert "threshold" in out.extra
        assert out.extra["prob"] == pytest.approx(0.82)


class TestPostprocessingOtherTypes:
    def test_segmentation_result(self) -> None:

        r = SegmentationResult(tool_id="t1", class_names=["bg", "tumour"])
        out = PostprocessingPipeline().apply(r)
        assert isinstance(out, PostprocessedResult)
        assert "egmentation" in out.interpretation

    def test_detection_result_count_in_interpretation(self) -> None:
        boxes = [
            {"box": [0, 0, 10, 10], "label": 1, "score": 0.9},
            {"box": [20, 20, 30, 30], "label": 1, "score": 0.8},
        ]
        r = DetectionResult(tool_id="det", boxes=boxes, class_names=["bg", "nodule"])
        out = PostprocessingPipeline().apply(r)
        assert "2" in out.interpretation

    def test_base_tool_result_passthrough(self) -> None:
        r = ToolResult(tool_id="unknown")
        out = PostprocessingPipeline().apply(r)
        assert out.raw is r
        assert out.interpretation == ""
