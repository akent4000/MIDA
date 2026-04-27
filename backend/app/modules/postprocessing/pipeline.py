"""Postprocessing — enriches raw ToolResult with clinical interpretation.

Turns raw model output (probabilities, masks, boxes) into a richer
PostprocessedResult with:
    * Human-readable interpretation string
    * Confidence band ("low" | "medium" | "high") based on probability margin
    * Disclaimer for positive findings (research prototype disclaimer)

The pipeline is intentionally tool-agnostic: it dispatches on the result type
(ClassificationResult, SegmentationResult, …) so new tools are supported
automatically without changes here.

CLI usage (headless test):
    python -m backend.app.modules.postprocessing.pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.modules.ml_tools.base import (
    ClassificationResult,
    DetectionResult,
    SegmentationResult,
    ToolResult,
)

_DISCLAIMER = "Research prototype — not a medical device."


@dataclass
class PostprocessedResult:
    """Enriched result ready for serialisation by the API layer."""

    raw: ToolResult
    interpretation: str = ""
    confidence_band: str = ""  # "low" | "medium" | "high"
    extra: dict[str, Any] = field(default_factory=dict)


class PostprocessingPipeline:
    """Apply interpretation rules to a raw ToolResult.

    Rules for classification:
        prob ≥ 0.8 or prob ≤ 0.2  → high confidence
        0.35 < prob < 0.65         → low confidence (near decision boundary)
        otherwise                  → medium confidence
    """

    def apply(self, result: ToolResult) -> PostprocessedResult:
        if isinstance(result, ClassificationResult):
            return self._apply_classification(result)
        if isinstance(result, SegmentationResult):
            return PostprocessedResult(
                raw=result,
                interpretation="Segmentation mask generated.",
                confidence_band="",
            )
        if isinstance(result, DetectionResult):
            n = len(result.boxes)
            return PostprocessedResult(
                raw=result,
                interpretation=f"{n} region(s) detected.",
                confidence_band="",
            )
        return PostprocessedResult(raw=result)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _apply_classification(self, result: ClassificationResult) -> PostprocessedResult:
        prob = result.prob
        band = self._confidence_band(prob)

        if result.label == 0:
            interpretation = (
                f"No findings detected ({result.label_name}, p={prob:.3f}). "
                f"{_DISCLAIMER}"
            )
        else:
            interpretation = (
                f"Findings detected ({result.label_name}, p={prob:.3f}). "
                f"{_DISCLAIMER}"
            )

        return PostprocessedResult(
            raw=result,
            interpretation=interpretation,
            confidence_band=band,
            extra={"prob": prob, "threshold": result.threshold},
        )

    @staticmethod
    def _confidence_band(prob: float) -> str:
        if prob >= 0.8 or prob <= 0.2:
            return "high"
        if 0.35 < prob < 0.65:
            return "low"
        return "medium"


# ---------------------------------------------------------------------------
# CLI — headless smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sample = ClassificationResult(
        tool_id="pneumonia_classifier_v1",
        prob=0.82,
        label=1,
        label_name="Pneumonia",
        threshold=0.44,
        class_names=["Normal", "Pneumonia"],
    )
    result = PostprocessingPipeline().apply(sample)
    print(f"Interpretation : {result.interpretation}")
    print(f"Confidence     : {result.confidence_band}")
    print(f"Extra          : {result.extra}")
