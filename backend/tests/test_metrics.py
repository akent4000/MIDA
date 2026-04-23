"""Unit tests for compute_metrics.

No DICOMs, no torch — pure numpy, so this runs on any clone regardless of
dataset availability. Each test targets one property of the metric definitions.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.ml.training.metrics import compute_metrics


def test_perfect_separator() -> None:
    # Probs identical to labels → perfect ranking.
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = labels.astype(np.float64)
    m = compute_metrics(probs, labels)
    assert m["auc"] == pytest.approx(1.0)
    assert m["sensitivity_at_youden"] == pytest.approx(1.0)
    assert m["specificity_at_youden"] == pytest.approx(1.0)
    assert m["sensitivity_at_spec85"] == pytest.approx(1.0)
    assert m["specificity_at_sens85"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)


def test_random_scores_near_chance() -> None:
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=5000)
    probs = rng.random(5000)
    m = compute_metrics(probs, labels)
    # Tolerance wide enough for sampling noise, tight enough to catch bugs.
    assert 0.45 <= m["auc"] <= 0.55


def test_constant_probs_undefined_youden_ok() -> None:
    # All-same probs → roc_curve still returns something; AUC = 0.5.
    labels = np.array([0, 0, 1, 1])
    probs = np.full(4, 0.5)
    m = compute_metrics(probs, labels)
    assert m["auc"] == pytest.approx(0.5)


def test_single_class_raises() -> None:
    labels = np.zeros(10, dtype=np.int64)
    probs = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="both classes"):
        compute_metrics(probs, labels)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match"):
        compute_metrics(np.zeros(5), np.zeros(6))


def test_sens_at_spec85_respects_constraint() -> None:
    # Hand-crafted example where one clear threshold hits FPR = 0.1 (spec 0.9).
    # Positives score 0.9, negatives mostly 0.1, one negative at 0.8.
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    probs = np.array([0.9, 0.9, 0.9, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = compute_metrics(probs, labels)
    # FPR=1/10=0.1 ≤ 0.15 with all positives captured → TPR = 1.0.
    assert m["sensitivity_at_spec85"] == pytest.approx(1.0)


def test_threshold_youden_in_prob_range() -> None:
    # Youden threshold comes from roc_curve thresholds; the first element can be
    # inf by sklearn convention — just assert it's a finite float when non-degenerate.
    rng = np.random.default_rng(0)
    labels = (rng.random(200) > 0.7).astype(np.int64)
    probs = rng.random(200)
    m = compute_metrics(probs, labels)
    assert np.isfinite(m["threshold_youden"])
