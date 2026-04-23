"""Binary classification metrics for RSNA Pneumonia.

Aligned with plan §8 quality bar: AUC ROC ≥ 0.90, Sensitivity ≥ 0.85.

Two sensitivity operating points are reported — they answer different questions:
    * sensitivity@youden: picks the threshold maximizing (TPR - FPR), i.e. the
      balanced operating point. This is what we check against the ≥ 0.85 bar.
    * sensitivity@spec=0.85: fixes a target specificity and asks "what TPR do we
      get?". Useful for diagnostic comparison across models.
    * specificity@sens=0.85: mirror of the above — fix a target recall.

Accuracy is reported at threshold=0.5 for quick sanity, not as a primary metric.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


class BinaryMetrics(TypedDict):
    auc: float
    accuracy: float
    threshold_youden: float
    sensitivity_at_youden: float
    specificity_at_youden: float
    sensitivity_at_spec85: float
    specificity_at_sens85: float


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> BinaryMetrics:
    """Compute binary classification metrics from probabilities and labels.

    Args:
        probs: (N,) float array of positive-class probabilities in [0, 1].
        labels: (N,) int array of ground-truth labels in {0, 1}.

    Raises:
        ValueError: if labels contain only one class (AUC undefined).
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if probs.shape != labels.shape:
        raise ValueError(f"probs {probs.shape} and labels {labels.shape} must match")
    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError("labels must contain both classes to compute AUC")

    auc = float(roc_auc_score(labels, probs))
    fpr, tpr, thresholds = roc_curve(labels, probs)

    # Youden's J — argmax(TPR - FPR) picks the balanced operating point.
    j = tpr - fpr
    youden_idx = int(np.argmax(j))
    sens_youden = float(tpr[youden_idx])
    spec_youden = float(1.0 - fpr[youden_idx])
    thr_youden = float(thresholds[youden_idx])

    # sensitivity@spec=0.85 — max TPR where FPR ≤ 0.15. roc_curve returns FPR
    # ascending, so filter first then take the max TPR in the filtered set.
    spec85_mask = fpr <= 0.15
    sens_at_spec85 = float(tpr[spec85_mask].max()) if spec85_mask.any() else 0.0

    # specificity@sens=0.85 — max (1-FPR) where TPR ≥ 0.85.
    sens85_mask = tpr >= 0.85
    spec_at_sens85 = float((1.0 - fpr[sens85_mask]).max()) if sens85_mask.any() else 0.0

    accuracy = float(((probs >= 0.5).astype(np.int64) == labels).mean())

    return BinaryMetrics(
        auc=auc,
        accuracy=accuracy,
        threshold_youden=thr_youden,
        sensitivity_at_youden=sens_youden,
        specificity_at_youden=spec_youden,
        sensitivity_at_spec85=sens_at_spec85,
        specificity_at_sens85=spec_at_sens85,
    )
