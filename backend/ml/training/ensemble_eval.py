"""Ensemble evaluation — average sigmoid outputs of multiple checkpoints.

Typical use: evaluate a 5-fold CV ensemble on the RSNA test split.

    python -m backend.ml.training.ensemble_eval \
        --checkpoints backend/ml/weights/<run>_fold1_*/best.pt \
                      backend/ml/weights/<run>_fold2_*/best.pt \
                      ... \
        --split test

Averaging is done on sigmoid probabilities (soft voting). The ensemble
threshold is the mean of per-fold Youden thresholds, so behaviour stays
calibrated across single-model and ensemble inference paths.

All fold checkpoints must use the same architecture and input size.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from backend.ml.training.eval import build_eval_loader, load_checkpoint, quality_bar_check
from backend.ml.training.metrics import compute_metrics

logger = logging.getLogger("ensemble_eval")


@torch.no_grad()
def predict_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    fmt = torch.channels_last if device.type == "cuda" else torch.contiguous_format
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ids: list[str] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True, memory_format=fmt)
        probs = torch.sigmoid(model(images))
        if tta:
            probs_flip = torch.sigmoid(model(torch.flip(images, dims=[-1])))
            probs = (probs + probs_flip) / 2.0
        all_probs.append(probs.cpu().numpy().ravel())
        all_labels.append(batch["label"].numpy().ravel().astype(np.int64))
        all_ids.extend(batch["patient_id"])
    return np.concatenate(all_probs), np.concatenate(all_labels), all_ids


def ensemble_evaluate(
    checkpoint_paths: list[Path],
    split: str = "test",
    device: torch.device | None = None,
    tta: bool = False,
) -> dict[str, Any]:
    """Soft-voting ensemble. Returns a result dict and writes JSON next to checkpoint[0]."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(checkpoint_paths) < 2:
        raise ValueError(
            f"ensemble_evaluate expects ≥2 checkpoints; got {len(checkpoint_paths)}. "
            "Use eval.py for single-model evaluation."
        )
    logger.info(
        "device=%s n_checkpoints=%d split=%s tta=%s",
        device,
        len(checkpoint_paths),
        split,
        tta,
    )

    # All fold checkpoints must share data config (image_size, splits) — load once.
    first_model, first_ckpt = load_checkpoint(checkpoint_paths[0], device)
    cfg = first_ckpt["config"]
    loader = build_eval_loader(cfg, split)
    logger.info("split_size=%d", len(loader.dataset))

    fold_probs: list[np.ndarray] = []
    fold_thresholds: list[float] = []
    shared_labels: np.ndarray | None = None
    shared_ids: list[str] | None = None

    for i, ckpt_path in enumerate(checkpoint_paths):
        if i == 0:
            model, ckpt = first_model, first_ckpt
        else:
            model, ckpt = load_checkpoint(ckpt_path, device)
        probs, labels, ids = predict_all(model, loader, device, tta=tta)
        fold_probs.append(probs)
        fold_thresholds.append(float(ckpt.get("threshold_youden", 0.5)))
        if shared_labels is None:
            shared_labels, shared_ids = labels, ids
        logger.info(
            "fold %d/%d done — probs.shape=%s threshold=%.4f",
            i + 1,
            len(checkpoint_paths),
            probs.shape,
            fold_thresholds[-1],
        )
        # free fold model before loading next to keep VRAM bounded
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Soft voting: average probabilities across folds
    mean_probs = np.mean(np.stack(fold_probs), axis=0)
    mean_threshold = float(np.mean(fold_thresholds))
    assert shared_labels is not None and shared_ids is not None

    metrics = compute_metrics(mean_probs, shared_labels)
    preds = (mean_probs >= mean_threshold).astype(np.int64)
    cm = confusion_matrix(shared_labels, preds).tolist()
    report = classification_report(
        shared_labels, preds, target_names=["no_pneumonia", "pneumonia"], zero_division=0
    )
    qb = quality_bar_check(metrics)

    # Also compute per-fold mean AUC as a diagnostic
    per_fold_aucs = [float(compute_metrics(p, shared_labels)["auc"]) for p in fold_probs]

    result: dict[str, Any] = {
        "split": split,
        "n_folds": len(checkpoint_paths),
        "checkpoints": [str(p) for p in checkpoint_paths],
        "ensemble_threshold": mean_threshold,
        "per_fold_thresholds": fold_thresholds,
        "per_fold_aucs": per_fold_aucs,
        "ensemble_metrics": dict(metrics),
        "confusion_matrix": cm,
        "classification_report": report,
        "quality_bar": qb,
        "quality_bar_passed": all(qb.values()),
        "patient_ids": shared_ids,
        "ensemble_probs": mean_probs.tolist(),
        "patient_labels": shared_labels.tolist(),
    }

    # Print summary
    print("=" * 60)
    print(f"  Ensemble eval split={split}  n_folds={len(checkpoint_paths)}")
    print("=" * 60)
    print(f"  Per-fold AUCs : {[f'{a:.4f}' for a in per_fold_aucs]}")
    print(f"  Ensemble AUC  : {metrics['auc']:.4f}")
    print(f"  Sens@Youden   : {metrics['sensitivity_at_youden']:.4f}")
    print(f"  Spec@Youden   : {metrics['specificity_at_youden']:.4f}")
    print(f"  Sens@Spec85   : {metrics['sensitivity_at_spec85']:.4f}")
    print(f"  Threshold     : {mean_threshold:.4f}")
    print(f"  Quality bar   : {'PASS' if result['quality_bar_passed'] else 'FAIL'}")
    print("=" * 60)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = checkpoint_paths[0].parent.parent / f"ensemble_eval_{split}_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("saved → %s", out_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble evaluation on RSNA split.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--tta", action="store_true", help="horizontal flip TTA per fold")
    args = parser.parse_args()
    ensemble_evaluate(args.checkpoints, split=args.split, tta=args.tta)


if __name__ == "__main__":
    main()
