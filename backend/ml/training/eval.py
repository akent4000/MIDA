"""Test-set evaluation for a trained RSNA Pneumonia classifier.

Run ONLY after selecting the best checkpoint via val AUC — never during
model selection to avoid leaking test signal into architecture decisions.

Run:
    python -m backend.ml.training.eval --checkpoint backend/ml/weights/<run>/best.pt
    python -m backend.ml.training.eval --checkpoint backend/ml/weights/<run>/best.pt --split val

Produces next to the checkpoint file:
    eval_<split>_<timestamp>.json  — full metrics, confusion matrix, quality bar pass/fail

Prints a human-readable summary to stdout.
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
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from backend.ml.training.datasets import build_rsna_datasets
from backend.ml.training.metrics import BinaryMetrics, compute_metrics
from backend.ml.training.model import build_model
from backend.ml.training.train_baseline import resolve_path

REPO_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger("eval")

# Quality bar from plan §8.
QUALITY_BAR = {"auc": 0.90, "sensitivity_at_youden": 0.85}


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: dict[str, Any] = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, ckpt


def build_eval_loader(cfg: dict[str, Any], split: str) -> DataLoader:
    d = cfg["data"]
    train_ds, val_ds, test_ds = build_rsna_datasets(
        splits_path=resolve_path(d["splits_path"]),
        image_dir=resolve_path(d["image_dir"]),
        labels_csv=resolve_path(d["labels_csv"]),
        image_size=d["image_size"],
    )
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
    return DataLoader(
        ds,
        batch_size=d["batch_size"],
        shuffle=False,
        num_workers=d.get("num_workers", 4),
        pin_memory=d.get("pin_memory", True),
        persistent_workers=d.get("num_workers", 4) > 0,
        prefetch_factor=d.get("prefetch_factor") if d.get("num_workers", 4) > 0 else None,
    )


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run inference over a loader.

    If tta=True, also predicts on horizontal flip and averages sigmoid outputs.
    Doubles inference time for a typical +0.002-0.005 AUC boost on CXR tasks.
    """
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ids: list[str] = []
    for batch in loader:
        fmt = torch.channels_last if device.type == "cuda" else torch.contiguous_format
        images = batch["image"].to(device, non_blocking=True, memory_format=fmt)
        logits = model(images)
        probs = torch.sigmoid(logits)
        if tta:
            flipped = torch.flip(images, dims=[-1])
            probs_flip = torch.sigmoid(model(flipped))
            probs = (probs + probs_flip) / 2.0
        all_probs.append(probs.cpu().numpy().ravel())
        all_labels.append(batch["label"].numpy().ravel().astype(np.int64))
        all_ids.extend(batch["patient_id"])
    return np.concatenate(all_probs), np.concatenate(all_labels), all_ids


def quality_bar_check(metrics: BinaryMetrics) -> dict[str, bool]:
    return {k: float(metrics[k]) >= v for k, v in QUALITY_BAR.items()}  # type: ignore[literal-required]


def format_report(
    metrics: BinaryMetrics,
    cm: list[list[int]],
    report: str,
    qb: dict[str, bool],
    split: str,
    ckpt_meta: dict[str, Any],
) -> str:
    lines = [
        f"{'=' * 60}",
        f"  Eval split={split}  checkpoint_epoch={ckpt_meta.get('epoch', '?')}",
        f"  val_auc_at_train={ckpt_meta.get('val_auc', float('nan')):.4f}",
        f"{'=' * 60}",
        "",
        "  Metrics",
        f"    AUC ROC             : {metrics['auc']:.4f}",
        f"    Accuracy (@0.5)     : {metrics['accuracy']:.4f}",
        f"    Threshold (Youden)  : {metrics['threshold_youden']:.4f}",
        f"    Sensitivity @Youden : {metrics['sensitivity_at_youden']:.4f}",
        f"    Specificity @Youden : {metrics['specificity_at_youden']:.4f}",
        f"    Sensitivity @Spec85 : {metrics['sensitivity_at_spec85']:.4f}",
        f"    Specificity @Sens85 : {metrics['specificity_at_sens85']:.4f}",
        "",
        "  Confusion matrix (rows=actual, cols=predicted) @Youden threshold",
        "                   Pred NEG   Pred POS",
        f"    Actual NEG  :   {cm[0][0]:6d}     {cm[0][1]:6d}",
        f"    Actual POS  :   {cm[1][0]:6d}     {cm[1][1]:6d}",
        "",
        "  Classification report",
    ]
    for line in report.splitlines():
        lines.append(f"    {line}")
    lines += [
        "",
        "  Quality bar (plan §8)",
    ]
    for metric, target in QUALITY_BAR.items():
        actual = float(metrics[metric])  # type: ignore[literal-required]
        status = "PASS" if qb[metric] else "FAIL"
        lines.append(f"    {metric:<30} target>={target:.2f}  actual={actual:.4f}  [{status}]")
    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def evaluate(
    checkpoint_path: Path,
    split: str = "test",
    device: torch.device | None = None,
    tta: bool = False,
) -> dict[str, Any]:
    """Full evaluation pipeline. Returns the result dict (also saved to JSON)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "device=%s checkpoint=%s split=%s tta=%s",
        device,
        checkpoint_path.name,
        split,
        tta,
    )

    model, ckpt = load_checkpoint(checkpoint_path, device)
    cfg = ckpt["config"]
    threshold = float(ckpt.get("threshold_youden", 0.5))

    loader = build_eval_loader(cfg, split)
    logger.info("split_size=%d", len(loader.dataset))

    probs, labels, patient_ids = run_inference(model, loader, device, tta=tta)
    metrics = compute_metrics(probs, labels)

    preds = (probs >= threshold).astype(np.int64)
    cm = confusion_matrix(labels, preds).tolist()
    report = classification_report(
        labels, preds, target_names=["no_pneumonia", "pneumonia"], zero_division=0
    )
    qb = quality_bar_check(metrics)

    ckpt_meta = {k: ckpt.get(k) for k in ("epoch", "val_auc", "threshold_youden")}
    print(format_report(metrics, cm, report, qb, split, ckpt_meta))

    result: dict[str, Any] = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "epoch": ckpt.get("epoch"),
        "val_auc_at_train": ckpt.get("val_auc"),
        "threshold_youden": threshold,
        "metrics": dict(metrics),
        "confusion_matrix": cm,
        "classification_report": report,
        "quality_bar": qb,
        "quality_bar_passed": all(qb.values()),
        # per-patient predictions for error analysis
        "patient_ids": patient_ids,
        "patient_probs": probs.tolist(),
        "patient_labels": labels.tolist(),
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = checkpoint_path.parent / f"eval_{split}_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("saved → %s", out_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained RSNA checkpoint on a split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--tta", action="store_true", help="horizontal flip TTA")
    args = parser.parse_args()
    evaluate(args.checkpoint, split=args.split, tta=args.tta)


if __name__ == "__main__":
    main()
