"""Evaluate a trained CheXpert model checkpoint on the validation split.

Run:
    python -m backend.ml.training.eval_chexpert \
        --checkpoint backend/ml/weights/<run>/best.pt

Reports per-class AUC for all 14 labels plus mean AUC for all classes and for the
5 priority classes (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion)
from the original CheXpert paper.

Outputs (written to the checkpoint's parent directory):
    eval_results.json  — per-class AUC + mean AUC
    roc_curves.png     — 14-panel ROC curve plot
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.ml.training.chexpert_dataset import (
    CHEXPERT_LABELS,
    PRIORITY_CLASSES,
    build_chexpert_datasets,
)
from backend.ml.training.model import build_model
from backend.ml.training.train_baseline import resolve_path

REPO_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger("eval_chexpert")


@torch.no_grad()
def run_inference(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs, labels) each of shape (N, 14)."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    fmt = torch.channels_last if device.type == "cuda" else torch.contiguous_format
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True, memory_format=fmt)
        logits = model(images)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch["label"].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Evaluate CheXpert model: per-class AUC + ROC curves."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Override data_dir from checkpoint config."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    logger.info(
        "checkpoint: epoch=%d  val_mean_auc=%.4f",
        ckpt["epoch"],
        ckpt.get("val_mean_auc", 0.0),
    )

    d = cfg["data"]
    data_dir = args.data_dir if args.data_dir is not None else resolve_path(d["data_dir"])

    _, val_ds = build_chexpert_datasets(
        data_dir=data_dir,
        val_fraction=d.get("val_fraction", 0.1),
        image_size=d.get("image_size", 320),
        u_strategy=d.get("u_strategy", "ignore"),
        seed=cfg["seed"],
        frontal_only=d.get("frontal_only", True),
    )
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    logger.info("val set: %d images", len(val_ds))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    probs, labels = run_inference(model, loader, device)

    from sklearn.metrics import roc_auc_score, roc_curve  # local import — optional dep

    results: dict[str, float] = {}
    valid_aucs: list[float] = []
    priority_aucs: list[float] = []

    for i, lbl in enumerate(CHEXPERT_LABELS):
        y = labels[:, i]
        if len(np.unique(y)) < 2:
            logger.info("%-32s  AUC=N/A (single class in val split)", lbl)
            continue
        auc = float(roc_auc_score(y, probs[:, i]))
        results[lbl] = auc
        valid_aucs.append(auc)
        if lbl in PRIORITY_CLASSES:
            priority_aucs.append(auc)
        marker = " ★" if lbl in PRIORITY_CLASSES else ""
        logger.info("%-32s  AUC=%.4f%s", lbl, auc, marker)

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    priority_mean = float(np.mean(priority_aucs)) if priority_aucs else 0.0
    logger.info("─" * 50)
    logger.info("mean AUC (all valid classes):   %.4f", mean_auc)
    logger.info("mean AUC (5 priority ★ classes): %.4f  (target ≥ 0.88)", priority_mean)

    out_dir = args.checkpoint.parent
    summary = {**results, "mean_auc": mean_auc, "priority_mean_auc": priority_mean}
    (out_dir / "eval_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ROC curve grid (3 rows × 5 cols = 15 panels; last panel unused)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 5, figsize=(22, 13))
        axes_flat = axes.flatten()
        for i, lbl in enumerate(CHEXPERT_LABELS):
            ax = axes_flat[i]
            y = labels[:, i]
            if lbl not in results or len(np.unique(y)) < 2:
                ax.set_title(f"{lbl}\nN/A", fontsize=7)
                ax.axis("off")
                continue
            fpr, tpr, _ = roc_curve(y, probs[:, i])
            auc = results[lbl]
            color = "darkgreen" if lbl in PRIORITY_CLASSES else "steelblue"
            ax.plot(fpr, tpr, color=color, lw=1.5)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.8)
            star = " ★" if lbl in PRIORITY_CLASSES else ""
            ax.set_title(f"{lbl}{star}\nAUC={auc:.3f}", fontsize=7, color=color)
            ax.set_xlabel("FPR", fontsize=6)
            ax.set_ylabel("TPR", fontsize=6)
            ax.tick_params(labelsize=6)

        for ax in axes_flat[len(CHEXPERT_LABELS) :]:
            ax.axis("off")

        fig.suptitle(
            f"CheXpert ROC Curves  ·  mean AUC={mean_auc:.4f}  ·  priority AUC={priority_mean:.4f}"
            f"\n(★ = priority class, target ≥ 0.88)",
            fontsize=10,
        )
        plt.tight_layout()
        roc_path = out_dir / "roc_curves.png"
        fig.savefig(roc_path, dpi=120, bbox_inches="tight")
        logger.info("ROC curves saved → %s", roc_path)
    except ImportError:
        logger.warning("matplotlib not installed — skipping ROC plot")


if __name__ == "__main__":
    main()
