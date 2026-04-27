"""5-fold cross-validation training wrapper.

Runs train() from train_baseline.py independently for each fold using
stratified patient-level splits. Each fold saves its own best.pt and
metrics.csv under a <run_name>_fold<k>_<timestamp>/ directory.

Run:
    python -m backend.ml.training.cv_train \
        --config backend/ml/configs/finetune_rsna_densenet121_384.yaml \
        --folds 5

After all folds complete, a summary CSV is written:
    backend/ml/runs/cv_<run_name>_<timestamp>/cv_summary.csv
with per-fold val AUC, sensitivity@youden, and the ensemble mean.

Usage for inference: average softmax probabilities from the 5 best.pt models.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from backend.ml.training.datasets import RSNAClassificationDataset
from backend.ml.training.train_baseline import load_config, resolve_path, train

REPO_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger("cv_train")


def run_cv(cfg: dict[str, Any], config_path: Path, n_folds: int = 5) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cv_run_dir = resolve_path(cfg["output"]["run_dir"]) / f"cv_{cfg['run_name']}_{timestamp}"
    cv_run_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset (all splits combined for CV)
    d = cfg["data"]
    # We use the train split only — val and test stay held-out
    train_ds_full = RSNAClassificationDataset(
        "train",
        splits_path=resolve_path(d["splits_path"]),
        image_dir=resolve_path(d["image_dir"]),
        labels_csv=resolve_path(d["labels_csv"]),
        image_size=d["image_size"],
    )
    labels = train_ds_full.labels()
    indices = np.arange(len(train_ds_full))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg["seed"])
    fold_results: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels), start=1):
        logger.info("=" * 60)
        logger.info(
            "FOLD %d / %d  (train=%d val=%d)", fold_idx, n_folds, len(train_idx), len(val_idx)
        )

        fold_train = Subset(train_ds_full, train_idx.tolist())
        fold_val = Subset(train_ds_full, val_idx.tolist())

        # Compute pos_weight for this fold up front and bake it into the fold
        # config — avoids having to attach a .labels() method to Subset (which
        # can't be pickled for Windows multiprocessing data-loader workers).
        fold_labels = labels[train_idx]
        n_pos = int((fold_labels == 1).sum())
        n_neg = int((fold_labels == 0).sum())
        if n_pos == 0:
            raise ValueError(f"fold {fold_idx} has no positive samples")
        fold_pos_weight = n_neg / n_pos

        fold_cfg = dict(cfg)
        fold_cfg["run_name"] = f"{cfg['run_name']}_fold{fold_idx}"
        # Deep-copy the loss sub-dict so each fold gets its own computed weight
        fold_cfg["loss"] = dict(cfg["loss"])
        fold_cfg["loss"]["pos_weight"] = fold_pos_weight

        fold_paths = train(
            fold_cfg,
            config_path,
            train_ds_override=fold_train,
            val_ds_override=fold_val,
        )

        # Read best epoch metrics from summary.json
        summary_path = fold_paths.run_dir / "summary.json"
        if summary_path.exists():
            import json

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {}

        # Also grab the last row from metrics.csv for detailed metrics
        best_val_auc = summary.get("best_val_auc", float("nan"))
        fold_results.append(
            {
                "fold": fold_idx,
                "best_val_auc": best_val_auc,
                "best_ckpt": str(fold_paths.best_ckpt),
            }
        )
        logger.info("FOLD %d done — best_val_auc=%.4f", fold_idx, best_val_auc)

    # Write CV summary
    summary_csv = cv_run_dir / "cv_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "best_val_auc", "best_ckpt"])
        w.writeheader()
        w.writerows(fold_results)

    aucs = [r["best_val_auc"] for r in fold_results if not np.isnan(r["best_val_auc"])]
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    logger.info("=" * 60)
    logger.info("CV COMPLETE — mean_val_auc=%.4f ± %.4f", mean_auc, std_auc)
    logger.info("summary → %s", summary_csv)
    return cv_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV training for RSNA Pneumonia.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_cv(cfg, args.config, n_folds=args.folds)


if __name__ == "__main__":
    main()
