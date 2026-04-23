"""Smoke test for the baseline training loop.

Verifies the script runs end-to-end on a tiny Subset in 2 epochs — catches
interface drift between datasets / metrics / scheduler / checkpointing
without needing a full dataset run.

Skipped when RSNA data isn't present so CI on fresh clones stays green.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILE = REPO_ROOT / "backend" / "ml" / "configs" / "splits_rsna_v1.json"
IMG_DIR = REPO_ROOT / "backend" / "ml" / "data" / "rsna" / "stage_2_train_images"
LABELS_CSV = REPO_ROOT / "backend" / "ml" / "data" / "rsna" / "stage_2_train_labels.csv"


def _have_data() -> bool:
    return SPLIT_FILE.exists() and IMG_DIR.exists() and LABELS_CSV.exists()


pytestmark = pytest.mark.skipif(not _have_data(), reason="RSNA data not present")


def _pick_balanced_indices(labels, n_per_class: int) -> list[int]:
    pos = [i for i, lbl in enumerate(labels) if lbl == 1][:n_per_class]
    neg = [i for i, lbl in enumerate(labels) if lbl == 0][:n_per_class]
    if len(pos) < n_per_class or len(neg) < n_per_class:
        pytest.skip("not enough samples per class for balanced smoke subset")
    return pos + neg


def test_smoke_train_two_epochs(tmp_path: Path) -> None:
    from torch.utils.data import Subset

    from backend.ml.training.datasets import RSNAClassificationDataset
    from backend.ml.training.train_baseline import train

    # Tiny balanced subsets so compute_metrics never gets a single-class val set.
    full_train = RSNAClassificationDataset("train")
    full_val = RSNAClassificationDataset("val")
    train_idx = _pick_balanced_indices(full_train.labels().tolist(), n_per_class=4)
    val_idx = _pick_balanced_indices(full_val.labels().tolist(), n_per_class=4)
    train_sub = Subset(full_train, train_idx)
    val_sub = Subset(full_val, val_idx)
    # Override labels() on the Subset-wrapped train_ds for pos_weight computation.
    train_sub.labels = lambda: full_train.labels()[train_idx]  # type: ignore[attr-defined]

    cfg = {
        "run_name": "smoke",
        "seed": 0,
        "model": {"arch": "resnet50", "weights": None, "num_classes": 1},
        "data": {
            "splits_path": "backend/ml/configs/splits_rsna_v1.json",
            "image_dir": "backend/ml/data/rsna/stage_2_train_images",
            "labels_csv": "backend/ml/data/rsna/stage_2_train_labels.csv",
            "image_size": 224,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        },
        "optim": {
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "warmup_epochs": 0,
        },
        "train": {
            "epochs": 2,
            "amp": False,
            "grad_clip": 1.0,
            "early_stop_patience": 99,
            "early_stop_metric": "val_auc",
            "log_every_steps": 1,
        },
        "loss": {"name": "bce_with_logits", "pos_weight": "auto"},
        "output": {
            "run_dir": str(tmp_path / "runs"),
            "weights_dir": str(tmp_path / "weights"),
        },
    }
    # train() needs a config file path to snapshot; write the YAML out.
    import yaml

    config_file = tmp_path / "smoke.yaml"
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    paths = train(cfg, config_file, train_ds_override=train_sub, val_ds_override=val_sub)

    # Artifacts exist
    assert paths.best_ckpt.exists(), "best.pt not written"
    assert paths.last_ckpt.exists(), "last.pt not written"
    assert paths.metrics_csv.exists(), "metrics.csv not written"
    assert paths.config_snapshot.exists(), "config snapshot not copied"
    assert (paths.run_dir / "summary.json").exists(), "summary.json not written"

    # CSV has a header + exactly 2 data rows
    with paths.metrics_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2, f"expected 2 epoch rows, got {len(rows)}"
    assert {"epoch", "val_auc", "train_loss"}.issubset(rows[0].keys())

    # summary.json has best_val_auc as a finite float
    summary = json.loads((paths.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert "best_val_auc" in summary
    assert summary["stopped_at_epoch"] == 2
