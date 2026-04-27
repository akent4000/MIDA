"""Smoke test for eval.py.

Trains a 1-epoch mini-model via the same path as test_train_baseline,
saves a fake checkpoint, then runs evaluate() on the val split.
Skipped when RSNA data isn't present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILE = REPO_ROOT / "backend" / "ml" / "configs" / "splits_rsna_v1.json"
IMG_DIR = REPO_ROOT / "backend" / "ml" / "data" / "rsna" / "stage_2_train_images"
LABELS_CSV = REPO_ROOT / "backend" / "ml" / "data" / "rsna" / "stage_2_train_labels.csv"


def _have_data() -> bool:
    return SPLIT_FILE.exists() and IMG_DIR.exists() and LABELS_CSV.exists()


pytestmark = pytest.mark.skipif(not _have_data(), reason="RSNA data not present")


def _balanced_indices(labels_arr, n_per_class: int) -> list[int]:
    pos = [i for i, v in enumerate(labels_arr) if v == 1][:n_per_class]
    neg = [i for i, v in enumerate(labels_arr) if v == 0][:n_per_class]
    if len(pos) < n_per_class or len(neg) < n_per_class:
        pytest.skip("not enough samples per class")
    return pos + neg


def test_smoke_evaluate(tmp_path: Path) -> None:
    from torch.utils.data import Subset

    from backend.ml.training.datasets import RSNAClassificationDataset
    from backend.ml.training.eval import evaluate
    from backend.ml.training.train_baseline import train

    cfg = {
        "run_name": "smoke_eval",
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
            "epochs": 1,
            "amp": False,
            "grad_clip": 1.0,
            "early_stop_patience": 99,
            "early_stop_metric": "val_auc",
            "log_every_steps": 999,
        },
        "loss": {"name": "bce_with_logits", "pos_weight": "auto"},
        "output": {
            "run_dir": str(tmp_path / "runs"),
            "weights_dir": str(tmp_path / "weights"),
        },
    }
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    full_train = RSNAClassificationDataset("train")
    full_val = RSNAClassificationDataset("val")
    train_idx = _balanced_indices(full_train.labels().tolist(), n_per_class=4)
    val_idx = _balanced_indices(full_val.labels().tolist(), n_per_class=4)
    train_sub = Subset(full_train, train_idx)
    val_sub = Subset(full_val, val_idx)
    train_sub.labels = lambda: full_train.labels()[train_idx]  # type: ignore[attr-defined]

    paths = train(cfg, config_file, train_ds_override=train_sub, val_ds_override=val_sub)
    assert paths.best_ckpt.exists()

    # evaluate() on val split (same tiny subset logic bypassed — uses full val via config)
    # We actually evaluate with the full val split here to test the data-loading path.
    result = evaluate(paths.best_ckpt, split="val", device=torch.device("cpu"))

    # Structural checks — not quality-bar checks on a 1-epoch untrained model.
    assert "metrics" in result
    assert "confusion_matrix" in result
    assert "quality_bar" in result
    assert isinstance(result["quality_bar_passed"], bool)

    cm = result["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2

    metrics = result["metrics"]
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["sensitivity_at_youden"] <= 1.0

    # JSON file was written next to the checkpoint.
    eval_files = list(paths.best_ckpt.parent.glob("eval_val_*.json"))
    assert len(eval_files) == 1
    saved = json.loads(eval_files[0].read_text(encoding="utf-8"))
    assert saved["split"] == "val"
    assert saved["quality_bar_passed"] == result["quality_bar_passed"]
