"""Smoke tests for RSNAClassificationDataset.

Requires the RSNA DICOMs + split config. Skipped on a fresh clone so CI
without the dataset stays green. Once data is present, these tests lock down
the contract: correct length, right tensor shape, binary label, deterministic
val transforms, stochastic train transforms.
"""

from __future__ import annotations

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


def test_length_matches_split() -> None:
    from backend.ml.training.datasets import RSNAClassificationDataset

    split_data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    ds = RSNAClassificationDataset("val")
    assert len(ds) == len(split_data["val"])


def test_item_shape_and_types() -> None:
    import torch

    from backend.ml.training.datasets import RSNAClassificationDataset

    ds = RSNAClassificationDataset("val", image_size=224)
    sample = ds[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].dtype == torch.long
    assert int(sample["label"]) in {0, 1}
    assert isinstance(sample["patient_id"], str)


def test_val_transforms_deterministic() -> None:
    import torch

    from backend.ml.training.datasets import RSNAClassificationDataset

    ds = RSNAClassificationDataset("val")
    a = ds[0]["image"]
    b = ds[0]["image"]
    assert torch.equal(a, b), "val transforms must be deterministic"


def test_train_transforms_stochastic() -> None:
    import torch

    from backend.ml.training.datasets import RSNAClassificationDataset

    ds = RSNAClassificationDataset("train")
    a = ds[0]["image"]
    b = ds[0]["image"]
    assert not torch.equal(a, b), "train transforms must apply random augmentations"


def test_labels_helper_matches_items() -> None:
    from backend.ml.training.datasets import RSNAClassificationDataset

    ds = RSNAClassificationDataset("test")
    labels = ds.labels()
    assert len(labels) == len(ds)
    assert set(labels.tolist()).issubset({0, 1})
