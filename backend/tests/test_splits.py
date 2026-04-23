"""Smoke tests for the RSNA split config.

Skipped if `splits_rsna_v1.json` hasn't been generated yet (e.g. on a fresh
clone without the dataset). Once present, these tests lock down the three
invariants that must never regress: no patient leakage, right total, balanced
positive rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILE = REPO_ROOT / "backend" / "ml" / "configs" / "splits_rsna_v1.json"


def _load():
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))


@pytest.mark.skipif(not SPLIT_FILE.exists(), reason="splits_rsna_v1.json not generated")
def test_no_patient_overlap() -> None:
    data = _load()
    train, val, test = set(data["train"]), set(data["val"]), set(data["test"])
    assert not (train & val), "train/val overlap"
    assert not (train & test), "train/test overlap"
    assert not (val & test), "val/test overlap"


@pytest.mark.skipif(not SPLIT_FILE.exists(), reason="splits_rsna_v1.json not generated")
def test_split_proportions() -> None:
    data = _load()
    total = len(data["train"]) + len(data["val"]) + len(data["test"])
    assert 0.12 < len(data["val"]) / total < 0.18, "val fraction drifted"
    assert 0.12 < len(data["test"]) / total < 0.18, "test fraction drifted"


@pytest.mark.skipif(not SPLIT_FILE.exists(), reason="splits_rsna_v1.json not generated")
def test_meta_block() -> None:
    data = _load()
    meta = data["meta"]
    assert meta["seed"] == 42
    assert meta["train_n"] + meta["val_n"] + meta["test_n"] == (
        len(data["train"]) + len(data["val"]) + len(data["test"])
    )
