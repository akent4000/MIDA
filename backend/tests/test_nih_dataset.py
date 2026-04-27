"""Tests for NIH ChestX-ray14 dataset and related modules.

Tests are split into two groups:
  - No-data tests: run on any clone (mock data, no NIH files needed)
  - Data tests: skipped when NIH data is absent
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
NIH_DIR = REPO_ROOT / "backend" / "ml" / "data" / "nih"


def _have_nih() -> bool:
    return (NIH_DIR / "Data_Entry_2017.csv").exists() and (NIH_DIR / "train_val_list.txt").exists()


# ---------------------------------------------------------------------------
# model.py — DenseNet-121 + backbone_checkpoint (no data needed)
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_densenet121_binary(self) -> None:
        from backend.ml.training.model import build_model

        cfg = {"model": {"arch": "densenet121", "weights": None, "num_classes": 1}}
        model = build_model(cfg)
        import torch

        x = torch.zeros(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1)

    def test_densenet121_14class(self) -> None:
        from backend.ml.training.model import build_model

        cfg = {"model": {"arch": "densenet121", "weights": None, "num_classes": 14}}
        model = build_model(cfg)
        x = torch.zeros(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 14)

    def test_backbone_checkpoint_loads_partial(self, tmp_path: Path) -> None:
        """backbone_checkpoint replaces ImageNet weights but not the head."""
        from backend.ml.training.model import build_model

        # Build a 14-class pretrain model and save it as a backbone checkpoint
        cfg_pretrain = {"model": {"arch": "densenet121", "weights": None, "num_classes": 14}}
        pretrain_model = build_model(cfg_pretrain)
        ckpt_path = tmp_path / "backbone.pt"
        torch.save({"model_state": pretrain_model.state_dict()}, ckpt_path)

        # Load as 1-class fine-tune model — head shape mismatch must be handled
        cfg_finetune = {
            "model": {
                "arch": "densenet121",
                "weights": None,
                "num_classes": 1,
                "backbone_checkpoint": str(ckpt_path),
            }
        }
        model = build_model(cfg_finetune)
        x = torch.zeros(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1)

    def test_unsupported_arch_raises(self) -> None:
        from backend.ml.training.model import build_model

        with pytest.raises(ValueError, match="Unsupported arch"):
            build_model({"model": {"arch": "vgg16", "weights": None, "num_classes": 1}})


# ---------------------------------------------------------------------------
# transforms.py — CLAHETransform (no data needed)
# ---------------------------------------------------------------------------


class TestCLAHETransform:
    def test_shape_preserved(self) -> None:
        from backend.ml.training.transforms import CLAHETransform

        t = CLAHETransform()
        arr = np.random.rand(1, 64, 64).astype(np.float32)
        out = t(arr)
        assert np.asarray(out).shape == (1, 64, 64)

    def test_dtype_float32(self) -> None:
        from backend.ml.training.transforms import CLAHETransform

        t = CLAHETransform()
        arr = np.random.rand(1, 64, 64).astype(np.float32)
        out = t(arr)
        assert np.asarray(out).dtype == np.float32

    def test_output_range(self) -> None:
        from backend.ml.training.transforms import CLAHETransform

        t = CLAHETransform()
        arr = np.random.rand(1, 64, 64).astype(np.float32)
        out = np.asarray(t(arr))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_works_with_metatensor(self) -> None:
        """CLAHETransform must handle MONAI MetaTensor (torch.Tensor subclass)."""
        from monai.data import MetaTensor

        from backend.ml.training.transforms import CLAHETransform

        t = CLAHETransform()
        arr = MetaTensor(torch.rand(1, 64, 64))
        out = t(arr)
        assert out is not None

    def test_deterministic(self) -> None:
        from backend.ml.training.transforms import CLAHETransform

        t = CLAHETransform()
        arr = np.random.rand(1, 64, 64).astype(np.float32)
        a = np.asarray(t(arr))
        b = np.asarray(t(arr))
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# nih_dataset.py — label parsing (no real images needed)
# ---------------------------------------------------------------------------


class TestNIHLabelParsing:
    def _make_df(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "Image Index": ["img1.png", "img2.png", "img3.png"],
                "Finding Labels": ["Pneumonia|Effusion", "No Finding", "Atelectasis"],
            }
        )

    def test_pneumonia_label_set(self, tmp_path: Path) -> None:
        from backend.ml.training.nih_dataset import LABEL_TO_IDX, NIHChestXray14Dataset

        df = self._make_df()
        # Provide a fake image_index with mock paths
        fake_index = {n: tmp_path / n for n in ["img1.png", "img2.png", "img3.png"]}
        ds = NIHChestXray14Dataset(["img1.png"], df, tmp_path, image_index=fake_index)
        lbl = ds._labels["img1.png"]
        assert lbl[LABEL_TO_IDX["Pneumonia"]] == 1.0
        assert lbl[LABEL_TO_IDX["Effusion"]] == 1.0
        assert lbl[LABEL_TO_IDX["Atelectasis"]] == 0.0

    def test_no_finding_all_zeros(self, tmp_path: Path) -> None:
        from backend.ml.training.nih_dataset import NIHChestXray14Dataset

        df = self._make_df()
        fake_index = {n: tmp_path / n for n in ["img1.png", "img2.png", "img3.png"]}
        ds = NIHChestXray14Dataset(["img2.png"], df, tmp_path, image_index=fake_index)
        assert ds._labels["img2.png"].sum() == 0.0

    def test_label_matrix_shape(self, tmp_path: Path) -> None:
        from backend.ml.training.nih_dataset import NUM_CLASSES, NIHChestXray14Dataset

        df = self._make_df()
        fake_index = {n: tmp_path / n for n in ["img1.png", "img2.png", "img3.png"]}
        ds = NIHChestXray14Dataset(
            ["img1.png", "img2.png", "img3.png"], df, tmp_path, image_index=fake_index
        )
        mat = ds.label_matrix()
        assert mat.shape == (3, NUM_CLASSES)
        assert mat.dtype == np.float32


# ---------------------------------------------------------------------------
# nih_dataset.py — with real data (skipped when NIH absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _have_nih(), reason="NIH data not present")
class TestNIHDatasetReal:
    def test_build_nih_datasets_sizes(self) -> None:
        from backend.ml.training.nih_dataset import build_nih_datasets

        train_ds, val_ds = build_nih_datasets(val_fraction=0.1)
        # train_val_list has ~86k entries; 10% val → ~8.6k val, ~77k train
        assert len(train_ds) > 70_000
        assert len(val_ds) > 5_000
        assert len(train_ds) + len(val_ds) > 85_000

    def test_item_shape(self) -> None:
        from backend.ml.training.nih_dataset import build_nih_datasets

        train_ds, _ = build_nih_datasets(val_fraction=0.1)
        sample = train_ds[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["image"].dtype == torch.float32
        assert sample["label"].shape == (14,)
        assert sample["label"].dtype == torch.float32

    def test_labels_binary(self) -> None:
        from backend.ml.training.nih_dataset import build_nih_datasets

        train_ds, _ = build_nih_datasets(val_fraction=0.1)
        mat = train_ds.label_matrix()
        assert set(np.unique(mat)).issubset({0.0, 1.0})
