"""RSNA Pneumonia classification dataset.

Wraps the DICOM directory + `splits_rsna_v1.json` into a `torch.utils.data.Dataset`
ready for ImageNet-pretrained backbones (ResNet / EfficientNet).

Per-patient binary label: `Target = max(Target across bboxes)`. A patient with
any bbox becomes positive; an image with no findings is negative. Per-box
supervision is discarded — this is the classification task, not detection.

DICOM handling:
    * pydicom.dcmread, then `apply_voi_lut` so the window/level preference
      baked into the DICOM by the radiologist is honored.
    * Invert when PhotometricInterpretation == "MONOCHROME1".
    * Per-image min-max scaling into [0, 1] — robust to the mix of 12/16-bit
      sources in RSNA without hard-coding a bit depth.

Transform pipeline (MONAI):
    train: EnsureChannelFirst → RandFlip → RandAffine → RandAdjustContrast
           → RandGaussianNoise → Resize → RepeatChannel(3) → Normalize(ImageNet)
    val / test: EnsureChannelFirst → Resize → RepeatChannel(3) → Normalize(ImageNet)

Usage:
    from backend.ml.training.datasets import build_rsna_datasets
    train_ds, val_ds, test_ds = build_rsna_datasets(image_size=224)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pydicom
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    NormalizeIntensity,
    RandAdjustContrast,
    RandAffine,
    RandFlip,
    RandGaussianNoise,
    RepeatChannel,
    Resize,
    Transform,
)
from pydicom.pixels import apply_voi_lut
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "backend" / "ml" / "data" / "rsna"
CONFIG_DIR = REPO_ROOT / "backend" / "ml" / "configs"

DEFAULT_IMG_DIR = DATA_DIR / "stage_2_train_images"
DEFAULT_LABELS_CSV = DATA_DIR / "stage_2_train_labels.csv"
DEFAULT_SPLITS = CONFIG_DIR / "splits_rsna_v1.json"

# ImageNet stats — expected by torchvision's pretrained ResNet/EfficientNet.
# Single-channel CXRs get repeated across 3 channels so the same mean/std applies.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

Split = Literal["train", "val", "test"]


def load_dicom_array(path: Path) -> np.ndarray:
    """Read a DICOM file → float32 ndarray in [0, 1], shape (H, W)."""
    ds = pydicom.dcmread(str(path))
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return arr


def build_transforms(split: Split, image_size: int = 224) -> Callable:
    """MONAI transform pipeline. Input: (H, W) float32 array in [0, 1]."""
    spatial = (image_size, image_size)
    ops: list[Transform] = [EnsureChannelFirst(channel_dim="no_channel")]
    if split == "train":
        ops += [
            RandFlip(spatial_axis=1, prob=0.5),
            RandAffine(
                prob=0.5,
                rotate_range=np.deg2rad(10),
                translate_range=(10, 10),
                scale_range=(0.05, 0.05),
                padding_mode="zeros",
            ),
            RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
            RandGaussianNoise(prob=0.2, mean=0.0, std=0.01),
        ]
    ops += [
        Resize(spatial_size=spatial),
        RepeatChannel(repeats=3),
        NormalizeIntensity(
            subtrahend=IMAGENET_MEAN,
            divisor=IMAGENET_STD,
            channel_wise=True,
        ),
    ]
    return Compose(ops)


@dataclass(frozen=True)
class RSNAItem:
    patient_id: str
    label: int


class RSNAClassificationDataset(Dataset):
    """Per-patient binary classification dataset for RSNA Pneumonia."""

    def __init__(
        self,
        split: Split,
        splits_path: Path = DEFAULT_SPLITS,
        image_dir: Path = DEFAULT_IMG_DIR,
        labels_csv: Path = DEFAULT_LABELS_CSV,
        transform: Callable | None = None,
        image_size: int = 224,
    ) -> None:
        self.split = split
        self.image_dir = Path(image_dir)
        self.transform = transform if transform is not None else build_transforms(split, image_size)

        with Path(splits_path).open(encoding="utf-8") as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(f"split='{split}' not present in {splits_path}")
        patient_ids: list[str] = splits[split]

        labels_df = pd.read_csv(labels_csv)
        per_patient = labels_df.groupby("patientId")["Target"].max().to_dict()
        missing = [pid for pid in patient_ids if pid not in per_patient]
        if missing:
            raise ValueError(
                f"{len(missing)} patient(s) in split='{split}' not found in {labels_csv}"
            )
        self.items: list[RSNAItem] = [
            RSNAItem(patient_id=pid, label=int(per_patient[pid])) for pid in patient_ids
        ]

    def __len__(self) -> int:
        return len(self.items)

    def labels(self) -> np.ndarray:
        """Array of binary labels in dataset order — useful for class weighting."""
        return np.fromiter((it.label for it in self.items), dtype=np.int64, count=len(self.items))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.items[idx]
        arr = load_dicom_array(self.image_dir / f"{item.patient_id}.dcm")
        image = self.transform(arr)
        # MONAI returns MetaTensor (Tensor subclass); normalize just in case.
        image_t = image if isinstance(image, torch.Tensor) else torch.as_tensor(np.asarray(image))
        return {
            "image": image_t.float(),
            "label": torch.tensor(item.label, dtype=torch.long),
            "patient_id": item.patient_id,
        }


def build_rsna_datasets(
    splits_path: Path = DEFAULT_SPLITS,
    image_dir: Path = DEFAULT_IMG_DIR,
    labels_csv: Path = DEFAULT_LABELS_CSV,
    image_size: int = 224,
) -> tuple[RSNAClassificationDataset, RSNAClassificationDataset, RSNAClassificationDataset]:
    """Convenience factory: (train, val, test) with default per-split transforms."""
    return (
        RSNAClassificationDataset(
            "train", splits_path, image_dir, labels_csv, image_size=image_size
        ),
        RSNAClassificationDataset("val", splits_path, image_dir, labels_csv, image_size=image_size),
        RSNAClassificationDataset(
            "test", splits_path, image_dir, labels_csv, image_size=image_size
        ),
    )
