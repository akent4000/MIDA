"""NIH ChestX-ray14 dataset for DenseNet-121 pretraining.

Dataset: https://www.kaggle.com/datasets/nih-chest-xrays/data
         112,120 frontal-view CXRs, 30,805 unique patients.

Labels: 14 pathologies (multi-label binary), sourced from Data_Entry_2017.csv.
        "No Finding" is treated as all-zeros target (not its own output class).

Expected directory layout after `kaggle datasets download nih-chest-xrays/data --unzip`:
    backend/ml/data/nih/
        images/                      ← all 112,120 PNG files (flat directory)
        Data_Entry_2017.csv
        train_val_list.txt           ← official train+val split (86,524 images)
        test_list.txt                ← official test split (25,596 images)

Usage:
    from backend.ml.training.nih_dataset import build_nih_datasets
    train_ds, val_ds = build_nih_datasets(val_fraction=0.1)
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from backend.ml.training.transforms import CLAHETransform

REPO_ROOT = Path(__file__).resolve().parents[3]
NIH_DIR = REPO_ROOT / "backend" / "ml" / "data" / "nih"

NIH_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NUM_CLASSES = len(NIH_LABELS)
LABEL_TO_IDX: dict[str, int] = {lbl: i for i, lbl in enumerate(NIH_LABELS)}


def _nih_transforms(image_size: int) -> Callable:
    """Minimal CXR transforms for NIH pretraining (no heavy augmentation).

    NIH images are already PNG at 1024×1024 so we:
      1. Apply CLAHE for local contrast enhancement
      2. Resize to target size
      3. Normalise to ImageNet stats (3-channel via simple expand)
    """
    from monai.transforms import Compose, NormalizeIntensity, RepeatChannel, Resize

    from backend.ml.training.datasets import IMAGENET_MEAN, IMAGENET_STD

    return Compose(
        [
            CLAHETransform(clip_limit=0.01),  # (1, H, W) float32 in [0,1]
            Resize(spatial_size=(image_size, image_size)),
            RepeatChannel(repeats=3),  # (1,H,W) → (3,H,W) before ImageNet norm
            NormalizeIntensity(
                subtrahend=IMAGENET_MEAN,
                divisor=IMAGENET_STD,
                channel_wise=True,
            ),
        ]
    )


def _build_image_index(data_dir: Path) -> dict[str, Path]:
    """Scan images_001..012/images/ and return filename → full path.

    The Kaggle NIH download splits 112k PNGs across 12 subdirectories
    (images_001/images/, images_002/images/, …). We build a lookup once at
    dataset construction so __getitem__ is O(1).
    """
    index: dict[str, Path] = {}
    for subdir in sorted(data_dir.glob("images_*/images")):
        for png in subdir.glob("*.png"):
            index[png.name] = png
    return index


class NIHChestXray14Dataset(Dataset):
    """Multi-label dataset for NIH ChestX-ray14."""

    def __init__(
        self,
        image_names: list[str],
        labels_df: pd.DataFrame,
        data_dir: Path,
        transform: Callable | None = None,
        image_size: int = 224,
        image_index: dict[str, Path] | None = None,
    ) -> None:
        self.image_names = image_names
        # image_index is shared across train/val splits to avoid scanning twice
        self._image_index: dict[str, Path] = (
            image_index if image_index is not None else _build_image_index(Path(data_dir))
        )
        self.transform = transform or _nih_transforms(image_size)

        # Pre-compute float32 label vectors (N, 14)
        self._labels: dict[str, np.ndarray] = {}
        for _, row in labels_df.iterrows():
            vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            findings = str(row["Finding Labels"]).split("|")
            for f in findings:
                idx = LABEL_TO_IDX.get(f.strip())
                if idx is not None:
                    vec[idx] = 1.0
            self._labels[row["Image Index"]] = vec

    def __len__(self) -> int:
        return len(self.image_names)

    def label_matrix(self) -> np.ndarray:
        """(N, 14) float32 array — used for computing per-class pos_weight."""
        return np.stack([self._labels[n] for n in self.image_names])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        name = self.image_names[idx]
        img_path = self._image_index[name]
        img = Image.open(img_path).convert("L")  # → 8-bit grey
        arr = np.array(img, dtype=np.float32) / 255.0  # → [0, 1]
        arr = arr[np.newaxis]  # (1, H, W) for CLAHETransform
        image = self.transform(arr)
        image_t = image if isinstance(image, torch.Tensor) else torch.as_tensor(np.asarray(image))
        label = torch.from_numpy(self._labels[name])  # (14,) float32
        return {"image": image_t.float(), "label": label, "image_name": name}


def build_nih_datasets(
    data_dir: Path = NIH_DIR,
    val_fraction: float = 0.1,
    image_size: int = 224,
    seed: int = 42,
) -> tuple[NIHChestXray14Dataset, NIHChestXray14Dataset]:
    """Build train/val splits from the official NIH train_val_list.txt.

    The official test_list.txt is NEVER loaded here — reserved for final eval.

    Args:
        data_dir:     Root of the NIH dataset (contains images/ and CSVs).
        val_fraction: Fraction of train_val to hold out as val.
        image_size:   Spatial size fed to the model.
        seed:         For reproducible train/val split.
    """
    data_dir = Path(data_dir)
    labels_df = pd.read_csv(data_dir / "Data_Entry_2017.csv")
    train_val_names = (data_dir / "train_val_list.txt").read_text().splitlines()
    train_val_names = [n.strip() for n in train_val_names if n.strip()]

    rng = random.Random(seed)
    rng.shuffle(train_val_names)
    n_val = max(1, int(len(train_val_names) * val_fraction))
    val_names = train_val_names[:n_val]
    train_names = train_val_names[n_val:]

    # Build index once and share between train/val to avoid scanning 112k files twice
    image_index = _build_image_index(data_dir)
    train_ds = NIHChestXray14Dataset(
        train_names, labels_df, data_dir, image_size=image_size, image_index=image_index
    )
    val_ds = NIHChestXray14Dataset(
        val_names, labels_df, data_dir, image_size=image_size, image_index=image_index
    )
    return train_ds, val_ds
