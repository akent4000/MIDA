"""CheXpert dataset for 14-class multi-label chest X-ray classification.

Dataset: CheXpert (Stanford) — 224,316 chest radiographs, 65,240 patients.
Kaggle:  ashery/chexpert (~11 GB, 320×320 small version)

Expected directory layout:
    backend/ml/data/chexpert/
        train.csv
        train/patient00001/study1/view1_frontal.jpg ...

Label encoding in train.csv:
    1.0  = positive,  0.0 = negative,  -1.0 = uncertain,  NaN = not mentioned

U-label strategy (U-Ignore default):
    mask=0 excludes uncertain/unmentioned entries from the loss.

------------------------------------------------------------------
Preload mode (preload=True in config)
------------------------------------------------------------------
All images are decoded, CLAHE-processed, and resized to target size
at dataset construction time and held in RAM as uint8 numpy arrays.

Memory estimate (train+val, 320×320 uint8):
    190 000 images × 102 400 bytes ≈ 19.5 GB

Preloading is parallelised with ThreadPoolExecutor (PIL/skimage
release the GIL, so threads are effective here).  Wall-clock time
is ~7–10 minutes with 8 threads on an i7-10700.

When preloaded:
  * __getitem__ skips disk I/O, JPEG decode, CLAHE, and Resize —
    only fast random augmentations and normalisation remain (~2 ms).
  * Set num_workers: 0 in the DataLoader config (no I/O to overlap).
  * Expected throughput: ~0.1–0.2 s/batch vs ~2.5 s/batch without.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from monai.transforms import (
    Compose,
    NormalizeIntensity,
    RandAdjustContrast,
    RandAffine,
    RandFlip,
    RandGaussianNoise,
    RepeatChannel,
    Resize,
    Transform,
)
from PIL import Image
from torch.utils.data import Dataset

from backend.ml.training.datasets import IMAGENET_MEAN, IMAGENET_STD
from backend.ml.training.transforms import CLAHETransform

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "backend" / "ml" / "data" / "chexpert"

logger = logging.getLogger(__name__)

CHEXPERT_LABELS: list[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
NUM_CLASSES = len(CHEXPERT_LABELS)
LABEL_TO_IDX: dict[str, int] = {lbl: i for i, lbl in enumerate(CHEXPERT_LABELS)}

PRIORITY_CLASSES: frozenset[str] = frozenset(
    {"Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"}
)

UStrategy = Literal["ignore", "zeros", "ones"]
Split = Literal["train", "val"]

_CSV_PATH_PREFIX = "CheXpert-v1.0-small/"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _resolve_image_path(csv_path: str, data_dir: Path) -> Path:
    rel = csv_path[len(_CSV_PATH_PREFIX) :] if csv_path.startswith(_CSV_PATH_PREFIX) else csv_path
    return data_dir / rel


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------


def _build_transforms(split: Split, image_size: int) -> Callable:
    """Full pipeline: CLAHE → [aug] → Resize → RepeatChannel → Normalize.

    Used when preload=False (images loaded from disk on every __getitem__).
    Input: float32 ndarray in [0, 1], shape (1, H, W).
    """
    spatial = (image_size, image_size)
    ops: list[Transform] = [CLAHETransform(clip_limit=0.01)]
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
        NormalizeIntensity(subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
    ]
    return Compose(ops)


def _build_augment_transforms(split: Split) -> Callable:
    """Augmentation-only pipeline for preloaded data.

    CLAHE and Resize are skipped — already baked into the cache.
    Input: float32 ndarray in [0, 1], shape (1, H, W) at target size.
    """
    ops: list[Transform] = []
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
        RepeatChannel(repeats=3),
        NormalizeIntensity(subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
    ]
    return Compose(ops)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheXpertItem:
    path: str
    labels: np.ndarray  # (14,) float32 — 0/1; uncertain/unmentioned → 0
    mask: np.ndarray  # (14,) float32 — 1=include in loss, 0=ignore


def _parse_labels(row: pd.Series, u_strategy: UStrategy) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(NUM_CLASSES, dtype=np.float32)
    mask = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, col in enumerate(CHEXPERT_LABELS):
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            pass  # unmentioned: label=0, mask=0
        elif float(val) == -1.0:
            if u_strategy == "zeros":
                mask[i] = 1.0
            elif u_strategy == "ones":
                labels[i] = 1.0
                mask[i] = 1.0
            # "ignore": leave label=0, mask=0
        else:
            labels[i] = float(val)
            mask[i] = 1.0
    return labels, mask


# ---------------------------------------------------------------------------
# Preloading
# ---------------------------------------------------------------------------


def _load_one(args: tuple[int, str, Path, int]) -> tuple[int, np.ndarray]:
    """Load, CLAHE-process, and resize one image. Returns (idx, uint8 array)."""
    from skimage.exposure import equalize_adapthist

    idx, csv_path, data_dir, image_size = args
    img_path = _resolve_image_path(csv_path, data_dir)
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr_clahe = equalize_adapthist(arr, clip_limit=0.01).astype(np.float32)
    img_out = Image.fromarray((arr_clahe * 255).clip(0, 255).astype(np.uint8))
    img_out = img_out.resize((image_size, image_size), Image.LANCZOS)
    return idx, np.array(img_out, dtype=np.uint8)


def preload_images(
    items: list[CheXpertItem],
    data_dir: Path,
    image_size: int,
    num_workers: int = 8,
    desc: str = "preloading",
    cache_file: Path | None = None,
) -> np.ndarray:
    """Load all images into RAM with parallel CLAHE + resize.

    Returns a contiguous (N, image_size, image_size) uint8 numpy array.

    Disk cache: if *cache_file* is given and already exists, loads instantly
    via memory-map (np.load mmap_mode='r') — OS keeps hot pages in RAM.
    On first run the result is computed and saved to *cache_file*.

    Uses ProcessPoolExecutor — CLAHE (skimage equalize_adapthist) is
    numpy/Python and holds the GIL, so threads cannot parallelize it.
    Separate processes each have their own GIL and run CLAHE truly in
    parallel across all CPU cores.
    """
    try:
        from tqdm import tqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False

    # --- disk cache hit ---
    if cache_file is not None and Path(cache_file).exists():
        logger.info("loading %s from disk cache %s", desc, cache_file)
        arr = np.load(cache_file, mmap_mode="r")
        assert arr.shape == (len(items), image_size, image_size), (
            f"cache shape mismatch: got {arr.shape}, "
            f"expected ({len(items)}, {image_size}, {image_size}). "
            "Delete the cache file and retry."
        )
        return arr

    # --- compute ---
    n = len(items)
    args = [(i, it.path, data_dir, image_size) for i, it in enumerate(items)]
    logger.info(
        "preloading %d images with %d processes (image_size=%d)… "
        "[CLAHE is GIL-bound → ProcessPoolExecutor for true parallelism]",
        n,
        num_workers,
        image_size,
    )

    # Pre-allocate contiguous array so workers write by index without sync
    cache = np.empty((n, image_size, image_size), dtype=np.uint8)

    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_load_one, a): a[0] for a in args}
        it = as_completed(futures)
        if use_tqdm:
            it = tqdm(it, total=n, desc=desc, unit="img", dynamic_ncols=True)
        for fut in it:
            idx, arr_one = fut.result()
            cache[idx] = arr_one
            completed += 1
            if not use_tqdm and completed % 10_000 == 0:
                logger.info("  %d / %d images loaded", completed, n)

    # --- save to disk cache ---
    if cache_file is not None:
        cache_file = Path(cache_file)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("saving cache → %s (%.1f GB)", cache_file, cache.nbytes / 1e9)
        np.save(cache_file, cache)

    return cache


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CheXpertDataset(Dataset):
    """Multi-label CheXpert dataset (frontal views only by default).

    When *preloaded_cache* is provided, images are served from RAM and
    __getitem__ only runs fast random augmentations + normalisation.
    """

    def __init__(
        self,
        items: list[CheXpertItem],
        data_dir: Path,
        split: Split,
        transform: Callable | None = None,
        image_size: int = 320,
        preloaded_cache: np.ndarray | None = None,
    ) -> None:
        self.items = items
        self.data_dir = Path(data_dir)

        if preloaded_cache is not None:
            assert len(preloaded_cache) == len(items)
            # Move to PyTorch shared memory so DataLoader workers share the
            # cache without copying 19.5 GB per worker (Windows spawn).
            # torch.Tensor.share_memory_() uses OS shared memory; pickle
            # sends only the handle — workers attach to the same pages.
            self._cache: torch.Tensor | None = torch.from_numpy(
                np.ascontiguousarray(preloaded_cache)
            ).share_memory_()
            self.transform = (
                transform if transform is not None else _build_augment_transforms(split)
            )
        else:
            self._cache = None
            self.transform = (
                transform if transform is not None else _build_transforms(split, image_size)
            )

    def __len__(self) -> int:
        return len(self.items)

    def label_matrix(self) -> np.ndarray:
        return np.stack([it.labels for it in self.items])

    def mask_matrix(self) -> np.ndarray:
        return np.stack([it.mask for it in self.items])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.items[idx]

        if self._cache is not None:
            # Fast path: read from shared memory (no copy), augment only
            arr = self._cache[idx].numpy().astype(np.float32) / 255.0  # (H, W)
            arr = arr[np.newaxis]  # (1, H, W)
        else:
            img_path = _resolve_image_path(item.path, self.data_dir)
            img = Image.open(img_path).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr[np.newaxis]  # (1, H, W) for CLAHETransform

        image = self.transform(arr)
        image_t = image if isinstance(image, torch.Tensor) else torch.as_tensor(np.asarray(image))
        return {
            "image": image_t.float(),
            "label": torch.from_numpy(item.labels),
            "mask": torch.from_numpy(item.mask),
            "path": item.path,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_chexpert_datasets(
    data_dir: Path = DATA_DIR,
    val_fraction: float = 0.1,
    image_size: int = 320,
    u_strategy: UStrategy = "ignore",
    seed: int = 42,
    frontal_only: bool = True,
    preload: bool = False,
    num_preload_workers: int = 8,
    cache_dir: Path | None = None,
) -> tuple[CheXpertDataset, CheXpertDataset]:
    """Patient-level train/val split from train.csv.

    Args:
        preload:             If True, preload all images into RAM.
        num_preload_workers: Worker processes for parallel CLAHE+resize.
        cache_dir:           Directory for numpy disk cache files.
                             If set and cache exists, preload is instant
                             (memory-map). If not set, cache is not saved.
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "train.csv")

    if frontal_only:
        df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

    df["patient_id"] = df["Path"].str.extract(r"(patient\d+)")
    patients = df["patient_id"].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_fraction))
    val_patients = set(patients[:n_val])

    train_items: list[CheXpertItem] = []
    val_items: list[CheXpertItem] = []
    for _, row in df.iterrows():
        labels, mask = _parse_labels(row, u_strategy)
        item = CheXpertItem(path=str(row["Path"]), labels=labels, mask=mask)
        if row["patient_id"] in val_patients:
            val_items.append(item)
        else:
            train_items.append(item)

    train_cache: np.ndarray | None = None
    val_cache: np.ndarray | None = None

    if preload:
        tag = f"frontal_{image_size}" if frontal_only else f"all_{image_size}"
        train_cf = Path(cache_dir) / f"train_{tag}.npy" if cache_dir else None
        val_cf = Path(cache_dir) / f"val_{tag}.npy" if cache_dir else None
        train_cache = preload_images(
            train_items,
            data_dir,
            image_size,
            num_preload_workers,
            desc="train preload",
            cache_file=train_cf,
        )
        val_cache = preload_images(
            val_items,
            data_dir,
            image_size,
            num_preload_workers,
            desc="val preload",
            cache_file=val_cf,
        )

    train_ds = CheXpertDataset(
        train_items,
        data_dir,
        "train",
        image_size=image_size,
        preloaded_cache=train_cache,
    )
    val_ds = CheXpertDataset(
        val_items,
        data_dir,
        "val",
        image_size=image_size,
        preloaded_cache=val_cache,
    )
    return train_ds, val_ds
