"""Train DenseNet-121 on CheXpert 14-class multi-label classification.

Run:
    python -m backend.ml.training.train_chexpert \
        --config backend/ml/configs/chexpert_densenet121.yaml

Produces:
    backend/ml/runs/<run_name>_<timestamp>/metrics.csv
    backend/ml/weights/<run_name>_<timestamp>/best.pt   (best val mean-AUC)
    backend/ml/weights/<run_name>_<timestamp>/last.pt

Loss: masked BCEWithLogitsLoss (U-Ignore strategy — uncertain labels are excluded
from the loss via a per-element binary mask returned by the dataset).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import yaml
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from backend.ml.training.chexpert_dataset import CHEXPERT_LABELS, build_chexpert_datasets
from backend.ml.training.model import build_model
from backend.ml.training.train_baseline import resolve_path, seed_everything, setup_run_dirs

REPO_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger("train_chexpert")

_CSV_FIELDS = ["epoch", "lr", "train_loss", "val_loss", "val_mean_auc"] + [
    f"val_auc_{lbl.lower().replace(' ', '_')}" for lbl in CHEXPERT_LABELS
]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCEWithLogitsLoss with per-element masking for U-Ignore.

    mask=1 for labels that are known (0 or 1), mask=0 for uncertain/unmentioned.
    Loss is averaged only over the valid (mask=1) positions.
    """
    loss = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_multilabel_aucs(probs: np.ndarray, labels: np.ndarray) -> tuple[float, list[float]]:
    """Mean AUC + per-class AUC. Classes with a single unique label are NaN."""
    per_class: list[float] = []
    for c in range(labels.shape[1]):
        y = labels[:, c]
        if len(np.unique(y)) < 2:
            per_class.append(float("nan"))
        else:
            per_class.append(float(roc_auc_score(y, probs[:, c])))
    valid = [v for v in per_class if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0, per_class


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    d = cfg["data"]
    preload = d.get("preload", False)
    cache_dir = resolve_path(d["cache_dir"]) if d.get("cache_dir") else None
    train_ds, val_ds = build_chexpert_datasets(
        data_dir=resolve_path(d["data_dir"]),
        val_fraction=d.get("val_fraction", 0.1),
        image_size=d["image_size"],
        u_strategy=d.get("u_strategy", "ignore"),
        seed=cfg["seed"],
        frontal_only=d.get("frontal_only", True),
        preload=preload,
        num_preload_workers=d.get("num_preload_workers", 8),
        cache_dir=cache_dir,
    )
    logger.info("chexpert train=%d  val=%d  preload=%s", len(train_ds), len(val_ds), preload)

    # With preload the cache lives in shared memory (share_memory_()) so
    # workers access it without copying. Use a modest worker count to
    # parallelise augmentations without spawning overhead dominating.
    num_workers = d.get("num_workers_preload", 4) if preload else d["num_workers"]
    prefetch = d.get("prefetch_factor") if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=d["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=d.get("pin_memory", True),
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=d["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=d.get("pin_memory", True),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    grad_clip: float,
    log_every: int,
) -> float:
    model.train()
    running = 0.0
    n_batches = 0
    fmt = torch.channels_last if device.type == "cuda" else torch.contiguous_format
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True, memory_format=fmt)
        labels = batch["label"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = masked_bce_loss(logits, labels, mask)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        running += float(loss.detach())
        n_batches += 1
        if log_every and step % log_every == 0:
            logger.info(
                "epoch=%d  step=%d/%d  loss=%.4f  lr=%.2e",
                epoch,
                step,
                len(loader),
                float(loss.detach()),
                optim.param_groups[0]["lr"],
            )
    return running / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list[float]]:
    model.eval()
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    fmt = torch.channels_last if device.type == "cuda" else torch.contiguous_format
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True, memory_format=fmt)
        labels = batch["label"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        losses.append(float(masked_bce_loss(logits, labels, mask).detach()))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    mean_auc, per_class = compute_multilabel_aucs(probs, labels_arr)
    return float(np.mean(losses)), mean_auc, per_class


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: dict[str, Any], config_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    seed_everything(cfg["seed"])
    paths = setup_run_dirs(cfg, config_path)
    logger.info("run_dir=%s", paths.run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info("device=%s", device)

    train_loader, val_loader = build_loaders(cfg)

    model = build_model(cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    t = cfg["train"]
    if t.get("compile", False) and hasattr(torch, "compile"):
        logger.info("torch.compile(model, mode='reduce-overhead') …")
        model = torch.compile(model, mode="reduce-overhead")

    o = cfg["optim"]
    optim = AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, t["epochs"] * steps_per_epoch)
    warmup_epochs = o.get("warmup_epochs", 1)
    if warmup_epochs == 0:
        scheduler: torch.optim.lr_scheduler.LRScheduler = CosineAnnealingLR(
            optim, T_max=total_steps
        )
    else:
        warmup_steps = warmup_epochs * steps_per_epoch
        cosine_steps = max(1, total_steps - warmup_steps)
        scheduler = SequentialLR(
            optim,
            schedulers=[
                LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optim, T_max=cosine_steps),
            ],
            milestones=[warmup_steps],
        )

    amp_enabled = bool(t.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(device=device.type, enabled=amp_enabled)

    best_auc = -np.inf
    epochs_no_improve = 0
    patience = t["early_stop_patience"]

    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optim,
            scheduler,
            scaler,
            device,
            epoch,
            amp_enabled,
            t.get("grad_clip", 1.0),
            t.get("log_every_steps", 100),
        )
        val_loss, mean_auc, per_class = evaluate(model, val_loader, device)
        dt = time.time() - t0

        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": optim.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_auc": mean_auc,
        }
        for lbl, auc in zip(CHEXPERT_LABELS, per_class, strict=False):
            row[f"val_auc_{lbl.lower().replace(' ', '_')}"] = auc

        write_header = not paths.metrics_csv.exists()
        with paths.metrics_csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)

        logger.info(
            "epoch=%d  train_loss=%.4f  val_loss=%.4f  val_mean_auc=%.4f  took=%.1fs",
            epoch,
            train_loss,
            val_loss,
            mean_auc,
            dt,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_mean_auc": mean_auc,
                "config": cfg,
            },
            paths.last_ckpt,
        )
        if mean_auc > best_auc:
            best_auc = mean_auc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_mean_auc": mean_auc,
                    "config": cfg,
                },
                paths.best_ckpt,
            )
            logger.info("new best mean_auc=%.4f → saved best.pt", best_auc)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("early stop at epoch=%d (best mean_auc=%.4f)", epoch, best_auc)
                break

    (paths.run_dir / "summary.json").write_text(
        json.dumps({"best_val_mean_auc": float(best_auc)}, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DenseNet-121 on CheXpert 14-class multi-label."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train(cfg, args.config)


if __name__ == "__main__":
    main()
