"""Baseline RSNA Pneumonia classifier — ResNet50 + BCEWithLogits + AMP.

Run:
    python -m backend.ml.training.train_baseline --config backend/ml/configs/baseline_resnet50.yaml

Produces:
    backend/ml/runs/<run_name>_<timestamp>/config.yaml
    backend/ml/runs/<run_name>_<timestamp>/metrics.csv
    backend/ml/weights/<run_name>_<timestamp>/best.pt   (by val AUC)
    backend/ml/weights/<run_name>_<timestamp>/last.pt

Scope: train + val only. Test set is untouched here on purpose — a separate
eval script will pick it up after model selection so we don't overfit to test.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

from backend.ml.training.datasets import RSNAClassificationDataset, build_rsna_datasets
from backend.ml.training.metrics import BinaryMetrics, compute_metrics
from backend.ml.training.model import build_model

REPO_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger("train_baseline")


# ---------------------------------------------------------------------------
# Config + setup
# ---------------------------------------------------------------------------


@dataclass
class RunPaths:
    run_dir: Path
    weights_dir: Path
    metrics_csv: Path
    config_snapshot: Path
    best_ckpt: Path
    last_ckpt: Path


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """Config paths are relative to REPO_ROOT unless already absolute."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def seed_everything(seed: int) -> None:
    # NOTE: cudnn.benchmark=True (set later) makes runs non-deterministic across
    # hardware. Add torch.backends.cudnn.deterministic=True here if exact
    # reproducibility is required (at ~20% throughput cost).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_run_dirs(cfg: dict[str, Any], config_path: Path) -> RunPaths:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{cfg['run_name']}_{timestamp}"
    run_dir = resolve_path(cfg["output"]["run_dir"]) / run_id
    weights_dir = resolve_path(cfg["output"]["weights_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    paths = RunPaths(
        run_dir=run_dir,
        weights_dir=weights_dir,
        metrics_csv=run_dir / "metrics.csv",
        config_snapshot=run_dir / "config.yaml",
        best_ckpt=weights_dir / "best.pt",
        last_ckpt=weights_dir / "last.pt",
    )
    shutil.copy2(config_path, paths.config_snapshot)
    return paths


# ---------------------------------------------------------------------------
# Data + model
# ---------------------------------------------------------------------------


def build_loaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader, RSNAClassificationDataset]:
    d = cfg["data"]
    # NOTE: build_rsna_datasets builds all 3 splits (train/val/test). test_ds is
    # discarded here to prevent any accidental use during training. The cost is
    # one extra CSV parse (~ms). Refactor if build time becomes a bottleneck.
    train_ds, val_ds, _test_ds = build_rsna_datasets(
        splits_path=resolve_path(d["splits_path"]),
        image_dir=resolve_path(d["image_dir"]),
        labels_csv=resolve_path(d["labels_csv"]),
        image_size=d["image_size"],
    )
    prefetch = d.get("prefetch_factor") if d["num_workers"] > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=d["batch_size"],
        shuffle=True,
        num_workers=d["num_workers"],
        pin_memory=d["pin_memory"],
        drop_last=True,
        persistent_workers=d["num_workers"] > 0,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=d["batch_size"],
        shuffle=False,
        num_workers=d["num_workers"],
        pin_memory=d["pin_memory"],
        persistent_workers=d["num_workers"] > 0,
        prefetch_factor=prefetch,
    )
    return train_loader, val_loader, train_ds


def resolve_pos_weight(cfg: dict[str, Any], train_ds: RSNAClassificationDataset) -> float:
    pw = cfg["loss"]["pos_weight"]
    if pw == "auto":
        labels = train_ds.labels()
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if n_pos == 0:
            raise ValueError("no positives in train split — pos_weight undefined")
        return n_neg / n_pos
    return float(pw)


def build_scheduler(
    optim: torch.optim.Optimizer, cfg: dict[str, Any], steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """LinearLR warmup → CosineAnnealingLR over the remaining epochs.

    Step-level scheduling: total_steps = epochs * steps_per_epoch, warmup uses
    the first `warmup_epochs * steps_per_epoch` steps.
    """
    epochs = cfg["train"]["epochs"]
    warmup_epochs = cfg["optim"]["warmup_epochs"]
    total_steps = max(1, epochs * steps_per_epoch)
    if warmup_epochs == 0:
        return CosineAnnealingLR(optim, T_max=total_steps)
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup = LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optim, T_max=cosine_steps)
    return SequentialLR(optim, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: dict[str, Any],
) -> float:
    model.train()
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    grad_clip = cfg["train"]["grad_clip"]
    log_every = cfg["train"]["log_every_steps"]
    running = 0.0
    n_batches = 0
    for step, batch in enumerate(loader):
        fmt = torch.channels_last if device.type == "cuda" else torch.contiguous_format
        images = batch["image"].to(device, non_blocking=True, memory_format=fmt)
        labels = batch["label"].to(device, non_blocking=True).float().unsqueeze(1)
        optim.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, labels)
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
            lr = optim.param_groups[0]["lr"]
            logger.info(
                "epoch=%d step=%d/%d loss=%.4f lr=%.2e",
                epoch,
                step,
                len(loader),
                float(loss.detach()),
                lr,
            )
    return running / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, BinaryMetrics]:
    model.eval()
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    fmt = torch.channels_last if device.type == "cuda" else torch.contiguous_format
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True, memory_format=fmt)
        labels = batch["label"].to(device, non_blocking=True).float().unsqueeze(1)
        logits = model(images)
        loss = loss_fn(logits, labels)
        losses.append(float(loss.detach()))
        all_probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
        all_labels.append(labels.cpu().numpy().ravel().astype(np.int64))
    probs = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    metrics = compute_metrics(probs, labels_arr)
    return float(np.mean(losses)), metrics


# ---------------------------------------------------------------------------
# Logging / checkpointing
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "val_loss",
    "val_auc",
    "val_accuracy",
    "val_sens_youden",
    "val_spec_youden",
    "val_threshold_youden",
    "val_sens_at_spec85",
    "val_spec_at_sens85",
]


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    val_auc: float,
    threshold_youden: float,
    cfg: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_auc": val_auc,
            "threshold_youden": threshold_youden,
            "config": cfg,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def train(
    cfg: dict[str, Any],
    config_path: Path,
    train_ds_override: RSNAClassificationDataset | None = None,
    val_ds_override: RSNAClassificationDataset | None = None,
) -> RunPaths:
    """Run full train+val loop. Overrides exist for smoke tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    seed_everything(cfg["seed"])
    paths = setup_run_dirs(cfg, config_path)
    logger.info("run_dir=%s", paths.run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    if device.type == "cuda":
        # Input shapes are constant across steps → benchmark picks fastest conv algo.
        torch.backends.cudnn.benchmark = True

    # Data
    if train_ds_override is not None and val_ds_override is not None:
        train_ds: RSNAClassificationDataset | Subset = train_ds_override
        val_ds: RSNAClassificationDataset | Subset = val_ds_override
        d = cfg["data"]
        # Respect cfg's num_workers / pin_memory / prefetch_factor on the override
        # path too — this is critical for CV performance. Smoke tests pass
        # num_workers=0 in cfg, so this doesn't break them.
        n_workers = d.get("num_workers", 0)
        pin = d.get("pin_memory", False)
        prefetch = d.get("prefetch_factor") if n_workers > 0 else None
        train_loader = DataLoader(
            train_ds,
            batch_size=d["batch_size"],
            shuffle=True,
            num_workers=n_workers,
            pin_memory=pin,
            drop_last=False,
            persistent_workers=n_workers > 0,
            prefetch_factor=prefetch,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=d["batch_size"],
            shuffle=False,
            num_workers=n_workers,
            pin_memory=pin,
            persistent_workers=n_workers > 0,
            prefetch_factor=prefetch,
        )
        pos_weight_ds = train_ds_override  # labels() for pos_weight
    else:
        train_loader, val_loader, pos_weight_ds = build_loaders(cfg)

    logger.info("train_size=%d val_size=%d", len(train_loader.dataset), len(val_loader.dataset))

    # Model / loss / optim
    model = build_model(cfg).to(device)
    if device.type == "cuda":
        # channels_last boosts Ampere+ conv throughput ~20-30% under AMP.
        model = model.to(memory_format=torch.channels_last)
    pos_weight = resolve_pos_weight(cfg, pos_weight_ds)
    logger.info("pos_weight=%.4f", pos_weight)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optim = AdamW(
        model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"]
    )
    scheduler = build_scheduler(optim, cfg, steps_per_epoch=max(1, len(train_loader)))
    scaler = GradScaler(
        device=device.type, enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"
    )

    # Loop
    best_auc = -np.inf
    epochs_no_improve = 0
    patience = cfg["train"]["early_stop_patience"]
    stop_epoch = cfg["train"]["epochs"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optim, scheduler, loss_fn, scaler, device, epoch, cfg
        )
        val_loss, metrics = evaluate(model, val_loader, loss_fn, device)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "lr": optim.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": metrics["auc"],
            "val_accuracy": metrics["accuracy"],
            "val_sens_youden": metrics["sensitivity_at_youden"],
            "val_spec_youden": metrics["specificity_at_youden"],
            "val_threshold_youden": metrics["threshold_youden"],
            "val_sens_at_spec85": metrics["sensitivity_at_spec85"],
            "val_spec_at_sens85": metrics["specificity_at_sens85"],
        }
        append_csv_row(paths.metrics_csv, row)
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_auc=%.4f "
            "sens@youden=%.3f spec@youden=%.3f sens@spec85=%.3f took=%.1fs",
            epoch,
            train_loss,
            val_loss,
            metrics["auc"],
            metrics["sensitivity_at_youden"],
            metrics["specificity_at_youden"],
            metrics["sensitivity_at_spec85"],
            dt,
        )

        save_checkpoint(
            paths.last_ckpt,
            model,
            optim,
            scheduler,
            scaler,
            epoch,
            metrics["auc"],
            metrics["threshold_youden"],
            cfg,
        )
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            epochs_no_improve = 0
            save_checkpoint(
                paths.best_ckpt,
                model,
                optim,
                scheduler,
                scaler,
                epoch,
                metrics["auc"],
                metrics["threshold_youden"],
                cfg,
            )
            logger.info("new best val_auc=%.4f → saved %s", best_auc, paths.best_ckpt.name)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    "early stop at epoch=%d (no val_auc improvement for %d epochs, best=%.4f)",
                    epoch,
                    patience,
                    best_auc,
                )
                stop_epoch = epoch
                break

    (paths.run_dir / "summary.json").write_text(
        json.dumps(
            {"best_val_auc": float(best_auc), "stopped_at_epoch": stop_epoch},
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RSNA Pneumonia baseline (ResNet50).")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.config)


if __name__ == "__main__":
    main()
