"""DenseNet-121 pretraining on NIH ChestX-ray14 (multi-label, 14 classes).

Run:
    python -m backend.ml.training.pretrain_nih \
        --config backend/ml/configs/pretrain_nih_densenet121.yaml

Produces:
    backend/ml/runs/<run_name>_<timestamp>/metrics.csv
    backend/ml/weights/<run_name>_<timestamp>/best.pt     (best val mean-AUC)
    backend/ml/weights/<run_name>_<timestamp>/last.pt
    backend/ml/weights/<run_name>_<timestamp>/backbone.pt (backbone only, no head)

backbone.pt is the file to reference in finetune_rsna_densenet121_384.yaml
under model.backbone_checkpoint.
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
import yaml
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from backend.ml.training.model import build_model
from backend.ml.training.nih_dataset import NIH_LABELS, build_nih_datasets
from backend.ml.training.train_baseline import resolve_path, seed_everything, setup_run_dirs

REPO_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger("pretrain_nih")

NIH_CSV_FIELDS = ["epoch", "lr", "train_loss", "val_loss", "val_mean_auc"] + [
    f"val_auc_{lbl.lower()}" for lbl in NIH_LABELS
]


def compute_multilabel_aucs(probs: np.ndarray, labels: np.ndarray) -> tuple[float, list[float]]:
    """Mean AUC + per-class AUC for multi-label output.

    Skips classes with only one label value present (AUC undefined).
    Returns (mean_auc, per_class_aucs).
    """
    per_class: list[float] = []
    for c in range(labels.shape[1]):
        y = labels[:, c]
        if len(np.unique(y)) < 2:
            per_class.append(float("nan"))
        else:
            per_class.append(float(roc_auc_score(y, probs[:, c])))
    valid = [v for v in per_class if not np.isnan(v)]
    mean_auc = float(np.mean(valid)) if valid else 0.0
    return mean_auc, per_class


def build_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    d = cfg["data"]
    train_ds, val_ds = build_nih_datasets(
        data_dir=resolve_path(d["data_dir"]),
        val_fraction=d.get("val_fraction", 0.1),
        image_size=d["image_size"],
        seed=cfg["seed"],
    )
    logger.info("nih train=%d val=%d", len(train_ds), len(val_ds))

    prefetch = d.get("prefetch_factor") if d["num_workers"] > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=d["batch_size"],
        shuffle=True,
        num_workers=d["num_workers"],
        pin_memory=d.get("pin_memory", True),
        drop_last=True,
        persistent_workers=d["num_workers"] > 0,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=d["batch_size"],
        shuffle=False,
        num_workers=d["num_workers"],
        pin_memory=d.get("pin_memory", True),
        persistent_workers=d["num_workers"] > 0,
        prefetch_factor=prefetch,
    )
    return train_loader, val_loader


def build_pos_weights(train_ds: Any, device: torch.device) -> torch.Tensor:
    """Per-class pos_weight = n_neg / n_pos, shape (14,)."""
    lm = train_ds.label_matrix()  # (N, 14)
    n_pos = lm.sum(axis=0).clip(min=1)
    n_neg = len(lm) - n_pos
    return torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
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
    for step, batch in enumerate(loader):
        fmt = torch.channels_last if device.type == "cuda" else torch.contiguous_format
        images = batch["image"].to(device, non_blocking=True, memory_format=fmt)
        labels = batch["label"].to(device, non_blocking=True)  # (B, 14) float32
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
            logger.info(
                "epoch=%d step=%d/%d loss=%.4f lr=%.2e",
                epoch,
                step,
                len(loader),
                float(loss.detach()),
                optim.param_groups[0]["lr"],
            )
    return running / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> tuple[float, float, list[float]]:
    model.eval()
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    fmt = torch.channels_last if device.type == "cuda" else torch.contiguous_format
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True, memory_format=fmt)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        losses.append(float(loss_fn(logits, labels).detach()))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    mean_auc, per_class = compute_multilabel_aucs(probs, labels_arr)
    return float(np.mean(losses)), mean_auc, per_class


def save_backbone(model: nn.Module, arch: str, path: Path) -> None:
    """Save only the backbone (no head) for downstream fine-tuning."""
    from backend.ml.training.model import _HEAD_KEY_PREFIXES

    head_prefixes = _HEAD_KEY_PREFIXES.get(arch, ())
    backbone_state = {
        k: v
        for k, v in model.state_dict().items()
        if not any(k.startswith(p) for p in head_prefixes)
    }
    torch.save({"model_state": backbone_state}, path)
    logger.info("backbone saved → %s (%d keys)", path.name, len(backbone_state))


def pretrain(cfg: dict[str, Any], config_path: Path) -> Path:
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
    train_ds = train_loader.dataset

    model = build_model(cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    pos_weights = build_pos_weights(train_ds, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    o = cfg["optim"]
    optim = AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])

    t = cfg["train"]
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
        warmup_sched = LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(optim, T_max=cosine_steps)
        scheduler = SequentialLR(
            optim, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
        )

    amp_enabled = bool(t.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(device=device.type, enabled=amp_enabled)

    best_auc = -np.inf
    epochs_no_improve = 0
    patience = t["early_stop_patience"]
    backbone_path = paths.weights_dir / "backbone.pt"

    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optim,
            scheduler,
            loss_fn,
            scaler,
            device,
            epoch,
            amp_enabled,
            t.get("grad_clip", 1.0),
            t.get("log_every_steps", 50),
        )
        val_loss, mean_auc, per_class = evaluate(model, val_loader, loss_fn, device)
        dt = time.time() - t0

        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": optim.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_auc": mean_auc,
        }
        for lbl, auc in zip(NIH_LABELS, per_class, strict=False):
            row[f"val_auc_{lbl.lower()}"] = auc

        write_header = not paths.metrics_csv.exists()
        with paths.metrics_csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=NIH_CSV_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)

        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_mean_auc=%.4f took=%.1fs",
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
            save_backbone(model, cfg["model"]["arch"], backbone_path)
            logger.info("new best mean_auc=%.4f → saved best.pt + backbone.pt", best_auc)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("early stop at epoch=%d (best mean_auc=%.4f)", epoch, best_auc)
                break

    (paths.run_dir / "summary.json").write_text(
        json.dumps({"best_val_mean_auc": float(best_auc)}, indent=2), encoding="utf-8"
    )
    return backbone_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain DenseNet-121 on NIH ChestX-ray14.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pretrain(cfg, args.config)


if __name__ == "__main__":
    main()
