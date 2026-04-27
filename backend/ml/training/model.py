"""Model factory for RSNA Pneumonia classifiers.

Single source of truth for architecture construction — used by train_baseline,
eval, and the inference backend.

Supported archs (cfg["model"]["arch"]):
    resnet50        — torchvision ResNet-50, replaces fc head
    efficientnet_b0 — torchvision EfficientNet-B0, replaces classifier[1]
    densenet121     — torchvision DenseNet-121 (CheXNet arch), replaces classifier

backbone_checkpoint (cfg["model"]["backbone_checkpoint"]):
    Path to a checkpoint saved by pretrain_nih.py. When set, the ImageNet
    initialisation is replaced by NIH-pretrained backbone weights.
    Only the backbone is loaded — the classification head is always freshly
    initialised so that num_classes can differ between pretrain and fine-tune.

All models output raw logits of shape (N, num_classes). For binary
classification (num_classes=1) pair with BCEWithLogitsLoss / sigmoid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

SUPPORTED_ARCHS = frozenset({"resnet50", "efficientnet_b0", "densenet121"})

# Keys whose weights belong to the head (not the backbone). These are skipped
# when loading a backbone checkpoint so that head size mismatches don't error.
_HEAD_KEY_PREFIXES: dict[str, tuple[str, ...]] = {
    "resnet50": ("fc.",),
    "efficientnet_b0": ("classifier.",),
    "densenet121": ("classifier.",),
}


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Construct a classification model from a training config dict.

    Args:
        cfg: dict containing a "model" sub-dict with keys:
            arch                (str)            — see SUPPORTED_ARCHS
            weights             (str | None)      — torchvision weights enum name
            num_classes         (int)             — output logits (1 for binary)
            backbone_checkpoint (str | None)      — path to NIH pretrain .pt

    Returns:
        nn.Module with classification head sized to num_classes, weights
        initialised from torchvision pretrain or backbone_checkpoint.
    """
    m = cfg["model"]
    arch: str = m["arch"]
    num_classes: int = int(m["num_classes"])
    weights_name: str | None = m.get("weights") or None
    backbone_ckpt: str | None = m.get("backbone_checkpoint") or None

    model = _build_arch(arch, weights_name, num_classes)

    if backbone_ckpt is not None:
        _load_backbone(model, arch, Path(backbone_ckpt))

    return model


def _build_arch(arch: str, weights_name: str | None, num_classes: int) -> nn.Module:
    if arch == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights[weights_name] if weights_name else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights[weights_name] if weights_name else None
        model = efficientnet_b0(weights=weights)
        in_features: int = model.classifier[1].in_features  # type: ignore[union-attr]
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "densenet121":
        from torchvision.models import DenseNet121_Weights, densenet121

        weights = DenseNet121_Weights[weights_name] if weights_name else None
        model = densenet121(weights=weights)
        # classifier is a single Linear(1024, 1000)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported arch {arch!r}. Supported: {sorted(SUPPORTED_ARCHS)}")


def _load_backbone(model: nn.Module, arch: str, ckpt_path: Path) -> None:
    """Load backbone weights from a pretrain checkpoint, skipping the head."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Support both raw state_dict and our checkpoint format {model_state: ...}
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

    head_prefixes = _HEAD_KEY_PREFIXES.get(arch, ())
    backbone_state = {
        k: v for k, v in state.items() if not any(k.startswith(pfx) for pfx in head_prefixes)
    }
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    head_keys = [k for k in missing if any(k.startswith(p) for p in head_prefixes)]
    real_missing = [k for k in missing if k not in head_keys]
    if real_missing:
        raise RuntimeError(f"backbone_checkpoint missing non-head keys: {real_missing[:5]}...")
