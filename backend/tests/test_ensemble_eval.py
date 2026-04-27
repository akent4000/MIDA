"""Unit tests for ensemble_eval.

Tests the soft-voting logic without needing real RSNA DICOMs by
monkeypatching build_eval_loader and load_checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


def _fake_loader(probs_sigmoid: float, labels: list[int]) -> DataLoader:
    """Build a tiny DataLoader that yields dicts matching the real RSNA format."""

    # Use a collate_fn that returns the right dict format
    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, idx):
            return {
                "image": torch.zeros(3, 8, 8),
                "label": torch.tensor(labels[idx], dtype=torch.long),
                "patient_id": f"p{idx}",
            }

    return DataLoader(_DS(), batch_size=4, shuffle=False)


def _fake_model(output_logit: float) -> torch.nn.Module:
    """Mock model that always returns a fixed logit."""
    m = MagicMock()

    def _forward(x):
        return torch.full((x.shape[0], 1), output_logit)

    m.side_effect = _forward
    m.return_value = None  # unused because side_effect drives behavior
    return m


class TestEnsembleEvaluate:
    def test_requires_two_checkpoints(self, tmp_path: Path) -> None:
        from backend.ml.training.ensemble_eval import ensemble_evaluate

        with pytest.raises(ValueError, match="≥2 checkpoints"):
            ensemble_evaluate([tmp_path / "only_one.pt"], split="test")

    def test_averages_probs_across_folds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two models emitting different logits → ensemble averages to mid."""
        from backend.ml.training import ensemble_eval
        from backend.ml.training import eval as eval_mod

        labels = [0, 0, 1, 1]
        loader = _fake_loader(probs_sigmoid=0.5, labels=labels)

        ckpt_meta = {"config": {"data": {"batch_size": 4}}, "threshold_youden": 0.5}
        # Fold 1 always predicts logit=-2 (prob ≈ 0.12); Fold 2 always logit=+2 (prob ≈ 0.88)
        # Mean prob ≈ 0.5 across both folds regardless of label → AUC = 0.5
        model_a = _fake_model(-2.0)
        model_b = _fake_model(+2.0)

        load_calls = [(model_a, ckpt_meta), (model_b, ckpt_meta)]

        def fake_load(path, device):
            return load_calls.pop(0)

        monkeypatch.setattr(ensemble_eval, "load_checkpoint", fake_load)
        monkeypatch.setattr(ensemble_eval, "build_eval_loader", lambda cfg, split: loader)
        # Also patch eval module in case ensemble_eval pulls from there
        monkeypatch.setattr(eval_mod, "build_eval_loader", lambda cfg, split: loader, raising=False)

        ckpt_paths = [tmp_path / "fold1.pt", tmp_path / "fold2.pt"]
        # create dummy files so tmp_path.parent.parent resolves
        for p in ckpt_paths:
            p.write_bytes(b"")

        result = ensemble_eval.ensemble_evaluate(
            ckpt_paths, split="test", device=torch.device("cpu")
        )

        # Two symmetric models → ensemble probs all ≈ 0.5 → AUC must be ≈ 0.5
        assert result["n_folds"] == 2
        assert len(result["per_fold_aucs"]) == 2
        # Ensemble AUC should be near 0.5 (random)
        assert 0.4 <= result["ensemble_metrics"]["auc"] <= 0.6
        assert len(result["ensemble_probs"]) == len(labels)
        assert result["patient_labels"] == labels

    def test_perfect_fold_gives_perfect_ensemble(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If both folds agree perfectly, ensemble is perfect."""
        from backend.ml.training import ensemble_eval
        from backend.ml.training import eval as eval_mod

        labels = [0, 0, 1, 1]

        class PerfectDS(torch.utils.data.Dataset):
            def __len__(self):
                return len(labels)

            def __getitem__(self, idx):
                return {
                    "image": torch.full((3, 8, 8), float(labels[idx])),  # image matches label
                    "label": torch.tensor(labels[idx], dtype=torch.long),
                    "patient_id": f"p{idx}",
                }

        loader = DataLoader(PerfectDS(), batch_size=4, shuffle=False)

        # Build a model that outputs a perfect logit based on input mean
        class PerfectModel(torch.nn.Module):
            def forward(self, x):
                # x is zeros for neg, ones for pos → mean == label
                return (x.mean(dim=[1, 2, 3]) * 20 - 10).unsqueeze(1)  # sharp sigmoid

        ckpt_meta = {"config": {"data": {"batch_size": 4}}, "threshold_youden": 0.5}
        load_calls = [(PerfectModel(), ckpt_meta), (PerfectModel(), ckpt_meta)]

        def fake_load(path, device):
            return load_calls.pop(0)

        monkeypatch.setattr(ensemble_eval, "load_checkpoint", fake_load)
        monkeypatch.setattr(ensemble_eval, "build_eval_loader", lambda cfg, split: loader)
        monkeypatch.setattr(eval_mod, "build_eval_loader", lambda cfg, split: loader, raising=False)

        ckpt_paths = [tmp_path / "fold1.pt", tmp_path / "fold2.pt"]
        for p in ckpt_paths:
            p.write_bytes(b"")

        result = ensemble_eval.ensemble_evaluate(
            ckpt_paths, split="test", device=torch.device("cpu")
        )

        assert result["ensemble_metrics"]["auc"] == pytest.approx(1.0)
        assert result["per_fold_aucs"] == [pytest.approx(1.0), pytest.approx(1.0)]


class TestTTAInference:
    def test_tta_in_run_inference_changes_result_for_asymmetric_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TTA averages original + flipped prediction; for an asymmetric model they differ."""
        from backend.ml.training.eval import run_inference

        class AsymModel(torch.nn.Module):
            """Outputs differ for original vs horizontally-flipped input."""

            def forward(self, x):
                # Sum the rightmost column — differs between original and flip
                return x[:, :, :, -1].sum(dim=[1, 2]).unsqueeze(1)

        class SimpleDS(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                img = torch.zeros(3, 4, 4)
                img[:, :, -1] = 1.0  # asymmetric
                return {
                    "image": img,
                    "label": torch.tensor(idx % 2, dtype=torch.long),
                    "patient_id": f"p{idx}",
                }

        loader = DataLoader(SimpleDS(), batch_size=2, shuffle=False)
        model = AsymModel()
        device = torch.device("cpu")

        probs_plain, _, _ = run_inference(model, loader, device, tta=False)
        probs_tta, _, _ = run_inference(model, loader, device, tta=True)

        # With TTA, flipping puts the 1-column on the left → logits differ → probs differ
        assert not np.allclose(probs_plain, probs_tta)
