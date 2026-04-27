"""Tests for PneumoniaTool — single↔ensemble mode toggle, lazy ensemble load."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.modules.inference.base import ModelInference, Prediction
from backend.app.modules.ml_tools.pneumonia.tool import (
    ENSEMBLE_THRESHOLD,
    MODE_ENSEMBLE,
    MODE_SINGLE,
    PneumoniaTool,
)
from backend.app.modules.ml_tools.settings import SettingType


def _fake_backend(prob: float, threshold: float = 0.5) -> MagicMock:
    b = MagicMock(spec=ModelInference)
    b.predict.return_value = Prediction(
        prob=prob, label=int(prob >= threshold), threshold=threshold
    )
    return b


@pytest.fixture()
def loaded_tool() -> PneumoniaTool:
    """A PneumoniaTool with a fake single backend already injected."""
    t = PneumoniaTool()
    t._single = _fake_backend(prob=0.7)
    t._loaded = True
    return t


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_includes_mode(self) -> None:
        schema = PneumoniaTool().get_settings_schema()
        keys = [f.key for f in schema]
        assert "mode" in keys

    def test_mode_field_is_select_with_two_options(self) -> None:
        schema = PneumoniaTool().get_settings_schema()
        mode = next(f for f in schema if f.key == "mode")
        assert mode.type == SettingType.SELECT
        opt_values = [o.value for o in mode.options]
        assert opt_values == [MODE_SINGLE, MODE_ENSEMBLE]
        assert mode.default == MODE_SINGLE


# ---------------------------------------------------------------------------
# Apply settings
# ---------------------------------------------------------------------------


class TestApplySettings:
    def test_default_mode_is_single(self) -> None:
        t = PneumoniaTool()
        assert t._mode == MODE_SINGLE

    def test_apply_switches_to_ensemble_when_paths_configured(
        self, loaded_tool: PneumoniaTool, tmp_path: Path
    ) -> None:
        loaded_tool._ensemble_paths = [tmp_path / "fold1.pt", tmp_path / "fold2.pt"]
        loaded_tool.apply_settings({"mode": MODE_ENSEMBLE})
        assert loaded_tool._mode == MODE_ENSEMBLE

    def test_apply_falls_back_when_no_ensemble_paths(self, loaded_tool: PneumoniaTool) -> None:
        loaded_tool._ensemble_paths = []
        loaded_tool.apply_settings({"mode": MODE_ENSEMBLE})
        assert loaded_tool._mode == MODE_SINGLE  # silently downgraded

    def test_apply_ignores_unknown_mode(self, loaded_tool: PneumoniaTool) -> None:
        loaded_tool.apply_settings({"mode": "junk"})
        assert loaded_tool._mode == MODE_SINGLE

    def test_apply_round_trip_single(self, loaded_tool: PneumoniaTool, tmp_path: Path) -> None:
        loaded_tool._ensemble_paths = [tmp_path / "fold1.pt", tmp_path / "fold2.pt"]
        loaded_tool.apply_settings({"mode": MODE_ENSEMBLE})
        loaded_tool.apply_settings({"mode": MODE_SINGLE})
        assert loaded_tool._mode == MODE_SINGLE


# ---------------------------------------------------------------------------
# Predict (single mode)
# ---------------------------------------------------------------------------


class TestPredictSingle:
    def test_single_returns_classification_result(self, loaded_tool: PneumoniaTool) -> None:
        img = np.zeros((3, 384, 384), dtype=np.float32)
        result = loaded_tool.predict(img)
        assert result.tool_id == PneumoniaTool.TOOL_ID
        assert result.prob == 0.7
        assert result.metadata["mode"] == MODE_SINGLE
        assert result.metadata["n_models"] == 1

    def test_predict_before_load_raises(self) -> None:
        t = PneumoniaTool()
        with pytest.raises(RuntimeError):
            t.predict(np.zeros((3, 384, 384), dtype=np.float32))


# ---------------------------------------------------------------------------
# Predict (ensemble mode) — lazy load + soft-vote
# ---------------------------------------------------------------------------


class TestPredictEnsemble:
    def test_ensemble_lazy_loads_and_soft_votes(
        self, loaded_tool: PneumoniaTool, tmp_path: Path
    ) -> None:
        # Three "fold" probs — mean = 0.6, above 0.4396 threshold → label 1
        fold_probs = [0.4, 0.6, 0.8]
        fake_backends = [_fake_backend(p) for p in fold_probs]

        paths = []
        for i, _ in enumerate(fold_probs):
            p = tmp_path / f"fold{i+1}.pt"
            p.write_bytes(b"fake")
            paths.append(p)
        loaded_tool._ensemble_paths = paths
        loaded_tool.apply_settings({"mode": MODE_ENSEMBLE})

        # Patch the backend factory used inside _ensure_ensemble_loaded
        with patch(
            "backend.app.modules.inference.get_inference_backend",
            side_effect=fake_backends,
        ):
            img = np.zeros((3, 384, 384), dtype=np.float32)
            result = loaded_tool.predict(img)

        assert result.metadata["mode"] == MODE_ENSEMBLE
        assert result.metadata["n_models"] == 3
        assert result.threshold == ENSEMBLE_THRESHOLD
        assert result.prob == pytest.approx(np.mean(fold_probs))
        assert result.label == 1  # 0.6 >= 0.4396

    def test_ensemble_caches_loaded_backends(
        self, loaded_tool: PneumoniaTool, tmp_path: Path
    ) -> None:
        path = tmp_path / "fold1.pt"
        path.write_bytes(b"fake")
        path2 = tmp_path / "fold2.pt"
        path2.write_bytes(b"fake")
        loaded_tool._ensemble_paths = [path, path2]
        loaded_tool.apply_settings({"mode": MODE_ENSEMBLE})

        factory = MagicMock(side_effect=[_fake_backend(0.5), _fake_backend(0.5)])
        with patch("backend.app.modules.inference.get_inference_backend", factory):
            img = np.zeros((3, 384, 384), dtype=np.float32)
            loaded_tool.predict(img)
            loaded_tool.predict(img)

        # Factory called only on first predict — cached for second
        assert factory.call_count == 2  # one per fold, NOT 4

    def test_ensemble_missing_path_raises(self, loaded_tool: PneumoniaTool, tmp_path: Path) -> None:
        loaded_tool._ensemble_paths = [tmp_path / "missing.pt"]
        loaded_tool._mode = MODE_ENSEMBLE  # bypass apply_settings fallback
        with pytest.raises(FileNotFoundError):
            loaded_tool.predict(np.zeros((3, 384, 384), dtype=np.float32))


# ---------------------------------------------------------------------------
# Env var parsing
# ---------------------------------------------------------------------------


class TestEnsemblePaths:
    def test_empty_env_means_no_ensemble(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PNEUMONIA_ENSEMBLE_PATHS", raising=False)
        t = PneumoniaTool()
        assert t._ensemble_paths == []

    def test_comma_separated_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "PNEUMONIA_ENSEMBLE_PATHS",
            "/tmp/a.pt, /tmp/b.pt , /tmp/c.pt",
        )
        t = PneumoniaTool()
        assert t._ensemble_paths == [Path("/tmp/a.pt"), Path("/tmp/b.pt"), Path("/tmp/c.pt")]

    def test_schema_describes_unavailable_when_no_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PNEUMONIA_ENSEMBLE_PATHS", raising=False)
        schema = PneumoniaTool().get_settings_schema()
        mode = next(f for f in schema if f.key == "mode")
        ensemble_opt = next(o for o in mode.options if o.value == MODE_ENSEMBLE)
        assert "Unavailable" in ensemble_opt.description
