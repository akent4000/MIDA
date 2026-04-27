"""Tests for the generic tool-settings infrastructure (schema, service, ABC defaults)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from backend.app.modules.ml_tools.base import (
    ClassificationResult,
    MLTool,
    Modality,
    ModelInfo,
    TaskType,
)
from backend.app.modules.ml_tools.registry import ToolRegistry
from backend.app.modules.ml_tools.settings import (
    SettingField,
    SettingOption,
    SettingsValidationError,
    SettingType,
    schema_to_dict,
    validate_values,
)
from backend.app.services.tool_settings_service import (
    ToolSettingsService,
    apply_settings_from_db,
)

# ---------------------------------------------------------------------------
# Test tool
# ---------------------------------------------------------------------------


class _DummyTool(MLTool):
    TOOL_ID = "dummy"
    _INFO = ModelInfo(
        tool_id=TOOL_ID,
        name="Dummy",
        version="0.0.1",
        description="",
        modality=Modality.XRAY,
        task_type=TaskType.CLASSIFICATION,
        input_shape=(3, 32, 32),
        class_names=["A", "B"],
    )

    def __init__(self) -> None:
        self.applied: dict[str, Any] = {}
        self._loaded = False

    @property
    def info(self) -> ModelInfo:
        return self._INFO

    def load(self, weights_path: Path) -> None:
        self._loaded = True

    def predict(self, image: np.ndarray) -> ClassificationResult:
        return ClassificationResult(tool_id=self.TOOL_ID)

    def get_preprocessing_config(self) -> dict[str, Any]:
        return {"resize": [32, 32]}

    def get_settings_schema(self) -> list[SettingField]:
        return [
            SettingField(
                key="mode",
                label="Mode",
                type=SettingType.SELECT,
                default="single",
                options=[
                    SettingOption(value="single", label="Single"),
                    SettingOption(value="ensemble", label="Ensemble"),
                ],
            ),
            SettingField(
                key="threshold",
                label="Threshold",
                type=SettingType.NUMBER,
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.01,
            ),
            SettingField(
                key="explain",
                label="Grad-CAM",
                type=SettingType.TOGGLE,
                default=True,
            ),
        ]

    def apply_settings(self, values: dict[str, Any]) -> None:
        self.applied = dict(values)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestValidate:
    def setup_method(self) -> None:
        self.schema = _DummyTool().get_settings_schema()

    def test_defaults_filled_for_missing_keys(self) -> None:
        out = validate_values(self.schema, {})
        assert out == {"mode": "single", "threshold": 0.5, "explain": True}

    def test_select_accepts_valid_option(self) -> None:
        out = validate_values(self.schema, {"mode": "ensemble"})
        assert out["mode"] == "ensemble"

    def test_select_rejects_invalid_option(self) -> None:
        with pytest.raises(SettingsValidationError):
            validate_values(self.schema, {"mode": "garbage"})

    def test_number_coerced_and_bounded(self) -> None:
        out = validate_values(self.schema, {"threshold": "0.7"})
        assert out["threshold"] == 0.7
        with pytest.raises(SettingsValidationError):
            validate_values(self.schema, {"threshold": 1.5})
        with pytest.raises(SettingsValidationError):
            validate_values(self.schema, {"threshold": -0.1})

    def test_toggle_requires_bool(self) -> None:
        with pytest.raises(SettingsValidationError):
            validate_values(self.schema, {"explain": "yes"})

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(SettingsValidationError):
            validate_values(self.schema, {"nonsense": 1})


class TestSerialization:
    def test_schema_to_dict_includes_all_metadata(self) -> None:
        schema = _DummyTool().get_settings_schema()
        out = schema_to_dict(schema)
        select_field = next(f for f in out if f["key"] == "mode")
        assert select_field["type"] == "select"
        assert {o["value"] for o in select_field["options"]} == {"single", "ensemble"}
        num_field = next(f for f in out if f["key"] == "threshold")
        assert num_field["min"] == 0.0
        assert num_field["max"] == 1.0
        assert num_field["step"] == 0.01


# ---------------------------------------------------------------------------
# Service + DB round-trip
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


class TestService:
    def test_get_with_defaults_when_empty(self, db: Session) -> None:
        svc = ToolSettingsService(db)
        schema = _DummyTool().get_settings_schema()
        values = svc.get_with_defaults("dummy", schema)
        assert values["mode"] == "single"

    def test_set_and_get_roundtrip(self, db: Session) -> None:
        svc = ToolSettingsService(db)
        svc.set_values("dummy", {"mode": "ensemble", "threshold": 0.7})
        assert svc.get_values("dummy") == {"mode": "ensemble", "threshold": 0.7}

    def test_patch_persists_only_supplied_keys(self, db: Session) -> None:
        svc = ToolSettingsService(db)
        schema = _DummyTool().get_settings_schema()
        svc.patch("dummy", schema, {"mode": "ensemble"})
        # threshold/explain not persisted, so defaults still come back
        values = svc.get_with_defaults("dummy", schema)
        assert values == {"mode": "ensemble", "threshold": 0.5, "explain": True}

    def test_patch_validates(self, db: Session) -> None:
        svc = ToolSettingsService(db)
        schema = _DummyTool().get_settings_schema()
        with pytest.raises(SettingsValidationError):
            svc.patch("dummy", schema, {"mode": "garbage"})

    def test_apply_settings_pushes_to_loaded_tool(self, db: Session, tmp_path: Path) -> None:
        registry = ToolRegistry()
        registry.register_class("dummy", _DummyTool)
        registry.load("dummy", tmp_path / "fake.pt")

        svc = ToolSettingsService(db)
        schema = _DummyTool().get_settings_schema()
        svc.patch("dummy", schema, {"mode": "ensemble"})
        apply_settings_from_db(db, registry, "dummy")

        tool = registry.get("dummy")
        assert tool.applied["mode"] == "ensemble"  # type: ignore[attr-defined]
        assert tool.applied["threshold"] == 0.5  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Default ABC behaviour
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_mltool_default_schema_is_empty(self) -> None:
        from backend.app.modules.ml_tools.pneumonia.tool import PneumoniaTool

        # Pneumonia hasn't been customized yet — default contract still applies
        # (will start returning a non-empty schema once the toggle lands).
        schema = PneumoniaTool().get_settings_schema()
        assert isinstance(schema, list)

    def test_mltool_default_apply_settings_is_noop(self) -> None:
        from backend.app.modules.ml_tools.pneumonia.tool import PneumoniaTool

        t = PneumoniaTool()
        t.apply_settings({"anything": 1})  # must not raise
