"""End-to-end tests for /api/v1/tools/{tool_id}/config."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.modules.ml_tools.base import MLTool, Modality, ModelInfo, TaskType
from backend.app.modules.ml_tools.settings import (
    SettingField,
    SettingOption,
    SettingType,
)


class _ToolWithSettings(MLTool):
    """Drop-in stand-in registered alongside pneumonia for these tests."""

    TOOL_ID = "tool_with_settings"
    _INFO = ModelInfo(
        tool_id=TOOL_ID,
        name="WithSettings",
        version="0.0.1",
        description="",
        modality=Modality.XRAY,
        task_type=TaskType.CLASSIFICATION,
        input_shape=(3, 16, 16),
        class_names=["A"],
    )

    @property
    def info(self) -> ModelInfo:
        return self._INFO

    def load(self, weights_path: Any) -> None: ...
    def predict(self, image: Any) -> Any: ...
    def get_preprocessing_config(self) -> dict[str, Any]:
        return {}

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
        ]


@pytest.fixture()
def client_with_settings_tool(client: TestClient) -> TestClient:
    """Inject an extra tool into the registry override used by the client fixture."""
    from backend.app.core.dependencies import get_registry

    # Reuse the same registry instance the client already has so changes stick
    # within one test (FastAPI calls the override factory per request).
    registry = client.app.dependency_overrides[get_registry]()  # type: ignore[no-any-return]
    registry.register_class(_ToolWithSettings.TOOL_ID, _ToolWithSettings)
    client.app.dependency_overrides[get_registry] = lambda: registry
    return client


def test_get_returns_schema_and_defaults(client_with_settings_tool: TestClient) -> None:
    resp = client_with_settings_tool.get("/api/v1/tools/tool_with_settings/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tool_id"] == "tool_with_settings"
    assert body["values"] == {"mode": "single"}
    assert body["schema"][0]["key"] == "mode"
    assert body["schema"][0]["type"] == "select"


def test_patch_persists_change(client_with_settings_tool: TestClient) -> None:
    c = client_with_settings_tool
    resp = c.patch(
        "/api/v1/tools/tool_with_settings/config",
        json={"values": {"mode": "ensemble"}},
    )
    assert resp.status_code == 200
    assert resp.json()["values"]["mode"] == "ensemble"

    # Re-read confirms persistence
    resp2 = c.get("/api/v1/tools/tool_with_settings/config")
    assert resp2.json()["values"]["mode"] == "ensemble"


def test_patch_invalid_value_returns_422(client_with_settings_tool: TestClient) -> None:
    resp = client_with_settings_tool.patch(
        "/api/v1/tools/tool_with_settings/config",
        json={"values": {"mode": "garbage"}},
    )
    assert resp.status_code == 422


def test_unknown_tool_returns_404(client: TestClient) -> None:
    resp = client.get("/api/v1/tools/nonexistent/config")
    assert resp.status_code == 404


class _NoSettingsTool(MLTool):
    TOOL_ID = "no_settings_tool"
    _INFO = ModelInfo(
        tool_id=TOOL_ID,
        name="NoSettings",
        version="0.0.1",
        description="",
        modality=Modality.XRAY,
        task_type=TaskType.CLASSIFICATION,
        input_shape=(3, 16, 16),
        class_names=["A"],
    )

    @property
    def info(self) -> ModelInfo:
        return self._INFO

    def load(self, weights_path: Any) -> None: ...
    def predict(self, image: Any) -> Any: ...
    def get_preprocessing_config(self) -> dict[str, Any]:
        return {}

    # Inherits the default empty get_settings_schema() from MLTool.


def test_tool_without_settings_patch_rejected(client: TestClient) -> None:
    from backend.app.core.dependencies import get_registry

    registry = client.app.dependency_overrides[get_registry]()
    registry.register_class(_NoSettingsTool.TOOL_ID, _NoSettingsTool)
    client.app.dependency_overrides[get_registry] = lambda: registry

    resp = client.patch(
        "/api/v1/tools/no_settings_tool/config",
        json={"values": {"foo": "bar"}},
    )
    assert resp.status_code == 400
