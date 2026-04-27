from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_models_returns_pneumonia(client: TestClient) -> None:
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    items = resp.json()
    assert isinstance(items, list)
    assert len(items) >= 1
    tool_ids = [item["tool_id"] for item in items]
    assert "pneumonia_classifier_v1" in tool_ids


def test_model_schema(client: TestClient) -> None:
    resp = client.get("/api/v1/models")
    item = next(i for i in resp.json() if i["tool_id"] == "pneumonia_classifier_v1")
    assert item["name"] == "RSNA Pneumonia Classifier"
    assert item["modality"] == "xray"
    assert item["task_type"] == "classification"
    assert item["class_names"] == ["Normal", "Pneumonia"]
    assert item["input_shape"] == [3, 384, 384]
    assert item["loaded"] is False  # no weights in test env


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
