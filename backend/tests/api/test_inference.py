from __future__ import annotations

import io
import uuid
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _upload_study(client: TestClient, png_bytes: bytes) -> str:
    resp = client.post(
        "/api/v1/studies",
        files={"file": ("chest.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def test_submit_inference(client: TestClient, png_bytes: bytes) -> None:
    study_id = _upload_study(client, png_bytes)

    mock_result = MagicMock()
    mock_result.id = str(uuid.uuid4())

    with patch(
        "backend.app.workers.tasks.run_inference.apply_async",
        return_value=mock_result,
    ):
        resp = client.post(
            f"/api/v1/studies/{study_id}/inference",
            json={"tool_id": "pneumonia_classifier_v1"},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert "inference_id" in data
    assert "task_id" in data
    assert data["status"] == "pending"
    assert data["tool_id"] == "pneumonia_classifier_v1"


def test_submit_inference_study_not_found(client: TestClient) -> None:
    resp = client.post(
        f"/api/v1/studies/{uuid.uuid4()}/inference",
        json={"tool_id": "pneumonia_classifier_v1"},
    )
    assert resp.status_code == 404


def test_submit_inference_unknown_tool(client: TestClient, png_bytes: bytes) -> None:
    study_id = _upload_study(client, png_bytes)
    resp = client.post(
        f"/api/v1/studies/{study_id}/inference",
        json={"tool_id": "nonexistent_tool"},
    )
    assert resp.status_code == 400


def test_get_inference_result(client: TestClient, png_bytes: bytes) -> None:
    study_id = _upload_study(client, png_bytes)

    mock_result = MagicMock()
    mock_result.id = str(uuid.uuid4())

    with patch(
        "backend.app.workers.tasks.run_inference.apply_async",
        return_value=mock_result,
    ):
        submit = client.post(
            f"/api/v1/studies/{study_id}/inference",
            json={"tool_id": "pneumonia_classifier_v1"},
        )
    inference_id = submit.json()["inference_id"]

    resp = client.get(f"/api/v1/inference/{inference_id}/result")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert data["tool_id"] == "pneumonia_classifier_v1"


def test_get_inference_result_not_found(client: TestClient) -> None:
    resp = client.get(f"/api/v1/inference/{uuid.uuid4()}/result")
    assert resp.status_code == 404


def test_get_explanation_no_gradcam(client: TestClient, png_bytes: bytes) -> None:
    study_id = _upload_study(client, png_bytes)

    mock_result = MagicMock()
    mock_result.id = str(uuid.uuid4())

    with patch(
        "backend.app.workers.tasks.run_inference.apply_async",
        return_value=mock_result,
    ):
        submit = client.post(
            f"/api/v1/studies/{study_id}/inference",
            json={"tool_id": "pneumonia_classifier_v1"},
        )
    inference_id = submit.json()["inference_id"]

    resp = client.get(f"/api/v1/inference/{inference_id}/explanation")
    assert resp.status_code == 404


def test_get_task_status(client: TestClient, png_bytes: bytes) -> None:
    study_id = _upload_study(client, png_bytes)

    task_id = str(uuid.uuid4())
    mock_result = MagicMock()
    mock_result.id = task_id

    with patch(
        "backend.app.workers.tasks.run_inference.apply_async",
        return_value=mock_result,
    ):
        submit = client.post(
            f"/api/v1/studies/{study_id}/inference",
            json={"tool_id": "pneumonia_classifier_v1"},
        )
    actual_task_id = submit.json()["task_id"]

    resp = client.get(f"/api/v1/tasks/{actual_task_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == actual_task_id
    assert data["status"] == "pending"


def test_get_task_not_found(client: TestClient) -> None:
    resp = client.get(f"/api/v1/tasks/{uuid.uuid4()}")
    assert resp.status_code == 404
