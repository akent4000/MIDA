from __future__ import annotations

import io

from fastapi.testclient import TestClient


def test_upload_png(client: TestClient, png_bytes: bytes) -> None:
    resp = client.post(
        "/api/v1/studies",
        files={"file": ("chest.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data
    assert data["file_format"] == "png"
    assert data["anonymized"] is True
    assert data["file_size"] == len(png_bytes)


def test_upload_empty_file(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/studies",
        files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert resp.status_code == 422


def test_upload_garbage(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/studies",
        files={"file": ("bad.dcm", io.BytesIO(b"\x00" * 10), "application/octet-stream")},
    )
    assert resp.status_code == 422


def test_get_study(client: TestClient, png_bytes: bytes) -> None:
    upload = client.post(
        "/api/v1/studies",
        files={"file": ("chest.png", io.BytesIO(png_bytes), "image/png")},
    )
    study_id = upload.json()["id"]

    resp = client.get(f"/api/v1/studies/{study_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == study_id


def test_get_study_not_found(client: TestClient) -> None:
    import uuid

    resp = client.get(f"/api/v1/studies/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_study_image(client: TestClient, png_bytes: bytes) -> None:
    upload = client.post(
        "/api/v1/studies",
        files={"file": ("chest.png", io.BytesIO(png_bytes), "image/png")},
    )
    study_id = upload.json()["id"]

    resp = client.get(f"/api/v1/studies/{study_id}/image")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 0


def test_get_image_not_found(client: TestClient) -> None:
    import uuid

    resp = client.get(f"/api/v1/studies/{uuid.uuid4()}/image")
    assert resp.status_code == 404
