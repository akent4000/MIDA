"""Celery inference task.

The task is intentionally self-contained: it rebuilds the registry and loads
weights on first call, then reuses them for subsequent calls in the same worker
process (module-level singleton, safe with --pool=solo).
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from celery import Task

from backend.app.workers.celery_app import celery_app

# ---------------------------------------------------------------------------
# Per-worker registry singleton (loaded once, reused across tasks)
# ---------------------------------------------------------------------------

_registry: Any = None  # ToolRegistry | None


def _get_registry() -> Any:
    global _registry
    if _registry is None:
        from backend.app.core.config import get_settings
        from backend.app.modules.ml_tools.registry import build_registry

        settings = get_settings()
        _registry = build_registry()
        if settings.PNEUMONIA_WEIGHTS_PATH:
            weights = Path(settings.PNEUMONIA_WEIGHTS_PATH)
            if weights.exists():
                _registry.load("pneumonia_classifier_v1", weights)
    return _registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _publish(task_id: str, data: dict[str, Any]) -> None:
    try:
        import redis as _redis

        from backend.app.core.config import get_settings

        r = _redis.Redis.from_url(get_settings().REDIS_URL)
        r.publish(f"task:{task_id}", json.dumps(data))
        r.close()
    except Exception:
        pass  # pub/sub is best-effort; don't let it break the task


def _update_db(engine: Any, inference_result_id: str, status: str, **kwargs: Any) -> None:
    from sqlmodel import Session

    from backend.app.models.inference_result import InferenceResult

    with Session(engine) as s:
        ir = s.get(InferenceResult, uuid.UUID(inference_result_id))
        if ir is None:
            return
        ir.status = status
        for k, v in kwargs.items():
            setattr(ir, k, v)
        if status in ("done", "failed"):
            ir.completed_at = datetime.now(UTC)
        s.add(ir)
        s.commit()


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@celery_app.task(bind=True, name="mida.run_inference")
def run_inference(
    self: Task,
    study_id: str,
    inference_result_id: str,
    tool_id: str,
) -> dict[str, Any]:
    """Full pipeline: download → load → preprocess → predict → postprocess → explain."""
    from sqlmodel import Session

    from backend.app.core.config import get_settings
    from backend.app.core.database import get_engine
    from backend.app.models.study import Study
    from backend.app.modules.dicom.service import DicomService
    from backend.app.modules.postprocessing.pipeline import PostprocessingPipeline
    from backend.app.modules.preprocessing.pipeline import PreprocessingPipeline
    from backend.app.modules.storage.service import StorageService

    task_id: str = self.request.id
    settings = get_settings()
    engine = get_engine()

    def _update(status: str, **kw: Any) -> None:
        _update_db(engine, inference_result_id, status, **kw)
        _publish(task_id, {"task_id": task_id, "status": status, **kw})

    try:
        _update("running")

        with Session(engine) as s:
            study = s.get(Study, uuid.UUID(study_id))
            if study is None:
                raise ValueError(f"Study {study_id} not found")
            file_key = study.file_key

        storage = StorageService(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            bucket=settings.MINIO_BUCKET,
            secure=settings.MINIO_SECURE,
        )
        raw = storage.download(file_key)

        dicom_svc = DicomService()
        loaded_study = dicom_svc.load(raw)

        registry = _get_registry()
        tool = registry.get(tool_id)

        # Apply user-configured settings before predict() so toggles made via
        # the API take effect on the very next task without a worker restart.
        with Session(engine) as s:
            from backend.app.services.tool_settings_service import apply_settings_from_db

            apply_settings_from_db(s, registry, tool_id)

        preprocessor = PreprocessingPipeline.from_config(tool.get_preprocessing_config())
        image = preprocessor.apply(loaded_study.pixel_data)

        raw_result = tool.predict(image)

        pp_result = PostprocessingPipeline().apply(raw_result)

        gradcam_key: str | None = None
        if settings.INFERENCE_BACKEND == "pytorch":
            gradcam_key = _try_gradcam(tool, image, storage, study_id, inference_result_id)

        result_dict: dict[str, Any] = {
            "interpretation": pp_result.interpretation,
            "confidence_band": pp_result.confidence_band,
            "extra": pp_result.extra,
            "raw": {
                "tool_id": pp_result.raw.tool_id,
                "prob": getattr(pp_result.raw, "prob", None),
                "label": getattr(pp_result.raw, "label", None),
                "label_name": getattr(pp_result.raw, "label_name", None),
                "threshold": getattr(pp_result.raw, "threshold", None),
                "class_names": getattr(pp_result.raw, "class_names", None),
                "metadata": getattr(pp_result.raw, "metadata", {}),
            },
        }

        _update("done", result_json=json.dumps(result_dict), gradcam_key=gradcam_key)
        return result_dict

    except Exception as exc:
        _update("failed", error_message=str(exc))
        raise


def _try_gradcam(
    tool: Any,
    image: Any,
    storage: Any,
    study_id: str,
    inference_result_id: str,
) -> str | None:
    """Compute Grad-CAM and upload to MinIO. Returns object key or None."""
    try:
        layer_name = tool.get_gradcam_target_layer()
        if not layer_name:
            return None

        from backend.app.modules.inference.pytorch_impl import PyTorchInference

        # Grad-CAM is computed off the single fine-tuned model regardless of
        # the active inference mode (ensemble/single) — visualising any one
        # of 5 fold gradients would be misleading and slow.
        inference_backend = getattr(tool, "_single", None)
        if not isinstance(inference_backend, PyTorchInference):
            return None

        model = getattr(inference_backend, "_model", None)
        if model is None:
            return None

        from backend.app.modules.explainability.gradcam import GradCAMExplainer

        with GradCAMExplainer.from_layer_name(model, layer_name) as cam:
            heatmap = cam.explain(image)

        png_bytes = _heatmap_to_png(heatmap)
        key = f"studies/{study_id}/inference/{inference_result_id}/gradcam.png"
        storage.upload(key, png_bytes, content_type="image/png")
        return key

    except Exception:
        return None


def _heatmap_to_png(heatmap: Any) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
    ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()
