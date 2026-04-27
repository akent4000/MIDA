from celery import Celery

from backend.app.core.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "mira",
    broker=_settings.REDIS_URL,
    backend=_settings.REDIS_URL,
    include=["backend.app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=1,
    worker_pool="solo",
)
