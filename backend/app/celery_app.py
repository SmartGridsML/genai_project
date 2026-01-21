import os

from celery import Celery

from backend.app.config import settings

celery_app = Celery(
    "cv_helper",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.autodiscover_tasks(["backend.app"])
celery_app.conf.imports = ("backend.app.tasks.application_tasks",)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_always_eager=bool(os.getenv("PYTEST_CURRENT_TEST")),
)
