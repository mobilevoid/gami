"""Celery application configuration for GAMI workers."""
from celery import Celery

celery_app = Celery(
    "gami",
    broker="redis://127.0.0.1:6380/0",
    backend="redis://127.0.0.1:6380/1",
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Task routing
    task_routes={
        "gami.parse_source": {"queue": "parse"},
        "gami.embed_segments": {"queue": "embed"},
        "gami.extract_entities": {"queue": "extract"},
        "gami.extract_from_segment": {"queue": "extract"},
        "gami.summarize_source": {"queue": "extract"},
        "gami.resolve_entities": {"queue": "background"},
        "gami.detect_contradictions": {"queue": "background"},
        "gami.update_importance": {"queue": "background"},
        "gami.warm_cache": {"queue": "background"},
    },
    # Celery Beat schedule for background optimization workers
    beat_schedule={
        "importance-scoring-hourly": {
            "task": "gami.update_importance",
            "schedule": 3600.0,
        },
        "cache-warming-6h": {
            "task": "gami.warm_cache",
            "schedule": 21600.0,
        },
        "entity-resolution-daily": {
            "task": "gami.resolve_entities",
            "schedule": 86400.0,
        },
        "contradiction-detection-daily": {
            "task": "gami.detect_contradictions",
            "schedule": 86400.0,
        },
    },
    # Concurrency limit for extract queue (don't overwhelm vLLM)
    worker_concurrency=2,
    # Result expiry: 24 hours
    result_expires=86400,
)

# Explicitly import task modules so @celery_app.task decorators register.
# autodiscover_tasks(["workers"]) only looks for workers/tasks.py, which doesn't
# exist — tasks are spread across *_worker.py files. Without these imports the
# worker boots with zero registered tasks and silently drops every job.
from workers import (  # noqa: E402,F401
    parser_worker,
    embedder_worker,
    extractor_worker,
    summarizer_worker,
    entity_resolver,
    contradiction_worker,
    importance_worker,
    cache_warmer,
)
