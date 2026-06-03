"""
Celery application.

Parse workers no longer load Docling at startup — documents are parsed
via the Jina Reader API (async HTTP, ~300ms/page) instead of local OCR.
This eliminates the ~600MB model download and ~30s/page parsing time.

Queue split:
  parse  → parse_document (I/O-bound Jina API calls, 8 workers safe)
  embed  → enrich_and_embed_text, embed_image_pages, etc. (8 workers)
"""
from __future__ import annotations

import os

import redis as redis_lib
from celery import Celery
from celery.signals import worker_process_init

from config.settings import settings

app = Celery(
    "rag_ingestion",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["ingestion.tasks"],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_routes={
        "ingestion.tasks.parse_document":        {"queue": "parse"},
        "ingestion.tasks.enrich_and_embed_text":  {"queue": "embed"},
        "ingestion.tasks.embed_image_pages":      {"queue": "embed"},
        "ingestion.tasks.generate_and_embed_summary": {"queue": "embed"},
        "ingestion.tasks.pandas_ingest":          {"queue": "embed"},
    },
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

_redis_client: redis_lib.Redis | None = None


@worker_process_init.connect
def init_worker(**kwargs):
    """Initialise Redis connection once per worker process. No ML models to load."""
    global _redis_client
    _redis_client = redis_lib.Redis.from_url(settings.redis_url)
    print(f"[Worker {os.getpid()}] Ready (Jina Reader mode — no local models).")


def get_redis() -> redis_lib.Redis:
    return _redis_client


# get_converter kept for backward-compat but returns None — not used anymore
def get_converter():
    return None
