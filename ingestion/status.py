"""
Status tracking for ingestion jobs.
Uses a synchronous SQLAlchemy engine — Celery workers are sync processes
and asyncpg connections cannot be shared across event loops created by asyncio.run().

Status ordering (monotonically increasing):
  queued → parsing → enriching → indexing → indexed
  failed / duplicate are terminal at any point.

Rules:
  - chunk_count always increments, never overwrites
  - status never goes backwards (no indexing after indexed, etc.)
  - INDEXED is always written, regardless of current status (it's terminal)
  - FAILED can always be written
"""
from __future__ import annotations

from enum import Enum
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config.settings import settings

# Strip ssl params from URL — psycopg2 uses sslmode differently
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse as _urlunparse

def _build_sync_url(url: str) -> tuple[str, dict]:
    url = url.replace("+asyncpg", "+psycopg2")
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    ssl_val = params.pop("ssl", None) or params.pop("sslmode", None)
    clean_query = urlencode({k: v[0] for k, v in params.items()})
    clean_url = _urlunparse(parsed._replace(query=clean_query))
    # psycopg2 uses sslmode in connect_args
    connect_args = {"sslmode": "require"} if ssl_val else {}
    return clean_url, connect_args

_sync_url, _sync_connect_args = _build_sync_url(settings.postgres_url)
_sync_engine = create_engine(_sync_url, pool_size=5, max_overflow=10,
                              connect_args=_sync_connect_args)
SyncSession = sessionmaker(_sync_engine)

# Status rank — higher = further along. Never go backwards.
_RANK = {
    "queued": 0, "parsing": 1, "enriching": 2,
    "indexing": 3, "indexed": 4,
    "failed": 99, "duplicate": 99,
}


class IngestionStatus(str, Enum):
    QUEUED    = "queued"
    PARSING   = "parsing"
    ENRICHING = "enriching"
    INDEXING  = "indexing"
    INDEXED   = "indexed"
    FAILED    = "failed"
    DUPLICATE = "duplicate"


def update_status(doc_id: str, status: IngestionStatus,
                  page_count: int = None, chunk_delta: int = 0,
                  error: str = None) -> None:
    now = datetime.utcnow()
    params: dict = {"doc_id": doc_id, "status": status.value,
                    "error": error, "now": now}

    set_clauses = ["error_message = :error"]

    # chunk_count always increments (never resets)
    if chunk_delta:
        set_clauses.append("chunk_count = COALESCE(chunk_count, 0) + :delta")
        params["delta"] = chunk_delta

    if page_count is not None:
        set_clauses.append("page_count = :page_count")
        params["page_count"] = page_count

    # Terminal statuses (FAILED, DUPLICATE, INDEXED) always write unconditionally
    terminal = {IngestionStatus.FAILED, IngestionStatus.DUPLICATE, IngestionStatus.INDEXED}
    if status in terminal:
        set_clauses.append("status = :status")
        if status == IngestionStatus.INDEXED:
            set_clauses.append("completed_at = :now")
        if status == IngestionStatus.PARSING:
            set_clauses.append("started_at = :now")
        sql = f"UPDATE ingestion_jobs SET {', '.join(set_clauses)} WHERE doc_id = :doc_id"
    else:
        # Non-terminal: only advance forward, never go backwards
        rank = _RANK.get(status.value, 0)
        params["rank"] = rank
        set_clauses.append("status = :status")
        if status == IngestionStatus.PARSING:
            set_clauses.append("started_at = :now")
        # Build status-rank lookup using CASE so we can compare in SQL
        sql = f"""
            UPDATE ingestion_jobs
            SET {', '.join(set_clauses)}
            WHERE doc_id = :doc_id
              AND CASE status
                    WHEN 'queued'    THEN 0
                    WHEN 'parsing'   THEN 1
                    WHEN 'enriching' THEN 2
                    WHEN 'indexing'  THEN 3
                    WHEN 'indexed'   THEN 4
                    ELSE 99
                  END < :rank
        """

    with SyncSession() as session:
        session.execute(text(sql), params)
        session.commit()
