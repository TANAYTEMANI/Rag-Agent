from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from config.settings import settings

engine = create_async_engine(settings.postgres_url, echo=False, pool_size=10, max_overflow=20)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


# ── Documents ──────────────────────────────────────────────────────────────

async def create_document(session: AsyncSession, doc_id: str, file_name: str,
                          file_hash: str, doc_type: str = None,
                          department: str = None, language: str = None) -> None:
    await session.execute(text("""
        INSERT INTO documents (doc_id, file_name, file_hash, doc_type, department, language)
        VALUES (:doc_id, :file_name, :file_hash, :doc_type, :department, :language)
        ON CONFLICT (doc_id) DO NOTHING
    """), {"doc_id": doc_id, "file_name": file_name, "file_hash": file_hash,
           "doc_type": doc_type, "department": department, "language": language})
    await session.commit()


async def get_document_by_hash(session: AsyncSession, file_hash: str) -> dict | None:
    result = await session.execute(
        text("SELECT * FROM documents WHERE file_hash = :h"), {"h": file_hash}
    )
    row = result.mappings().first()
    return dict(row) if row else None


async def supersede_document(session: AsyncSession, old_doc_id: str, new_doc_id: str) -> None:
    await session.execute(text("""
        UPDATE documents SET superseded_by = :new_id WHERE doc_id = :old_id
    """), {"new_id": new_doc_id, "old_id": old_doc_id})
    await session.commit()


# ── Ingestion jobs ─────────────────────────────────────────────────────────

async def create_job(session: AsyncSession, doc_id: str, file_name: str,
                     file_hash: str) -> None:
    await session.execute(text("""
        INSERT INTO ingestion_jobs (doc_id, file_name, file_hash, status)
        VALUES (:doc_id, :file_name, :file_hash, 'queued')
        ON CONFLICT (doc_id) DO NOTHING
    """), {"doc_id": doc_id, "file_name": file_name, "file_hash": file_hash})
    await session.commit()


async def update_job_status(session: AsyncSession, doc_id: str, status: str,
                            page_count: int = None, chunk_delta: int = 0,
                            error: str = None) -> None:
    now = datetime.utcnow()
    updates: dict[str, Any] = {"doc_id": doc_id, "status": status, "now": now, "error": error}

    set_clauses = ["status = :status", "error_message = :error"]
    if status == "parsing":
        set_clauses.append("started_at = :now")
    if status == "indexed":
        set_clauses.append("completed_at = :now")
    if page_count is not None:
        set_clauses.append("page_count = :page_count")
        updates["page_count"] = page_count
    if chunk_delta:
        set_clauses.append("chunk_count = COALESCE(chunk_count, 0) + :delta")
        updates["delta"] = chunk_delta

    await session.execute(
        text(f"UPDATE ingestion_jobs SET {', '.join(set_clauses)} WHERE doc_id = :doc_id"),
        updates
    )
    await session.commit()


async def get_job(session: AsyncSession, doc_id: str) -> dict | None:
    result = await session.execute(
        text("SELECT * FROM ingestion_jobs WHERE doc_id = :id"), {"id": doc_id}
    )
    row = result.mappings().first()
    return dict(row) if row else None


async def get_job_status_summary(session: AsyncSession) -> list[dict]:
    result = await session.execute(text("""
        SELECT status,
               COUNT(*) AS count,
               ROUND(AVG(EXTRACT(EPOCH FROM (completed_at - started_at)))::numeric, 2) AS avg_secs
        FROM ingestion_jobs
        GROUP BY status
        ORDER BY status
    """))
    return [dict(r) for r in result.mappings().all()]


# ── Tabular tables ─────────────────────────────────────────────────────────

async def upsert_tabular_table(session: AsyncSession, table_id: str, doc_id: str,
                               file_name: str, sheet_name: str | None,
                               parquet_path: str, row_count: int, col_count: int,
                               columns: list[str], description: str) -> None:
    await session.execute(text("""
        INSERT INTO tabular_tables
            (table_id, doc_id, file_name, sheet_name, parquet_path,
             row_count, col_count, columns, description)
        VALUES (:tid, :did, :fn, :sn, :pp, :rc, :cc, :cols, :desc)
        ON CONFLICT (table_id) DO UPDATE SET
            parquet_path = EXCLUDED.parquet_path,
            description  = EXCLUDED.description
    """), {"tid": table_id, "did": doc_id, "fn": file_name, "sn": sheet_name,
           "pp": parquet_path, "rc": row_count, "cc": col_count,
           "cols": columns, "desc": description})
    await session.commit()


async def search_tabular_tables(session: AsyncSession, keywords: list[str],
                                limit: int = 5) -> list[dict]:
    if not keywords:
        return []
    pattern = " | ".join(keywords)
    result = await session.execute(text("""
        SELECT table_id, doc_id, file_name, sheet_name, description
        FROM tabular_tables
        WHERE to_tsvector('english', description) @@ to_tsquery('english', :q)
        LIMIT :lim
    """), {"q": pattern, "lim": limit})
    return [dict(r) for r in result.mappings().all()]


async def get_tabular_table(session: AsyncSession, table_id: str) -> dict | None:
    result = await session.execute(
        text("SELECT * FROM tabular_tables WHERE table_id = :id"), {"id": table_id}
    )
    row = result.mappings().first()
    return dict(row) if row else None
