"""
Celery tasks.

parse_document       → parse queue  (I/O-bound Jina Reader API, 8 workers)
enrich_and_embed_text → embed queue (I/O-bound API calls, 8 workers)
embed_image_pages    → embed queue
generate_and_embed_summary → embed queue
pandas_ingest        → embed queue

KEY CHANGE vs previous version:
  parse_document now uses Jina Reader API instead of local Docling OCR.
  - Before: Docling loads 600MB models, runs OCR per page → ~30-100s/doc
  - After:  Jina sends each page to r.jina.ai concurrently  → ~3-10s/doc
  - Speedup: 5-15x depending on page count
"""
from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from ingestion.worker import app, get_redis
from ingestion.dedup import file_hash, is_already_indexed, mark_as_indexed
from ingestion.status import IngestionStatus, update_status
from ingestion.content_extractor import chunk_batches
from config.settings import settings


# ── helpers ──────────────────────────────────────────────────────────────

def _qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        https=settings.qdrant_api_key is not None,
        check_compatibility=False,
    )


# ── Task 1: parse_document (parse queue) ─────────────────────────────────

@app.task(bind=True, max_retries=3, queue="parse", name="ingestion.tasks.parse_document")
def parse_document(self, doc_path: str, doc_id: str, file_name: str):
    """
    Fast path: Jina Reader API parses the document page-by-page in parallel.
    Each page is ~300ms. A 20-page PDF takes ~3-5s total (all pages concurrent).
    No local ML models. No Docker memory issues.
    """
    r = get_redis()

    fhash = file_hash(doc_path)
    if is_already_indexed(fhash, r):
        update_status(doc_id, IngestionStatus.DUPLICATE)
        return {"status": "duplicate", "doc_id": doc_id}

    try:
        update_status(doc_id, IngestionStatus.PARSING)
        ext = Path(doc_path).suffix.lower()

        # ── Excel/CSV → Pandas engine directly, skip text parsing ─────
        if ext in (".xlsx", ".xls", ".csv"):
            pandas_ingest.apply_async(args=[doc_path, doc_id, file_name], queue="embed")
            mark_as_indexed(fhash, doc_id, r)
            return {"status": "queued_pandas", "doc_id": doc_id}

        # ── Parse via Jina Reader (parallel async HTTP) ────────────────
        async def _parse():
            from ingestion.jina_parser import parse_document_file, pages_to_chunks
            pages = await parse_document_file(doc_path)
            doc_markdown = "\n\n".join(p["text"] for p in pages if p["text"])
            text_chunks  = pages_to_chunks(pages, doc_id, doc_markdown)
            image_pages  = [p for p in pages if p.get("b64")]
            return pages, doc_markdown, text_chunks, image_pages

        pages, doc_markdown, text_chunks, image_pages = asyncio.run(_parse())
        page_count = len(pages)

        if not text_chunks:
            # Empty document — still create a summary so it's searchable
            update_status(doc_id, IngestionStatus.ENRICHING, page_count=page_count)
            generate_and_embed_summary.apply_async(
                args=[doc_id, file_name, doc_markdown[:5000] or f"Empty document: {file_name}"],
                queue="embed",
            )
            mark_as_indexed(fhash, doc_id, r)
            return {"status": "processing", "doc_id": doc_id, "pages": page_count}

        # ── Parent-child grouping (every 4 pages → one parent) ────────
        from chunking.parent_child import build_parent_child_chunks
        all_chunks = build_parent_child_chunks(text_chunks, parent_size=4)

        # ── Dispatch embed tasks ───────────────────────────────────────
        for batch in chunk_batches(all_chunks, size=settings.chunk_batch_size):
            enrich_and_embed_text.apply_async(args=[batch, doc_id], queue="embed")

        if image_pages:
            image_data = [
                {"id": str(uuid.uuid4()), "doc_id": doc_id,
                 "page_no": p["page_no"], "b64": p["b64"]}
                for p in image_pages
            ]
            for batch in chunk_batches(image_data, size=10):
                embed_image_pages.apply_async(args=[batch, doc_id], queue="embed")

        generate_and_embed_summary.apply_async(
            args=[doc_id, file_name, doc_markdown[:5000]],
            queue="embed",
        )

        update_status(doc_id, IngestionStatus.ENRICHING, page_count=page_count)
        mark_as_indexed(fhash, doc_id, r)
        return {"status": "processing", "doc_id": doc_id, "pages": page_count}

    except Exception as exc:
        update_status(doc_id, IngestionStatus.FAILED, error=str(exc))
        raise self.retry(exc=exc, countdown=30)


def _store_large_tables(table_chunks: list[dict], doc_id: str, file_name: str):
    """Store large PDF-embedded tables as DataFrames for Pandas engine.
    Uses synchronous DB writes — safe inside Celery worker processes."""
    import json
    import pandas as pd
    from pathlib import Path as P
    from sqlalchemy import text as sql_text
    from ingestion.status import SyncSession

    store_dir = P(settings.table_store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    with SyncSession() as session:
        for chunk in table_chunks:
            df = pd.read_json(chunk["json_repr"])
            table_id = f"{doc_id}_table_{chunk['id'][:8]}"
            path = store_dir / f"{table_id}.parquet"
            df.to_parquet(path)
            session.execute(sql_text("""
                INSERT INTO tabular_tables
                    (table_id, doc_id, file_name, parquet_path,
                     row_count, col_count, columns, description)
                VALUES (:tid, :did, :fn, :pp, :rc, :cc, :cols, :desc)
                ON CONFLICT (table_id) DO UPDATE SET parquet_path = EXCLUDED.parquet_path
            """), {
                "tid": table_id, "did": doc_id, "fn": file_name,
                "pp": str(path), "rc": len(df), "cc": len(df.columns),
                "cols": list(df.columns),
                "desc": f"Table from {file_name}, page {chunk.get('page_no')}",
            })
        session.commit()


# ── Sync Jina helpers (no asyncio needed) ────────────────────────────────

def _jina_embed_sync(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    """Synchronous Jina embedding — safe to call from Celery tasks."""
    import httpx
    resp = httpx.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.jina_api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": settings.jina_embed_model,
            "input": texts,
            "task": task,
            "late_chunking": True,
            "dimensions": settings.embed_dimensions,
            "embedding_type": "float",
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def _jina_embed_images_sync(b64_images: list[str]) -> list[list[float]]:
    import httpx
    resp = httpx.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.jina_api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": settings.jina_image_model,
            "input": [{"image": b64} for b64 in b64_images],
            "task": "retrieval.passage",
            "dimensions": settings.embed_dimensions,
            "embedding_type": "float",
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


# ── Task 2: enrich_and_embed_text (embed queue) ───────────────────────────

@app.task(bind=True, max_retries=3, queue="embed", name="ingestion.tasks.enrich_and_embed_text")
def enrich_and_embed_text(self, chunk_batch: list[dict], doc_id: str):
    try:
        update_status(doc_id, IngestionStatus.ENRICHING)

        # Step 1: Contextual enrichment via Anthropic (async-only SDK, isolated run)
        to_enrich = [c for c in chunk_batch if c.get("chunk_type") not in ("parent", "summary")]
        skip      = [c for c in chunk_batch if c.get("chunk_type") in ("parent", "summary")]

        if to_enrich:
            async def _enrich():
                from chunking.contextual_enricher import enrich_batch_concurrent
                return await enrich_batch_concurrent(to_enrich)
            asyncio.run(_enrich())

        for c in skip:
            c["contextualized_text"] = c.get("text", "")

        all_chunks = to_enrich + skip
        texts = [c.get("contextualized_text") or c["text"] for c in all_chunks]

        # Step 2: Embed via Jina (synchronous HTTP)
        vectors = _jina_embed_sync(texts, task="retrieval.passage")

        # Step 3: Write to Qdrant (sync client)
        from index.qdrant_store import upsert_text_chunks_sync
        qdrant = _qdrant_client()
        points = []
        for chunk, vec in zip(all_chunks, vectors):
            points.append({
                "id":           chunk["id"],
                "dense_vector": vec,
                "payload": {
                    "text":            chunk.get("contextualized_text") or chunk["text"],
                    "original_text":   chunk["text"],
                    "doc_id":          chunk["doc_id"],
                    "chunk_type":      chunk.get("chunk_type", "text"),
                    "element_type":    chunk.get("element_type"),
                    "page_no":         chunk.get("page_no"),
                    "bbox":            chunk.get("bbox"),
                    "section_heading": chunk.get("section_heading"),
                    "parent_id":       chunk.get("parent_id"),
                },
            })
        upsert_text_chunks_sync(qdrant, points)
        update_status(doc_id, IngestionStatus.INDEXING, chunk_delta=len(points))

    except Exception as exc:
        update_status(doc_id, IngestionStatus.FAILED, error=str(exc))
        raise self.retry(exc=exc, countdown=30)


# ── Task 3: embed_image_pages (embed queue) ───────────────────────────────

@app.task(bind=True, max_retries=3, queue="embed", name="ingestion.tasks.embed_image_pages")
def embed_image_pages(self, image_batch: list[dict], doc_id: str):
    try:
        b64s    = [p["b64"] for p in image_batch]
        vectors = _jina_embed_images_sync(b64s)

        from index.qdrant_store import upsert_image_pages_sync
        qdrant = _qdrant_client()
        points = [
            {
                "id":            page["id"],
                "multi_vectors": [vec],
                "payload": {
                    "doc_id":     page["doc_id"],
                    "page_no":    page["page_no"],
                    "chunk_type": "image",
                },
            }
            for page, vec in zip(image_batch, vectors)
        ]
        upsert_image_pages_sync(qdrant, points)

    except Exception as exc:
        raise self.retry(exc=exc, countdown=30)


# ── Task 4: generate_and_embed_summary (embed queue) ─────────────────────

@app.task(bind=True, max_retries=3, queue="embed", name="ingestion.tasks.generate_and_embed_summary")
def generate_and_embed_summary(self, doc_id: str, file_name: str, doc_text: str):
    try:
        # Generate summary (Anthropic async SDK, isolated asyncio.run)
        async def _gen():
            from chunking.summary_generator import generate_doc_summary
            return await generate_doc_summary(doc_id, file_name, doc_text)

        summary_chunk = asyncio.run(_gen())

        # Embed and store (sync)
        vectors = _jina_embed_sync([summary_chunk["text"]], task="retrieval.passage")

        from index.qdrant_store import upsert_text_chunks_sync
        qdrant = _qdrant_client()
        upsert_text_chunks_sync(qdrant, [{
            "id":           summary_chunk["id"],
            "dense_vector": vectors[0],
            "payload": {
                "text":            summary_chunk["text"],
                "original_text":   summary_chunk["text"],
                "doc_id":          doc_id,
                "chunk_type":      "summary",
                "element_type":    "summary",
                "page_no":         None,
                "bbox":            None,
                "section_heading": None,
                "parent_id":       None,
            },
        }])
        update_status(doc_id, IngestionStatus.INDEXED, chunk_delta=1)

    except Exception as exc:
        raise self.retry(exc=exc, countdown=30)


# ── Task 5: pandas_ingest (embed queue) ──────────────────────────────────

@app.task(bind=True, max_retries=3, queue="embed", name="ingestion.tasks.pandas_ingest")
def pandas_ingest(self, file_path: str, doc_id: str, file_name: str):
    try:
        async def _ingest():
            from pandas_engine.query_engine import PandasQueryEngine
            engine = PandasQueryEngine()
            await engine.ingest_file(file_path, doc_id, file_name)

        asyncio.run(_ingest())
        update_status(doc_id, IngestionStatus.INDEXED)

    except Exception as exc:
        update_status(doc_id, IngestionStatus.FAILED, error=str(exc))
        raise self.retry(exc=exc, countdown=30)
