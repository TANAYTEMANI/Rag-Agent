"""
Ingestion endpoints.
Uses FastAPI BackgroundTasks for async processing — no Celery required.
Works on any single-container deployment (Railway, Render, etc).
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db
from api.schemas import BatchStatusResponse, DocStatusResponse, IngestResponse
from config.settings import settings
from index.postgres_store import (
    create_document, create_job, get_job, get_job_status_summary,
    get_document_by_hash,
)
from ingestion.dedup import file_hash

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_ALLOWED = {
    ".pdf", ".docx", ".pptx", ".ppt", ".doc",
    ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg",
}


async def _run_ingestion(doc_path: str, doc_id: str, file_name: str):
    """Full ingestion pipeline — runs as a FastAPI background task."""
    from ingestion.jina_parser import parse_document_file, pages_to_chunks
    from ingestion.content_extractor import chunk_batches
    from chunking.parent_child import build_parent_child_chunks
    from chunking.contextual_enricher import enrich_batch_concurrent
    from chunking.summary_generator import generate_doc_summary
    from embedding.jina_client import late_chunk_embed, embed_images
    from index.qdrant_store import get_qdrant_client, setup_collections, upsert_text_chunks, upsert_image_pages
    from index.postgres_store import AsyncSessionLocal, update_job_status
    from ingestion.dedup import mark_as_indexed
    import redis as redis_lib

    r = redis_lib.Redis.from_url(settings.redis_url)

    async def update(status, **kw):
        async with AsyncSessionLocal() as s:
            await update_job_status(s, doc_id, status, **kw)

    try:
        ext = Path(doc_path).suffix.lower()
        await update("parsing")

        # Tabular files → pandas engine
        if ext in (".xlsx", ".xls", ".csv"):
            from pandas_engine.query_engine import PandasQueryEngine
            await update("enriching")
            engine = PandasQueryEngine()
            await engine.ingest_file(doc_path, doc_id, file_name)
            await update("indexed")
            mark_as_indexed(file_hash(doc_path), doc_id, r)
            return

        # Parse document
        pages = await parse_document_file(doc_path)
        doc_markdown = "\n\n".join(p["text"] for p in pages if p["text"])
        text_chunks = pages_to_chunks(pages, doc_id, doc_markdown)
        image_pages = [p for p in pages if p.get("b64")]
        page_count  = len(pages)

        if not text_chunks:
            await update("enriching", page_count=page_count)
        else:
            # Build parent-child hierarchy
            all_chunks = build_parent_child_chunks(text_chunks, parent_size=4)

            # Contextual enrichment (concurrent)
            await update("enriching", page_count=page_count)
            to_enrich = [c for c in all_chunks if c.get("chunk_type") not in ("parent",)]
            skip      = [c for c in all_chunks if c.get("chunk_type") in ("parent",)]
            if to_enrich:
                await enrich_batch_concurrent(to_enrich)
            for c in skip:
                c["contextualized_text"] = c.get("text", "")

            # Embed all chunks
            await update("indexing")
            texts   = [c.get("contextualized_text") or c["text"] for c in all_chunks]
            vectors = await late_chunk_embed(texts, task="retrieval.passage")

            # Write to Qdrant
            qdrant = get_qdrant_client()
            await setup_collections(qdrant)
            points = [
                {
                    "id": c["id"],
                    "dense_vector": v,
                    "payload": {
                        "text":            c.get("contextualized_text") or c["text"],
                        "original_text":   c["text"],
                        "doc_id":          doc_id,
                        "chunk_type":      c.get("chunk_type", "text"),
                        "element_type":    c.get("element_type"),
                        "page_no":         c.get("page_no"),
                        "bbox":            c.get("bbox"),
                        "section_heading": c.get("section_heading"),
                        "parent_id":       c.get("parent_id"),
                    },
                }
                for c, v in zip(all_chunks, vectors)
            ]
            await upsert_text_chunks(qdrant, points)
            await update("indexing", chunk_delta=len(points))

            # Embed image pages
            if image_pages:
                img_vecs = await embed_images([p["b64"] for p in image_pages])
                img_points = [
                    {
                        "id":            str(uuid.uuid4()),
                        "multi_vectors": [v],
                        "payload": {"doc_id": doc_id, "page_no": p["page_no"], "chunk_type": "image"},
                    }
                    for p, v in zip(image_pages, img_vecs)
                ]
                await upsert_image_pages(qdrant, img_points)

        # Document summary
        summary = await generate_doc_summary(doc_id, file_name, doc_markdown[:5000])
        sum_vecs = await late_chunk_embed([summary["text"]], task="retrieval.passage")
        qdrant   = get_qdrant_client()
        await upsert_text_chunks(qdrant, [{
            "id": summary["id"], "dense_vector": sum_vecs[0],
            "payload": {
                "text": summary["text"], "original_text": summary["text"],
                "doc_id": doc_id, "chunk_type": "summary", "element_type": "summary",
                "page_no": None, "bbox": None, "section_heading": None, "parent_id": None,
            },
        }])

        await update("indexed", chunk_delta=1)
        mark_as_indexed(file_hash(doc_path), doc_id, r)

    except Exception as exc:
        async with AsyncSessionLocal() as s:
            await update_job_status(s, doc_id, "failed", error=str(exc)[:500])


@router.post("", response_model=IngestResponse)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    department: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id  = str(uuid.uuid4())
    dest    = upload_dir / f"{doc_id}{ext}"
    content = await file.read()
    dest.write_bytes(content)

    fhash    = file_hash(str(dest))
    doc_type = ext.lstrip(".")

    existing = await get_document_by_hash(db, fhash)
    if existing:
        return IngestResponse(
            doc_id=str(existing["doc_id"]),
            file_name=file.filename,
            status="duplicate",
            message="Document already indexed.",
        )

    await create_document(db, doc_id, file.filename, fhash,
                          doc_type=doc_type, department=department)
    await create_job(db, doc_id, file.filename, fhash)

    # Run ingestion as a background task — no Celery needed
    background_tasks.add_task(_run_ingestion, str(dest), doc_id, file.filename)

    return IngestResponse(
        doc_id=doc_id,
        file_name=file.filename,
        status="queued",
        message="Document queued for ingestion.",
    )


@router.get("/status", response_model=BatchStatusResponse)
async def batch_status(db: AsyncSession = Depends(get_db)):
    rows = await get_job_status_summary(db)
    summary = {
        r["status"]: {"status": r["status"], "count": r["count"],
                      "avg_secs": r.get("avg_secs")}
        for r in rows
    }
    return BatchStatusResponse(summary=summary)


@router.get("/{doc_id}", response_model=DocStatusResponse)
async def doc_status(doc_id: str, db: AsyncSession = Depends(get_db)):
    job = await get_job(db, doc_id)
    if not job:
        raise HTTPException(404, "Document not found")
    return DocStatusResponse(**{k: job.get(k) for k in DocStatusResponse.model_fields})


@router.post("/{doc_id}/supersede")
async def supersede(
    doc_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    from ingestion.updater import supersede as do_supersede
    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    upload_dir = Path(settings.upload_dir)
    new_doc_id = str(uuid.uuid4())
    dest = upload_dir / f"{new_doc_id}{ext}"
    dest.write_bytes(await file.read())

    fhash = file_hash(str(dest))
    await create_document(db, new_doc_id, file.filename, fhash, doc_type=ext.lstrip("."))
    await create_job(db, new_doc_id, file.filename, fhash)
    await do_supersede(doc_id, new_doc_id)

    background_tasks.add_task(_run_ingestion, str(dest), new_doc_id, file.filename)
    return {"old_doc_id": doc_id, "new_doc_id": new_doc_id, "status": "queued"}
