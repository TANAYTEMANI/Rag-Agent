"""
Ingestion endpoints:
  POST /ingest        — upload a file, create DB record, dispatch Celery task
  GET  /ingest/status — batch summary by status
  GET  /ingest/{id}   — single document status
  POST /ingest/{id}/supersede — replace document with a new version
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db
from api.schemas import BatchStatusResponse, DocStatusResponse, IngestResponse
from config.settings import settings
from index.postgres_store import (
    create_document, create_job, get_job, get_job_status_summary,
    get_document_by_hash,
)
from ingestion.dedup import file_hash
from ingestion.worker import app as celery_app

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_ALLOWED = {
    ".pdf", ".docx", ".pptx", ".ppt", ".doc",
    ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg",
}


@router.post("", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    department: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id   = str(uuid.uuid4())
    dest     = upload_dir / f"{doc_id}{ext}"

    content = await file.read()
    dest.write_bytes(content)

    fhash    = file_hash(str(dest))
    doc_type = ext.lstrip(".")

    # Check if already indexed (by hash)
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

    celery_app.send_task(
        "ingestion.tasks.parse_document",
        args=[str(dest), doc_id, file.filename],
        queue="parse",
    )

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
async def supersede(doc_id: str, file: UploadFile = File(...),
                    db: AsyncSession = Depends(get_db)):
    """Replace an existing document with a new version."""
    from ingestion.updater import supersede as do_supersede

    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    upload_dir = Path(settings.upload_dir)
    new_doc_id = str(uuid.uuid4())
    dest = upload_dir / f"{new_doc_id}{ext}"
    dest.write_bytes(await file.read())

    fhash = file_hash(str(dest))
    await create_document(db, new_doc_id, file.filename, fhash,
                          doc_type=ext.lstrip("."))
    await create_job(db, new_doc_id, file.filename, fhash)
    await do_supersede(doc_id, new_doc_id)

    celery_app.send_task("ingestion.tasks.parse_document",
                         args=[str(dest), new_doc_id, file.filename], queue="parse")
    return {"old_doc_id": doc_id, "new_doc_id": new_doc_id, "status": "queued"}
