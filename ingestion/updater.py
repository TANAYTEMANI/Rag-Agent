"""Handles replacing an existing document with a new version."""
from __future__ import annotations

import asyncio
from qdrant_client import QdrantClient
from index.qdrant_store import delete_by_doc_id
from index.postgres_store import AsyncSessionLocal, supersede_document
from config.settings import settings


async def supersede(old_doc_id: str, new_doc_id: str) -> None:
    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    await delete_by_doc_id(qdrant, old_doc_id)
    async with AsyncSessionLocal() as session:
        await supersede_document(session, old_doc_id, new_doc_id)
