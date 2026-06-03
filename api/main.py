"""
FastAPI application entry point.
Lifespan: sets up Qdrant collections and warms up services on startup.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, ingest, query
from api.deps import get_qdrant, get_redis
from config.settings import settings

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────
    log.info("Starting RAG agent backend…")

    qdrant = get_qdrant()
    from index.qdrant_store import setup_collections
    await setup_collections(qdrant)
    log.info("Qdrant collections ready")

    # Pre-warm: embed a dummy string so Jina SDK is initialised
    try:
        from embedding.jina_client import embed_query as _eq
        await _eq("warmup")
        log.info("Jina client warmed up")
    except Exception as e:
        log.warning("Jina warmup failed", error=str(e))

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    r = await get_redis()
    await r.aclose()
    log.info("Shutdown complete")


app = FastAPI(
    title="RAG Agent API",
    description="Large-scale multimodal RAG backend",
    version="1.0.0",
    lifespan=lifespan,
    redirect_slashes=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
