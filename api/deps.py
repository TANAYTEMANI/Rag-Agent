"""FastAPI dependency injection for all shared clients."""
from __future__ import annotations

from functools import lru_cache
from typing import AsyncIterator

import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from index.postgres_store import AsyncSessionLocal


# ── Qdrant ────────────────────────────────────────────────────────────────

@lru_cache
def get_qdrant() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,      # None for local, set for Qdrant Cloud
        https=settings.qdrant_api_key is not None,  # Cloud uses HTTPS
    )


# ── Redis ─────────────────────────────────────────────────────────────────

_redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(settings.redis_url, decode_responses=False)
    return _redis_pool


# ── PostgreSQL ────────────────────────────────────────────────────────────

async def get_db() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session
