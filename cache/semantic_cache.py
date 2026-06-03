"""L2 cache: semantic similarity via Qdrant query_cache collection."""
import hashlib
from qdrant_client import AsyncQdrantClient
from index.qdrant_store import COLLECTION_CACHE, cache_search, cache_upsert
from config.settings import settings


def _point_id(vector: list[float]) -> str:
    raw = str(vector[:8]).encode()
    return hashlib.md5(raw).hexdigest()


async def semantic_cache_get(client: AsyncQdrantClient,
                              vector: list[float]) -> str | None:
    return await cache_search(client, vector, threshold=settings.semantic_cache_threshold)


async def semantic_cache_set(client: AsyncQdrantClient,
                              vector: list[float], response: str) -> None:
    await cache_upsert(client, point_id=_point_id(vector), vector=vector, response=response)
