"""L3 cache: cached chunk IDs for a query embedding (skips retrieval + reranking)."""
import hashlib
import json
import redis.asyncio as aioredis


def _key(vector: list[float]) -> str:
    raw = str(vector[:8]).encode()
    return f"retrieval:{hashlib.md5(raw).hexdigest()}"


async def retrieval_cache_get(r: aioredis.Redis,
                               vector: list[float]) -> list[str] | None:
    val = await r.get(_key(vector))
    return json.loads(val) if val else None


async def retrieval_cache_set(r: aioredis.Redis, vector: list[float],
                               chunk_ids: list[str], ttl: int = 1800) -> None:
    await r.setex(_key(vector), ttl, json.dumps(chunk_ids))
