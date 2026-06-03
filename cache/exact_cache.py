"""L1 cache: exact query string match via Redis. <1ms lookup."""
import hashlib
import redis.asyncio as aioredis
from config.settings import settings


def _key(query: str) -> str:
    norm = query.strip().lower()
    return f"exact:{hashlib.md5(norm.encode()).hexdigest()}"


async def exact_cache_get(r: aioredis.Redis, query: str) -> str | None:
    val = await r.get(_key(query))
    return val.decode() if val else None


async def exact_cache_set(r: aioredis.Redis, query: str,
                           response: str, ttl: int = 3600) -> None:
    await r.setex(_key(query), ttl, response)
