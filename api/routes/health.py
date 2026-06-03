from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient

from api.deps import get_db, get_redis, get_qdrant
from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    db: AsyncSession = Depends(get_db),
    r: aioredis.Redis = Depends(get_redis),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
):
    results = {"qdrant": "error", "postgres": "error", "redis": "error"}

    try:
        await qdrant.get_collections()
        results["qdrant"] = "ok"
    except Exception:
        pass

    try:
        await db.execute(text("SELECT 1"))
        results["postgres"] = "ok"
    except Exception:
        pass

    try:
        await r.ping()
        results["redis"] = "ok"
    except Exception:
        pass

    overall = "ok" if all(v == "ok" for v in results.values()) else "degraded"
    return HealthResponse(status=overall, **results)
