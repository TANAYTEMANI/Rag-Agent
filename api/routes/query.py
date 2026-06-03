"""
POST /query — the full RAG query pipeline.

Three-level cache stack:
  L1: exact Redis match  (<1ms)
  L2: semantic Qdrant    (~15ms)
  L3: retrieval result cache (skips retrieval + reranking)

Slow path: embed → HyDE → hybrid retrieve → rerank → parent expand → generate
Streaming via Server-Sent Events.
"""
from __future__ import annotations

import asyncio
import json

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_qdrant, get_redis
from api.schemas import QueryRequest, QueryResponse
from cache.exact_cache import exact_cache_get, exact_cache_set
from cache.semantic_cache import semantic_cache_get, semantic_cache_set
from cache.retrieval_cache import retrieval_cache_get, retrieval_cache_set
from config.settings import settings
from embedding.jina_client import embed_query
from generation.generator import generate_complete, generate_streaming
from index.qdrant_store import build_metadata_filter, fetch_points_by_ids
from query.hybrid_retriever import hybrid_retrieve
from query.hyde import hyde_embed
from query.parent_expander import expand_to_parents
from query.reranker import rerank_and_filter, NoRelevantChunksError
from query.router import route_query

router = APIRouter(prefix="/query", tags=["query"])


async def _full_pipeline(
    query: str,
    doc_type: str | None,
    department: str | None,
    qdrant: AsyncQdrantClient,
    r: aioredis.Redis,
    db: AsyncSession,
) -> tuple[list[dict], str | None, str | None, list[float] | None]:
    """
    Returns: (chunks, from_cache, query_type, query_vec)
    chunks     — context chunks for generation
    from_cache — "exact" | "semantic" | "retrieval" | None
    query_type — classified query type string
    query_vec  — query embedding (needed by caller to populate L1/L2 cache)
    """

    # ── L1: exact match — no embedding needed ────────────────────────────
    cached = await exact_cache_get(r, query)
    if cached:
        return [{"id": "cache", "payload": {"text": cached}}], "exact", None, None

    # ── Parallel: embed + route + build filter ────────────────────────────
    query_vec, route = await asyncio.gather(
        embed_query(query),
        route_query(query),
    )
    payload_filter = build_metadata_filter(doc_type=doc_type, department=department)

    # ── L2: semantic cache ────────────────────────────────────────────────
    cached_semantic = await semantic_cache_get(qdrant, query_vec)
    if cached_semantic:
        return [{"id": "cache", "payload": {"text": cached_semantic}}], "semantic", route.query_type, query_vec

    # ── L3: retrieval result cache ────────────────────────────────────────
    cached_ids = await retrieval_cache_get(r, query_vec)
    if cached_ids:
        chunks = await fetch_points_by_ids(qdrant, cached_ids)
        return chunks, "retrieval", route.query_type, query_vec

    # ── Pandas engine path ────────────────────────────────────────────────
    if route.needs_pandas:
        from index.postgres_store import search_tabular_tables
        keywords = [w for w in query.lower().split() if len(w) > 3]
        tables = await search_tabular_tables(db, keywords, limit=3)
        if tables:
            from pandas_engine.query_engine import PandasQueryEngine
            engine = PandasQueryEngine()
            results = []
            for table in tables:
                answer = await engine.query(table["table_id"], query)
                results.append({
                    "id": table["table_id"],
                    "payload": {
                        "text": answer,
                        "doc_id": str(table.get("doc_id", "")),
                        "chunk_type": "pandas_result",
                        "element_type": "table",
                        "page_no": None, "bbox": None,
                        "section_heading": None, "parent_id": None,
                    },
                })
            return results, None, "aggregation", query_vec

    # ── HyDE: hypothetical document embedding ────────────────────────────
    hyde_vec = await hyde_embed(query)

    # ── Hybrid retrieval ──────────────────────────────────────────────────
    raw_chunks = await hybrid_retrieve(
        qdrant, query, hyde_vec,
        payload_filter=payload_filter,
        needs_visual=route.needs_visual,
    )

    # ── Rerank + relevance gate ───────────────────────────────────────────
    try:
        reranked = await rerank_and_filter(query, raw_chunks)
    except NoRelevantChunksError:
        return [], None, route.query_type, query_vec

    # ── Parent expansion ──────────────────────────────────────────────────
    expanded = await expand_to_parents(qdrant, reranked)

    # ── Store in L3 retrieval cache ───────────────────────────────────────
    await retrieval_cache_set(r, query_vec, [c["id"] for c in expanded])

    return expanded, None, route.query_type, query_vec


async def _store_response_in_caches(
    r: aioredis.Redis, qdrant: AsyncQdrantClient,
    query: str, query_vec: list[float], answer: str,
) -> None:
    """Store the final answer string in L1 (exact) and L2 (semantic) caches."""
    await asyncio.gather(
        exact_cache_set(r, query, answer),
        semantic_cache_set(qdrant, query_vec, answer),
        return_exceptions=True,
    )


@router.post("")
async def query_endpoint(
    req: QueryRequest,
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    r: aioredis.Redis = Depends(get_redis),
    db: AsyncSession = Depends(get_db),
):
    chunks, from_cache, query_type, query_vec = await _full_pipeline(
        req.query, req.doc_type, req.department, qdrant, r, db
    )

    if not chunks:
        no_answer = "I could not find relevant information in the available documents."
        if req.stream:
            async def _no_ctx_stream():
                yield f"data: {json.dumps({'type': 'token', 'text': no_answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'from_cache': None})}\n\n"
            return StreamingResponse(_no_ctx_stream(), media_type="text/event-stream")
        return QueryResponse(answer=no_answer, citations=[], from_cache=from_cache, query_type=query_type)

    # Cache hit — return the pre-stored answer directly
    if from_cache in ("exact", "semantic"):
        answer = chunks[0]["payload"]["text"]
        if req.stream:
            async def _cached_stream():
                yield f"data: {json.dumps({'type': 'token', 'text': answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'from_cache': from_cache})}\n\n"
            return StreamingResponse(_cached_stream(), media_type="text/event-stream")
        return QueryResponse(answer=answer, citations=[], from_cache=from_cache, query_type=query_type)

    if req.stream:
        async def _stream():
            full_answer = []
            citations = []
            async for event in generate_streaming(req.query, chunks):
                if event["type"] == "token":
                    full_answer.append(event["text"])
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "citations":
                    citations = event["data"]
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "done":
                    event["from_cache"] = from_cache
                    yield f"data: {json.dumps(event)}\n\n"
            # Populate L1 + L2 caches with the final answer
            if query_vec and full_answer:
                asyncio.create_task(_store_response_in_caches(
                    r, qdrant, req.query, query_vec, "".join(full_answer)
                ))

        return StreamingResponse(_stream(), media_type="text/event-stream")

    answer, citations = await generate_complete(req.query, chunks)

    # Populate L1 + L2 caches
    if query_vec:
        asyncio.create_task(_store_response_in_caches(r, qdrant, req.query, query_vec, answer))

    return QueryResponse(
        answer=answer,
        citations=citations,
        from_cache=from_cache,
        query_type=query_type,
    )
