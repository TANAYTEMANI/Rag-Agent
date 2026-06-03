from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    MultiVectorConfig,
    MultiVectorComparator,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
    PointStruct,
    SparseVector,
    NamedVector,
    NamedSparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    DatetimeRange,
    SearchRequest,
    SearchParams,
    HasIdCondition,
)

from config.settings import settings

COLLECTION_TEXT   = "text_chunks"
COLLECTION_IMAGES = "image_pages"
COLLECTION_CACHE  = "query_cache"


def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        https=settings.qdrant_api_key is not None,
    )


async def setup_collections(client: AsyncQdrantClient) -> None:
    existing = {c.name for c in (await client.get_collections()).collections}

    if COLLECTION_TEXT not in existing:
        await client.create_collection(
            collection_name=COLLECTION_TEXT,
            vectors_config={
                "dense": VectorParams(size=settings.embed_dimensions, distance=Distance.COSINE),
            },
            on_disk_payload=True,
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
        )
        # Payload indexes for fast pre-filtering
        for field, ftype in [
            ("doc_id",       PayloadSchemaType.KEYWORD),
            ("doc_type",     PayloadSchemaType.KEYWORD),
            ("chunk_type",   PayloadSchemaType.KEYWORD),
            ("element_type", PayloadSchemaType.KEYWORD),
            ("department",   PayloadSchemaType.KEYWORD),
            ("page_no",      PayloadSchemaType.INTEGER),
        ]:
            await client.create_payload_index(COLLECTION_TEXT, field, ftype)

        # Full-text index on `text` field — powers Qdrant BM25 sparse search
        await client.create_payload_index(
            COLLECTION_TEXT, "text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )

    if COLLECTION_IMAGES not in existing:
        await client.create_collection(
            collection_name=COLLECTION_IMAGES,
            vectors_config={
                "colpali": VectorParams(
                    size=settings.embed_dimensions,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                )
            },
            on_disk_payload=True,
        )
        for field, ftype in [
            ("doc_id",   PayloadSchemaType.KEYWORD),
            ("page_no",  PayloadSchemaType.INTEGER),
        ]:
            await client.create_payload_index(COLLECTION_IMAGES, field, ftype)

    if COLLECTION_CACHE not in existing:
        await client.create_collection(
            collection_name=COLLECTION_CACHE,
            vectors_config=VectorParams(size=settings.embed_dimensions, distance=Distance.COSINE),
        )


# ── Upsert ─────────────────────────────────────────────────────────────────

async def upsert_text_chunks(client: AsyncQdrantClient, points: list[dict]) -> None:
    """Async version — used by API/query path."""
    qdrant_points = [
        PointStruct(
            id=p["id"],
            vector={"dense": p["dense_vector"]},
            payload=p["payload"],
        )
        for p in points
    ]
    await client.upsert(collection_name=COLLECTION_TEXT, points=qdrant_points)


def upsert_text_chunks_sync(client, points: list[dict]) -> None:
    """Sync version — used by Celery worker tasks."""
    from qdrant_client import QdrantClient as SyncClient
    qdrant_points = [
        PointStruct(
            id=p["id"],
            vector={"dense": p["dense_vector"]},
            payload=p["payload"],
        )
        for p in points
    ]
    client.upsert(collection_name=COLLECTION_TEXT, points=qdrant_points)


async def upsert_image_pages(client: AsyncQdrantClient, points: list[dict]) -> None:
    """Async version — used by API/query path."""
    qdrant_points = [
        PointStruct(
            id=p["id"],
            vector={"colpali": p["multi_vectors"]},
            payload=p["payload"],
        )
        for p in points
    ]
    await client.upsert(collection_name=COLLECTION_IMAGES, points=qdrant_points)


def upsert_image_pages_sync(client, points: list[dict]) -> None:
    """Sync version — used by Celery worker tasks."""
    qdrant_points = [
        PointStruct(
            id=p["id"],
            vector={"colpali": p["multi_vectors"]},
            payload=p["payload"],
        )
        for p in points
    ]
    client.upsert(collection_name=COLLECTION_IMAGES, points=qdrant_points)


# ── Search ─────────────────────────────────────────────────────────────────

async def search_dense(client: AsyncQdrantClient, vector: list[float],
                       payload_filter: Filter | None = None,
                       top_k: int = 20) -> list[dict]:
    results = await client.search(
        collection_name=COLLECTION_TEXT,
        query_vector=NamedVector(name="dense", vector=vector),
        query_filter=payload_filter,
        limit=top_k,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=128, exact=False),
    )
    return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results]


async def search_sparse(client: AsyncQdrantClient, query_text: str,
                        payload_filter: Filter | None = None,
                        top_k: int = 20) -> list[dict]:
    """BM25 full-text search using Qdrant's built-in text payload index."""
    from qdrant_client.models import ScrollRequest, MatchText
    # Use scroll with full-text match condition for keyword search
    conditions = [FieldCondition(key="text", match=MatchText(text=query_text))]
    if payload_filter and payload_filter.must:
        conditions.extend(payload_filter.must)

    results, _ = await client.scroll(
        collection_name=COLLECTION_TEXT,
        scroll_filter=Filter(must=conditions),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    # Assign a uniform score of 1.0 — RRF handles the ranking
    return [{"id": str(r.id), "score": 1.0, "payload": r.payload} for r in results]


async def search_images(client: AsyncQdrantClient, query_vectors: list[list[float]],
                        top_k: int = 10) -> list[dict]:
    results = await client.search(
        collection_name=COLLECTION_IMAGES,
        query_vector=NamedVector(name="colpali", vector=query_vectors),
        limit=top_k,
        with_payload=True,
    )
    return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results]


async def fetch_points_by_ids(client: AsyncQdrantClient, ids: list[str]) -> list[dict]:
    """Fetch full payloads for specific chunk IDs (used for parent expansion + L3 cache)."""
    results = await client.retrieve(
        collection_name=COLLECTION_TEXT,
        ids=ids,
        with_payload=True,
    )
    return [{"id": str(r.id), "payload": r.payload} for r in results]


async def delete_by_doc_id(client: AsyncQdrantClient, doc_id: str) -> None:
    from qdrant_client.models import FilterSelector
    await client.delete(
        collection_name=COLLECTION_TEXT,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        ),
    )
    await client.delete(
        collection_name=COLLECTION_IMAGES,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        ),
    )


# ── Semantic cache ─────────────────────────────────────────────────────────

async def cache_search(client: AsyncQdrantClient, vector: list[float],
                       threshold: float) -> str | None:
    results = await client.search(
        collection_name=COLLECTION_CACHE,
        query_vector=vector,
        limit=1,
        score_threshold=threshold,
        with_payload=True,
    )
    if results:
        return results[0].payload.get("response")
    return None


async def cache_upsert(client: AsyncQdrantClient, point_id: str,
                       vector: list[float], response: str) -> None:
    await client.upsert(
        collection_name=COLLECTION_CACHE,
        points=[PointStruct(id=point_id, vector=vector,
                            payload={"response": response})],
    )


def build_metadata_filter(doc_type: str = None, department: str = None,
                           chunk_type: str = None) -> Filter | None:
    conditions = []
    if doc_type:
        conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
    if department:
        conditions.append(FieldCondition(key="department", match=MatchValue(value=department)))
    if chunk_type:
        conditions.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))
    return Filter(must=conditions) if conditions else None
