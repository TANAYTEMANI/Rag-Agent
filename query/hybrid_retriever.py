"""
Hybrid retrieval: parallel dense + sparse + image search.
Results fused with Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter

from index.qdrant_store import search_dense, search_sparse, search_images
from config.settings import settings


async def hybrid_retrieve(
    client: AsyncQdrantClient,
    query: str,
    dense_vector: list[float],
    payload_filter: Filter | None = None,
    needs_visual: bool = False,
    top_k: int = None,
) -> list[dict]:
    """Run dense + sparse (+ optional image) search in parallel, fuse with RRF."""
    top_k = top_k or settings.top_k_retrieval

    tasks = [
        search_dense(client, dense_vector, payload_filter, top_k),
        search_sparse(client, query, payload_filter, top_k),
    ]
    if needs_visual:
        from embedding.jina_client import embed_query_for_images
        img_vec = await embed_query_for_images(query)
        tasks.append(search_images(client, img_vec, top_k=10))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    result_lists = []
    for r in results:
        if isinstance(r, Exception):
            result_lists.append([])
        else:
            result_lists.append(r)

    return reciprocal_rank_fusion(result_lists, k=60)[:top_k]


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            item_id = item["id"]
            scores[item_id] += 1.0 / (k + rank + 1)
            items[item_id] = item

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [items[i] for i in sorted_ids]
