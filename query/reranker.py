"""
Jina Reranker v3 — cross-encoder reranking with score-threshold relevance gate.
The threshold replaces CRAG (per-chunk LLM calls): zero extra latency.
"""
from __future__ import annotations

from embedding.jina_client import rerank as jina_rerank
from config.settings import settings


class NoRelevantChunksError(Exception):
    pass


async def rerank_and_filter(
    query: str,
    chunks: list[dict],
    top_n: int = None,
    score_threshold: float = None,
) -> list[dict]:
    """
    Rerank top_k candidates → keep top_n above threshold.
    Raises NoRelevantChunksError if nothing passes (caller should fallback or say IDK).
    """
    top_n = top_n or settings.top_n_reranked
    score_threshold = score_threshold or settings.reranker_score_threshold

    if not chunks:
        raise NoRelevantChunksError("No chunks to rerank")

    texts = [c["payload"]["text"] for c in chunks]
    ranked = await jina_rerank(query, texts, top_n=len(chunks))

    results = []
    for orig_idx, score in ranked:
        if score >= score_threshold:
            chunk = dict(chunks[orig_idx])
            chunk["reranker_score"] = score
            results.append(chunk)

    if not results:
        raise NoRelevantChunksError(
            f"All {len(chunks)} chunks scored below threshold {score_threshold:.2f}"
        )

    return results[:top_n]
