"""
Jina AI API client.
- late_chunk_embed: text embedding with late_chunking=True (jina-embeddings-v3)
- embed_query: single query embedding (task=retrieval.query)
- embed_images: multimodal page images (jina-embeddings-v4)
- rerank: Jina Reranker v3
All calls are async with httpx, retried on 429/5xx via tenacity.
"""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from config.settings import settings

_EMBED_URL  = "https://api.jina.ai/v1/embeddings"
_RERANK_URL = "https://api.jina.ai/v1/rerank"

_headers = lambda: {"Authorization": f"Bearer {settings.jina_api_key}",
                    "Content-Type": "application/json"}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout))


@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=1, max=30),
       retry=retry_if_exception(_is_retryable),
       reraise=True)
async def _post(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    resp = await client.post(url, headers=_headers(), json=payload, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


async def late_chunk_embed(texts: list[str],
                           task: str = "retrieval.passage") -> list[list[float]]:
    """Embed a batch of text chunks with late_chunking=True.
    All chunks are treated as segments of a single long document so each
    embedding carries full cross-chunk context."""
    if not texts:
        return []
    payload = {
        "model": settings.jina_embed_model,
        "input": texts,
        "task": task,
        "late_chunking": True,
        "dimensions": settings.embed_dimensions,
        "embedding_type": "float",
    }
    async with httpx.AsyncClient() as client:
        data = await _post(client, _EMBED_URL, payload)
    return [item["embedding"] for item in data["data"]]


async def embed_query(text: str) -> list[float]:
    """Single query embedding without late chunking."""
    payload = {
        "model": settings.jina_embed_model,
        "input": [text],
        "task": "retrieval.query",
        "late_chunking": False,
        "dimensions": settings.embed_dimensions,
        "embedding_type": "float",
    }
    async with httpx.AsyncClient() as client:
        data = await _post(client, _EMBED_URL, payload)
    return data["data"][0]["embedding"]


async def embed_images(b64_images: list[str]) -> list[list[float]]:
    """Embed document page images using jina-embeddings-v4 (multimodal)."""
    if not b64_images:
        return []
    payload = {
        "model": settings.jina_image_model,
        "input": [{"image": b64} for b64 in b64_images],
        "task": "retrieval.passage",
        "dimensions": settings.embed_dimensions,
        "embedding_type": "float",
    }
    async with httpx.AsyncClient() as client:
        data = await _post(client, _EMBED_URL, payload)
    return [item["embedding"] for item in data["data"]]


async def embed_query_for_images(text: str) -> list[float]:
    """Query embedding against jina-embeddings-v4 image collection."""
    payload = {
        "model": settings.jina_image_model,
        "input": [{"text": text}],
        "task": "retrieval.query",
        "dimensions": settings.embed_dimensions,
        "embedding_type": "float",
    }
    async with httpx.AsyncClient() as client:
        data = await _post(client, _EMBED_URL, payload)
    return data["data"][0]["embedding"]


async def rerank(query: str, documents: list[str],
                 top_n: int | None = None) -> list[tuple[int, float]]:
    """
    Returns list of (original_index, relevance_score) sorted by score desc.
    """
    if not documents:
        return []
    payload = {
        "model": settings.jina_rerank_model,
        "query": query,
        "documents": documents,
        "top_n": top_n or len(documents),
    }
    async with httpx.AsyncClient() as client:
        data = await _post(client, _RERANK_URL, payload)
    return [(r["index"], r["relevance_score"]) for r in data["results"]]
