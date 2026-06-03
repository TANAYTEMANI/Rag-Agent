"""Expand leaf chunks to their parent chunks for richer LLM context."""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from index.qdrant_store import fetch_points_by_ids


async def expand_to_parents(client: AsyncQdrantClient,
                             chunks: list[dict]) -> list[dict]:
    """
    Replace child chunks with parent chunks where available.
    Deduplicates: multiple children with the same parent return that parent once.
    Falls back to the child if no parent is recorded.
    """
    parent_ids = {
        c["payload"].get("parent_id")
        for c in chunks
        if c["payload"].get("parent_id")
    }

    if not parent_ids:
        return chunks

    parents_raw = await fetch_points_by_ids(client, list(parent_ids))
    parent_map = {p["id"]: p for p in parents_raw}

    seen: set[str] = set()
    result: list[dict] = []

    for chunk in chunks:
        pid = chunk["payload"].get("parent_id")
        if pid and pid in parent_map:
            if pid not in seen:
                seen.add(pid)
                result.append(parent_map[pid])
        else:
            cid = chunk["id"]
            if cid not in seen:
                seen.add(cid)
                result.append(chunk)

    return result
