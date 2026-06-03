"""
Builds parent-child chunk relationships.
Parent chunks are 3-5 consecutive child chunks merged together.
Children are searched; parents are returned to the LLM for richer context.
"""
from __future__ import annotations

import uuid


def build_parent_child_chunks(chunks: list[dict], parent_size: int = 4) -> list[dict]:
    """
    Group consecutive text chunks into parents.
    Returns all chunks (both children with parent_id set, and parent chunks).
    """
    all_chunks = []
    for i in range(0, len(chunks), parent_size):
        group = chunks[i: i + parent_size]
        if not group:
            continue

        parent_id = str(uuid.uuid4())
        parent_text = "\n\n".join(c["text"] for c in group)

        parent = {
            "id":             parent_id,
            "text":           parent_text,
            "document_text":  group[0].get("document_text", ""),
            "doc_id":         group[0]["doc_id"],
            "chunk_type":     "parent",
            "element_type":   "parent",
            "page_no":        group[0].get("page_no"),
            "bbox":           group[0].get("bbox"),
            "section_heading": group[0].get("section_heading"),
            "parent_id":      None,
            "contextualized_text": parent_text,
        }
        all_chunks.append(parent)

        for child in group:
            child["parent_id"] = parent_id
            all_chunks.append(child)

    return all_chunks
