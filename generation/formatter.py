"""Build the context string passed to Claude and the citation metadata."""
from __future__ import annotations

from generation.prompts import CONTEXT_BLOCK_TMPL


def format_context_blocks(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Returns:
        context_str   — formatted <context> blocks for the LLM prompt
        citations     — list of citation metadata dicts
    """
    blocks = []
    citations = []

    for idx, chunk in enumerate(chunks, start=1):
        payload = chunk.get("payload", chunk)
        text    = payload.get("text", "")
        doc_id  = payload.get("doc_id", "unknown")
        page_no = payload.get("page_no")
        section = payload.get("section_heading") or "—"
        etype   = payload.get("element_type", "text")
        bbox    = payload.get("bbox")

        block = CONTEXT_BLOCK_TMPL.format(
            idx=idx,
            source_doc=doc_id,
            page=page_no if page_no is not None else "—",
            section=section,
            element_type=etype,
            text=text,
        )
        blocks.append(block)
        citations.append({
            "context_id":    idx,
            "doc_id":        doc_id,
            "page":          page_no,
            "section":       section,
            "element_type":  etype,
            "bbox":          bbox,
            "chunk_id":      chunk.get("id"),
            "reranker_score": chunk.get("reranker_score"),
        })

    return "\n\n".join(blocks), citations
