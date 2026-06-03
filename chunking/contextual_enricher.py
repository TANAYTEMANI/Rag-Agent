"""
Anthropic Contextual Retrieval enrichment.
Prepends a 2-3 sentence context to each chunk using Claude Haiku.
The document body is prompt-cached so only the chunk text varies per call.
All chunks in a batch are enriched concurrently via asyncio.gather.
"""
from __future__ import annotations

import asyncio
import anthropic

from config.settings import settings

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

# Cap concurrent Anthropic calls across all embed workers.
# 16 workers × up to 20 chunks each = up to 320 simultaneous calls without this.
# Anthropic Haiku: 1000 RPM on tier-1 paid, 50 RPM on free.
# 40 concurrent is safe at tier-1 and won't cause 429s.
_ANTHROPIC_SEM = asyncio.Semaphore(40)

_CONTEXT_SYSTEM = "You are a document indexing assistant. Be concise."

_CONTEXT_USER_TMPL = """\
Here is a chunk from this document:
<chunk>
{chunk_text}
</chunk>

Write exactly 2-3 sentences that situate this chunk within the document above.
Focus on: what section/topic it belongs to, key entities/concepts it references,
and what questions it can answer. Output ONLY the context text, no preamble."""


async def _enrich_single(chunk_text: str, doc_markdown: str) -> str:
    async with _ANTHROPIC_SEM:
        response = await _client.messages.create(
            model=settings.enrichment_model,
            max_tokens=200,
            system=_CONTEXT_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<document>\n{doc_markdown[:6000]}\n</document>\n\n",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": _CONTEXT_USER_TMPL.format(chunk_text=chunk_text),
                    },
                ],
            }],
        )
        context = response.content[0].text.strip()
        return f"{context}\n\n{chunk_text}"


async def enrich_batch_concurrent(chunks: list[dict]) -> list[dict]:
    """
    Fire one Haiku call per chunk, all concurrently.
    ~20x faster than serial for a batch of 20 chunks.
    The document body is prompt-cached — cost is ~10x lower.
    """
    if not chunks:
        return chunks

    doc_markdown = chunks[0].get("document_text", "")

    tasks = [
        _enrich_single(c["text"], doc_markdown)
        for c in chunks
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for chunk, result in zip(chunks, results):
        if isinstance(result, Exception):
            chunk["contextualized_text"] = chunk["text"]
        else:
            chunk["contextualized_text"] = result
    return chunks
