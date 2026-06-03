"""HyDE: generate a hypothetical answer passage, embed it for dense retrieval."""
from __future__ import annotations

import anthropic
from embedding.jina_client import embed_query
from config.settings import settings
from generation.prompts import HYDE_PROMPT

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


async def hyde_embed(query: str) -> list[float]:
    """Generate a hypothetical document passage and return its embedding."""
    response = await _client.messages.create(
        model=settings.enrichment_model,
        max_tokens=200,
        messages=[{"role": "user",
                    "content": HYDE_PROMPT.format(query=query)}],
    )
    hypothetical = response.content[0].text.strip()
    return await embed_query(hypothetical)
