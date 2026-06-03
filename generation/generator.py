"""
Claude Sonnet generation with:
- Prompt caching on the system prompt (reduces repeated cost ~80%)
- Streaming via SSE
- Structured citation JSON in the final event
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import anthropic

from config.settings import settings
from generation.prompts import SYSTEM_PROMPT
from generation.formatter import format_context_blocks

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


async def generate_streaming(
    query: str,
    chunks: list[dict],
) -> AsyncIterator[dict]:
    """
    Yields SSE-style dicts:
      {"type": "token",    "text": "..."}
      {"type": "citations","data": [...]}
      {"type": "done",     "from_cache": false}
    """
    context_str, citations = format_context_blocks(chunks)

    full_prompt = f"{context_str}\n\nQuestion: {query}"

    async with _client.messages.stream(
        model=settings.generation_model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": full_prompt,
                },
            ],
        }],
    ) as stream:
        async for text in stream.text_stream:
            yield {"type": "token", "text": text}

    yield {"type": "citations", "data": citations}
    yield {"type": "done", "from_cache": False}


async def generate_complete(query: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """Non-streaming variant for cache population."""
    context_str, citations = format_context_blocks(chunks)
    full_prompt = f"{context_str}\n\nQuestion: {query}"

    response = await _client.messages.create(
        model=settings.generation_model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": full_prompt},
            ],
        }],
    )
    return response.content[0].text, citations
