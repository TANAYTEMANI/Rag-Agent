"""Generates a single document-level summary chunk stored alongside leaf chunks."""
from __future__ import annotations

import uuid
import anthropic

from config.settings import settings

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

_SUMMARY_PROMPT = """\
Summarize this document in 4-6 sentences. Include:
1. The document's main topic and purpose
2. Key entities, dates, and decisions mentioned
3. What specific questions this document can answer

Document title: {title}

{text}"""


async def generate_doc_summary(doc_id: str, title: str,
                                full_text: str) -> dict:
    response = await _client.messages.create(
        model=settings.enrichment_model,
        max_tokens=300,
        messages=[{"role": "user", "content":
            _SUMMARY_PROMPT.format(title=title, text=full_text[:5000])}],
    )
    summary_text = response.content[0].text.strip()
    return {
        "id":                 str(uuid.uuid4()),
        "text":               summary_text,
        "contextualized_text": summary_text,
        "document_text":      "",
        "doc_id":             doc_id,
        "chunk_type":         "summary",
        "element_type":       "summary",
        "page_no":            None,
        "bbox":               None,
        "section_heading":    None,
        "parent_id":          None,
    }
