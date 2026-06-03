"""
Query router: classifies intent and detects if Pandas engine is needed.
LLM call for semantic type; keyword matching for tabular detection (free).
"""
from __future__ import annotations

import json
import anthropic
from dataclasses import dataclass, field

from config.settings import settings
from generation.prompts import ROUTER_PROMPT

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

_TABULAR_KEYWORDS = {
    "total", "sum", "average", "avg", "count", "how many",
    "compare", "highest", "lowest", "maximum", "minimum",
    "between", "per quarter", "per region", "breakdown", "revenue",
    "profit", "spend", "cost", "sales", "number of", "percentage",
}


@dataclass
class QueryRoute:
    query_type: str = "simple_factual"
    needs_visual: bool = False
    needs_decomposition: bool = False
    needs_pandas: bool = False
    doc_type_hint: str | None = None
    department: str | None = None


async def route_query(query: str) -> QueryRoute:
    query_lower = query.lower()
    needs_pandas = any(kw in query_lower for kw in _TABULAR_KEYWORDS)

    try:
        response = await _client.messages.create(
            model=settings.enrichment_model,
            max_tokens=80,
            messages=[{"role": "user",
                        "content": ROUTER_PROMPT.format(query=query)}],
        )
        data = json.loads(response.content[0].text.strip())
        return QueryRoute(
            query_type=data.get("type", "simple_factual"),
            needs_visual=bool(data.get("needs_visual", False)),
            needs_decomposition=bool(data.get("needs_decomposition", False)),
            needs_pandas=needs_pandas,
        )
    except Exception:
        return QueryRoute(needs_pandas=needs_pandas)
