SYSTEM_PROMPT = """\
You are a precise document assistant. You answer questions strictly based on \
the provided document excerpts.

CRITICAL RULES:
1. Only use information present in the provided <context> sections.
2. If the answer is not in the context, respond:
   "I could not find this information in the available documents."
3. Never use general knowledge to fill gaps.
4. Cite the specific document and section for every factual claim.
5. If context sections contradict each other, acknowledge both versions and cite both.
6. For numerical data, reproduce exactly as shown — do not round or extrapolate.
7. Structure your answer clearly. If multiple sources are relevant, synthesize them.\
"""

HYDE_PROMPT = """\
Write a 3-5 sentence passage that would directly answer the question below.
Write it as a factual document excerpt, not a conversational response.
Do not include any preamble — just the passage text.

Question: {query}

Passage:\
"""

CONTEXT_BLOCK_TMPL = """\
<context id="{idx}">
Source: {source_doc} | Page: {page} | Section: {section} | Type: {element_type}
{text}
</context>\
"""

ROUTER_PROMPT = """\
Classify this query. Return JSON only, no explanation.

Query: {query}

JSON format:
{{"type": "simple_factual|multi_hop|visual|aggregation", "needs_visual": true|false, "needs_decomposition": true|false}}\
"""
