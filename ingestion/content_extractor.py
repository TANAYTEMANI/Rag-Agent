from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

# Docling is only available in the worker image, not the API image.
# Use TYPE_CHECKING so the type hint is available for IDEs/linters
# without triggering a runtime import on non-worker containers.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from docling.datamodel.document import DoclingDocument


def _nearest_heading(element, doc) -> str | None:
    """Walk upward through the document tree to find the nearest section heading."""
    try:
        for heading in doc.texts:
            if heading.label == "section-header":
                return heading.text
    except Exception:
        pass
    return None


def _bbox_payload(prov) -> dict | None:
    if not prov:
        return None
    b = prov.bbox
    return {"x0": round(b.l, 4), "y0": round(b.t, 4),
            "x1": round(b.r, 4), "y1": round(b.b, 4)}


def extract_text_chunks(doc, doc_id: str, doc_markdown: str) -> list[dict]:
    """Extract text elements with bbox grounding.
    Accepts all text-bearing labels so simple PDFs (fpdf, scanned) are not silently empty.
    """
    # Broad label set — Docling uses different labels for different PDF generators
    # Docling uses underscore_case for label values (e.g. section_header, list_item)
    # but also hyphen-case in some versions — accept both
    allowed_labels = {
        "paragraph", "section_header", "section-header",
        "list_item", "list-item", "code", "formula",
        "text", "title", "caption", "footnote",
        "page_header", "page-header", "page_footer", "page-footer",
    }
    chunks = []
    for element, _ in doc.iterate_items():
        if element.label not in allowed_labels:
            continue
        text = getattr(element, "text", None)
        if not text or len(text.strip()) < 10:  # lowered from 20 → 10
            continue

        prov = element.prov[0] if element.prov else None
        chunk_id = str(uuid.uuid4())
        chunks.append({
            "id":            chunk_id,
            "text":          text.strip(),
            "document_text": doc_markdown,
            "doc_id":        doc_id,
            "chunk_type":    "text",
            "element_type":  element.label,
            "page_no":       prov.page_no if prov else None,
            "bbox":          _bbox_payload(prov),
            "section_heading": _nearest_heading(element, doc),
            "parent_id":     None,  # filled by parent_child.py
        })
    return chunks


def extract_table_chunks(doc, doc_id: str) -> list[dict]:
    """Extract tables as markdown text + JSON payload with bbox."""
    chunks = []
    for table in doc.tables:
        try:
            df = table.export_to_dataframe()
            md = table.export_to_markdown()
        except Exception:
            continue

        prov = table.prov[0] if table.prov else None
        chunk_id = str(uuid.uuid4())
        chunks.append({
            "id":           chunk_id,
            "text":         md,
            "document_text": "",
            "doc_id":       doc_id,
            "chunk_type":   "table",
            "element_type": "table",
            "page_no":      prov.page_no if prov else None,
            "bbox":         _bbox_payload(prov),
            "section_heading": None,
            "parent_id":    None,
            "row_count":    len(df),
            "col_count":    len(df.columns),
            "json_repr":    df.to_json(orient="records"),
            "_df":          df,  # used by tasks.py for large-table DataFrame storage
        })
    return chunks


def extract_image_pages(doc, doc_id: str) -> list[dict]:
    """Render pages as base64 images for Jina v4 multimodal embedding."""
    pages = []
    try:
        for page_no, page in doc.pages.items():
            if page.image is None:
                continue
            buf = io.BytesIO()
            page.image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            pages.append({
                "id":      str(uuid.uuid4()),
                "doc_id":  doc_id,
                "page_no": page_no,
                "b64":     b64,
            })
    except Exception:
        pass
    return pages


def chunk_batches(items: list, size: int) -> list[list]:
    return [items[i: i + size] for i in range(0, len(items), size)]
