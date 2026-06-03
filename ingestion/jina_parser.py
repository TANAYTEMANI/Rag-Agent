"""
Fast document parser using PyMuPDF (fitz).

Replaces Docling's slow OCR pipeline for the common case of digital PDFs.
PyMuPDF extracts embedded text from PDFs in <10ms per page — no ML models,
no network calls, no Docker memory issues.

Fallback chain:
  1. PyMuPDF text extraction (digital PDFs) — ~5ms/page
  2. pypdf fallback (if fitz unavailable)   — ~50ms/page
  3. Page rendered as PNG for image embedding (via pdf2image + poppler)

Scanned PDFs (image-only, no embedded text) will return minimal text.
For those, the image embedding path (Jina v4) handles retrieval.
"""
from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path


def _extract_pages_pymupdf(pdf_path: str) -> list[dict]:
    """
    Extract per-page text + PNG renders using PyMuPDF.
    Returns list of {page_no, text, b64}.
    """
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        # Extract embedded text (fast — no OCR)
        text = page.get_text("text").strip()

        # Render to PNG for image embedding (used by Jina v4)
        mat  = fitz.Matrix(1.5, 1.5)   # 1.5x scale → good quality without huge size
        pix  = page.get_pixmap(matrix=mat)
        b64  = base64.b64encode(pix.tobytes("png")).decode()

        pages.append({"page_no": i, "text": text, "b64": b64})

    doc.close()
    return pages


def _extract_pages_pypdf(pdf_path: str) -> list[dict]:
    """Fallback if pymupdf is not installed."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append({"page_no": i, "text": text, "b64": None})
    return pages


def _extract_pages_docx(file_path: str) -> list[dict]:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"page_no": 1, "text": text, "b64": None}]
    except Exception as e:
        return [{"page_no": 1, "text": f"[DOCX extraction failed: {e}]", "b64": None}]


def _extract_pages_pptx(file_path: str) -> list[dict]:
    """Extract text from PPTX slide by slide."""
    try:
        from pptx import Presentation
        prs = Presentation(file_path)
        pages = []
        for i, slide in enumerate(prs.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            pages.append({"page_no": i, "text": "\n".join(texts), "b64": None})
        return pages
    except Exception as e:
        return [{"page_no": 1, "text": f"[PPTX extraction failed: {e}]", "b64": None}]


def _extract_pages_text(file_path: str) -> list[dict]:
    """Plain text / markdown files."""
    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return [{"page_no": 1, "text": text, "b64": None}]
    except Exception as e:
        return [{"page_no": 1, "text": f"[Text extraction failed: {e}]", "b64": None}]


async def parse_document_file(file_path: str) -> list[dict]:
    """
    Parse any supported format. Returns list[{page_no, text, b64}].
    All extraction is synchronous and fast — no API calls, no ML models.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        try:
            return _extract_pages_pymupdf(file_path)
        except ImportError:
            return _extract_pages_pypdf(file_path)

    if ext in (".docx", ".doc"):
        return _extract_pages_docx(file_path)

    if ext in (".pptx", ".ppt"):
        return _extract_pages_pptx(file_path)

    if ext in (".txt", ".md", ".html"):
        return _extract_pages_text(file_path)

    # Unknown format — try pymupdf (handles many formats)
    try:
        return _extract_pages_pymupdf(file_path)
    except Exception:
        return [{"page_no": 1,
                 "text": f"[Unsupported format: {ext}]",
                 "b64": None}]


def pages_to_chunks(pages: list[dict], doc_id: str,
                    doc_markdown: str) -> list[dict]:
    """
    Convert extracted pages into chunk dicts for the rest of the pipeline.
    One chunk per page — self-contained context windows for the reranker.
    """
    chunks = []
    for page in pages:
        text = page.get("text", "").strip()
        if not text or len(text) < 10:
            continue
        chunks.append({
            "id":              str(uuid.uuid4()),
            "text":            text,
            "document_text":   doc_markdown,
            "doc_id":          doc_id,
            "chunk_type":      "text",
            "element_type":    "page",
            "page_no":         page["page_no"],
            "bbox":            None,
            "section_heading": None,
            "parent_id":       None,
        })
    return chunks
