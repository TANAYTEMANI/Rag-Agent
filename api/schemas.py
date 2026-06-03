from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from pydantic import ConfigDict
from typing import Any


class _StrCoerce(BaseModel):
    """Base model that coerces UUID/int fields to str where annotated as str."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("doc_id", mode="before", check_fields=False)
    @classmethod
    def coerce_uuid(cls, v):
        return str(v) if v is not None else v


class IngestResponse(_StrCoerce):
    doc_id: str
    file_name: str
    status: str
    message: str


class JobStatusItem(BaseModel):
    status: str
    count: int
    avg_secs: float | None = None


class BatchStatusResponse(BaseModel):
    summary: dict[str, JobStatusItem]


class DocStatusResponse(_StrCoerce):
    doc_id: str
    file_name: str
    status: str
    page_count: int | None
    chunk_count: int | None
    error_message: str | None
    queued_at: Any
    started_at: Any
    completed_at: Any


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    doc_type: str | None = None
    department: str | None = None
    stream: bool = True


class Citation(BaseModel):
    context_id: int
    doc_id: str
    page: int | None
    section: str | None
    element_type: str | None
    bbox: dict | None
    chunk_id: str | None
    reranker_score: float | None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    from_cache: str | None = None
    query_type: str | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    postgres: str
    redis: str
