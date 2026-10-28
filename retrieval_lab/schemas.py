from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_name: str
    content_sha256: str
    page_count: int
    chunk_count: int
    chunk_ids: list[int]


class VectorHit(BaseModel):
    id: int
    content: str
    document_name: str
    page: int | None
    metadata: dict[str, Any]
    similarity: float
    rank: int


class LexicalHit(BaseModel):
    id: int
    content: str
    document_name: str
    page: int | None
    metadata: dict[str, Any]
    lexical_score: float
    rank: int


class HybridHit(BaseModel):
    id: int
    content: str
    document_name: str
    page: int | None
    metadata: dict[str, Any]
    rrf_score: float
    rank: int
    vector_rank: int | None
    lexical_rank: int | None


class LatencyMs(BaseModel):
    embed_ms: float
    vector_ms: float
    lexical_ms: float
    fusion_ms: float
    hybrid_ms: float


class DocumentRef(BaseModel):
    document_name: str
    pages: list[int]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=10000)
    candidate_k: int | None = Field(default=None, ge=1, le=200)


class QueryResponse(BaseModel):
    question: str
    top_k: int
    rrf_k: int
    candidate_k: int
    vector: list[VectorHit]
    lexical: list[LexicalHit]
    hybrid: list[HybridHit]
    latency_ms: LatencyMs
    documents: list[DocumentRef]


class HealthResponse(BaseModel):
    status: str
    database: str
