import time
from typing import Any

from retrieval_lab.db import DocumentStore
from retrieval_lab.embeddings import Embedder
from retrieval_lab.errors import DatabaseError, EmbeddingError, RetrievalError
from retrieval_lab.rrf import fuse_rankings


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _as_int(value: Any) -> int:
    return int(value)


def _as_float(value: Any) -> float:
    return float(value)


def _chunk_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _as_int(row["id"]),
        "content": row.get("content") or "",
        "document_name": row.get("document_name") or "",
        "page": None if row.get("page") is None else _as_int(row["page"]),
        "metadata": row.get("metadata") or {},
    }


def _vector_hit(row: dict[str, Any]) -> dict[str, Any]:
    hit = _chunk_fields(row)
    hit["similarity"] = _as_float(row["similarity"])
    hit["rank"] = _as_int(row.get("result_rank", row.get("rank")))
    return hit


def _lexical_hit(row: dict[str, Any]) -> dict[str, Any]:
    hit = _chunk_fields(row)
    hit["lexical_score"] = _as_float(row["lexical_score"])
    hit["rank"] = _as_int(row.get("result_rank", row.get("rank")))
    return hit


def _document_refs(*hit_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages_by_name: dict[str, set[int]] = {}
    for hits in hit_lists:
        for hit in hits:
            name = hit["document_name"]
            pages_by_name.setdefault(name, set())
            if hit["page"] is not None:
                pages_by_name[name].add(hit["page"])
    return [
        {"document_name": name, "pages": sorted(pages)}
        for name, pages in sorted(pages_by_name.items())
    ]


def resolve_candidate_k(top_k: int, candidate_k: int | None) -> int:
    if top_k < 1:
        raise RetrievalError(422, "top_k must be a positive integer.")
    if candidate_k is None:
        return min(max(top_k, 20), 200)
    if candidate_k < top_k:
        raise RetrievalError(422, "candidate_k must be greater than or equal to top_k.")
    if candidate_k > 200:
        raise RetrievalError(422, "candidate_k must be 200 or less.")
    return candidate_k


def query_retrieval(
    question: str,
    store: DocumentStore,
    embedder: Embedder,
    *,
    filter_collections: list[str],
    top_k: int = 5,
    rrf_k: int = 60,
    candidate_k: int | None = None,
) -> dict[str, Any]:
    cleaned = question.strip()
    if not cleaned:
        raise RetrievalError(422, "question must not be empty.")
    if rrf_k < 1:
        raise RetrievalError(422, "rrf_k must be a positive integer.")
    resolved_candidate_k = resolve_candidate_k(top_k, candidate_k)
    if not filter_collections:
        return {
            "question": cleaned,
            "top_k": top_k,
            "rrf_k": rrf_k,
            "candidate_k": resolved_candidate_k,
            "filter_collections": [],
            "vector": [],
            "lexical": [],
            "hybrid": [],
            "latency_ms": {
                "embed_ms": 0.0,
                "vector_ms": 0.0,
                "lexical_ms": 0.0,
                "fusion_ms": 0.0,
                "hybrid_ms": 0.0,
            },
            "documents": [],
        }

    embed_started = time.perf_counter()
    try:
        query_embedding = embedder.embed_texts([cleaned])[0]
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError() from exc
    embed_ms = _elapsed_ms(embed_started)

    vector_started = time.perf_counter()
    try:
        vector_rows = store.match_documents(
            query_embedding, resolved_candidate_k, filter_collections
        )
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError() from exc
    vector_ms = _elapsed_ms(vector_started)

    lexical_started = time.perf_counter()
    try:
        lexical_rows = store.search_fts(
            cleaned, resolved_candidate_k, filter_collections
        )
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError() from exc
    lexical_ms = _elapsed_ms(lexical_started)

    try:
        vector_hits = [_vector_hit(row) for row in vector_rows]
        lexical_hits = [_lexical_hit(row) for row in lexical_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise DatabaseError("Retrieval response was missing required fields.") from exc

    fusion_started = time.perf_counter()
    by_id = {hit["id"]: hit for hit in lexical_hits}
    by_id.update({hit["id"]: hit for hit in vector_hits})
    vector_ranks = {hit["id"]: hit["rank"] for hit in vector_hits}
    lexical_ranks = {hit["id"]: hit["rank"] for hit in lexical_hits}
    fused = fuse_rankings([vector_ranks, lexical_ranks], rrf_k=rrf_k)[:top_k]
    hybrid_hits = []
    for rank, (chunk_id, rrf_score) in enumerate(fused, start=1):
        source = by_id[chunk_id]
        hybrid_hits.append(
            {
                "id": chunk_id,
                "content": source["content"],
                "document_name": source["document_name"],
                "page": source["page"],
                "metadata": source["metadata"],
                "rrf_score": rrf_score,
                "rank": rank,
                "vector_rank": vector_ranks.get(chunk_id),
                "lexical_rank": lexical_ranks.get(chunk_id),
            }
        )
    fusion_ms = _elapsed_ms(fusion_started)

    return {
        "question": cleaned,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "candidate_k": resolved_candidate_k,
        "filter_collections": filter_collections,
        "vector": vector_hits,
        "lexical": lexical_hits,
        "hybrid": hybrid_hits,
        "latency_ms": {
            "embed_ms": embed_ms,
            "vector_ms": vector_ms,
            "lexical_ms": lexical_ms,
            "fusion_ms": fusion_ms,
            "hybrid_ms": round(vector_ms + lexical_ms + fusion_ms, 3),
        },
        "documents": _document_refs(vector_hits, lexical_hits),
    }
