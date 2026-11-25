from retrieval_lab.rrf import fuse_rankings

STANDALONE_LATENCY_LIMITATION = (
    "Estimativa montada com os componentes da execução compartilhada. "
    "Não substitui um benchmark isolado com repetições."
)


def derive_hits(bundle: dict, strategy: str, top_k: int, rrf_k: int) -> list[dict]:
    vector = list(bundle.get("vector") or [])
    lexical = list(bundle.get("lexical") or [])
    if strategy == "vector":
        return vector[:top_k]
    if strategy == "lexical":
        return lexical[:top_k]
    if strategy != "hybrid":
        raise ValueError(f"Unknown strategy: {strategy}")
    by_id = {hit["id"]: hit for hit in lexical}
    by_id.update({hit["id"]: hit for hit in vector})
    fused = fuse_rankings(
        [
            {hit["id"]: hit["rank"] for hit in vector},
            {hit["id"]: hit["rank"] for hit in lexical},
        ],
        rrf_k=rrf_k,
    )[:top_k]
    hits = []
    for rank, (chunk_id, rrf_score) in enumerate(fused, start=1):
        source = by_id[chunk_id]
        hits.append(
            {
                "id": chunk_id,
                "content": source.get("content") or "",
                "document_name": source.get("document_name") or "",
                "page": source.get("page"),
                "metadata": source.get("metadata") or {},
                "rrf_score": rrf_score,
                "rank": rank,
            }
        )
    return hits


def attribute_latency(bundle: dict, strategy: str, *, embed_source: str) -> dict:
    """Separate the shared-run clock from a standalone estimate.

    `measured_execution_latency_ms` sums clocks that this shared execution
    actually spent for the strategy. A cached embedding stays visible and is
    not added again. `estimated_standalone_latency_ms` adds that cached
    embedding back for vector and hybrid, as an estimate of one independent
    query. Lexical search does not embed the query.
    """
    latency = bundle.get("latency_ms") or {}
    embed_ms = float(latency.get("embed_ms") or 0.0)
    vector_ms = float(latency.get("vector_ms") or 0.0)
    lexical_ms = float(latency.get("lexical_ms") or 0.0)
    fusion_ms = float(latency.get("fusion_ms") or 0.0)
    components = []
    measured = 0.0
    standalone = 0.0

    def add(name: str, ms: float, source: str, *, in_measured: bool, in_standalone: bool) -> None:
        nonlocal measured, standalone
        components.append(
            {
                "name": name,
                "ms": round(ms, 3),
                "source": source,
                "counted_in_measured_execution": in_measured,
                "included_in_standalone_estimate": in_standalone,
                "counted_in_total": in_measured,
            }
        )
        if in_measured:
            measured += ms
        if in_standalone:
            standalone += ms

    if strategy in {"vector", "hybrid"}:
        if embed_source == "measured":
            add("embed_ms", embed_ms, "measured", in_measured=True, in_standalone=True)
        else:
            add("embed_ms", embed_ms, "shared_cache", in_measured=False, in_standalone=True)
    if strategy in {"vector", "hybrid"}:
        add("vector_retrieval_ms", vector_ms, "measured_shared_call", in_measured=True, in_standalone=True)
    if strategy in {"lexical", "hybrid"}:
        add("lexical_retrieval_ms", lexical_ms, "measured_shared_call", in_measured=True, in_standalone=True)
    if strategy == "hybrid":
        add("fusion_ms", fusion_ms, "measured", in_measured=True, in_standalone=True)
    measured_ms = round(measured, 3)
    return {
        "components": components,
        "measured_execution_latency_ms": measured_ms,
        "measured_execution_latency_basis": "measured_shared_execution_excludes_cached_embed",
        "estimated_standalone_latency_ms": round(standalone, 3),
        "estimated_standalone_latency_basis": "estimated_from_shared_call_components",
        "standalone_latency_limitation": STANDALONE_LATENCY_LIMITATION,
        "strategy_total_ms": measured_ms,
        "strategy_total_basis": "alias_of_measured_execution_latency_ms",
        "embed_ms_measured": embed_ms if embed_source == "measured" and strategy in {"vector", "hybrid"} else None,
        "embed_ms_shared_cache": embed_ms if embed_source == "shared_cache" and strategy in {"vector", "hybrid"} else None,
    }
