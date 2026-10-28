def fuse_rankings(
    rankings: list[dict[int, int]],
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse 1-based rankings with Reciprocal Rank Fusion.

    Each mapping is chunk id to rank inside one retrieved list. A chunk that
    is absent from a list contributes nothing from that list. Ties break by
    chunk id ascending. Scores from different strategies are not rescaled.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer.")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for chunk_id, rank in ranking.items():
            if rank < 1:
                raise ValueError("ranks are 1-based.")
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
