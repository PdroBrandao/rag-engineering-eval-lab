def precision_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be a positive integer.")
    relevant = set(relevant_ids)
    hits = sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in relevant)
    return hits / k


def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be a positive integer.")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved_ids: list[int], relevant_ids: list[int]) -> float:
    relevant = set(relevant_ids)
    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def score_ranking(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> dict[str, float]:
    return {
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
    }
