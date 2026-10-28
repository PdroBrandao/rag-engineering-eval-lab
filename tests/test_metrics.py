import json
from pathlib import Path

import pytest

from retrieval_lab.evaluate import evaluate_cases, load_dataset
from retrieval_lab.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


def test_precision_recall_and_mrr():
    retrieved = [3, 1, 2, 9]
    relevant = [1, 2]
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert mean_reciprocal_rank(retrieved, relevant) == 0.5
    assert recall_at_k(retrieved, [], 2) == 0.0
    assert mean_reciprocal_rank([4, 5], relevant) == 0.0


def test_evaluate_cases_compares_vector_and_hybrid_at_k():
    def query_fn(question: str, k: int):
        assert k == 2
        return {
            "vector": [{"id": 3}, {"id": 1}, {"id": 2}],
            "lexical": [{"id": 8}],
            "hybrid": [{"id": 1}, {"id": 2}, {"id": 9}],
        }

    report = evaluate_cases(
        [{"question": "risks", "relevant_chunk_ids": [1, 2]}],
        2,
        query_fn,
    )
    assert report["cases"][0]["strategies"]["vector"]["recall_at_k"] == 0.5
    assert report["cases"][0]["strategies"]["hybrid"]["recall_at_k"] == 1.0
    assert report["mean"]["hybrid"]["precision_at_k"] == 1.0


def test_load_dataset_rejects_missing_questions(tmp_path: Path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"k": 5, "cases": [{"relevant_chunk_ids": [1]}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)
