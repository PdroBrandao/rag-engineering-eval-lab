import json
from pathlib import Path

from retrieval_lab.experiments.compare_runs import classify_change, compare_runs


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _retrieval(question_id: str, config_id: str, strategy: str, chunks: list[int], recall: float) -> dict:
    return {
        "status": "success",
        "question_id": question_id,
        "config_id": config_id,
        "strategy": strategy,
        "chunk_ids": chunks,
        "recall_at_k": recall,
        "mrr": recall,
        "ndcg_at_k": recall,
        "evidence_coverage": recall,
        "strategy_total_ms": 10.0,
    }


def _e2e(question_id: str, config_id: str, answer: int, support: str) -> dict:
    return {
        "status": "success",
        "stage": "generation",
        "question_id": question_id,
        "config_id": config_id,
        "answer_correct": answer,
        "answer_support": support,
        "e2e_latency_ms": 30.0,
        "api_cost_usd_chat_estimated": 0.001,
    }


def test_comparison_separates_lexical_changes_from_llm_variance(tmp_path: Path):
    assert classify_change("lexical", [1], [1], 1, 0) == "llm_variance"
    assert classify_change("lexical", [1], [2], 1, 0) == "lexical_change"
    assert classify_change("hybrid", [1], [2], 1, 1) == "hybrid_lexical_change"
    assert classify_change("vector", [1], [2], 1, 0) == "vector_variance"
    assert classify_change("vector", [1], [1], 1, 1) == "unchanged"

    baseline = tmp_path / "v01"
    candidate = tmp_path / "v02"
    baseline.mkdir()
    candidate.mkdir()
    _write(
        baseline / "retrieval_detail.jsonl",
        [
            _retrieval("Q001", "cs512_top5_lexical", "lexical", [1], 0.0),
            _retrieval("Q001", "cs512_top5_hybrid", "hybrid", [1], 0.0),
            _retrieval("Q001", "cs512_top5_vector", "vector", [4], 1.0),
        ],
    )
    _write(
        candidate / "retrieval_detail.jsonl",
        [
            _retrieval("Q001", "cs512_top5_lexical", "lexical", [9], 1.0),
            _retrieval("Q001", "cs512_top5_hybrid", "hybrid", [9], 1.0),
            _retrieval("Q001", "cs512_top5_vector", "vector", [4], 1.0),
        ],
    )
    _write(
        baseline / "e2e_detail.jsonl",
        [
            _e2e("Q001", "cs512_top5_lexical", 0, "incorrect"),
            _e2e("Q001", "cs512_top5_hybrid", 1, "correct_unsupported"),
            _e2e("Q001", "cs512_top5_vector", 1, "supported"),
        ],
    )
    _write(
        candidate / "e2e_detail.jsonl",
        [
            _e2e("Q001", "cs512_top5_lexical", 1, "supported"),
            _e2e("Q001", "cs512_top5_hybrid", 1, "supported"),
            _e2e("Q001", "cs512_top5_vector", 0, "incorrect"),
        ],
    )
    compared = compare_runs(baseline, candidate)
    classes = {row["config_id"]: row["change_class"] for row in compared["pairs"]}
    assert classes == {
        "cs512_top5_lexical": "lexical_change",
        "cs512_top5_hybrid": "hybrid_lexical_change",
        "cs512_top5_vector": "llm_variance",
    }
    summary = {row["config_id"]: row for row in compared["summary"]}
    assert summary["cs512_top5_lexical"]["attribution"] == "lexical_method"
    assert summary["cs512_top5_lexical"]["delta_recall_at_k"] == 1.0
    assert summary["cs512_top5_hybrid"]["attribution"] == "hybrid_lexical_input"
    assert summary["cs512_top5_hybrid"]["baseline_correct_unsupported"] == 1
    assert summary["cs512_top5_hybrid"]["candidate_correct_unsupported"] == 0
    assert summary["cs512_top5_vector"]["attribution"] == "not_lexical"
    assert summary["cs512_top5_vector"]["delta_recall_at_k"] == 0.0
    assert summary["cs512_top5_vector"]["n_llm_variance"] == 1
    assert summary["cs512_top5_lexical"]["mrr_definition"] == "legacy_field_name_for_reciprocal_full_evidence_rank"
    assert summary["cs512_top5_lexical"]["ndcg_comparable"] is True

    newer = tmp_path / "v04"
    newer.mkdir()
    upgraded = _retrieval("Q001", "cs512_top5_lexical", "lexical", [9], 1.0)
    upgraded["metrics_contract"] = "v0.4"
    upgraded["reciprocal_full_evidence_rank"] = 1.0
    _write(newer / "retrieval_detail.jsonl", [upgraded])
    _write(newer / "e2e_detail.jsonl", [_e2e("Q001", "cs512_top5_lexical", None, "inconclusive")])
    mixed = compare_runs(baseline, newer)
    mixed_summary = mixed["summary"][0]
    assert mixed_summary["ndcg_comparable"] is False
    assert mixed_summary["delta_ndcg_at_k"] == ""
    assert mixed_summary["delta_mrr"] == ""
    assert mixed_summary["delta_answer_correct"] == ""
