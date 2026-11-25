"""Compare two finished v0.3 runs without rescoring or calling a model.

Retrieval changes on lexical and hybrid are attributed to the lexical version.
Vector retrieval changes are not. An answer that changes while the retrieved
chunk ids stay the same is treated as variation between LLM calls.
"""

import argparse
import csv
import json
from pathlib import Path

from retrieval_lab.experiments.metrics_v3 import percentile
from retrieval_lab.experiments.reliability import latest_success, load_jsonl


def _sum_known(values) -> float | str:
    collected = list(values)
    if any(value is None or value == "" for value in collected):
        return ""
    return round(sum(float(value) for value in collected), 8)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _contract(row: dict) -> str:
    return row.get("metrics_contract") or "v0.3-legacy"


def _full_evidence_rank(row: dict) -> float | None:
    if "reciprocal_full_evidence_rank" in row:
        return row.get("reciprocal_full_evidence_rank")
    return row.get("mrr")


def classify_change(
    strategy: str,
    baseline_chunks: list[int],
    candidate_chunks: list[int],
    baseline_answer: int | None,
    candidate_answer: int | None,
) -> str:
    chunks_changed = list(baseline_chunks) != list(candidate_chunks)
    answer_changed = baseline_answer != candidate_answer
    if not chunks_changed and not answer_changed:
        return "unchanged"
    if not chunks_changed and answer_changed:
        return "llm_variance"
    if strategy == "vector":
        return "vector_variance"
    if strategy == "hybrid":
        return "hybrid_lexical_change"
    return "lexical_change"


def _index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["question_id"], row["config_id"]): row for row in rows}


def compare_runs(baseline_dir: Path, candidate_dir: Path) -> dict[str, list[dict]]:
    baseline_retrieval = _index(latest_success(load_jsonl(baseline_dir / "retrieval_detail.jsonl")))
    candidate_retrieval = _index(latest_success(load_jsonl(candidate_dir / "retrieval_detail.jsonl")))
    baseline_e2e = _index(latest_success(load_jsonl(baseline_dir / "e2e_detail.jsonl")))
    candidate_e2e = _index(latest_success(load_jsonl(candidate_dir / "e2e_detail.jsonl")))
    keys = sorted(set(baseline_retrieval) & set(candidate_retrieval))
    pairs = []
    for key in keys:
        left = baseline_retrieval[key]
        right = candidate_retrieval[key]
        left_e2e = baseline_e2e.get(key, {})
        right_e2e = candidate_e2e.get(key, {})
        left_answer = left_e2e.get("answer_correct")
        right_answer = right_e2e.get("answer_correct")
        change = classify_change(
            left["strategy"],
            left.get("chunk_ids") or [],
            right.get("chunk_ids") or [],
            left_answer,
            right_answer,
        )
        pairs.append(
            {
                "question_id": key[0],
                "config_id": key[1],
                "strategy": left["strategy"],
                "change_class": change,
                "baseline_chunk_ids": left.get("chunk_ids") or [],
                "candidate_chunk_ids": right.get("chunk_ids") or [],
                "baseline_answer_correct": left_answer,
                "candidate_answer_correct": right_answer,
                "baseline_answer_support": left_e2e.get("answer_support"),
                "candidate_answer_support": right_e2e.get("answer_support"),
                "baseline_metrics_contract": _contract(left),
                "candidate_metrics_contract": _contract(right),
                "baseline_recall_at_k": left.get("recall_at_k"),
                "candidate_recall_at_k": right.get("recall_at_k"),
                "baseline_reciprocal_full_evidence_rank": _full_evidence_rank(left),
                "candidate_reciprocal_full_evidence_rank": _full_evidence_rank(right),
                "baseline_ndcg_at_k": left.get("ndcg_at_k"),
                "candidate_ndcg_at_k": right.get("ndcg_at_k"),
                "ndcg_comparable": _contract(left) == _contract(right),
                "baseline_evidence_coverage": left.get("evidence_coverage"),
                "candidate_evidence_coverage": right.get("evidence_coverage"),
                "baseline_strategy_total_ms": left.get("strategy_total_ms"),
                "candidate_strategy_total_ms": right.get("strategy_total_ms"),
                "baseline_e2e_latency_ms": left_e2e.get("e2e_latency_ms"),
                "candidate_e2e_latency_ms": right_e2e.get("e2e_latency_ms"),
                "baseline_chat_cost_usd": left_e2e.get("api_cost_usd_chat_estimated"),
                "candidate_chat_cost_usd": right_e2e.get("api_cost_usd_chat_estimated"),
            }
        )
    by_config: dict[str, list[dict]] = {}
    for row in pairs:
        by_config.setdefault(row["config_id"], []).append(row)
    summary = []
    for config_id in sorted(by_config):
        group = by_config[config_id]
        sample = group[0]

        def delta(name: str) -> float | str:
            left = [item[f"baseline_{name}"] for item in group if item.get(f"baseline_{name}") is not None]
            right = [item[f"candidate_{name}"] for item in group if item.get(f"candidate_{name}") is not None]
            if not left or not right:
                return ""
            return round(_mean([float(item) for item in right]) - _mean([float(item) for item in left]), 6)

        def paired_answer_delta() -> float | str:
            pairs = [
                item
                for item in group
                if item.get("baseline_answer_correct") is not None and item.get("candidate_answer_correct") is not None
            ]
            if not pairs:
                return ""
            return round(
                _mean([float(item["candidate_answer_correct"]) for item in pairs])
                - _mean([float(item["baseline_answer_correct"]) for item in pairs]),
                6,
            )

        ndcg_comparable = all(item["ndcg_comparable"] for item in group)
        same_legacy_name = all(
            item["baseline_metrics_contract"] == "v0.3-legacy"
            and item["candidate_metrics_contract"] == "v0.3-legacy"
            for item in group
        )
        baseline_latency = [
            float(item["baseline_strategy_total_ms"])
            for item in group
            if item.get("baseline_strategy_total_ms") is not None
        ]
        candidate_latency = [
            float(item["candidate_strategy_total_ms"])
            for item in group
            if item.get("candidate_strategy_total_ms") is not None
        ]
        baseline_e2e = [float(item["baseline_e2e_latency_ms"]) for item in group if item.get("baseline_e2e_latency_ms") is not None]
        candidate_e2e = [float(item["candidate_e2e_latency_ms"]) for item in group if item.get("candidate_e2e_latency_ms") is not None]
        summary.append(
            {
                "config_id": config_id,
                "strategy": sample["strategy"],
                "attribution": {
                    "lexical": "lexical_method",
                    "hybrid": "hybrid_lexical_input",
                    "vector": "not_lexical",
                }[sample["strategy"]],
                "delta_recall_at_k": delta("recall_at_k"),
                "delta_reciprocal_full_evidence_rank": delta("reciprocal_full_evidence_rank"),
                "delta_mrr": delta("reciprocal_full_evidence_rank") if same_legacy_name else "",
                "mrr_definition": (
                    "legacy_field_name_for_reciprocal_full_evidence_rank"
                    if same_legacy_name
                    else "conventional_mrr_not_in_contract"
                ),
                "delta_ndcg_at_k": delta("ndcg_at_k") if ndcg_comparable else "",
                "ndcg_comparable": ndcg_comparable,
                "delta_evidence_coverage": delta("evidence_coverage"),
                "delta_answer_correct": paired_answer_delta(),
                "baseline_correct_unsupported": sum(
                    1 for item in group if item["baseline_answer_support"] == "correct_unsupported"
                ),
                "candidate_correct_unsupported": sum(
                    1 for item in group if item["candidate_answer_support"] == "correct_unsupported"
                ),
                "baseline_chat_cost_usd": _sum_known(item.get("baseline_chat_cost_usd") for item in group),
                "candidate_chat_cost_usd": _sum_known(item.get("candidate_chat_cost_usd") for item in group),
                "baseline_latency_p50": round(percentile(baseline_latency, 0.50), 3) if baseline_latency else "",
                "baseline_latency_p95": round(percentile(baseline_latency, 0.95), 3) if baseline_latency else "",
                "candidate_latency_p50": round(percentile(candidate_latency, 0.50), 3) if candidate_latency else "",
                "candidate_latency_p95": round(percentile(candidate_latency, 0.95), 3) if candidate_latency else "",
                "baseline_e2e_latency_p50": round(percentile(baseline_e2e, 0.50), 3) if baseline_e2e else "",
                "baseline_e2e_latency_p95": round(percentile(baseline_e2e, 0.95), 3) if baseline_e2e else "",
                "candidate_e2e_latency_p50": round(percentile(candidate_e2e, 0.50), 3) if candidate_e2e else "",
                "candidate_e2e_latency_p95": round(percentile(candidate_e2e, 0.95), 3) if candidate_e2e else "",
                "n_unchanged": sum(1 for item in group if item["change_class"] == "unchanged"),
                "n_llm_variance": sum(1 for item in group if item["change_class"] == "llm_variance"),
                "n_lexical_change": sum(1 for item in group if item["change_class"] == "lexical_change"),
                "n_hybrid_lexical_change": sum(
                    1 for item in group if item["change_class"] == "hybrid_lexical_change"
                ),
                "n_vector_variance": sum(1 for item in group if item["change_class"] == "vector_variance"),
            }
        )
    return {"pairs": pairs, "summary": summary}


def write_comparison(baseline_dir: Path, candidate_dir: Path, out_dir: Path) -> None:
    compared = compare_runs(baseline_dir, candidate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("pairs", "summary"):
        rows = compared[name]
        path = out_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["config_id"])
            writer.writeheader()
            for row in rows:
                printable = {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                    for key, value in row.items()
                }
                writer.writerow(printable)
    (out_dir / "comparison_meta.json").write_text(
        json.dumps(
            {
                "baseline_dir": str(baseline_dir),
                "candidate_dir": str(candidate_dir),
                "lexical_and_hybrid_retrieval_deltas": "attributed_to_lexical_version",
                "vector_retrieval_deltas": "not_attributed_to_lexical_version",
                "same_chunks_different_answer": "llm_variance",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two finished v0.3 experiment runs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_comparison(args.baseline, args.candidate, args.out)


if __name__ == "__main__":
    main()
