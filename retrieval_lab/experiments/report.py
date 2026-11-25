import csv
from pathlib import Path

from retrieval_lab.experiments.metrics_v3 import percentile
from retrieval_lab.experiments.reliability import latest_success, load_jsonl
from retrieval_lab.experiments.uncertainty import wilson_interval


def _present(values: list) -> list[float]:
    return [float(value) for value in values if value is not None and value != ""]


def _mean(values: list) -> float | None:
    numbers = _present(values)
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _fmt(value: float | None, digits: int) -> float | str:
    if value is None:
        return ""
    return round(value, digits)


def _full_evidence_rank(item: dict) -> float | None:
    if "reciprocal_full_evidence_rank" in item:
        return item.get("reciprocal_full_evidence_rank")
    return item.get("mrr")


def _sum_optional(values: list) -> float | None:
    if any(value is None or value == "" for value in values):
        return None
    return sum(float(value) for value in values)


def _cost_basis(values: list) -> str:
    if any(value is None or value == "" for value in values):
        return "unknown"
    return "sum_of_row_costs"


def _number(*values: object) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        return float(value)
    return None


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_reports(run_dir: Path, *, run_id: str, baseline_config_id: str) -> None:
    retrieval_rows = latest_success(load_jsonl(run_dir / "retrieval_detail.jsonl"))
    e2e_rows = latest_success(load_jsonl(run_dir / "e2e_detail.jsonl"))
    by_config_retrieval: dict[str, list[dict]] = {}
    for row in retrieval_rows:
        by_config_retrieval.setdefault(row["config_id"], []).append(row)
    by_config_e2e: dict[str, list[dict]] = {}
    for row in e2e_rows:
        by_config_e2e.setdefault(row["config_id"], []).append(row)

    retrieval_csv = []
    e2e_csv = []
    for config_id in sorted(by_config_retrieval):
        group = by_config_retrieval[config_id]
        sample = group[0]
        measured_latency = _present(item.get("measured_execution_latency_ms", item.get("strategy_total_ms")) for item in group)
        standalone_latency = _present(item.get("estimated_standalone_latency_ms") for item in group)
        measured_embeds = [
            float(item["embed_ms_measured"])
            for item in group
            if item.get("embed_ms_measured") is not None
        ]
        retrieval_csv.append(
            {
                "run_id": run_id,
                "metrics_contract": sample.get("metrics_contract") or "v0.3-legacy",
                "config_id": config_id,
                "chunk_size": sample["chunk_size"],
                "index_set_id": sample["index_set_id"],
                "top_k": sample["top_k"],
                "strategy": sample["strategy"],
                "n_questions": len(group),
                "recall_at_k_mean": _fmt(_mean([item["recall_at_k"] for item in group]), 6),
                "reciprocal_full_evidence_rank_mean": _fmt(
                    _mean([_full_evidence_rank(item) for item in group]), 6
                ),
                "ndcg_at_k_mean": _fmt(_mean([item["ndcg_at_k"] for item in group]), 6),
                "evidence_coverage_mean": _fmt(_mean([item["evidence_coverage"] for item in group]), 6),
                "context_tokens_mean": _fmt(_mean([item["context_tokens"] for item in group]), 3),
                "measured_execution_latency_ms_p50": round(percentile(measured_latency, 0.50), 3) if measured_latency else "",
                "measured_execution_latency_ms_p95": round(percentile(measured_latency, 0.95), 3) if measured_latency else "",
                "measured_execution_latency_basis": "measured_shared_execution_excludes_cached_embed",
                "estimated_standalone_latency_ms_p50": round(percentile(standalone_latency, 0.50), 3) if standalone_latency else "",
                "estimated_standalone_latency_ms_p95": round(percentile(standalone_latency, 0.95), 3) if standalone_latency else "",
                "estimated_standalone_latency_basis": "estimated_from_shared_call_components_not_an_isolated_benchmark",
                "strategy_total_ms_p50": round(percentile(measured_latency, 0.50), 3) if measured_latency else "",
                "strategy_total_ms_p95": round(percentile(measured_latency, 0.95), 3) if measured_latency else "",
                "strategy_total_basis": "alias_of_measured_execution_latency_ms",
                "embed_ms_measured_mean": round(_mean(measured_embeds), 3) if measured_embeds else "",
                "embed_shared_cache_rate": _fmt(
                    _mean([1.0 if item.get("embed_ms_shared_cache") is not None else 0.0 for item in group]),
                    6,
                ),
                "needs_review_rate": _fmt(
                    _mean([1.0 if item["needs_review"] else 0.0 for item in group]), 6
                ),
                "recall_mean_includes_unconfirmed_needs_review": True,
            }
        )
        e2e_group = by_config_e2e.get(config_id, [])
        if not e2e_group:
            continue
        graded = [item for item in e2e_group if item.get("answer_correct") in (0, 1)]
        llm = _present(item.get("llm_latency_ms") for item in e2e_group)
        measured_e2e = _present(
            item.get("measured_e2e_execution_latency_ms", item.get("e2e_latency_ms")) for item in e2e_group
        )
        standalone_e2e = _present(item.get("estimated_standalone_e2e_latency_ms") for item in e2e_group)
        judge_latency = _present(item.get("judge_latency_ms") for item in e2e_group)
        generation_costs = [item.get("generation_cost_usd", item.get("api_cost_usd_chat_estimated")) for item in e2e_group]
        judge_costs = [item.get("judge_cost_usd") for item in e2e_group]
        e2e_csv.append(
            {
                "run_id": run_id,
                "metrics_contract": e2e_group[0].get("metrics_contract") or "v0.3-legacy",
                "config_id": config_id,
                "chunk_size": sample["chunk_size"],
                "index_set_id": sample["index_set_id"],
                "top_k": sample["top_k"],
                "strategy": sample["strategy"],
                "generate_model": e2e_group[0].get("generate_model"),
                "n_questions": len(e2e_group),
                "n_graded_auto": len(graded),
                "answer_correct_rate": _fmt(_mean([item["answer_correct"] for item in graded]), 6),
                "answer_uncertain_rate": _fmt(
                    _mean([1.0 if item.get("uncertain") else 0.0 for item in e2e_group]), 6
                ),
                "correct_unsupported_rate": _fmt(
                    _mean(
                        [
                            1.0 if item.get("answer_support") == "correct_unsupported" else 0.0
                            for item in graded
                        ]
                    ),
                    6,
                ),
                "human_pending_count": sum(1 for item in e2e_group if item.get("judge_pending")),
                "prompt_tokens_measured_mean": _fmt(
                    _mean([item.get("prompt_tokens_measured") for item in e2e_group]), 3
                ),
                "completion_tokens_measured_mean": _fmt(
                    _mean([item.get("completion_tokens_measured") for item in e2e_group]), 3
                ),
                "llm_latency_ms_p50_measured": round(percentile(llm, 0.50), 3) if llm else "",
                "llm_latency_ms_p95_measured": round(percentile(llm, 0.95), 3) if llm else "",
                "measured_e2e_execution_latency_ms_p50": round(percentile(measured_e2e, 0.50), 3) if measured_e2e else "",
                "measured_e2e_execution_latency_ms_p95": round(percentile(measured_e2e, 0.95), 3) if measured_e2e else "",
                "estimated_standalone_e2e_latency_ms_p50": round(percentile(standalone_e2e, 0.50), 3) if standalone_e2e else "",
                "estimated_standalone_e2e_latency_ms_p95": round(percentile(standalone_e2e, 0.95), 3) if standalone_e2e else "",
                "judge_latency_ms_p50_measured": round(percentile(judge_latency, 0.50), 3) if judge_latency else "",
                "e2e_latency_ms_p50": round(percentile(measured_e2e, 0.50), 3) if measured_e2e else "",
                "e2e_latency_ms_p95": round(percentile(measured_e2e, 0.95), 3) if measured_e2e else "",
                "e2e_latency_basis": "alias_of_measured_e2e_execution_excludes_cached_embed_and_judge",
                "standalone_latency_limitation": "estimated_standalone_e2e_latency_ms is not an isolated repeated benchmark",
                "generation_cost_usd": _fmt(_sum_optional(generation_costs), 8),
                "embedding_cost_usd_shared_not_additive": e2e_group[0].get(
                    "embedding_cost_usd", e2e_group[0].get("api_cost_usd_embed_shared_estimated")
                ),
                "judge_cost_usd": _fmt(_sum_optional(judge_costs), 8),
                "judge_cost_basis": _cost_basis(judge_costs),
                "api_cost_usd_chat_estimated": _fmt(_sum_optional(generation_costs), 8),
                "chat_cost_basis": "alias_of_generation_cost_usd",
                "api_cost_usd_embed_shared_estimated": e2e_group[0].get(
                    "api_cost_usd_embed_shared_estimated"
                ),
                "embed_cost_basis": "shared_estimated_do_not_sum_across_configs",
                "empty_or_refusal_rate": _fmt(
                    _mean([1.0 if item.get("empty_or_refusal") else 0.0 for item in e2e_group]), 6
                ),
            }
        )

    retrieval_fields = list(retrieval_csv[0].keys()) if retrieval_csv else ["run_id"]
    e2e_fields = list(e2e_csv[0].keys()) if e2e_csv else ["run_id"]
    _write_csv(run_dir / "retrieval.csv", retrieval_csv, retrieval_fields)
    if e2e_csv:
        _write_csv(run_dir / "end_to_end.csv", e2e_csv, e2e_fields)

    baseline = next((row for row in retrieval_csv if row["config_id"] == baseline_config_id), None)
    baseline_e2e = next((row for row in e2e_csv if row["config_id"] == baseline_config_id), None)
    comparison = []
    for row in retrieval_csv:
        e2e = next((item for item in e2e_csv if item["config_id"] == row["config_id"]), {})
        recall = float(row["recall_at_k_mean"])
        base_recall = float(baseline["recall_at_k_mean"]) if baseline else 0.0
        answer = e2e.get("answer_correct_rate")
        base_answer = baseline_e2e.get("answer_correct_rate") if baseline_e2e else ""
        delta_answer = ""
        if answer not in ("", None) and base_answer not in ("", None):
            delta_answer = round(float(answer) - float(base_answer), 6)
        latency = _number(e2e.get("measured_e2e_execution_latency_ms_p95"), e2e.get("e2e_latency_ms_p95"), row["strategy_total_ms_p95"])
        base_latency = _number(
            (baseline_e2e or {}).get("measured_e2e_execution_latency_ms_p95"),
            (baseline_e2e or {}).get("e2e_latency_ms_p95"),
            (baseline or {}).get("strategy_total_ms_p95"),
        )
        delta_latency = "" if latency is None or base_latency is None else round(latency - base_latency, 3)
        chat_cost = e2e.get("generation_cost_usd", e2e.get("api_cost_usd_chat_estimated"))
        base_chat = (baseline_e2e or {}).get("generation_cost_usd", (baseline_e2e or {}).get("api_cost_usd_chat_estimated"))
        delta_chat = ""
        if chat_cost not in ("", None) and base_chat not in ("", None):
            delta_chat = round(float(chat_cost) - float(base_chat), 8)
        promising = False
        if baseline and delta_answer != "" and base_latency is not None:
            latency_limit = base_latency * 1.10
            promising = (
                (recall - base_recall) > 0.05
                and float(delta_answer) >= 0
                and latency is not None
                and latency <= latency_limit
            )
        comparison.append(
            {
                "run_id": run_id,
                "config_id": row["config_id"],
                "is_baseline": row["config_id"] == baseline_config_id,
                "recall_at_k_mean": row["recall_at_k_mean"],
                "evidence_coverage_mean": row["evidence_coverage_mean"],
                "reciprocal_full_evidence_rank_mean": row["reciprocal_full_evidence_rank_mean"],
                "ndcg_at_k_mean": row["ndcg_at_k_mean"],
                "answer_correct_rate": answer,
                "correct_unsupported_rate": e2e.get("correct_unsupported_rate", ""),
                "context_tokens_mean": row["context_tokens_mean"],
                "measured_execution_latency_ms_p95": row["measured_execution_latency_ms_p95"],
                "estimated_standalone_latency_ms_p95": row["estimated_standalone_latency_ms_p95"],
                "strategy_total_ms_p95": row["strategy_total_ms_p95"],
                "measured_e2e_execution_latency_ms_p95": e2e.get("measured_e2e_execution_latency_ms_p95", ""),
                "estimated_standalone_e2e_latency_ms_p95": e2e.get("estimated_standalone_e2e_latency_ms_p95", ""),
                "e2e_latency_ms_p95": e2e.get("e2e_latency_ms_p95", ""),
                "latency_gate_basis": "measured_shared_execution_excludes_cached_embed_not_a_production_benchmark",
                "generation_cost_usd": e2e.get("generation_cost_usd", ""),
                "judge_cost_usd": e2e.get("judge_cost_usd", ""),
                "api_cost_usd_chat_estimated": e2e.get("api_cost_usd_chat_estimated", ""),
                "delta_recall_vs_baseline": round(recall - base_recall, 6),
                "delta_coverage_vs_baseline": round(
                    float(row["evidence_coverage_mean"])
                    - float(baseline["evidence_coverage_mean"] if baseline else 0),
                    6,
                ),
                "delta_answer_correct_vs_baseline": delta_answer,
                "delta_e2e_latency_p95_vs_baseline": delta_latency,
                "delta_generation_cost_usd_vs_baseline": delta_chat,
                "promising": promising,
                "verdict": "inconclusive",
            }
        )
    if comparison:
        _write_csv(run_dir / "comparison.csv", comparison, list(comparison[0].keys()))
    _write_extended_reports(run_dir, retrieval_rows, load_jsonl(run_dir / "e2e_detail.jsonl"))


def _write_extended_reports(run_dir: Path, retrieval_rows: list[dict], e2e_rows: list[dict]) -> None:
    """Extra reports. Existing CSV columns stay as they are.

    Failed rows, judge failures and unsupported rubrics stay out of the means.
    """
    status_rows = []
    by_config: dict[str, list[dict]] = {}
    for row in e2e_rows:
        by_config.setdefault(row.get("config_id") or "", []).append(row)
    for config_id in sorted(by_config):
        group = by_config[config_id]
        counts = {"valid": 0, "inconclusive": 0, "not_evaluated": 0}
        for row in group:
            state = row.get("evaluation_status")
            if row.get("status", "success") != "success" or row.get("unsupported_rubric") or row.get("judge_pending"):
                state = "not_evaluated"
            elif state not in counts:
                state = "inconclusive" if row.get("answer_correct") is None else "valid"
            counts[state] += 1
        status_rows.append({"config_id": config_id, **counts, "n_rows": len(group)})
    if status_rows:
        _write_csv(run_dir / "evaluation_status.csv", status_rows, list(status_rows[0].keys()))

    category_rows = []
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in retrieval_rows:
        grouped.setdefault((row.get("category") or "unspecified", row["config_id"]), []).append(row)
    for (category, config_id), group in sorted(grouped.items()):
        category_rows.append(
            {
                "category": category,
                "config_id": config_id,
                "strategy": group[0].get("strategy"),
                "n_questions": len(group),
                "recall_at_k_mean": _fmt(_mean([item.get("recall_at_k") for item in group]), 6),
                "reciprocal_full_evidence_rank_mean": _fmt(
                    _mean([_full_evidence_rank(item) for item in group]), 6
                ),
                "ndcg_at_k_mean": _fmt(_mean([item.get("ndcg_at_k") for item in group]), 6),
                "evidence_coverage_mean": _fmt(_mean([item.get("evidence_coverage") for item in group]), 6),
                "needs_review_count": sum(1 for item in group if item.get("needs_review")),
            }
        )
    if category_rows:
        _write_csv(run_dir / "category_summary.csv", category_rows, list(category_rows[0].keys()))

    success_e2e = [row for row in e2e_rows if row.get("status", "success") == "success"]
    by_question: dict[str, list[dict]] = {}
    for row in success_e2e:
        by_question.setdefault(row["question_id"], []).append(row)
    cost_rows = []
    for question_id in sorted(by_question):
        rows = by_question[question_id]
        generation_values = [row.get("generation_cost_usd", row.get("api_cost_usd_chat_estimated")) for row in rows]
        judge_values = [row.get("judge_cost_usd") for row in rows]
        embed_once: dict[str, float] = {}
        embed_unknown = False
        for row in rows:
            embed = row.get("embedding_cost_usd", row.get("api_cost_usd_embed_shared_estimated"))
            if embed is None or embed == "":
                embed_unknown = True
                continue
            embed_once[row.get("index_set_id") or ""] = float(embed)
        generation = _sum_optional(generation_values)
        judge = _sum_optional(judge_values)
        embedding = None if embed_unknown else round(sum(embed_once.values()), 8)
        operational = _sum_optional([generation, embedding])
        total = _sum_optional([operational, judge])
        cost_rows.append(
            {
                "question_id": question_id,
                "generation_cost_usd": _fmt(generation, 8),
                "embedding_cost_usd": _fmt(embedding, 8),
                "judge_cost_usd": _fmt(judge, 8),
                "rag_operational_cost_usd": _fmt(operational, 8),
                "evaluation_total_cost_usd": _fmt(total, 8),
                "cost_basis": "generation_per_answer_plus_embed_once_per_index_set_plus_judge",
                "rag_operational_cost_basis": "generation_plus_embedding_excludes_judge",
            }
        )
    if cost_rows:
        _write_csv(run_dir / "cost_by_question.csv", cost_rows, list(cost_rows[0].keys()))

    uncertainty_rows = []
    by_retrieval: dict[str, list[dict]] = {}
    for row in retrieval_rows:
        by_retrieval.setdefault(row["config_id"], []).append(row)
    for config_id, group in sorted(by_retrieval.items()):
        satisfied = sum(1 for item in group if item.get("evidence_satisfied"))
        low, high = wilson_interval(satisfied, len(group))
        uncertainty_rows.append(
            {
                "config_id": config_id,
                "metric": "evidence_satisfied_rate",
                "n": len(group),
                "estimate": round(satisfied / len(group), 6) if group else "",
                "low_95": round(low, 6),
                "high_95": round(high, 6),
                "limitation": "Wilson interval. table_ambiguous matches stay unconfirmed and are not answer grades.",
            }
        )
    by_answer: dict[str, list[dict]] = {}
    for row in success_e2e:
        if row.get("answer_correct") in (0, 1) and not row.get("unsupported_rubric"):
            by_answer.setdefault(row["config_id"], []).append(row)
    for config_id, group in sorted(by_answer.items()):
        correct = sum(1 for item in group if item.get("answer_correct") == 1)
        low, high = wilson_interval(correct, len(group))
        uncertainty_rows.append(
            {
                "config_id": config_id,
                "metric": "answer_correct_rate_valid_only",
                "n": len(group),
                "estimate": round(correct / len(group), 6),
                "low_95": round(low, 6),
                "high_95": round(high, 6),
                "limitation": "Excludes inconclusive judge verdicts, unsupported rubrics and failed calls.",
            }
        )
    if uncertainty_rows:
        _write_csv(run_dir / "uncertainty.csv", uncertainty_rows, list(uncertainty_rows[0].keys()))

    review_rows = []
    for row in e2e_rows:
        reasons = []
        if row.get("status") == "failed":
            reasons.append("failed")
        if row.get("unsupported_rubric"):
            reasons.append("unsupported_rubric")
        if row.get("judge_pending"):
            reasons.append("judge_pending")
        if row.get("uncertain") and row.get("answer_correct") is None and not row.get("unsupported_rubric"):
            reasons.append("judge_uncertain")
        if reasons:
            review_rows.append(
                {
                    "question_id": row.get("question_id"),
                    "config_id": row.get("config_id"),
                    "reasons": "|".join(reasons),
                    "answer_correct": row.get("answer_correct"),
                }
            )
    for row in retrieval_rows:
        if row.get("needs_review"):
            review_rows.append(
                {
                    "question_id": row.get("question_id"),
                    "config_id": row.get("config_id"),
                    "reasons": "needs_review",
                    "answer_correct": "",
                }
            )
    if review_rows:
        _write_csv(run_dir / "review_cases.csv", review_rows, list(review_rows[0].keys()))
