import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import tiktoken

from retrieval_lab.experiments.derive import attribute_latency, derive_hits
from retrieval_lab.experiments.grading import (
    context_from_hits,
    evaluation_costs,
    generate_answer,
    grade_answer,
    judge_with_client,
    support_label,
)
from retrieval_lab.experiments.ground_truth import load_suite
from retrieval_lab.experiments.manifest import effective_collections, load_manifest
from retrieval_lab.experiments.metrics_v3 import METRICS_CONTRACT, PRICING_VERSION, score_hits
from retrieval_lab.experiments.reliability import (
    CALL_TIMEOUT_SECONDS,
    ProgressLog,
    append_jsonl,
    call_with_retry,
    fingerprint,
    load_jsonl,
    load_or_check_checkpoint,
    success_keys,
)
from retrieval_lab.experiments.report import write_reports
from retrieval_lab.retrieval import query_lexical_pt_or, query_retrieval

QueryFn = Callable[[str, str], dict[str, Any]]
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text or ""))


def _add_latency(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) + float(right), 3)


def _sum_known(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return round(sum(float(value) for value in values), 8)


def _embed_cost(question: str, pricing: dict) -> float:
    tokens = count_tokens(question)
    return round(tokens * float(pricing["embedding_input"]) / 1_000_000, 8)


def _failure_fields(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:300],
        "metrics_contract": METRICS_CONTRACT,
        "recall_at_k": None,
        "reciprocal_full_evidence_rank": None,
        "ndcg_at_k": None,
        "evidence_coverage": None,
        "answer_correct": None,
    }


def run_experiment(
    *,
    run_id: str,
    manifest: dict,
    questions: list[dict],
    out_dir: Path,
    query_fn: QueryFn,
    layers: set[str],
    read_collections: list[str],
    generate_fn: Callable[[str, str], dict] | None = None,
    judge_fn: Callable[[str, str, str], dict] | None = None,
    progress_stream: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> dict[str, int]:
    collections = effective_collections(manifest["collections"], read_collections)
    out_dir.mkdir(parents=True, exist_ok=True)
    retrieval = manifest["retrieval"]
    lexical_version = retrieval.get("lexical_version", "english_and")
    if lexical_version not in {"english_and", "portuguese_or"}:
        raise ValueError("lexical_version must be english_and or portuguese_or.")
    load_or_check_checkpoint(
        out_dir / "checkpoint.json",
        fingerprint(manifest, questions, layers),
        run_id=run_id,
        lexical_version=lexical_version,
    )
    retrieval_path = out_dir / "retrieval_detail.jsonl"
    e2e_path = out_dir / "e2e_detail.jsonl"
    done_retrieval = success_keys(load_jsonl(retrieval_path), "retrieval")
    done_e2e = success_keys(load_jsonl(e2e_path), "generation")
    selected_strategies = set(manifest.get("strategies") or ("lexical", "vector", "hybrid"))
    selected_config_ids = manifest.get("config_ids")
    active_configs = [
        config for config in manifest["configs"] if config["strategy"] in selected_strategies
    ]
    if selected_config_ids:
        wanted = set(selected_config_ids)
        unknown = wanted - {config["config_id"] for config in manifest["configs"]}
        if unknown:
            raise ValueError("Unknown config_ids: " + ", ".join(sorted(unknown)))
        active_configs = [config for config in active_configs if config["config_id"] in wanted]
        if not active_configs:
            raise ValueError("config_ids did not match any active configuration.")
    generate = manifest["generate"]
    prompt_hash = hashlib.sha256(generate["system_prompt"].encode("utf-8")).hexdigest()
    embed_cost_once: dict[tuple[str, str], float] = {}
    progress = ProgressLog(len(questions) * len(active_configs), progress_stream)
    progress.completed = sum(
        1
        for question in questions
        for config in active_configs
        if (question["question_id"], config["config_id"]) in done_retrieval
        and ("end_to_end" not in layers or (question["question_id"], config["config_id"]) in done_e2e)
    )
    question_success = {question["question_id"]: 0 for question in questions}
    question_failed = {question["question_id"]: 0 for question in questions}

    def _retry(operation: str, fn: Callable[[], Any]) -> Any:
        def _on_retry(name: str, attempt: int, reason: str, delay: float) -> None:
            progress.retry(name, attempt, reason, delay)

        result = call_with_retry(operation, fn, sleep=sleep, rng=rng, on_retry=_on_retry)
        progress.calls += 1
        return result

    for question_index, question in enumerate(questions, start=1):
        label = f"{question['question_id']}/{len(questions)}"
        bundles: dict[str, dict] = {}
        bundle_errors: dict[str, BaseException] = {}
        measured_embed_for: set[str] = set()
        question_success[question["question_id"]] = 0
        question_failed[question["question_id"]] = 0
        for config in active_configs:
            index_set_id = config["index_set_id"]
            key = (question["question_id"], config["config_id"])
            retrieval_done = key in done_retrieval
            e2e_done = key in done_e2e
            if index_set_id not in bundles and index_set_id not in bundle_errors and not retrieval_done:
                progress.status(question=label, stage="embedding", config_id=index_set_id)
                progress.status(question=label, stage="retrieval", config_id=index_set_id)
                try:
                    bundles[index_set_id] = _retry(
                        "retrieval",
                        lambda index_set_id=index_set_id: query_fn(question["question"], index_set_id),
                    )
                except Exception as exc:
                    bundle_errors[index_set_id] = exc
                    progress.failures += 1
                    progress.line(
                        f"{label} stage=retrieval config={index_set_id} status=failed "
                        f"error={type(exc).__name__}"
                    )
            if index_set_id in bundle_errors and not retrieval_done:
                failed = {
                    "run_id": run_id,
                    "stage": "retrieval",
                    "question_id": question["question_id"],
                    "config_id": config["config_id"],
                    "chunk_size": config["chunk_size"],
                    "index_set_id": index_set_id,
                    "top_k": config["top_k"],
                    "strategy": config["strategy"],
                    **_failure_fields(bundle_errors[index_set_id]),
                }
                append_jsonl(retrieval_path, failed)
                if "end_to_end" in layers and not e2e_done:
                    append_jsonl(
                        e2e_path,
                        {
                            **failed,
                            "stage": "generation",
                            "answer_correct": None,
                        },
                    )
                question_failed[question["question_id"]] += 1
                progress.completed += 1
                continue
            if retrieval_done and ("end_to_end" not in layers or e2e_done):
                question_success[question["question_id"]] += 1
                continue
            if retrieval_done:
                bundle = bundles.get(index_set_id)
                if bundle is None and index_set_id not in bundle_errors:
                    progress.status(question=label, stage="retrieval", config_id=index_set_id)
                    try:
                        bundles[index_set_id] = _retry(
                            "retrieval",
                            lambda index_set_id=index_set_id: query_fn(question["question"], index_set_id),
                        )
                    except Exception as exc:
                        bundle_errors[index_set_id] = exc
                        progress.failures += 1
                bundle = bundles.get(index_set_id)
                if bundle is None:
                    question_failed[question["question_id"]] += 1
                    progress.completed += 1
                    continue
            else:
                bundle = bundles[index_set_id]
            progress.status(question=label, stage="evaluation", config_id=config["config_id"])
            try:
                ranking_bundle = bundle
                if lexical_version == "portuguese_or" and config["strategy"] in {"lexical", "hybrid"}:
                    ranking_bundle = dict(bundle)
                    ranking_bundle["lexical"] = list(bundle.get("lexical_pt_or") or [])
                    clocks = dict(bundle.get("latency_ms") or {})
                    if "lexical_pt_or_ms" in clocks:
                        clocks["lexical_ms"] = clocks["lexical_pt_or_ms"]
                    ranking_bundle["latency_ms"] = clocks
                hits = derive_hits(
                    ranking_bundle,
                    config["strategy"],
                    config["top_k"],
                    retrieval["rrf_k"],
                )
                uses_embed = config["strategy"] in {"vector", "hybrid"}
                if uses_embed and index_set_id not in measured_embed_for:
                    embed_source = "measured"
                    measured_embed_for.add(index_set_id)
                else:
                    embed_source = "shared_cache"
                latency = attribute_latency(
                    ranking_bundle, config["strategy"], embed_source=embed_source
                )
                if not uses_embed:
                    latency["embed_ms_measured"] = None
                    latency["embed_ms_shared_cache"] = None
                scored = score_hits(question, hits, config["top_k"])
                context = context_from_hits(hits)
            except Exception as exc:
                progress.failures += 1
                append_jsonl(
                    retrieval_path,
                    {
                        "run_id": run_id,
                        "stage": "evaluation",
                        "question_id": question["question_id"],
                        "config_id": config["config_id"],
                        "index_set_id": index_set_id,
                        "top_k": config["top_k"],
                        "strategy": config["strategy"],
                        "chunk_size": config["chunk_size"],
                        **_failure_fields(exc),
                    },
                )
                question_failed[question["question_id"]] += 1
                progress.completed += 1
                continue
            cost_key = (question["question_id"], index_set_id)
            if cost_key not in embed_cost_once:
                embed_cost_once[cost_key] = _embed_cost(
                    question["question"], generate["pricing_usd_per_1m"]
                )
            retrieval_row = {
                "run_id": run_id,
                "stage": "retrieval",
                "status": "success",
                "question_id": question["question_id"],
                "config_id": config["config_id"],
                "chunk_size": config["chunk_size"],
                "index_set_id": index_set_id,
                "top_k": config["top_k"],
                "strategy": config["strategy"],
                "category": question.get("category") or "",
                "metrics_contract": METRICS_CONTRACT,
                "lexical_version": lexical_version,
                "filter_collections": collections,
                "chunk_ids": [hit["id"] for hit in hits],
                "context_tokens": count_tokens(context),
                "recall_at_k": scored["recall_at_k"],
                "reciprocal_full_evidence_rank": scored["reciprocal_full_evidence_rank"],
                "ndcg_at_k": scored["ndcg_at_k"],
                "evidence_coverage": scored["evidence_coverage"],
                "needs_review": scored["needs_review"],
                "evidence_satisfied": scored["evidence_satisfied"],
                "coverage_state": scored["coverage_state"],
                "spans": scored["spans"],
                "measured_execution_latency_ms": latency["measured_execution_latency_ms"],
                "measured_execution_latency_basis": latency["measured_execution_latency_basis"],
                "estimated_standalone_latency_ms": latency["estimated_standalone_latency_ms"],
                "estimated_standalone_latency_basis": latency["estimated_standalone_latency_basis"],
                "standalone_latency_limitation": latency["standalone_latency_limitation"],
                "strategy_total_ms": latency["strategy_total_ms"],
                "strategy_total_basis": latency["strategy_total_basis"],
                "latency_components": latency["components"],
                "embed_ms_measured": latency["embed_ms_measured"],
                "embed_ms_shared_cache": latency["embed_ms_shared_cache"],
            }
            if not retrieval_done and "retrieval" in layers:
                append_jsonl(retrieval_path, retrieval_row)
                done_retrieval.add(key)
            if "end_to_end" not in layers or e2e_done:
                question_success[question["question_id"]] += 1
                if not (retrieval_done and e2e_done):
                    progress.completed += 1
                continue
            if generate_fn is None:
                raise RuntimeError("end_to_end requires a generate function.")
            progress.status(question=label, stage="generation", config_id=config["config_id"])
            try:
                user_prompt = generate["user_template"].format(
                    context=context, question=question["question"]
                )
                generated = _retry(
                    "generation",
                    lambda user_prompt=user_prompt: generate_fn(generate["system_prompt"], user_prompt),
                )
                progress.status(question=label, stage="evaluation", config_id=config["config_id"])

                def _judge(q: str, gold: str, answer: str, key_concepts: list[str] | None = None) -> dict:
                    if key_concepts:
                        return _retry("judge", lambda: judge_fn(q, gold, answer, key_concepts))
                    return _retry("judge", lambda: judge_fn(q, gold, answer))

                graded = grade_answer(
                    question,
                    generated["model_answer"],
                    judge_fn=_judge if judge_fn is not None else None,
                )
            except Exception as exc:
                progress.failures += 1
                append_jsonl(
                    e2e_path,
                    {
                        "run_id": run_id,
                        "stage": "generation",
                        "status": "failed",
                        "question_id": question["question_id"],
                        "config_id": config["config_id"],
                        "index_set_id": index_set_id,
                        "top_k": config["top_k"],
                        "strategy": config["strategy"],
                        "chunk_size": config["chunk_size"],
                        "answer_correct": None,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:300],
                    },
                )
                question_failed[question["question_id"]] += 1
                progress.completed += 1
                continue
            costs = evaluation_costs(
                graded["grading_mode"],
                graded,
                generated.get("prompt_tokens_measured"),
                generated.get("completion_tokens_measured"),
                generate["pricing_usd_per_1m"],
            )
            generation_latency = generated.get("llm_latency_ms")
            judge_latency = graded.get("judge_latency_ms")
            measured_e2e = _add_latency(latency["measured_execution_latency_ms"], generation_latency)
            standalone_e2e = _add_latency(latency["estimated_standalone_latency_ms"], generation_latency)
            answer_correct = graded["answer_correct"]
            if graded.get("unsupported_rubric") or graded.get("judge_pending"):
                answer_support = "not_evaluated"
            elif answer_correct is None:
                answer_support = "inconclusive"
            else:
                answer_support = support_label(answer_correct, scored["evidence_satisfied"])
            e2e_row = {
                "run_id": run_id,
                "stage": "generation",
                "status": "success",
                "question_id": question["question_id"],
                "config_id": config["config_id"],
                "chunk_size": config["chunk_size"],
                "index_set_id": index_set_id,
                "top_k": config["top_k"],
                "strategy": config["strategy"],
                "category": question.get("category") or "",
                "metrics_contract": METRICS_CONTRACT,
                "lexical_version": lexical_version,
                "evaluation_status": graded.get("evaluation_status")
                or ("inconclusive" if answer_correct is None else "valid"),
                "unsupported_rubric": bool(graded.get("unsupported_rubric")),
                "measured_execution_latency_ms": retrieval_row["measured_execution_latency_ms"],
                "estimated_standalone_latency_ms": retrieval_row["estimated_standalone_latency_ms"],
                "strategy_total_ms": retrieval_row["strategy_total_ms"],
                "strategy_total_basis": retrieval_row["strategy_total_basis"],
                "embed_ms_measured": retrieval_row["embed_ms_measured"],
                "embed_ms_shared_cache": retrieval_row["embed_ms_shared_cache"],
                "latency_components": retrieval_row["latency_components"],
                "evidence_satisfied": retrieval_row["evidence_satisfied"],
                "needs_review": retrieval_row["needs_review"],
                "generate_model": generate["model"],
                "prompt_hash": prompt_hash,
                "model_answer": generated["model_answer"],
                "grading_mode": graded["grading_mode"],
                "answer_correct": answer_correct,
                "uncertain": graded["uncertain"],
                "empty_or_refusal": graded["empty_or_refusal"],
                "judge_pending": graded.get("judge_pending", False),
                "answer_support": answer_support,
                "prompt_tokens_measured": generated.get("prompt_tokens_measured"),
                "completion_tokens_measured": generated.get("completion_tokens_measured"),
                "llm_latency_ms": generation_latency,
                "llm_latency_source": generated.get("llm_latency_source", "measured"),
                "judge_latency_ms": judge_latency,
                "judge_latency_source": graded.get("judge_latency_source"),
                "judge_latency_role": "evaluation_only_excluded_from_e2e",
                "measured_e2e_execution_latency_ms": measured_e2e,
                "measured_e2e_execution_latency_basis": "measured_retrieval_plus_measured_generation",
                "estimated_standalone_e2e_latency_ms": standalone_e2e,
                "estimated_standalone_e2e_latency_basis": "estimated_standalone_retrieval_plus_measured_generation",
                "standalone_latency_limitation": latency["standalone_latency_limitation"],
                "e2e_latency_ms": measured_e2e,
                "e2e_latency_basis": "alias_of_measured_e2e_execution_latency_ms",
                **costs,
                "embedding_cost_usd": embed_cost_once[cost_key],
                "embedding_cost_basis": "estimated_token_count_times_versioned_price_shared_per_question_index_set",
                "embedding_cost_scope": "shared_once_per_question_and_index_set",
                "api_cost_usd_embed_shared_estimated": embed_cost_once[cost_key],
                "embed_cost_basis": "shared_estimated",
                "pricing_version": PRICING_VERSION,
            }
            append_jsonl(e2e_path, e2e_row)
            done_e2e.add(key)
            question_success[question["question_id"]] += 1
            progress.completed += 1
        progress.question_summary(
            label,
            question_success[question["question_id"]],
            question_failed[question["question_id"]],
        )

    saved_e2e = [
        row
        for row in load_jsonl(e2e_path)
        if row.get("status", "success") == "success"
    ]
    embed_keys: dict[tuple[str, str], float] = {}
    for row in saved_e2e:
        value = row.get("embedding_cost_usd")
        if value is None:
            continue
        embed_keys[(row["question_id"], row["index_set_id"])] = float(value)
    embed_keys.update(embed_cost_once)
    embed_total = round(sum(embed_keys.values()), 8)
    generation_values = [row.get("generation_cost_usd") for row in saved_e2e]
    judge_values = [row.get("judge_cost_usd") for row in saved_e2e]
    generation_total = _sum_known(generation_values)
    judge_total = _sum_known(judge_values)
    operational = _sum_known([generation_total, embed_total])
    evaluation_total = _sum_known([operational, judge_total])
    meta = {
        "run_id": run_id,
        "metrics_contract": METRICS_CONTRACT,
        "pricing_version": PRICING_VERSION,
        "pricing_usd_per_1m": generate["pricing_usd_per_1m"],
        "n_questions": len(questions),
        "question_ids": [item["question_id"] for item in questions],
        "effective_collections": collections,
        "baseline_config_id": manifest["baseline_config_id"],
        "prompt_hash": prompt_hash,
        "verdict_policy": "inconclusive",
        "embed_cost_usd_estimated_by_question_index_set": {
            f"{question_id}:{index_set_id}": cost
            for (question_id, index_set_id), cost in sorted(embed_keys.items())
        },
        "embedding_cost_usd": embed_total,
        "embed_cost_usd_estimated_total": embed_total,
        "embed_cost_basis": "estimated_from_token_count_counted_once_per_question_and_index_set",
        "generation_cost_usd": generation_total,
        "judge_cost_usd": judge_total,
        "rag_operational_cost_usd": operational,
        "rag_operational_cost_basis": "estimated_generation_plus_shared_embedding_excludes_judge",
        "evaluation_total_cost_usd": evaluation_total,
        "evaluation_total_cost_basis": "rag_operational_plus_judge",
        "chat_cost_basis": "estimated_unit_price_times_measured_tokens",
        "latency_note": (
            "measured_execution_latency_ms excludes a cached embedding. "
            "estimated_standalone_latency_ms puts it back for vector and hybrid "
            "and is not an isolated repeated benchmark. Judge latency is evaluation-only."
        ),
        "q005_policy": "table matches stay needs_review and are not confirmed hits",
        "lexical_version": lexical_version,
        "strategies": sorted(selected_strategies),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    progress.status(question="run", stage="export")
    write_reports(out_dir, run_id=run_id, baseline_config_id=manifest["baseline_config_id"])
    success_count = sum(question_success.values())
    failed_count = sum(question_failed.values())
    progress.final_summary(success_count, failed_count)
    return {"success": success_count, "failed": failed_count, "retries": progress.retries}


def _build_query_fn(
    store,
    embedder,
    collections: list[str],
    candidate_k: int,
    rrf_k: int,
    lexical_version: str,
    strategies: set[str],
):
    lexical_only = strategies == {"lexical"} and lexical_version == "portuguese_or"

    def query(question: str, index_set_id: str) -> dict:
        if lexical_only:
            return query_lexical_pt_or(
                question,
                store,
                filter_collections=collections,
                candidate_k=candidate_k,
                filter_index_set_id=index_set_id,
            )
        return query_retrieval(
            question,
            store,
            embedder,
            filter_collections=collections,
            top_k=10,
            rrf_k=rrf_k,
            candidate_k=candidate_k,
            filter_index_set_id=index_set_id,
            lexical_version=lexical_version,
        )
    return query


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v0.3 retrieval and end-to-end suite.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--question-ids", default="")
    parser.add_argument("--layers", default="retrieval")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--read-collections", default="itau_2025")
    parser.add_argument("--strategies", default="")
    parser.add_argument("--lexical-version", default="")
    parser.add_argument("--config-ids", default="")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.lexical_version:
        if args.lexical_version not in {"english_and", "portuguese_or"}:
            parser.error("lexical-version must be english_and or portuguese_or.")
        manifest["retrieval"]["lexical_version"] = args.lexical_version
    if args.strategies:
        selected = [item.strip() for item in args.strategies.split(",") if item.strip()]
        unknown = set(selected) - {"lexical", "vector", "hybrid"}
        if unknown or not selected:
            parser.error("strategies must be a comma-separated subset of lexical,vector,hybrid.")
        manifest["strategies"] = selected
    if args.config_ids:
        selected_ids = [item.strip() for item in args.config_ids.split(",") if item.strip()]
        known = {item["config_id"] for item in manifest["configs"]}
        unknown = [item for item in selected_ids if item not in known]
        if unknown or not selected_ids:
            parser.error("config-ids must list config ids from the 27 configurations.")
        manifest["config_ids"] = selected_ids
    question_ids = [item for item in args.question_ids.split(",") if item] or None
    questions = load_suite(args.ground_truth, question_ids)
    layers = {item.strip() for item in args.layers.split(",") if item.strip()}
    out_dir = args.out_dir or Path("runs") / args.run_id
    read_collections = [item for item in args.read_collections.split(",") if item]

    from openai import OpenAI

    from retrieval_lab.config import load_settings
    from retrieval_lab.db import SupabaseDocumentStore, create_supabase
    from retrieval_lab.embeddings import OpenAIEmbedder

    settings = load_settings()
    store = SupabaseDocumentStore(create_supabase(settings))
    embedder = OpenAIEmbedder(
        OpenAI(api_key=settings.openai_api_key, timeout=CALL_TIMEOUT_SECONDS),
        settings.embedding_model,
    )
    client = OpenAI(api_key=settings.openai_api_key, timeout=CALL_TIMEOUT_SECONDS)
    generate_cfg = manifest["generate"]

    def generate_fn(system_prompt: str, user_prompt: str) -> dict:
        return generate_answer(
            client,
            model=generate_cfg["model"],
            temperature=generate_cfg["temperature"],
            max_output_tokens=generate_cfg["max_output_tokens"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def judge_fn(question: str, gold: str, answer: str, key_concepts: list[str] | None = None) -> dict:
        return judge_with_client(
            client,
            generate_cfg["judge_model"],
            question,
            gold,
            answer,
            key_concepts=key_concepts,
        )

    run_experiment(
        run_id=args.run_id,
        manifest=manifest,
        questions=questions,
        out_dir=out_dir,
        query_fn=_build_query_fn(
            store,
            embedder,
            effective_collections(manifest["collections"], read_collections),
            manifest["retrieval"]["candidate_k"],
            manifest["retrieval"]["rrf_k"],
            manifest["retrieval"].get("lexical_version", "english_and"),
            set(manifest.get("strategies") or ("lexical", "vector", "hybrid")),
        ),
        layers=layers,
        read_collections=read_collections,
        generate_fn=generate_fn if "end_to_end" in layers else None,
        judge_fn=judge_fn if "end_to_end" in layers else None,
    )


if __name__ == "__main__":
    main()
