import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from retrieval_lab.metrics import score_ranking

QueryFn = Callable[[str, int], dict[str, Any]]
STRATEGIES = ("vector", "lexical", "hybrid")


def load_dataset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Dataset must be a JSON object.")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset cases must be a non-empty list.")
    k = data.get("k", 5)
    rrf_k = data.get("rrf_k", 60)
    if not isinstance(k, int) or k < 1:
        raise ValueError("Dataset k must be a positive integer.")
    if not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("Dataset rrf_k must be a positive integer.")
    normalized = []
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be an object.")
        question = case.get("question")
        relevant = case.get("relevant_chunk_ids")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Case {index} needs a question.")
        if not isinstance(relevant, list) or not all(isinstance(item, int) for item in relevant):
            raise ValueError(f"Case {index} relevant_chunk_ids must be a list of integers.")
        normalized.append(
            {"question": question.strip(), "relevant_chunk_ids": relevant}
        )
    return {"k": k, "rrf_k": rrf_k, "cases": normalized}


def evaluate_cases(cases: list[dict[str, Any]], k: int, query_fn: QueryFn) -> dict[str, Any]:
    rows = []
    for case in cases:
        result = query_fn(case["question"], k)
        strategies = {}
        for name in STRATEGIES:
            retrieved = [hit["id"] for hit in result[name]]
            strategies[name] = score_ranking(retrieved, case["relevant_chunk_ids"], k)
        rows.append(
            {
                "question": case["question"],
                "relevant_chunk_ids": case["relevant_chunk_ids"],
                "strategies": strategies,
            }
        )
    mean: dict[str, dict[str, float]] = {}
    for name in STRATEGIES:
        mean[name] = {
            metric: sum(row["strategies"][name][metric] for row in rows) / len(rows)
            for metric in ("recall_at_k", "precision_at_k", "mrr")
        }
    return {"k": k, "cases": rows, "mean": mean}


def format_report(report: dict[str, Any]) -> str:
    lines = [f"k={report['k']} cases={len(report['cases'])}"]
    for index, case in enumerate(report["cases"], start=1):
        lines.append(f"[{index}] {case['question']}")
        if not case["relevant_chunk_ids"]:
            lines.append("  no relevant chunk ids; metrics for this case are zero")
        for name in STRATEGIES:
            scores = case["strategies"][name]
            lines.append(
                "  {name:<8} recall_at_k={recall_at_k:.3f} "
                "precision_at_k={precision_at_k:.3f} mrr={mrr:.3f}".format(name=name, **scores)
            )
    lines.append("mean")
    for name in STRATEGIES:
        scores = report["mean"][name]
        lines.append(
            "  {name:<8} recall_at_k={recall_at_k:.3f} "
            "precision_at_k={precision_at_k:.3f} mrr={mrr:.3f}".format(name=name, **scores)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare vector, lexical, and hybrid retrieval.")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)

    from openai import OpenAI

    from retrieval_lab.config import load_settings
    from retrieval_lab.db import SupabaseDocumentStore, create_supabase
    from retrieval_lab.embeddings import OpenAIEmbedder
    from retrieval_lab.retrieval import query_retrieval

    dataset = load_dataset(args.dataset)
    settings = load_settings()
    store = SupabaseDocumentStore(create_supabase(settings))
    embedder = OpenAIEmbedder(OpenAI(api_key=settings.openai_api_key), settings.embedding_model)

    eval_collections = [
        item.strip()
        for item in os.environ.get("EVAL_COLLECTIONS", "legacy").split(",")
        if item.strip()
    ]

    def query_fn(question: str, k: int) -> dict[str, Any]:
        return query_retrieval(
            question,
            store,
            embedder,
            filter_collections=eval_collections,
            top_k=k,
            rrf_k=dataset["rrf_k"],
            candidate_k=max(k, 20),
        )

    report = evaluate_cases(dataset["cases"], dataset["k"], query_fn)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
