"""Retrieval metrics for contract v0.4.

Grades assigned by `_chunk_grade` are unchanged from v0.3:

- 3: the chunk alone confirms every required span.
- 2: the chunk fully matches at least one required span, but not every span,
  and the question is not table-ambiguous.
- 1: table overlap that stays in needs_review, or only a partial fragment match.
- 0: no textual overlap.

`satisfied_without_chunk` does not change the grade. A chunk is graded from
its own text. nDCG uses these grades and normalizes by their ideal order.
It does not compare the list with an unseen corpus ideal.
"""

import math

from retrieval_lab.experiments.evidence_match import match_spans

METRICS_CONTRACT = "v0.4"
PRICING_VERSION = "experiment_manifest.v0.3#generate.pricing_usd_per_1m"


def _dcg(grades: list[float]) -> float:
    total = 0.0
    for index, grade in enumerate(grades, start=1):
        total += grade / math.log2(index + 1)
    return total


def _chunk_grade(question: dict, hit: dict, satisfied_without_chunk: bool) -> float:
    alone = match_spans(question, [hit])
    if alone["evidence_satisfied"]:
        return 3.0
    if alone["needs_review"]:
        return 1.0
    overlaps = [item["text_overlap"] for item in alone["spans"]]
    if "text_match" in overlaps and not alone["evidence_satisfied"]:
        return 2.0 if not question.get("table_ambiguous") else 1.0
    if "partial_coverage" in overlaps:
        return 1.0
    if satisfied_without_chunk:
        return 0.0
    return 0.0


def score_hits(question: dict, hits: list[dict], k: int) -> dict:
    window = hits[:k]
    full = match_spans(question, window)
    required = full["required_span_count"]
    recall = (full["confirmed_span_count"] / required) if required else 0.0
    coverage = 1.0 if full["evidence_satisfied"] else 0.0

    first_rank = None
    for rank in range(1, len(window) + 1):
        if match_spans(question, window[:rank])["evidence_satisfied"]:
            first_rank = rank
            break
    reciprocal = 0.0 if first_rank is None else 1.0 / first_rank

    grades = []
    for index, hit in enumerate(window):
        others = window[:index] + window[index + 1 :]
        grades.append(_chunk_grade(question, hit, match_spans(question, others)["evidence_satisfied"]))
    ideal = sorted(grades, reverse=True)
    idcg = _dcg(ideal)
    ndcg = 0.0 if idcg == 0 else _dcg(grades) / idcg

    return {
        "metrics_contract": METRICS_CONTRACT,
        "recall_at_k": recall,
        "reciprocal_full_evidence_rank": reciprocal,
        "ndcg_at_k": ndcg,
        "evidence_coverage": coverage,
        "needs_review": full["needs_review"],
        "evidence_satisfied": full["evidence_satisfied"],
        "coverage_state": full["coverage_state"],
        "spans": full["spans"],
        "relevance_grades": grades,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * fraction
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)
