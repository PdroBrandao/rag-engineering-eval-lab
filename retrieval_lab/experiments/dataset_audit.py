"""Audit the 50-question set against the runner and the extracted PDF.

The audit reports incompatibilities. It does not rewrite questions, evidence,
answers or rubrics.
"""

import json
from pathlib import Path

import tiktoken

from retrieval_lab.chunking import chunk_pages
from retrieval_lab.experiments.dataset_split import difficulty_of
from retrieval_lab.experiments.evidence_match import equivalent_targets
from retrieval_lab.experiments.textutil import normalize_text, span_fragments

INDEX_SETS = ((256, 26, "cs256"), (512, 51, "cs512"), (1024, 102, "cs1024"))
ENCODER = tiktoken.get_encoding("cl100k_base")


def audit_question(row: dict, pages: dict[int, str]) -> dict:
    page_number = int(row["pdf_page"])
    page_text = pages.get(page_number, "")
    normalized_page = normalize_text(page_text)
    quote_contiguous = normalize_text(row.get("evidence_quote")) in normalized_page
    issues = []
    rubric_risks = []
    fragments = []
    grading = row.get("grading") or {}
    mode = grading.get("mode")
    if mode == "semantic_review":
        rubric_risks.append("judge_uncertain_stays_inconclusive")
    if mode == "contains_all":
        for phrase in grading.get("required_phrases") or []:
            if len(normalize_text(phrase)) <= 4:
                rubric_risks.append(f"short_phrase:{phrase}")
    if row.get("table_ambiguous"):
        issues.append("table_ambiguous_stays_unconfirmed")
    for span in row.get("minimal_support_spans") or []:
        listed_pages = span.get("pdf_pages") if isinstance(span.get("pdf_pages"), list) else None
        allowed_pages = [int(item) for item in listed_pages] if listed_pages else [int(span["pdf_page"])]
        equivalents = span.get("equivalent_texts") if isinstance(span.get("equivalent_texts"), list) else None
        if equivalents:
            specs = equivalent_targets(span)
        else:
            specs = [(fragment, allowed_pages) for fragment in span_fragments(span)]
        matched_any = False
        for fragment, target_pages in specs:
            needle = normalize_text(fragment)
            in_page = any(
                needle and needle in normalize_text(pages.get(number, ""))
                for number in target_pages
            )
            other_pages = [
                number
                for number, text in pages.items()
                if number not in target_pages and needle and needle in normalize_text(text)
            ]
            chunk_hits = {}
            for size, overlap, name in INDEX_SETS:
                found = False
                for number in target_pages:
                    chunks = chunk_pages([(number, pages.get(number, ""))], size, overlap)
                    if any(needle and needle in normalize_text(chunk.content) for chunk in chunks):
                        found = True
                        break
                chunk_hits[name] = found
            tokens = len(ENCODER.encode(fragment))
            matched_any = matched_any or in_page
            if not equivalents and not in_page:
                issues.append(f"{span.get('span_id')}_absent_from_page_extraction")
            missing = [name for name, found in chunk_hits.items() if in_page and not found]
            if not equivalents and missing:
                issues.append(f"{span.get('span_id')}_not_inside_one_chunk:" + ",".join(missing))
            if len(normalize_text(fragment)) <= 12:
                rubric_risks.append(f"short_span:{fragment}")
            fragments.append(
                {
                    "span_id": span.get("span_id"),
                    "tokens": tokens,
                    "in_declared_page": in_page,
                    "other_pages": other_pages[:8],
                    "chunks": chunk_hits,
                }
            )
        if equivalents and not matched_any:
            issues.append(f"{span.get('span_id')}_absent_from_page_extraction")
        if equivalents and matched_any:
            missing_sets = []
            for _size, _overlap, name in INDEX_SETS:
                if not any(item["chunks"][name] for item in fragments if item["span_id"] == span.get("span_id")):
                    missing_sets.append(name)
            if missing_sets:
                issues.append(f"{span.get('span_id')}_not_inside_one_chunk:" + ",".join(missing_sets))
    if not quote_contiguous and not any(item.endswith("absent_from_page_extraction") for item in issues):
        issues.append("evidence_quote_is_not_contiguous_in_extraction")
    if any(item.get("other_pages") for item in fragments):
        issues.append("span_also_appears_on_other_pages")
    return {
        "question_id": row["question_id"],
        "category": row.get("category"),
        "difficulty": difficulty_of(row),
        "grading_mode": mode,
        "table_ambiguous": bool(row.get("table_ambiguous")),
        "review_notes": row.get("review_notes") or "",
        "quote_contiguous": quote_contiguous,
        "issues": sorted(set(issues)),
        "rubric_risks": rubric_risks,
        "fragments": fragments,
    }


def write_audit(dataset: Path, pdf: Path, destination: Path) -> list[dict]:
    from retrieval_lab.experiments.ground_truth import load_suite
    from retrieval_lab.pdf_extract import extract_pages

    rows = load_suite(dataset)
    pages = dict(extract_pages(pdf.read_bytes()))
    report = [audit_question(row, pages) for row in rows]
    destination.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in report),
        encoding="utf-8",
    )
    return report
