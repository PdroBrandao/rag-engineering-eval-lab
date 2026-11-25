import json
from pathlib import Path

GRADING_MODES = {"contains_all", "numeric", "judge", "semantic_review"}
EXPECTED_DATASET_IDS = tuple(f"Q{index:03d}" for index in list(range(1, 24)) + list(range(26, 53)))


def validate_question(row: dict, *, line_number: int) -> None:
    question_id = row.get("question_id")
    if not isinstance(question_id, str) or not question_id.startswith("Q"):
        raise ValueError(f"Line {line_number} has an invalid question_id.")
    if not row.get("question") or not row.get("reference_answer"):
        raise ValueError(f"{question_id} is missing the question or the reference answer.")
    if not row.get("evidence_quote"):
        raise ValueError(f"{question_id} is missing evidence_quote.")
    spans = row.get("minimal_support_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{question_id} needs at least one minimal_support_span.")
    if not any(span.get("required", True) for span in spans):
        raise ValueError(f"{question_id} needs a required span.")
    for span in spans:
        if "pdf_page" not in span:
            raise ValueError(f"{question_id} has a span without pdf_page.")
        has_text = isinstance(span.get("text"), str) and span.get("text", "").strip()
        has_lines = isinstance(span.get("lines"), list) and any(str(line).strip() for line in span["lines"])
        if not has_text and not has_lines:
            raise ValueError(f"{question_id} has a span without text.")
    grading = row.get("grading") or {}
    mode = grading.get("mode")
    if mode not in GRADING_MODES:
        raise ValueError(f"{question_id} has an unsupported grading mode.")
    if mode == "contains_all" and not grading.get("required_phrases"):
        raise ValueError(f"{question_id} contains_all grading needs required_phrases.")
    if mode == "numeric" and "value" not in grading:
        raise ValueError(f"{question_id} numeric grading needs a value.")
    if mode == "semantic_review" and not grading.get("key_concepts"):
        raise ValueError(f"{question_id} semantic_review grading needs key_concepts.")


def load_suite(path: Path, question_ids: list[str] | None = None) -> list[dict]:
    rows = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        validate_question(row, line_number=line_number)
        question_id = row["question_id"]
        if question_id in seen:
            raise ValueError(f"Duplicate question_id {question_id}.")
        seen.add(question_id)
        rows.append(row)
    if question_ids is None:
        return rows
    by_id = {row["question_id"]: row for row in rows}
    missing = [item for item in question_ids if item not in by_id]
    if missing:
        raise ValueError("Missing questions: " + ", ".join(missing))
    return [by_id[item] for item in question_ids]
