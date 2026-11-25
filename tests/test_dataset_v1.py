import json
from pathlib import Path

import pytest

from retrieval_lab.chunking import chunk_pages
from retrieval_lab.experiments.dataset_audit import audit_question
from retrieval_lab.experiments.textutil import normalize_text
from retrieval_lab.pdf_extract import extract_pages
from retrieval_lab.experiments.evidence_match import match_spans
from retrieval_lab.experiments.dataset_split import LOCKED_DEVELOPMENT, SEED, sha256_file, split_dataset
from retrieval_lab.experiments.grading import grade_answer
from retrieval_lab.experiments.ground_truth import EXPECTED_DATASET_IDS, load_suite
from retrieval_lab.experiments.manifest import load_manifest
from retrieval_lab.experiments.runner import run_experiment
from retrieval_lab.experiments.uncertainty import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval/ground_truth/itau_50_v1.0.jsonl"
SOURCE_V11 = ROOT / "eval/ground_truth/itau_50_v1.1.jsonl"
SPLIT_V11 = ROOT / "eval/ground_truth/dataset_split_v1.1.json"
DEVELOPMENT_V11 = ROOT / "eval/ground_truth/development_v1.1.jsonl"
FROZEN_V11 = ROOT / "eval/ground_truth/frozen_benchmark_v1.1.jsonl"
AUDIT_V11 = ROOT / "eval/ground_truth/validation_report_v1.1.jsonl"
SOURCE_V12 = ROOT / "eval/ground_truth/itau_50_v1.2.jsonl"
SPLIT_V12 = ROOT / "eval/ground_truth/dataset_split_v1.2.json"
DEVELOPMENT_V12 = ROOT / "eval/ground_truth/development_v1.2.jsonl"
FROZEN_V12 = ROOT / "eval/ground_truth/frozen_benchmark_v1.2.jsonl"
SOURCE_V13 = ROOT / "eval/ground_truth/itau_50_v1.3.jsonl"
SPLIT_V13 = ROOT / "eval/ground_truth/dataset_split_v1.3.json"
DEVELOPMENT_V13 = ROOT / "eval/ground_truth/development_v1.3.jsonl"
FROZEN_V13 = ROOT / "eval/ground_truth/frozen_benchmark_v1.3.jsonl"
SPLIT = ROOT / "eval/ground_truth/dataset_split_v1.json"
DEVELOPMENT = ROOT / "eval/ground_truth/development_v1.jsonl"
FROZEN = ROOT / "eval/ground_truth/frozen_benchmark_v1.jsonl"
AUDIT = ROOT / "eval/ground_truth/validation_report_v1.jsonl"


def test_split_is_deterministic_and_locks_the_first_ten():
    rows = load_suite(SOURCE)
    assert [row["question_id"] for row in rows] == list(EXPECTED_DATASET_IDS)
    first = split_dataset(rows, seed=SEED)
    second = split_dataset(rows, seed=SEED)
    assert first["development_ids"] == second["development_ids"]
    assert first["frozen_ids"] == second["frozen_ids"]
    assert set(LOCKED_DEVELOPMENT).issubset(first["development_ids"])
    assert set(first["development_ids"]).isdisjoint(first["frozen_ids"])
    assert len(first["development_ids"]) == 30
    assert len(first["frozen_ids"]) == 20
    manifest = json.loads(SPLIT.read_text(encoding="utf-8"))
    assert manifest["seed"] == SEED
    assert manifest["development"]["ids"] == first["development_ids"]
    assert manifest["frozen_benchmark"]["ids"] == first["frozen_ids"]
    assert manifest["frozen_benchmark"]["immutable"] is True
    assert sha256_file(DEVELOPMENT) == manifest["development"]["sha256"]
    assert sha256_file(FROZEN) == manifest["frozen_benchmark"]["sha256"]
    assert sha256_file(SOURCE) == manifest["source_sha256"]


def test_semantic_review_uses_the_judge_and_keeps_uncertain_inconclusive():
    pending = grade_answer(
        {"grading": {"mode": "semantic_review", "key_concepts": ["2023"]}, "question": "q", "reference_answer": "a"},
        "O estudo foi em 2023.",
    )
    assert pending["answer_correct"] is None
    assert pending["evaluation_status"] == "not_evaluated"
    seen = {}

    def judge(question: str, gold: str, answer: str, key_concepts: list[str] | None = None) -> dict:
        seen["key_concepts"] = key_concepts
        return {"verdict": "uncertain"}

    graded = grade_answer(
        {
            "grading": {"mode": "semantic_review", "key_concepts": ["2023", "segundo semestre de 2026"]},
            "question": "q",
            "reference_answer": "a",
        },
        "O estudo foi em 2023.",
        judge_fn=judge,
    )
    assert seen["key_concepts"] == ["2023", "segundo semestre de 2026"]
    assert graded["answer_correct"] is None
    assert graded["evaluation_status"] == "inconclusive"
    assert graded["uncertain"] is True

    def accept(_question: str, _gold: str, _answer: str, key_concepts: list[str] | None = None) -> dict:
        assert key_concepts == ["2023"]
        return {"verdict": "correct"}

    accepted = grade_answer(
        {"grading": {"mode": "semantic_review", "key_concepts": ["2023"]}, "question": "q", "reference_answer": "a"},
        "O estudo foi em 2023.",
        judge_fn=accept,
    )
    assert accepted["answer_correct"] == 1
    assert accepted["evaluation_status"] == "valid"
    report = [json.loads(line) for line in AUDIT.read_text(encoding="utf-8").splitlines()]
    by_id = {row["question_id"]: row for row in report}
    assert len(by_id) == 50
    assert any("absent_from_page_extraction" in issue for issue in by_id["Q030"]["issues"])
    assert "table_ambiguous_stays_unconfirmed" in by_id["Q005"]["issues"]
    assert "semantic_review_not_scored" in by_id["Q048"]["issues"]
    missing = audit_question(
        {
            "question_id": "Q030",
            "pdf_page": 1,
            "category": "factual",
            "table_ambiguous": False,
            "grading": {"mode": "judge"},
            "evidence_quote": "frase ausente",
            "minimal_support_spans": [{"span_id": "S2", "required": True, "pdf_page": 1, "text": "frase ausente"}],
        },
        {1: "outro texto"},
    )
    assert "S2_absent_from_page_extraction" in missing["issues"]


def test_v1_1_corrections_keep_v1_and_the_same_split():
    previous = json.loads(SPLIT.read_text(encoding="utf-8"))
    current = json.loads(SPLIT_V11.read_text(encoding="utf-8"))
    assert sha256_file(SOURCE) == previous["source_sha256"]
    assert sha256_file(DEVELOPMENT) == previous["development"]["sha256"]
    assert sha256_file(FROZEN) == previous["frozen_benchmark"]["sha256"]
    assert current["version"] == "v1.1"
    assert current["supersedes"] == "v1"
    assert current["development"]["ids"] == previous["development"]["ids"]
    assert current["frozen_benchmark"]["ids"] == previous["frozen_benchmark"]["ids"]
    assert sha256_file(SOURCE_V11) == current["source_sha256"]
    assert sha256_file(DEVELOPMENT_V11) == current["development"]["sha256"]
    assert sha256_file(FROZEN_V11) == current["frozen_benchmark"]["sha256"]
    changed = set(current["changed_question_ids"])
    assert changed == {"Q030", "Q037", "Q040"}
    old_lines = {
        json.loads(line)["question_id"]: line
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    new_lines = {
        json.loads(line)["question_id"]: line
        for line in SOURCE_V11.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert set(old_lines) == set(new_lines)
    assert {qid for qid, line in new_lines.items() if line != old_lines[qid]} == changed
    rows = {row["question_id"]: row for row in load_suite(SOURCE_V11)}
    assert rows["Q030"]["minimal_support_spans"][1]["text"].startswith("compartilhada entre os diferentes")
    assert rows["Q037"]["grading"]["required_phrases"] == ["50% dos clientes", "50% da exposição"]
    assert grade_answer(rows["Q037"], "cerca de 50%")["answer_correct"] == 0
    assert grade_answer(rows["Q037"], "50% dos clientes e 50% da exposição")["answer_correct"] == 1
    span = rows["Q040"]["minimal_support_spans"][0]
    pages = span["pdf_pages"]
    assert pages == [86, 89, 90, 93, 95]
    matched = match_spans(
        rows["Q040"],
        [{
            "id": 1,
            "content": " ".join(span["lines"]),
            "page": 89,
            "metadata": {},
            "document_name": "report.pdf",
        }],
    )
    assert matched["evidence_satisfied"] is True
    rejected = match_spans(
        rows["Q040"],
        [{"id": 1, "content": " ".join(span["lines"]), "page": 3, "metadata": {}, "document_name": "report.pdf"}],
    )
    assert rejected["evidence_satisfied"] is False
    report = {row["question_id"]: row for row in (json.loads(line) for line in AUDIT_V11.read_text().splitlines())}
    assert not any("absent_from_page_extraction" in issue for issue in report["Q030"]["issues"])
    assert "short_phrase:50%" not in report["Q037"]["rubric_risks"]
    assert "semantic_review_not_scored" not in report["Q048"]["issues"]


def _hit(content: str, page: int) -> dict:
    return {"id": 1, "content": content, "page": page, "metadata": {}, "document_name": "report.pdf"}


def test_q041_accepts_each_wording_only_on_its_page():
    previous = json.loads(SPLIT_V11.read_text(encoding="utf-8"))
    current = json.loads(SPLIT_V12.read_text(encoding="utf-8"))
    assert sha256_file(SOURCE_V11) == previous["source_sha256"]
    assert sha256_file(DEVELOPMENT_V11) == previous["development"]["sha256"]
    assert sha256_file(FROZEN_V11) == previous["frozen_benchmark"]["sha256"]
    assert current["version"] == "v1.2"
    assert current["changed_question_ids"] == ["Q041"]
    assert current["development"]["ids"] == previous["development"]["ids"]
    assert current["frozen_benchmark"]["ids"] == previous["frozen_benchmark"]["ids"]
    assert sha256_file(SOURCE_V12) == current["source_sha256"]
    assert sha256_file(FROZEN_V12) == sha256_file(FROZEN_V11)
    old_lines = {
        json.loads(line)["question_id"]: line
        for line in SOURCE_V11.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    new_lines = {
        json.loads(line)["question_id"]: line
        for line in SOURCE_V12.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert {qid for qid, line in new_lines.items() if line != old_lines[qid]} == {"Q041"}
    question = next(row for row in load_suite(SOURCE_V12) if row["question_id"] == "Q041")
    span = question["minimal_support_spans"][0]
    page5 = span["text"]
    page20 = next(item["text"] for item in span["equivalent_texts"] if item["pdf_page"] == 20)
    assert span["pdf_page"] == 5
    assert question["grading"]["mode"] == "semantic_review"
    assert question["grading"]["key_concepts"] == ["33,7", "72,0%"]
    assert match_spans(question, [_hit(page5, 5)])["evidence_satisfied"] is True
    assert match_spans(question, [_hit(page20, 20)])["evidence_satisfied"] is True
    assert match_spans(question, [_hit(page5, 20)])["evidence_satisfied"] is False
    assert match_spans(question, [_hit(page20, 5)])["evidence_satisfied"] is False
    assert match_spans(question, [_hit(page5, 3)])["evidence_satisfied"] is False
    assert match_spans(question, [_hit(page20, 3)])["evidence_satisfied"] is False
    pending = grade_answer(question, "R$ 33,7 bilhões e payout bruto de 72,0%.")
    assert pending["answer_correct"] is None
    assert pending["evaluation_status"] == "not_evaluated"
    q005 = next(row for row in load_suite(SOURCE_V12) if row["question_id"] == "Q005")
    table = match_spans(
        q005,
        [_hit(
            "Informações de resultado (R$ bilhões em IFRS) 2025 2024 variação (%) YoY (25–24) "
            "Receitas de prestação de serviços e resultados de contratos de seguros e previdência privada³ "
            "55,7 54,1 3,1 %",
            11,
        )],
    )
    assert table["needs_review"] is True
    assert table["evidence_satisfied"] is False
    loose = {
        "table_ambiguous": False,
        "minimal_support_spans": [{
            "span_id": "S1",
            "required": True,
            "pdf_page": 5,
            "pdf_pages": [5, 20],
            "text": page5,
            "equivalent_texts": [page5],
        }],
    }
    assert match_spans(loose, [_hit(page5, 20)])["evidence_satisfied"] is True


def test_q041_v1_3_accepts_the_chunk_that_drops_the_currency_prefix():
    previous = json.loads(SPLIT_V12.read_text(encoding="utf-8"))
    current = json.loads(SPLIT_V13.read_text(encoding="utf-8"))
    assert sha256_file(SOURCE_V12) == previous["source_sha256"]
    assert sha256_file(DEVELOPMENT_V12) == previous["development"]["sha256"]
    assert sha256_file(FROZEN_V12) == previous["frozen_benchmark"]["sha256"]
    assert current["version"] == "v1.3"
    assert current["changed_question_ids"] == ["Q041"]
    assert current["development"]["ids"] == previous["development"]["ids"]
    assert current["frozen_benchmark"]["ids"] == previous["frozen_benchmark"]["ids"]
    assert sha256_file(SOURCE_V13) == current["source_sha256"]
    assert sha256_file(DEVELOPMENT_V13) == current["development"]["sha256"]
    assert FROZEN_V13.read_bytes() == FROZEN_V12.read_bytes()
    old_lines = {
        json.loads(line)["question_id"]: line
        for line in SOURCE_V12.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    new_lines = {
        json.loads(line)["question_id"]: line
        for line in SOURCE_V13.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert {qid for qid, line in new_lines.items() if line != old_lines[qid]} == {"Q041"}
    question = next(row for row in load_suite(SOURCE_V13) if row["question_id"] == "Q041")
    span = question["minimal_support_spans"][0]
    page5 = next(item for item in span["equivalent_texts"] if item["pdf_page"] == 5)
    page20 = next(item for item in span["equivalent_texts"] if item["pdf_page"] == 20)
    assert page5["text"] == span["text"]
    assert page5["text"].startswith("Em 2025, foram R$ 33,7 bilhões")
    assert page20["text"] == "33,7 bilhões de distribuição bruta de JCP e dividendos² com um payout de 72,0%"
    corpus = ROOT / "data/Relatório Anual Integrado 2025.pdf"
    if not corpus.is_file():
        pytest.skip("optional local corpus is unavailable")
    pages = dict(extract_pages(corpus.read_bytes()))
    windows = [
        chunk.content
        for chunk in chunk_pages([(20, pages[20])], 256, 26)
        if normalize_text(page20["text"]) in normalize_text(chunk.content)
    ]
    assert len(windows) == 2
    assert any(normalize_text(window).startswith("33,7 bilhões") for window in windows)
    assert any("r$ 33,7 bilhões" in normalize_text(window) for window in windows)
    assert all(match_spans(question, [_hit(window, 20)])["evidence_satisfied"] for window in windows)
    assert all(not match_spans(question, [_hit(window, 5)])["evidence_satisfied"] for window in windows)
    assert all(not match_spans(question, [_hit(window, 3)])["evidence_satisfied"] for window in windows)
    assert match_spans(question, [_hit(page5["text"], 5)])["evidence_satisfied"] is True
    assert match_spans(question, [_hit(page5["text"], 20)])["evidence_satisfied"] is False
    assert match_spans(question, [_hit("33,7 bilhões", 20)])["evidence_satisfied"] is False
    q005 = next(row for row in load_suite(SOURCE_V13) if row["question_id"] == "Q005")
    table = match_spans(
        q005,
        [_hit(
            "Informações de resultado (R$ bilhões em IFRS) 2025 2024 variação (%) YoY (25–24) "
            "Receitas de prestação de serviços e resultados de contratos de seguros e previdência privada³ "
            "55,7 54,1 3,1 %",
            11,
        )],
    )
    assert table["needs_review"] is True
    assert table["evidence_satisfied"] is False


def test_frozen_config_filter_does_not_run_all_27(tmp_path: Path):
    manifest = load_manifest(ROOT / "eval/experiment_manifest.v0.3.json")
    manifest["config_ids"] = ["cs512_top5_hybrid", "cs512_top5_vector"]
    question = load_suite(ROOT / "eval/ground_truth/suite_q001_q010.jsonl", ["Q001"])[0]

    def query(_question: str, index_set_id: str) -> dict:
        return {
            "vector": [
                {
                    "id": 7,
                    "content": question["minimal_support_spans"][0]["text"],
                    "document_name": "report.pdf",
                    "page": 3,
                    "metadata": {},
                    "rank": 1,
                    "similarity": 0.9,
                }
            ],
            "lexical": [],
            "latency_ms": {"embed_ms": 1.0, "vector_ms": 1.0, "lexical_ms": 1.0, "fusion_ms": 0.1},
            "filter_index_set_id": index_set_id,
        }

    out = tmp_path / "frozen"
    run_experiment(
        run_id="frozen-filter",
        manifest=manifest,
        questions=[question],
        out_dir=out,
        query_fn=query,
        layers={"retrieval"},
        read_collections=["itau_2025"],
    )
    rows = [json.loads(line) for line in (out / "retrieval_detail.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["config_id"] for row in rows} == {"cs512_top5_hybrid", "cs512_top5_vector"}
    low, high = wilson_interval(8, 10)
    assert low <= 0.8 <= high
    assert wilson_interval(0, 0) == (0.0, 0.0)
