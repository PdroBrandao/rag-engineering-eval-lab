import csv
import json
import math
from pathlib import Path

from retrieval_lab.experiments.grading import evaluation_costs
from retrieval_lab.experiments.metrics_v3 import _chunk_grade, score_hits
from retrieval_lab.experiments.report import write_reports
from retrieval_lab.experiments.runner import run_experiment
from retrieval_lab.experiments.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
PRICING = {"input": 0.15, "output": 0.6, "embedding_input": 0.02}


def _span_question(*lines: str, table: bool = False) -> dict:
    span = {"span_id": "S1", "required": True, "pdf_page": 1}
    if len(lines) == 1:
        span["text"] = lines[0]
    else:
        span["lines"] = list(lines)
    return {
        "question_id": "QX",
        "question": "pergunta",
        "reference_answer": "resposta",
        "table_ambiguous": table,
        "minimal_support_spans": [span],
        "grading": {"mode": "contains_all", "required_phrases": ["resposta"]},
    }


def _hit(chunk_id: int, content: str) -> dict:
    return {"id": chunk_id, "page": 1, "content": content, "metadata": {}, "document_name": "report.pdf"}


def test_ndcg_stays_inside_zero_and_one_and_ranks_ideal_first():
    question = _span_question("evidencia completa")
    relevant = _hit(1, "evidencia completa")
    other = _hit(2, "texto sem relacao")
    weak = _hit(3, "evidencia")

    ideal = score_hits(question, [relevant, other, other], 3)
    worse = score_hits(question, [other, relevant, other], 3)
    empty = score_hits(question, [other, other], 2)
    duplicated = score_hits(question, [relevant, relevant], 2)
    partial = _span_question("alfa unico zz", "beta unico zz")
    grade_one = _hit(4, "alfa unico zz")
    grade_three = _hit(5, "alfa unico zz beta unico zz")

    for scored in (ideal, worse, empty, duplicated):
        assert 0.0 <= scored["ndcg_at_k"] <= 1.0
    assert ideal["ndcg_at_k"] == 1.0
    assert duplicated["ndcg_at_k"] == 1.0
    old_idcg = 3.0 / math.log2(2)
    old_dcg = 3.0 / math.log2(2) + 3.0 / math.log2(3)
    assert old_dcg / old_idcg > 1.0
    assert worse["ndcg_at_k"] < ideal["ndcg_at_k"]
    assert empty["ndcg_at_k"] == 0.0
    assert score_hits(partial, [grade_three, grade_one], 2)["ndcg_at_k"] == 1.0
    assert score_hits(partial, [grade_one, grade_three], 2)["ndcg_at_k"] < 1.0
    assert "mrr" not in ideal
    assert ideal["reciprocal_full_evidence_rank"] == 1.0
    assert worse["reciprocal_full_evidence_rank"] == 0.5


def test_chunk_grades_keep_the_existing_scale():
    full = _span_question("alfa unico zz", "beta unico zz")
    alone = _hit(1, "alfa unico zz beta unico zz")
    half = _hit(2, "alfa unico zz")
    unrelated = _hit(3, "nada")
    table = _span_question("celula da tabela", table=True)
    assert _chunk_grade(full, alone, False) == 3.0
    assert _chunk_grade(full, half, False) == 1.0
    assert _chunk_grade(full, unrelated, True) == 0.0
    assert _chunk_grade(table, _hit(4, "celula da tabela"), False) == 1.0
    multi = {
        "table_ambiguous": False,
        "minimal_support_spans": [
            {"span_id": "S1", "required": True, "pdf_page": 1, "text": "primeiro trecho longo"},
            {"span_id": "S2", "required": True, "pdf_page": 1, "text": "segundo trecho longo"},
        ],
    }
    assert _chunk_grade(multi, _hit(5, "primeiro trecho longo"), False) == 2.0


def test_full_evidence_rank_waits_for_every_required_span():
    question = {
        "table_ambiguous": False,
        "minimal_support_spans": [
            {"span_id": "S1", "required": True, "pdf_page": 1, "text": "primeiro trecho longo"},
            {"span_id": "S2", "required": True, "pdf_page": 1, "text": "segundo trecho longo"},
        ],
    }
    scored = score_hits(
        question,
        [_hit(1, "primeiro trecho longo"), _hit(2, "segundo trecho longo")],
        2,
    )
    assert scored["reciprocal_full_evidence_rank"] == 0.5
    assert scored["evidence_coverage"] == 1.0
    assert score_hits(question, [_hit(1, "nada")], 1)["reciprocal_full_evidence_rank"] == 0.0


def test_unknown_judge_cost_is_not_zero_and_optional_judge_is_zero():
    pricing = PRICING
    missing = evaluation_costs("judge", {"judge_called": True}, 10, 2, pricing)
    assert missing["generation_cost_usd"] == round(10 * 0.15 / 1_000_000 + 2 * 0.6 / 1_000_000, 8)
    assert missing["judge_cost_usd"] is None
    assert missing["judge_cost_basis"] == "unknown"
    assert missing["evaluation_row_cost_usd"] is None
    skipped = evaluation_costs("contains_all", {}, 10, 2, pricing)
    assert skipped["judge_cost_usd"] == 0.0
    assert skipped["judge_cost_basis"] == "not_applicable"
    pending = evaluation_costs("semantic_review", {"judge_called": False}, 10, 2, pricing)
    assert pending["judge_cost_usd"] is None
    assert pending["judge_cost_basis"] == "not_evaluated"
    judged = evaluation_costs(
        "semantic_review",
        {"judge_called": True, "judge_prompt_tokens": 20, "judge_completion_tokens": 5},
        10,
        2,
        pricing,
    )
    assert judged["judge_cost_usd"] == round(20 * 0.15 / 1_000_000 + 5 * 0.6 / 1_000_000, 8)


def test_inconclusive_answers_stay_out_of_the_mean(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    retrieval = [
        {
            "status": "success",
            "question_id": "Q1",
            "config_id": "cs512_top5_hybrid",
            "chunk_size": 512,
            "index_set_id": "cs512",
            "top_k": 5,
            "strategy": "hybrid",
            "metrics_contract": "v0.4",
            "recall_at_k": 0.0,
            "reciprocal_full_evidence_rank": 0.0,
            "ndcg_at_k": 1.0,
            "evidence_coverage": 0.0,
            "context_tokens": 10,
            "needs_review": True,
            "evidence_satisfied": False,
            "measured_execution_latency_ms": 4.0,
            "estimated_standalone_latency_ms": 14.0,
            "strategy_total_ms": 4.0,
        }
    ]
    e2e = [
        {
            "status": "success",
            "question_id": "Q1",
            "config_id": "cs512_top5_hybrid",
            "answer_correct": None,
            "evaluation_status": "inconclusive",
            "uncertain": True,
            "answer_support": "inconclusive",
            "generation_cost_usd": 0.01,
            "judge_cost_usd": None,
            "embedding_cost_usd": 0.001,
            "index_set_id": "cs512",
            "llm_latency_ms": 5,
            "measured_e2e_execution_latency_ms": 9,
        },
        {
            "status": "success",
            "question_id": "Q2",
            "config_id": "cs512_top5_hybrid",
            "answer_correct": 1,
            "evaluation_status": "valid",
            "uncertain": False,
            "answer_support": "supported",
            "generation_cost_usd": 0.02,
            "judge_cost_usd": 0.003,
            "embedding_cost_usd": 0.001,
            "index_set_id": "cs512",
            "llm_latency_ms": 5,
            "measured_e2e_execution_latency_ms": 9,
        },
    ]
    (run / "retrieval_detail.jsonl").write_text("".join(json.dumps(row) + "\n" for row in retrieval), encoding="utf-8")
    (run / "e2e_detail.jsonl").write_text("".join(json.dumps(row) + "\n" for row in e2e), encoding="utf-8")
    write_reports(run, run_id="fixture", baseline_config_id="cs512_top5_hybrid")
    e2e_rows = list(csv.DictReader((run / "end_to_end.csv").open(encoding="utf-8")))
    assert e2e_rows[0]["answer_correct_rate"] == "1.0"
    assert e2e_rows[0]["n_graded_auto"] == "1"
    assert e2e_rows[0]["judge_cost_usd"] == ""
    costs = list(csv.DictReader((run / "cost_by_question.csv").open(encoding="utf-8")))
    by_id = {row["question_id"]: row for row in costs}
    assert by_id["Q1"]["judge_cost_usd"] == ""
    assert by_id["Q1"]["evaluation_total_cost_usd"] == ""
    assert by_id["Q1"]["generation_cost_usd"] == "0.01"
    assert float(by_id["Q2"]["evaluation_total_cost_usd"]) == 0.024


def test_runner_records_measured_judge_tokens(tmp_path: Path):
    manifest = load_manifest(ROOT / "eval/experiment_manifest.v0.3.json")
    manifest["config_ids"] = ["cs512_top5_vector"]
    question = {
        "question_id": "QJ",
        "question": "Qual é o trecho?",
        "reference_answer": "o trecho certo",
        "evidence_quote": "o trecho certo",
        "table_ambiguous": False,
        "category": "factual",
        "minimal_support_spans": [
            {"span_id": "S1", "required": True, "pdf_page": 3, "text": "o trecho certo"}
        ],
        "grading": {"mode": "semantic_review", "key_concepts": ["trecho"]},
    }

    def query(_question: str, index_set_id: str) -> dict:
        return {
            "vector": [_hit(7, "o trecho certo") | {"rank": 1, "similarity": 0.9}],
            "lexical": [],
            "latency_ms": {"embed_ms": 10.0, "vector_ms": 2.0, "lexical_ms": 1.0, "fusion_ms": 0.1},
            "filter_index_set_id": index_set_id,
        }

    def generate(_system: str, _user: str) -> dict:
        return {
            "model_answer": "o trecho certo",
            "llm_latency_ms": 20.0,
            "prompt_tokens_measured": 100,
            "completion_tokens_measured": 10,
        }

    def judge(_question: str, _gold: str, _answer: str, key_concepts=None) -> dict:
        assert key_concepts == ["trecho"]
        return {
            "verdict": "uncertain",
            "prompt_tokens": 30,
            "completion_tokens": 4,
            "judge_latency_ms": 8.0,
            "judge_latency_source": "measured",
        }

    out = tmp_path / "judge"
    run_experiment(
        run_id="judge-cost",
        manifest=manifest,
        questions=[question],
        out_dir=out,
        query_fn=query,
        layers={"retrieval", "end_to_end"},
        read_collections=["itau_2025"],
        generate_fn=generate,
        judge_fn=judge,
    )
    row = json.loads((out / "e2e_detail.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["answer_correct"] is None
    assert row["evaluation_status"] == "inconclusive"
    assert row["judge_cost_usd"] == round(30 * 0.15 / 1_000_000 + 4 * 0.6 / 1_000_000, 8)
    assert row["judge_latency_ms"] == 8.0
    assert row["measured_e2e_execution_latency_ms"] == 32.0
    assert row["estimated_standalone_e2e_latency_ms"] == 32.0
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["metrics_contract"] == "v0.4"
    assert meta["judge_cost_usd"] == row["judge_cost_usd"]
    assert meta["evaluation_total_cost_usd"] == round(
        meta["rag_operational_cost_usd"] + meta["judge_cost_usd"], 8
    )
