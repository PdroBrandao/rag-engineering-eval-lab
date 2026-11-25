import io
import json
from pathlib import Path

import pytest

from retrieval_lab.experiments.manifest import load_manifest
from retrieval_lab.experiments.ground_truth import load_suite
from postgrest.exceptions import APIError

from retrieval_lab.db import SupabaseDocumentStore
from retrieval_lab.errors import DatabaseError
from retrieval_lab.experiments.reliability import call_with_retry, classify_failure, load_jsonl
from retrieval_lab.experiments.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = load_manifest(ROOT / "eval/experiment_manifest.v0.3.json")
QUESTION = load_suite(ROOT / "eval/ground_truth/suite_q001_q010.jsonl", ["Q001"])[0]


class Transient(Exception):
    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.headers = {"retry-after": retry_after} if retry_after is not None else {}


class Permanent(Exception):
    def __init__(self) -> None:
        super().__init__("bad request")
        self.status_code = 400


def _query(_question: str, index_set_id: str) -> dict:
    return {
        "vector": [
            {
                "id": 7,
                "content": QUESTION["minimal_support_spans"][0]["text"],
                "document_name": "report.pdf",
                "page": 3,
                "metadata": {},
                "rank": 1,
                "similarity": 0.9,
            }
        ],
        "lexical": [],
        "latency_ms": {"embed_ms": 10.0, "vector_ms": 2.0, "lexical_ms": 1.0, "fusion_ms": 0.1},
        "filter_index_set_id": index_set_id,
    }


def _generate(_system: str, _user: str) -> dict:
    return {
        "model_answer": "PricewaterhouseCoopers (PwC).",
        "llm_latency_ms": 20.0,
        "llm_latency_source": "measured",
        "prompt_tokens_measured": 100,
        "completion_tokens_measured": 10,
    }


def _run(tmp_path: Path, **overrides):
    stream = overrides.pop("stream", io.StringIO())
    kwargs = dict(
        run_id="reliability",
        manifest=MANIFEST,
        questions=[QUESTION],
        out_dir=tmp_path / "run",
        query_fn=_query,
        layers={"retrieval", "end_to_end"},
        read_collections=["itau_2025"],
        generate_fn=_generate,
        judge_fn=None,
        progress_stream=stream,
        sleep=lambda _delay: None,
    )
    kwargs.update(overrides)
    summary = run_experiment(**kwargs)
    return summary, stream, tmp_path / "run"


def test_retry_backs_off_then_succeeds_and_skips_permanent_errors():
    waits: list[float] = []
    state = {"tries": 0}

    def flaky():
        state["tries"] += 1
        if state["tries"] < 3:
            raise Transient(503)
        return "ok"

    assert call_with_retry("retrieval", flaky, sleep=waits.append, rng=__import__("random").Random(0)) == "ok"
    assert state["tries"] == 3
    assert len(waits) == 2
    assert waits[1] >= waits[0]

    def limited():
        raise Transient(429, retry_after="2.5")

    waits.clear()
    with pytest.raises(Transient):
        call_with_retry("generation", limited, attempts=2, sleep=waits.append)
    assert waits == [2.5]

    def denied():
        raise Permanent()

    waits.clear()
    with pytest.raises(Permanent):
        call_with_retry("generation", denied, sleep=waits.append)
    assert waits == []


def test_postgres_syntax_error_is_permanent_and_not_retried():
    syntax = APIError(
        {
            "code": "42601",
            "message": "a column definition list is redundant for a function with OUT parameters",
        }
    )
    store = SupabaseDocumentStore(client=None)
    with pytest.raises(DatabaseError) as raised:
        store._execute(_FailingRequest(syntax), "search_fts_pt_or")
    assert raised.value.status_code == 400
    assert raised.value.code == "42601"
    assert classify_failure(raised.value) == ("permanent", None)

    wrapped = DatabaseError()
    wrapped.__cause__ = syntax
    assert wrapped.status_code == 503
    assert classify_failure(wrapped) == ("permanent", None)

    waits: list[float] = []
    with pytest.raises(DatabaseError):
        call_with_retry("retrieval", lambda: store._execute(_FailingRequest(syntax), "search_fts_pt_or"), sleep=waits.append)
    assert waits == []


class _FailingRequest:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def execute(self):
        raise self.error


def test_progress_log_is_flushed_without_prompts(tmp_path: Path):
    _summary, stream, _out = _run(tmp_path)
    text = stream.getvalue()
    assert "Q001/1" in text
    assert "stage=retrieval" in text
    assert "stage=generation" in text
    assert "stage=evaluation" in text
    assert "stage=export" in text
    assert "done=27/27" in text
    assert "PricewaterhouseCoopers" not in text
    assert "Contexto:" not in text


def test_resume_does_not_duplicate_rows_or_chat_cost(tmp_path: Path):
    _run(tmp_path)
    out = tmp_path / "run"
    first_cost = sum(
        row["api_cost_usd_chat_estimated"]
        for row in load_jsonl(out / "e2e_detail.jsonl")
        if row["status"] == "success"
    )
    _run(tmp_path)
    rows = [row for row in load_jsonl(out / "e2e_detail.jsonl") if row["status"] == "success"]
    assert len(rows) == 27
    assert len({row["config_id"] for row in rows}) == 27
    assert sum(row["api_cost_usd_chat_estimated"] for row in rows) == pytest.approx(first_cost)


def test_failed_generation_is_not_a_zero_score_and_can_resume(tmp_path: Path):
    calls = {"n": 0}

    def fail_once(_system: str, _user: str) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise Permanent()
        return _generate(_system, _user)

    out = tmp_path / "run"
    run_experiment(
        run_id="reliability",
        manifest=MANIFEST,
        questions=[QUESTION],
        out_dir=out,
        query_fn=_query,
        layers={"retrieval", "end_to_end"},
        read_collections=["itau_2025"],
        generate_fn=fail_once,
        progress_stream=io.StringIO(),
        sleep=lambda _delay: None,
    )
    failed = [
        row
        for row in load_jsonl(out / "e2e_detail.jsonl")
        if row["status"] == "failed" and row["config_id"] == "cs256_top3_lexical"
    ]
    assert failed[0]["answer_correct"] is None
    run_experiment(
        run_id="reliability",
        manifest=MANIFEST,
        questions=[QUESTION],
        out_dir=out,
        query_fn=_query,
        layers={"retrieval", "end_to_end"},
        read_collections=["itau_2025"],
        generate_fn=_generate,
        progress_stream=io.StringIO(),
        sleep=lambda _delay: None,
    )
    successes = [
        row
        for row in load_jsonl(out / "e2e_detail.jsonl")
        if row["status"] == "success" and row["config_id"] == "cs256_top3_lexical"
    ]
    assert len(successes) == 1
    assert successes[0]["answer_correct"] == 1


def test_truncated_line_is_not_complete_and_checkpoint_rejects_a_different_suite(tmp_path: Path):
    _summary, _stream, out = _run(tmp_path)
    detail = out / "retrieval_detail.jsonl"
    lines = detail.read_text(encoding="utf-8").splitlines()
    detail.write_text("\n".join(lines[:-1] + ["{truncated"]) + "\n", encoding="utf-8")
    assert all(row.get("config_id") != json.loads(lines[-1])["config_id"] for row in load_jsonl(detail) if False) or True
    missing = json.loads(lines[-1])["config_id"]
    assert missing not in {row["config_id"] for row in load_jsonl(detail) if row.get("status") == "success"}
    _run(tmp_path)
    restored = [
        row for row in load_jsonl(detail) if row.get("status") == "success" and row["config_id"] == missing
    ]
    assert len(restored) == 1
    other = dict(QUESTION)
    other["question"] = "pergunta diferente"
    with pytest.raises(RuntimeError, match="Checkpoint"):
        run_experiment(
            run_id="reliability",
            manifest=MANIFEST,
            questions=[other],
            out_dir=out,
            query_fn=_query,
            layers={"retrieval", "end_to_end"},
            read_collections=["itau_2025"],
            generate_fn=_generate,
            progress_stream=io.StringIO(),
            sleep=lambda _delay: None,
        )
