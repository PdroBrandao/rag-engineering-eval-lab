import json
from pathlib import Path

import pytest

from retrieval_lab.experiments.derive import attribute_latency, derive_hits
from retrieval_lab.experiments.evidence_match import match_spans
from retrieval_lab.experiments.grading import grade_answer, support_label
from retrieval_lab.experiments.ground_truth import load_suite
from retrieval_lab.experiments.manifest import effective_collections, load_manifest
from retrieval_lab.experiments.metrics_v3 import score_hits
from retrieval_lab.experiments.reliability import fingerprint
from retrieval_lab.experiments.runner import run_experiment
from retrieval_lab.retrieval import query_retrieval
from retrieval_lab.rrf import fuse_rankings
from tests.fakes import FakeEmbedder, MemoryStore

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "eval/ground_truth/suite_q001_q010.jsonl"
MANIFEST = ROOT / "eval/experiment_manifest.v0.3.json"


def _question(question_id: str) -> dict:
    return next(row for row in load_suite(SUITE) if row["question_id"] == question_id)


def test_suite_is_exactly_the_first_ten_questions():
    rows = load_suite(SUITE)
    assert [row["question_id"] for row in rows] == [f"Q{i:03d}" for i in range(1, 11)]
    assert all(row["evidence_quote"] for row in rows)
    assert len(_question("Q004")["minimal_support_spans"]) == 2
    assert _question("Q005")["table_ambiguous"] is True


def test_q005_table_overlap_is_not_a_confirmed_hit():
    question = _question("Q005")
    chunk = {
        "id": 1,
        "page": 11,
        "content": (
            "Informações de resultado (R$ bilhões em IFRS) 2025 2024 variação (%) YoY (25–24) "
            "Receitas de prestação de serviços e resultados de contratos de seguros e previdência privada³ "
            "55,7 54,1 3,1 %"
        ),
    }
    matched = match_spans(question, [chunk])
    assert matched["needs_review"] is True
    assert matched["evidence_satisfied"] is False
    assert all(item["confirmed"] is False for item in matched["spans"])
    scored = score_hits(question, [chunk], 3)
    assert scored["recall_at_k"] == 0.0
    assert scored["evidence_coverage"] == 0.0


def test_multi_span_requires_every_span():
    question = _question("Q004")
    first = question["minimal_support_spans"][0]["text"]
    second = question["minimal_support_spans"][1]["text"]
    partial = match_spans(question, [{"id": 1, "page": 10, "content": first}])
    assert partial["evidence_satisfied"] is False
    full = match_spans(
        question,
        [
            {"id": 1, "page": 10, "content": first},
            {"id": 2, "page": 10, "content": second},
        ],
    )
    assert full["evidence_satisfied"] is True


def test_derived_hybrid_matches_direct_fusion():
    manifest = load_manifest(MANIFEST)
    store = MemoryStore()
    vector = []
    lexical = []
    for rank, chunk_id in enumerate([5, 4, 3, 2, 1], start=1):
        vector.append(
            {
                "id": chunk_id,
                "content": f"vector {chunk_id}",
                "document_name": "report.pdf",
                "page": 3,
                "metadata": {},
                "collection": "itau_2025",
                "index_set_id": "cs512",
                "similarity": 1 / rank,
                "result_rank": rank,
            }
        )
    for rank, chunk_id in enumerate([1, 2, 3, 8, 9], start=1):
        lexical.append(
            {
                "id": chunk_id,
                "content": f"lexical {chunk_id}",
                "document_name": "report.pdf",
                "page": 3,
                "metadata": {},
                "collection": "itau_2025",
                "index_set_id": "cs512",
                "lexical_score": 1 / rank,
                "result_rank": rank,
            }
        )
    store.vector_rows = vector
    store.lexical_rows = lexical
    shared = query_retrieval(
        "asseguração",
        store,
        FakeEmbedder(),
        filter_collections=["itau_2025"],
        top_k=10,
        rrf_k=60,
        candidate_k=20,
        filter_index_set_id="cs512",
    )
    direct = query_retrieval(
        "asseguração",
        store,
        FakeEmbedder(),
        filter_collections=["itau_2025"],
        top_k=3,
        rrf_k=60,
        candidate_k=20,
        filter_index_set_id="cs512",
    )
    derived = derive_hits(shared, "hybrid", 3, manifest["retrieval"]["rrf_k"])
    assert [hit["id"] for hit in derived] == [hit["id"] for hit in direct["hybrid"]]
    assert [hit["id"] for hit in derive_hits(shared, "vector", 3, 60)] == [
        hit["id"] for hit in direct["vector"][:3]
    ]
    ranks = fuse_rankings(
        [
            {hit["id"]: hit["rank"] for hit in shared["vector"]},
            {hit["id"]: hit["rank"] for hit in shared["lexical"]},
        ],
        rrf_k=60,
    )[:3]
    assert [chunk_id for chunk_id, _score in ranks] == [hit["id"] for hit in derived]


def test_latency_marks_cached_embed_separately_from_measured_time():
    bundle = {
        "latency_ms": {"embed_ms": 12.5, "vector_ms": 4.0, "lexical_ms": 3.0, "fusion_ms": 0.2}
    }
    measured = attribute_latency(bundle, "vector", embed_source="measured")
    cached = attribute_latency(bundle, "vector", embed_source="shared_cache")
    assert measured["embed_ms_measured"] == 12.5
    assert measured["strategy_total_ms"] == 16.5
    assert measured["measured_execution_latency_ms"] == 16.5
    assert measured["estimated_standalone_latency_ms"] == 16.5
    assert cached["embed_ms_shared_cache"] == 12.5
    assert cached["embed_ms_measured"] is None
    assert cached["strategy_total_ms"] == 4.0
    assert cached["measured_execution_latency_ms"] == 4.0
    assert cached["estimated_standalone_latency_ms"] == 16.5
    assert cached["components"][0]["source"] == "shared_cache"
    assert cached["components"][0]["counted_in_measured_execution"] is False
    assert cached["components"][0]["included_in_standalone_estimate"] is True
    lexical = attribute_latency(bundle, "lexical", embed_source="shared_cache")
    assert all(item["name"] != "embed_ms" for item in lexical["components"])
    assert lexical["strategy_total_ms"] == 3.0
    assert lexical["estimated_standalone_latency_ms"] == 3.0


def test_grading_and_unsupported_answer():
    question = _question("Q001")
    graded = grade_answer(question, "A asseguração foi da PricewaterhouseCoopers (PwC).")
    assert graded["answer_correct"] == 1
    assert support_label(1, False) == "correct_unsupported"
    assert support_label(1, True) == "supported"
    numeric = grade_answer(_question("Q003"), "Atuava em 18 países.")
    assert numeric["answer_correct"] == 1


def test_runner_rejects_collection_outside_the_grant(tmp_path: Path):
    manifest = load_manifest(MANIFEST)
    try:
        effective_collections(manifest["collections"], ["finance"])
    except PermissionError:
        pass
    else:
        raise AssertionError("expected permission error")


def test_runner_writes_inconclusive_comparison_and_resumes(tmp_path: Path):
    manifest = load_manifest(MANIFEST)
    question = _question("Q001")
    span = question["minimal_support_spans"][0]["text"]

    def query(_question: str, index_set_id: str) -> dict:
        return {
            "vector": [
                {
                    "id": 7,
                    "content": span,
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

    def generate(_system: str, _user: str) -> dict:
        return {
            "model_answer": "PricewaterhouseCoopers (PwC).",
            "llm_latency_ms": 20.0,
            "llm_latency_source": "measured",
            "prompt_tokens_measured": 100,
            "completion_tokens_measured": 10,
        }

    out = tmp_path / "run"
    kwargs = dict(
        run_id="smoke",
        manifest=manifest,
        questions=[question],
        out_dir=out,
        query_fn=query,
        layers={"retrieval", "end_to_end"},
        read_collections=["itau_2025"],
        generate_fn=generate,
        judge_fn=None,
    )
    run_experiment(**kwargs)
    run_experiment(**kwargs)
    retrieval_lines = [
        json.loads(line)
        for line in (out / "retrieval_detail.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(retrieval_lines) == 27
    comparison = (out / "comparison.csv").read_text(encoding="utf-8")
    assert comparison.count("\n") == 28
    assert "inconclusive" in comparison
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["verdict_policy"] == "inconclusive"
    assert len(meta["embed_cost_usd_estimated_by_question_index_set"]) == 3
    e2e = [
        json.loads(line)
        for line in (out / "e2e_detail.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    vector_rows = [row for row in e2e if row["strategy"] == "vector" and row["index_set_id"] == "cs256"]
    measured = [row for row in vector_rows if row["embed_ms_measured"] == 10.0]
    cached = [row for row in vector_rows if row["embed_ms_shared_cache"] == 10.0]
    assert len(measured) == 1
    assert len(cached) == 2
    assert measured[0]["estimated_standalone_latency_ms"] == measured[0]["measured_execution_latency_ms"]
    assert cached[0]["estimated_standalone_latency_ms"] > cached[0]["measured_execution_latency_ms"]
    assert all(row["chat_cost_basis"] == "estimated_unit_price_times_measured_tokens" for row in e2e)
    assert all(row["judge_cost_usd"] == 0 for row in e2e)
    assert all(row["judge_cost_basis"] == "not_applicable" for row in e2e)
    assert all(row["metrics_contract"] == "v0.4" for row in retrieval_lines)
    assert "mrr" not in retrieval_lines[0]
    assert meta["evaluation_total_cost_usd"] == meta["rag_operational_cost_usd"]


def _hit(chunk_id: int, rank: int, content: str, *, kind: str) -> dict:
    row = {
        "id": chunk_id,
        "content": content,
        "document_name": "report.pdf",
        "page": 3,
        "metadata": {},
        "collection": "itau_2025",
        "index_set_id": "cs512",
        "result_rank": rank,
    }
    if kind == "vector":
        row["similarity"] = 1 / rank
    else:
        row["lexical_score"] = 1 / rank
    return row


def test_portuguese_or_feeds_hybrid_and_keeps_english_reproducible(tmp_path: Path):
    manifest = load_manifest(MANIFEST)
    question = _question("Q001")
    span = question["minimal_support_spans"][0]["text"]
    store = MemoryStore()
    store.vector_rows = [_hit(chunk_id, rank, span, kind="vector") for rank, chunk_id in enumerate([1, 3, 4, 5, 6, 7], start=1)]
    store.lexical_rows = [_hit(chunk_id, rank, "termo em ingles", kind="lexical") for rank, chunk_id in enumerate([2, 8], start=1)]
    store.pt_or_rows = [
        _hit(chunk_id, rank, span, kind="lexical")
        for rank, chunk_id in enumerate([9, 1, 30, 31, 32, 33], start=1)
    ]
    embedder = FakeEmbedder()
    english = query_retrieval(
        "quem realizou a asseguração",
        store,
        embedder,
        filter_collections=["itau_2025"],
        top_k=10,
        rrf_k=60,
        candidate_k=20,
        filter_index_set_id="cs512",
    )
    assert english["lexical_version"] == "english_and"
    assert english["candidate_k"] == 20
    assert english["rrf_k"] == 60
    assert [hit["id"] for hit in english["lexical"]] == [2, 8]
    assert english["lexical_pt_or"] == []
    assert all(call.get("rpc") != "search_documents_fts_pt_or" for call in store.fts_calls)
    english_hybrid = [hit["id"] for hit in english["hybrid"]]
    assert english_hybrid == [
        chunk_id
        for chunk_id, _score in fuse_rankings(
            [
                {hit["id"]: hit["rank"] for hit in english["vector"]},
                {hit["id"]: hit["rank"] for hit in english["lexical"]},
            ],
            rrf_k=60,
        )[:10]
    ]
    assert 9 not in english_hybrid

    store.fts_calls.clear()
    selected = {}
    for top_k in (3, 5, 10):
        selected = query_retrieval(
            "quem realizou a asseguração",
            store,
            embedder,
            filter_collections=["itau_2025"],
            top_k=top_k,
            rrf_k=60,
            candidate_k=20,
            filter_index_set_id="cs512",
            lexical_version="portuguese_or",
        )
        assert selected["vector"] == english["vector"]
        assert [hit["id"] for hit in selected["lexical_pt_or"]] == [9, 1, 30, 31, 32, 33]
        assert all(call.get("rpc") == "search_documents_fts_pt_or" for call in store.fts_calls)
        assert {call["match_count"] for call in store.fts_calls} == {20}
        fused = [
            chunk_id
            for chunk_id, _score in fuse_rankings(
                [
                    {hit["id"]: hit["rank"] for hit in selected["vector"]},
                    {hit["id"]: hit["rank"] for hit in selected["lexical_pt_or"]},
                ],
                rrf_k=60,
            )
        ]
        assert [hit["id"] for hit in selected["hybrid"]] == fused[:top_k]
        assert len(selected["hybrid"]) == len({hit["id"] for hit in selected["hybrid"]})
        assert [hit["id"] for hit in selected["hybrid"]].count(1) == 1
    assert fuse_rankings([{10: 1}, {10: 1}], rrf_k=60) == [(10, 2 / 61)]
    top3 = query_retrieval(
        "quem realizou a asseguração",
        store,
        embedder,
        filter_collections=["itau_2025"],
        top_k=3,
        rrf_k=60,
        candidate_k=20,
        filter_index_set_id="cs512",
        lexical_version="portuguese_or",
    )
    assert [hit["id"] for hit in top3["hybrid"]] == [hit["id"] for hit in selected["hybrid"]][:3]

    manifest["retrieval"]["lexical_version"] = "portuguese_or"

    def query(_question: str, _index_set_id: str) -> dict:
        return {
            "vector": english["vector"],
            "lexical": english["lexical"],
            "lexical_pt_or": selected["lexical_pt_or"],
            "latency_ms": {
                "embed_ms": 1.0,
                "vector_ms": 2.0,
                "lexical_ms": 40.0,
                "lexical_pt_or_ms": 7.0,
                "fusion_ms": 0.1,
            },
        }

    out = tmp_path / "pt-or"
    kwargs = dict(
        run_id="pt-or",
        manifest=manifest,
        questions=[question],
        out_dir=out,
        query_fn=query,
        layers={"retrieval"},
        read_collections=["itau_2025"],
    )
    run_experiment(**kwargs)
    run_experiment(**kwargs)
    rows = [
        json.loads(line)
        for line in (out / "retrieval_detail.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 27
    lexical = next(row for row in rows if row["config_id"] == "cs512_top10_lexical")
    hybrid = next(row for row in rows if row["config_id"] == "cs512_top10_hybrid")
    hybrid_top3 = next(row for row in rows if row["config_id"] == "cs512_top3_hybrid")
    vector = next(row for row in rows if row["config_id"] == "cs512_top10_vector")
    assert lexical["chunk_ids"] == [9, 1, 30, 31, 32, 33]
    assert lexical["lexical_version"] == "portuguese_or"
    assert lexical["strategy_total_ms"] == 7.0
    assert hybrid["lexical_version"] == "portuguese_or"
    assert hybrid["chunk_ids"] == [hit["id"] for hit in selected["hybrid"]]
    assert hybrid["chunk_ids"][:3] == hybrid_top3["chunk_ids"]
    assert len(hybrid["chunk_ids"]) == len(set(hybrid["chunk_ids"]))
    assert 2 not in hybrid["chunk_ids"]
    assert hybrid["strategy_total_ms"] == 9.1
    assert hybrid["measured_execution_latency_ms"] == 9.1
    assert hybrid["estimated_standalone_latency_ms"] == 10.1
    assert vector["chunk_ids"] == [1, 3, 4, 5, 6, 7]
    assert vector["lexical_version"] == "portuguese_or"
    checkpoint = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["lexical_version"] == "portuguese_or"
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["lexical_version"] == "portuguese_or"
    base = load_manifest(MANIFEST)
    assert fingerprint(base, [question], {"retrieval"}) == fingerprint(
        {**base, "retrieval": {**base["retrieval"], "lexical_version": "english_and"}},
        [question],
        {"retrieval"},
    )
    assert fingerprint(manifest, [question], {"retrieval"}) != fingerprint(
        base, [question], {"retrieval"}
    )
    english_manifest = load_manifest(MANIFEST)
    with pytest.raises(RuntimeError, match="lexical version"):
        run_experiment(
            run_id="pt-or",
            manifest=english_manifest,
            questions=[question],
            out_dir=out,
            query_fn=query,
            layers={"retrieval"},
            read_collections=["itau_2025"],
        )
