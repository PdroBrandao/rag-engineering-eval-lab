import pytest
from fastapi.testclient import TestClient

from retrieval_lab.errors import IngestError
from retrieval_lab.ingest import ingest_pdf
from retrieval_lab.main import create_app
from retrieval_lab.pdf_extract import extract_pages
from tests.conftest import TEST_ISSUER
from tests.fakes import FakeEmbedder, MemoryStore
from tests.pdf_bytes import pdf_with_pages


@pytest.fixture
def lab(full_access_store, jwt_validator, test_settings, mint_token):
    store = full_access_store
    embedder = FakeEmbedder()
    app = create_app()
    app.state.store = store
    app.state.embedder = embedder
    app.state.settings = test_settings
    app.state.jwt_validator = jwt_validator
    app.state.max_upload_bytes = 1024
    headers = {"Authorization": f"Bearer {mint_token(subject='admin-all')}"}
    return TestClient(app), store, embedder, headers


def test_health_reports_database_status(lab):
    client, store, _embedder, _headers = lab
    assert client.get("/health").json() == {"status": "ok", "database": "ok"}
    store.fail_ping = True
    failed = client.get("/health")
    assert failed.status_code == 503
    assert failed.json() == {"status": "ok", "database": "error"}


def test_openapi_lists_the_lab_endpoints(lab):
    client, _store, _embedder, _headers = lab
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/ingest", "/query"} <= set(paths)


def test_ingest_rejects_invalid_empty_and_duplicate_pdfs(lab):
    client, store, embedder, headers = lab
    invalid = client.post(
        "/ingest",
        files={"file": ("notes.pdf", b"not a pdf", "application/pdf")},
        data={"collection": "legacy"},
        headers=headers,
    )
    assert invalid.status_code == 400
    assert embedder.calls == []

    empty = client.post(
        "/ingest",
        files={"file": ("blank.pdf", pdf_with_pages([""]), "application/pdf")},
        data={"collection": "legacy"},
        headers=headers,
    )
    assert empty.status_code == 400
    assert store.rows == []

    pdf = pdf_with_pages(["Vector retrieval and lexical retrieval can be fused."])
    first = client.post(
        "/ingest",
        files={"file": ("../../paper.pdf", pdf, "application/pdf")},
        data={"chunk_size": "50", "chunk_overlap": "5", "collection": "legacy"},
        headers=headers,
    )
    assert first.status_code == 200
    body = first.json()
    assert body["document_name"] == "paper.pdf"
    assert body["collection"] == "legacy"
    assert body["chunk_count"] == 1
    assert body["chunk_ids"] == [1]
    assert store.rows[0]["page"] == 1
    assert store.rows[0]["collection"] == "legacy"
    assert len(embedder.calls) == 1

    second = client.post(
        "/ingest",
        files={"file": ("paper.pdf", pdf, "application/pdf")},
        data={"collection": "legacy"},
        headers=headers,
    )
    assert second.status_code == 409
    assert len(embedder.calls) == 1
    assert len(store.rows) == 1


def test_replace_true_does_not_touch_stored_chunks(lab):
    client, store, embedder, headers = lab
    pdf = pdf_with_pages(["Keep this text."])
    assert (
        client.post(
            "/ingest",
            files={"file": ("keep.pdf", pdf, "application/pdf")},
            data={"collection": "legacy"},
            headers=headers,
        ).status_code
        == 200
    )
    stored = list(store.rows)
    replaced = client.post(
        "/ingest",
        files={"file": ("keep.pdf", pdf, "application/pdf")},
        data={"replace": "true", "collection": "legacy"},
        headers=headers,
    )
    assert replaced.status_code == 400
    assert store.rows == stored
    assert len(embedder.calls) == 1


def test_upload_limit_and_bad_chunk_params(lab):
    client, _store, embedder, headers = lab
    oversized = client.post(
        "/ingest",
        files={"file": ("big.pdf", b"x" * 1025, "application/pdf")},
        data={"collection": "legacy"},
        headers=headers,
    )
    assert oversized.status_code == 400
    overlap = client.post(
        "/ingest",
        files={"file": ("notes.pdf", pdf_with_pages(["text"]), "application/pdf")},
        data={"chunk_size": "4", "chunk_overlap": "4", "collection": "legacy"},
        headers=headers,
    )
    assert overlap.status_code == 400
    assert embedder.calls == []


def test_query_keeps_strategy_scores_separate(lab):
    client, store, embedder, headers = lab
    store.vector_rows = [
        {
            "id": 1,
            "collection": "legacy",
            "content": "vector chunk",
            "document_name": "paper.pdf",
            "page": 2,
            "metadata": {"content_sha256": "abc"},
            "similarity": 0.91,
            "result_rank": 1,
        },
        {
            "id": 2,
            "collection": "legacy",
            "content": "second vector",
            "document_name": "paper.pdf",
            "page": 3,
            "metadata": {},
            "similarity": 0.5,
            "result_rank": 2,
        },
    ]
    store.lexical_rows = [
        {
            "id": 2,
            "collection": "legacy",
            "content": "second vector",
            "document_name": "paper.pdf",
            "page": 3,
            "metadata": {},
            "lexical_score": 0.2,
            "result_rank": 1,
        }
    ]
    response = client.post(
        "/query",
        json={"question": "What are the risks?", "top_k": 5, "rrf_k": 60},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["candidate_k"] == 20
    assert body["vector"][0]["similarity"] == 0.91
    assert store.match_calls[-1]["filter_collections"]
    assert "rrf_score" not in body["vector"][0]
    assert [hit["id"] for hit in body["hybrid"]] == [2, 1]
    assert embedder.calls == [["What are the risks?"]]


def test_query_rejects_blank_question_and_small_candidate_pool(lab):
    client, _store, _embedder, headers = lab
    blank = client.post("/query", json={"question": "   "}, headers=headers)
    assert blank.status_code == 422
    small_pool = client.post(
        "/query",
        json={"question": "risks", "top_k": 10, "candidate_k": 5},
        headers=headers,
    )
    assert small_pool.status_code == 422


def test_extract_pages_reads_text_and_rejects_garbage():
    pages = extract_pages(pdf_with_pages(["Page one", "Page two"]))
    assert [page for page, _text in pages] == [1, 2]
    try:
        extract_pages(b"not a pdf")
    except IngestError as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("invalid PDF was accepted")


def test_ingest_service_stops_before_embedding_on_duplicate():
    store = MemoryStore()
    store.set_access(
        TEST_ISSUER,
        "svc",
        permissions=["documents:write"],
        read_collections=[],
        write_collections=["legacy"],
    )
    embedder = FakeEmbedder()
    pdf = pdf_with_pages(["Same bytes."])
    ingest_pdf(pdf, "a.pdf", store, embedder, "legacy")
    try:
        ingest_pdf(pdf, "b.pdf", store, embedder, "legacy")
    except IngestError as exc:
        assert exc.status_code == 409
    assert len(embedder.calls) == 1
