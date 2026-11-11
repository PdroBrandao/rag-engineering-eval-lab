import jwt
import pytest
from fastapi.testclient import TestClient

from retrieval_lab.auth.guards import resolve_query_collections
from retrieval_lab.auth.models import AccessContext, Principal
from retrieval_lab.auth.service import resolve_access
from retrieval_lab.ingest import ingest_pdf
from retrieval_lab.main import create_app
from retrieval_lab.retrieval import query_retrieval
from tests.conftest import TEST_ISSUER, TEST_AUDIENCE, mint_access_token
from tests.fakes import FakeEmbedder, MemoryStore
from tests.pdf_bytes import pdf_with_pages


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def app_client(embedder, jwt_validator, test_settings, mint_token):
    store = MemoryStore()
    store.set_access(
        TEST_ISSUER,
        "reader-finance",
        permissions=["documents:read"],
        read_collections=["finance", "legacy"],
        write_collections=[],
    )
    store.set_access(
        TEST_ISSUER,
        "writer-legal",
        permissions=["documents:write"],
        read_collections=[],
        write_collections=["legal"],
    )
    store.set_access(
        TEST_ISSUER,
        "admin-all",
        permissions=["documents:read", "documents:write"],
        read_collections=["finance", "legal", "legacy"],
        write_collections=["finance", "legal", "legacy"],
    )

    app = create_app()
    app.state.store = store
    app.state.embedder = embedder
    app.state.settings = test_settings
    app.state.jwt_validator = jwt_validator
    app.state.max_upload_bytes = 1024 * 1024
    client = TestClient(app)
    return client, store, mint_token


def _bearer(mint_token, subject: str, **kwargs) -> dict[str, str]:
    return {"Authorization": f"Bearer {mint_token(subject=subject, **kwargs)}"}


def test_missing_jwt_returns_401(app_client):
    client, store, _mint = app_client
    assert client.post("/query", json={"question": "risks"}).status_code == 401
    ingest = client.post(
        "/ingest",
        files={"file": ("x.pdf", pdf_with_pages(["text"]), "application/pdf")},
        data={"collection": "legacy"},
    )
    assert ingest.status_code == 401
    assert store.match_calls == []


def test_expired_jwt_returns_401(app_client, mint_token):
    client, _store, _mint = app_client
    headers = _bearer(mint_token, "admin-all", expired=True)
    assert client.post("/query", json={"question": "risks"}, headers=headers).status_code == 401


def test_invalid_audience_returns_401(app_client, jwt_keypair):
    client, _store, _mint = app_client
    private_pem, _ = jwt_keypair
    token = mint_access_token(private_pem, subject="admin-all", audience="wrong-audience")
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/query", json={"question": "risks"}, headers=headers).status_code == 401


def test_valid_jwt_without_grants_returns_403(app_client, mint_token):
    client, _store, _mint = app_client
    headers = _bearer(mint_token, "unknown-user")
    assert client.post("/query", json={"question": "risks"}, headers=headers).status_code == 403


def test_reader_cannot_ingest(app_client, mint_token):
    client, store, _mint = app_client
    headers = _bearer(mint_token, "reader-finance")
    response = client.post(
        "/ingest",
        files={"file": ("doc.pdf", pdf_with_pages(["finance doc"]), "application/pdf")},
        data={"collection": "finance"},
        headers=headers,
    )
    assert response.status_code == 403
    assert store.rows == []


def test_writer_without_read_cannot_query(app_client, mint_token):
    client, _store, _mint = app_client
    headers = _bearer(mint_token, "writer-legal")
    assert client.post("/query", json={"question": "contract"}, headers=headers).status_code == 403


def test_authorized_query_only_returns_allowed_collections(app_client, mint_token, embedder):
    client, store, _mint = app_client
    store.vector_rows = [
        {
            "id": 1,
            "collection": "finance",
            "content": "finance",
            "document_name": "f.pdf",
            "page": 1,
            "metadata": {},
            "similarity": 0.9,
            "result_rank": 1,
        },
        {
            "id": 2,
            "collection": "legal",
            "content": "legal",
            "document_name": "l.pdf",
            "page": 1,
            "metadata": {},
            "similarity": 0.8,
            "result_rank": 2,
        },
    ]
    store.lexical_rows = [
        {
            "id": 1,
            "collection": "finance",
            "content": "finance",
            "document_name": "f.pdf",
            "page": 1,
            "metadata": {},
            "lexical_score": 0.7,
            "result_rank": 1,
        },
        {
            "id": 2,
            "collection": "legal",
            "content": "legal",
            "document_name": "l.pdf",
            "page": 1,
            "metadata": {},
            "lexical_score": 0.6,
            "result_rank": 2,
        },
    ]
    headers = _bearer(mint_token, "reader-finance")
    response = client.post("/query", json={"question": "docs", "top_k": 5}, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["filter_collections"] == ["finance", "legacy"]
    assert {hit["id"] for hit in body["vector"]} == {1}
    assert {hit["id"] for hit in body["hybrid"]} == {1}
    assert store.match_calls[-1]["filter_collections"] == ["finance", "legacy"]


def test_query_collection_spoofing_returns_403(app_client, mint_token):
    client, _store, _mint = app_client
    headers = _bearer(mint_token, "reader-finance")
    response = client.post(
        "/query",
        json={"question": "docs", "collections": ["legal"]},
        headers=headers,
    )
    assert response.status_code == 403


def test_unauthorized_ingest_collection_rejected(app_client, mint_token):
    client, store, _mint = app_client
    headers = _bearer(mint_token, "writer-legal")
    response = client.post(
        "/ingest",
        files={"file": ("doc.pdf", pdf_with_pages(["legal text"]), "application/pdf")},
        data={"collection": "finance"},
        headers=headers,
    )
    assert response.status_code == 403
    assert store.rows == []


def test_health_stays_unauthenticated(app_client):
    client, _store, _mint = app_client
    assert client.get("/health").status_code == 200


def test_duplicate_hash_is_scoped_per_collection(embedder):
    store = MemoryStore()
    pdf = pdf_with_pages(["Same hash text."])
    ingest_pdf(pdf, "a.pdf", store, embedder, "finance")
    ingest_pdf(pdf, "b.pdf", store, embedder, "legal")
    assert len(store.rows) == 2
    from retrieval_lab.errors import IngestError

    with pytest.raises(IngestError) as exc:
        ingest_pdf(pdf, "c.pdf", store, embedder, "finance")
    assert exc.value.status_code == 409


def test_retrieval_never_calls_legacy_unscoped_rpc(embedder):
    store = MemoryStore()
    store.vector_rows = [
        {
            "id": 9,
            "collection": "finance",
            "content": "hidden",
            "document_name": "f.pdf",
            "page": 1,
            "metadata": {},
            "similarity": 0.99,
            "result_rank": 1,
        }
    ]
    with pytest.raises(Exception):
        store.try_legacy_unscoped_match([0.1, 0.2, 0.3], 5)
    result = query_retrieval(
        "question",
        store,
        embedder,
        filter_collections=["legal"],
        top_k=5,
    )
    assert result["vector"] == []
    assert store.match_calls[-1]["filter_collections"] == ["legal"]


def test_store_refuses_empty_collection_filter(embedder):
    store = MemoryStore()
    store.vector_rows = [{"id": 1, "collection": "finance", "similarity": 1, "result_rank": 1}]
    assert store.match_documents([0.1], 5, []) == []
    assert store.search_fts("x", 5, []) == []


def test_permissions_derived_from_database_not_token_claims(embedder, jwt_keypair):
    private_pem, _ = jwt_keypair
    store = MemoryStore()
    store.set_access(
        TEST_ISSUER,
        "real-reader",
        permissions=["documents:read"],
        read_collections=["legacy"],
        write_collections=[],
    )
    token = mint_access_token(private_pem, subject="real-reader")
    principal = Principal(issuer=TEST_ISSUER, subject="real-reader")
    access = resolve_access(store, principal)
    assert access.has_permission("documents:read")
    decoded = jwt.decode(
        token,
        options={"verify_signature": False},
        algorithms=["RS256"],
    )
    decoded["permissions"] = ["documents:write"]
    spoofed = Principal(issuer=TEST_ISSUER, subject=decoded["sub"])
    spoofed_access = resolve_access(store, spoofed)
    assert not spoofed_access.has_permission("documents:write")
    filters = resolve_query_collections(access, None)
    assert filters == ["legacy"]
