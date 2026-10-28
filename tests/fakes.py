from typing import Any

from postgrest.exceptions import APIError

from retrieval_lab.errors import DatabaseError


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


class MemoryStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.vector_rows: list[dict[str, Any]] = []
        self.lexical_rows: list[dict[str, Any]] = []
        self.fail_ping = False
        self.fail_insert = False

    def ping(self) -> None:
        if self.fail_ping:
            raise DatabaseError()

    def find_by_hash(self, content_sha256: str) -> list[dict[str, Any]]:
        for row in self.rows:
            if row["metadata"].get("content_sha256") == content_sha256:
                return [{"id": row["id"], "document_name": row["document_name"]}]
        return []

    def insert_chunks(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.fail_insert:
            raise DatabaseError()
        inserted = []
        for row in rows:
            for existing in self.rows:
                same_document = (
                    existing["metadata"]["content_sha256"] == row["metadata"]["content_sha256"]
                    and existing["chunk_index"] == row["chunk_index"]
                )
                if same_document:
                    raise APIError({"code": "23505", "message": "duplicate key"})
            stored = dict(row)
            stored["id"] = len(self.rows) + 1
            self.rows.append(stored)
            inserted.append({"id": stored["id"]})
        return inserted

    def match_documents(self, query_embedding: list[float], match_count: int) -> list[dict[str, Any]]:
        return self.vector_rows[:match_count]

    def search_fts(self, query_text: str, match_count: int) -> list[dict[str, Any]]:
        return self.lexical_rows[:match_count]
