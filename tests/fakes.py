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
        self.access_by_principal: dict[tuple[str, str], dict[str, Any]] = {}
        self.match_calls: list[dict[str, Any]] = []
        self.fts_calls: list[dict[str, Any]] = []
        self.legacy_rpc_blocked = True

    def ping(self) -> None:
        if self.fail_ping:
            raise DatabaseError()

    def set_access(
        self,
        issuer: str,
        subject: str,
        *,
        permissions: list[str],
        read_collections: list[str],
        write_collections: list[str],
    ) -> None:
        self.access_by_principal[(issuer, subject)] = {
            "permissions": permissions,
            "read_collections": read_collections,
            "write_collections": write_collections,
        }

    def resolve_lab_access(self, issuer: str, subject: str) -> dict[str, Any]:
        return self.access_by_principal.get(
            (issuer, subject),
            {
                "permissions": [],
                "read_collections": [],
                "write_collections": [],
            },
        )

    def find_by_hash(self, content_sha256: str, collection: str) -> list[dict[str, Any]]:
        for row in self.rows:
            if (
                row.get("collection") == collection
                and row["metadata"].get("content_sha256") == content_sha256
            ):
                return [
                    {
                        "id": row["id"],
                        "document_name": row["document_name"],
                        "collection": row.get("collection"),
                    }
                ]
        return []

    def insert_chunks(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.fail_insert:
            raise DatabaseError()
        inserted = []
        for row in rows:
            for existing in self.rows:
                same_document = (
                    existing.get("collection") == row.get("collection")
                    and existing["metadata"]["content_sha256"]
                    == row["metadata"]["content_sha256"]
                    and existing["chunk_index"] == row["chunk_index"]
                )
                if same_document:
                    raise APIError({"code": "23505", "message": "duplicate key"})
            stored = dict(row)
            stored["id"] = len(self.rows) + 1
            self.rows.append(stored)
            inserted.append({"id": stored["id"]})
        return inserted

    def match_documents(
        self,
        query_embedding: list[float],
        match_count: int,
        filter_collections: list[str],
    ) -> list[dict[str, Any]]:
        self.match_calls.append(
            {
                "match_count": match_count,
                "filter_collections": list(filter_collections),
            }
        )
        if not filter_collections:
            return []
        allowed = set(filter_collections)
        ranked = [row for row in self.vector_rows if row.get("collection") in allowed]
        return ranked[:match_count]

    def search_fts(
        self,
        query_text: str,
        match_count: int,
        filter_collections: list[str],
    ) -> list[dict[str, Any]]:
        self.fts_calls.append(
            {
                "query_text": query_text,
                "match_count": match_count,
                "filter_collections": list(filter_collections),
            }
        )
        if not filter_collections:
            return []
        allowed = set(filter_collections)
        ranked = [row for row in self.lexical_rows if row.get("collection") in allowed]
        return ranked[:match_count]

    def try_legacy_unscoped_match(self, query_embedding: list[float], match_count: int):
        if self.legacy_rpc_blocked:
            raise DatabaseError("Legacy match_documents RPC signature is not available.")
        return self.vector_rows[:match_count]

    def try_legacy_unscoped_fts(self, query_text: str, match_count: int):
        if self.legacy_rpc_blocked:
            raise DatabaseError("Legacy search_documents_fts RPC signature is not available.")
        return self.lexical_rows[:match_count]
