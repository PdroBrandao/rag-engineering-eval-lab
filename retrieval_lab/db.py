import logging
from typing import Any, Protocol

from postgrest.exceptions import APIError
from supabase import Client, create_client

from retrieval_lab.config import Settings
from retrieval_lab.errors import DatabaseError

logger = logging.getLogger(__name__)


class DocumentStore(Protocol):
    def ping(self) -> None: ...

    def resolve_lab_access(self, issuer: str, subject: str) -> dict[str, Any]: ...

    def find_by_hash(
        self,
        content_sha256: str,
        collection: str,
        index_set_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def insert_chunks(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def match_documents(
        self,
        query_embedding: list[float],
        match_count: int,
        filter_collections: list[str],
        filter_index_set_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def search_fts(
        self,
        query_text: str,
        match_count: int,
        filter_collections: list[str],
        filter_index_set_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def search_fts_pt_or(
        self,
        query_text: str,
        match_count: int,
        filter_collections: list[str],
        filter_index_set_id: str | None = None,
    ) -> list[dict[str, Any]]: ...


def create_supabase(settings: Settings) -> Client:
    return create_client(settings.supabase_url, settings.supabase_key)


class SupabaseDocumentStore:
    def __init__(self, client: Client) -> None:
        self._client = client

    def ping(self) -> None:
        self._execute(
            self._client.table("documents").select("id").limit(1),
            "ping",
        )

    def resolve_lab_access(self, issuer: str, subject: str) -> dict[str, Any]:
        response = self._execute(
            self._client.rpc(
                "resolve_lab_access",
                {"p_issuer": issuer, "p_subject": subject},
            ),
            "resolve_lab_access",
        )
        data = response.data
        if isinstance(data, dict):
            return data
        return {
            "permissions": [],
            "read_collections": [],
            "write_collections": [],
        }

    def find_by_hash(
        self,
        content_sha256: str,
        collection: str,
        index_set_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = (
            self._client.table("documents")
            .select("id,document_name,collection")
            .eq("metadata->>content_sha256", content_sha256)
            .eq("collection", collection)
        )
        if index_set_id:
            query = query.eq("index_set_id", index_set_id)
        response = self._execute(query.limit(1), "find_by_hash")
        return list(response.data or [])

    def insert_chunks(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        response = self._execute(
            self._client.table("documents").insert(rows),
            "insert_chunks",
        )
        return list(response.data or [])

    def match_documents(
        self,
        query_embedding: list[float],
        match_count: int,
        filter_collections: list[str],
        filter_index_set_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not filter_collections:
            return []
        payload = {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "filter_collections": filter_collections,
        }
        if filter_index_set_id:
            payload["filter_index_set_id"] = filter_index_set_id
        response = self._execute(
            self._client.rpc("match_documents", payload),
            "match_documents",
        )
        return list(response.data or [])

    def search_fts(
        self,
        query_text: str,
        match_count: int,
        filter_collections: list[str],
        filter_index_set_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not filter_collections:
            return []
        payload = {
            "query_text": query_text,
            "match_count": match_count,
            "filter_collections": filter_collections,
        }
        if filter_index_set_id:
            payload["filter_index_set_id"] = filter_index_set_id
        response = self._execute(
            self._client.rpc("search_documents_fts", payload),
            "search_fts",
        )
        return list(response.data or [])

    def search_fts_pt_or(
        self,
        query_text: str,
        match_count: int,
        filter_collections: list[str],
        filter_index_set_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not filter_collections:
            return []
        payload = {
            "query_text": query_text,
            "match_count": match_count,
            "filter_collections": filter_collections,
        }
        if filter_index_set_id:
            payload["filter_index_set_id"] = filter_index_set_id
        response = self._execute(
            self._client.rpc("search_documents_fts_pt_or", payload),
            "search_fts_pt_or",
        )
        return list(response.data or [])

    def _execute(self, request: Any, operation: str) -> Any:
        try:
            return request.execute()
        except APIError as exc:
            if operation == "insert_chunks" and exc.code == "23505":
                raise
            logger.error("database request failed operation=%s code=%s", operation, exc.code)
            if str(exc.code or "") == "42601":
                raise DatabaseError(
                    "Database request failed.",
                    status_code=400,
                    code="42601",
                ) from exc
            raise DatabaseError() from exc
        except DatabaseError:
            raise
        except Exception as exc:
            logger.error("database request failed operation=%s", operation)
            raise DatabaseError() from exc
