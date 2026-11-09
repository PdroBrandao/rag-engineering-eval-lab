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

    def find_by_hash(self, content_sha256: str, collection: str) -> list[dict[str, Any]]: ...

    def insert_chunks(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def match_documents(
        self,
        query_embedding: list[float],
        match_count: int,
        filter_collections: list[str],
    ) -> list[dict[str, Any]]: ...

    def search_fts(
        self,
        query_text: str,
        match_count: int,
        filter_collections: list[str],
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

    def find_by_hash(self, content_sha256: str, collection: str) -> list[dict[str, Any]]:
        response = self._execute(
            self._client.table("documents")
            .select("id,document_name,collection")
            .eq("metadata->>content_sha256", content_sha256)
            .eq("collection", collection)
            .limit(1),
            "find_by_hash",
        )
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
    ) -> list[dict[str, Any]]:
        if not filter_collections:
            return []
        response = self._execute(
            self._client.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": match_count,
                    "filter_collections": filter_collections,
                },
            ),
            "match_documents",
        )
        return list(response.data or [])

    def search_fts(
        self,
        query_text: str,
        match_count: int,
        filter_collections: list[str],
    ) -> list[dict[str, Any]]:
        if not filter_collections:
            return []
        response = self._execute(
            self._client.rpc(
                "search_documents_fts",
                {
                    "query_text": query_text,
                    "match_count": match_count,
                    "filter_collections": filter_collections,
                },
            ),
            "search_fts",
        )
        return list(response.data or [])

    def _execute(self, request: Any, operation: str) -> Any:
        try:
            return request.execute()
        except APIError as exc:
            if operation == "insert_chunks" and exc.code == "23505":
                raise
            logger.error("database request failed operation=%s code=%s", operation, exc.code)
            raise DatabaseError() from exc
        except DatabaseError:
            raise
        except Exception as exc:
            logger.error("database request failed operation=%s", operation)
            raise DatabaseError() from exc
