import hashlib
from typing import Any

from retrieval_lab.chunking import chunk_pages, validate_chunk_params
from retrieval_lab.db import DocumentStore
from retrieval_lab.embeddings import Embedder
from retrieval_lab.errors import DatabaseError, IngestError
from retrieval_lab.pdf_extract import extract_pages, pdf_filename


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest_pdf(
    data: bytes,
    filename: str | None,
    store: DocumentStore,
    embedder: Embedder,
    collection: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    replace: bool = False,
    embedding_model: str = "text-embedding-3-small",
    max_upload_bytes: int = 20 * 1024 * 1024,
) -> dict[str, Any]:
    if replace:
        raise IngestError(
            400,
            "replace=true is disabled in v1. Replacement is not atomic, so this "
            "endpoint does not delete stored chunks. Omit replace. Uploading the "
            "same PDF bytes again returns 409.",
        )
    if len(data) > max_upload_bytes:
        raise IngestError(400, "PDF exceeds the 20 MB upload limit.")

    try:
        validate_chunk_params(chunk_size, chunk_overlap)
    except ValueError as exc:
        raise IngestError(400, str(exc)) from exc

    document_hash = content_sha256(data)
    existing = store.find_by_hash(document_hash, collection)
    if existing:
        stored_name = existing[0].get("document_name") or "the previous upload"
        raise IngestError(
            409,
            f"This PDF was already ingested as {stored_name}. replace=true is "
            "disabled, so the stored chunks were left unchanged.",
        )

    pages = extract_pages(data)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        raise IngestError(400, "The PDF does not contain extractable text.")

    vectors = embedder.embed_texts([chunk.content for chunk in chunks])
    if len(vectors) != len(chunks):
        raise IngestError(502, "Embedding response did not match the chunk count.")

    document_name = pdf_filename(filename)
    rows = [
        {
            "document_name": document_name,
            "collection": collection,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "metadata": {
                "content_sha256": document_hash,
                "embedding_model": embedding_model,
                "token_count": chunk.token_count,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    try:
        inserted = store.insert_chunks(rows)
    except DatabaseError:
        raise
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code == "23505":
            raise IngestError(
                409,
                "This PDF was already ingested. replace=true is disabled, so the "
                "stored chunks were left unchanged.",
            ) from exc
        raise DatabaseError() from exc

    chunk_ids = [row["id"] for row in inserted if "id" in row]
    if len(chunk_ids) != len(chunks):
        raise DatabaseError("Database insert did not return chunk ids.")

    return {
        "document_name": document_name,
        "content_sha256": document_hash,
        "collection": collection,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "chunk_ids": chunk_ids,
    }
