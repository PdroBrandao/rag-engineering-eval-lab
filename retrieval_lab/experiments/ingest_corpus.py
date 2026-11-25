import argparse
from pathlib import Path

from retrieval_lab.experiments.manifest import INDEX_SETS


def ingest_index_sets(
    data: bytes,
    filename: str,
    store,
    embedder,
    *,
    collection: str,
    document_id: str,
    embedding_model: str,
) -> list[dict]:
    from retrieval_lab.ingest import ingest_pdf

    results = []
    for spec in INDEX_SETS:
        results.append(
            ingest_pdf(
                data,
                filename,
                store,
                embedder,
                collection,
                chunk_size=spec["chunk_size"],
                chunk_overlap=spec["chunk_overlap"],
                embedding_model=embedding_model,
                index_set_id=spec["index_set_id"],
                document_id=document_id,
                max_upload_bytes=80 * 1024 * 1024,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest one PDF into the three experiment index sets.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--collection", default="itau_2025")
    parser.add_argument("--document-id", default="itau_annual_report_2025")
    args = parser.parse_args()
    from openai import OpenAI

    from retrieval_lab.config import load_settings
    from retrieval_lab.db import SupabaseDocumentStore, create_supabase
    from retrieval_lab.embeddings import OpenAIEmbedder

    settings = load_settings()
    store = SupabaseDocumentStore(create_supabase(settings))
    embedder = OpenAIEmbedder(OpenAI(api_key=settings.openai_api_key), settings.embedding_model)
    data = args.pdf.read_bytes()
    results = ingest_index_sets(
        data,
        args.pdf.name,
        store,
        embedder,
        collection=args.collection,
        document_id=args.document_id,
        embedding_model=settings.embedding_model,
    )
    for result in results:
        print(
            f"{result['index_set_id']}: {result['chunk_count']} chunks "
            f"in {result['collection']}"
        )


if __name__ == "__main__":
    main()
