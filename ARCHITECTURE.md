# Architecture

This document describes the code in this repository.

## Process

The API is the FastAPI app in [retrieval_lab/main.py](retrieval_lab/main.py). `uvicorn retrieval_lab.main:app --host 127.0.0.1 --port 8000` and `python -m retrieval_lab.main` listen on localhost. The service has no frontend, API authentication, agent, or answer generation.

[retrieval_lab/config.py](retrieval_lab/config.py) reads credentials from `.env` when the first live request needs Supabase or OpenAI. Tests inject fakes and do not load those variables.

## Table

The existing table is `public.documents`. The PostgREST OpenAPI spec showed these columns, and the table was empty when this code was written:

- `id` bigint, primary key
- `document_name` text
- `page` integer
- `chunk_index` integer
- `content` text
- `metadata` jsonb
- `embedding` `public.vector(1536)`
- `created_at` timestamptz, default `now()`
- `fts` tsvector

The insert in [retrieval_lab/db.py](retrieval_lab/db.py) sends the name, page, index, content, metadata, and embedding. The database fills `id`, `created_at`, and `fts`. RLS stays enabled. The API uses the server secret key, which bypasses RLS in Supabase. The search functions are `security invoker`, and `EXECUTE` is granted only to `service_role`.

The unique index, the optional `fts` trigger, and the functions `match_documents` and `search_documents_fts` are in [sql/001_retrieval.sql](sql/001_retrieval.sql). That migration has been applied. `fts` is a stored generated column, so the script did not add a trigger. An HNSW index using `vector_cosine_ops` was already present, and the migration checks that operator class before creating another index, so the existing cosine index was left in place. The GIN index on `fts` was also already present. The new object is the partial unique index on `(metadata->>'content_sha256', chunk_index)`. Until those functions exist, ingestion can insert rows, but `POST /query` cannot run `<=>` or `ts_rank` through PostgREST.

## Ingestion

`POST /ingest` calls `ingest_pdf` in [retrieval_lab/ingest.py](retrieval_lab/ingest.py).

1. `replace=true` returns 400 before any database read or embedding call. v1 does not delete rows. A replacement would be a `delete` followed by an `insert`; a failed insert would leave the previous chunks already removed.
2. Files larger than 20 MB return 400.
3. The SHA-256 of the PDF bytes is looked up in `metadata->>content_sha256`. A match returns 409 and the embedder is not called.
4. [retrieval_lab/pdf_extract.py](retrieval_lab/pdf_extract.py) reads each page with `pypdf`. An invalid or encrypted PDF returns 400. Pages with no text are skipped in the next step. If no page has text, the response is 400.
5. [retrieval_lab/chunking.py](retrieval_lab/chunking.py) tokenizes with `cl100k_base`, the encoding used by `text-embedding-3-small`. Splits stay inside a page, so overlap does not mix text from different pages. `chunk_index` increases across the whole document. The defaults are 500 tokens and an overlap of 50. Stored page numbers are 1-based.
6. [retrieval_lab/embeddings.py](retrieval_lab/embeddings.py) calls `text-embedding-3-small` in batches of 64 and reorders each batch by the index returned by the API.
7. Every chunk goes into one PostgREST `insert`. That request is one transaction: all rows are stored, or none are. Each row's metadata stores the hash, model, `token_count`, `chunk_size`, and `chunk_overlap`.

If two ingestions of the same file both pass the lookup, the unique index on `(metadata->>'content_sha256', chunk_index)` makes the second insert fail with `23505`, and the API returns 409. That index exists only after the migration.

`page_count` in the response is the number of PDF pages, including empty pages. `chunk_count` counts only chunks that contain text.

## Query

`POST /query` calls `query_retrieval` in [retrieval_lab/retrieval.py](retrieval_lab/retrieval.py).

1. The question is embedded. That call is `embed_ms`.
2. `match_documents` orders by cosine distance `<=>` and returns at most `candidate_k` rows. `similarity` is `1 - distance`. `result_rank` is 1 for the smallest distance. The `limit` runs before `row_number`, so the planner can use the cosine HNSW index when that index exists.
3. `search_documents_fts` filters with `fts @@ websearch_to_tsquery('english', question)` and orders by `ts_rank`. The `english` configuration matches the stemming configured for the column. Stop words from that configuration are removed from the query. `lexical_score` is `ts_rank`.
4. [retrieval_lab/rrf.py](retrieval_lab/rrf.py) adds `1 / (rrf_k + rank)` for each list that contains the chunk. A missing list adds zero. The default `rrf_k` is 60. Ties break by chunk id. The fused list is cut to `top_k`.

The default `candidate_k` is `max(top_k, 20)`, capped at 200. The `vector` and `lexical` lists in the response are at most that long. `hybrid` is at most `top_k`. Fusion does not see the rest of the table.

Score fields stay separate: `similarity`, `lexical_score`, and `rrf_score`. A hybrid hit also stores `vector_rank` and `lexical_rank`. Either rank is null when the chunk was absent from that list.

Measured latencies:

- `embed_ms`: the OpenAI call
- `vector_ms`: the vector RPC
- `lexical_ms`: the lexical RPC
- `fusion_ms`: the Python RRF step
- `hybrid_ms`: the two RPCs plus fusion, run in sequence

Lexical search does not need the embedding, but the current code runs the two RPCs one after the other. `hybrid_ms` does not include `embed_ms`.

`documents` lists the names and pages that appeared in either candidate list.

OpenAI failures become 502. Database failures become 503. The error body is one short sentence. A provider status code may be written to the log. Keys, payloads, and SQL text are not.

## Evaluation

[retrieval_lab/metrics.py](retrieval_lab/metrics.py) calculates:

- Precision@K: hits in the first K results, divided by K
- Recall@K: hits in the first K results, divided by the number of relevant ids. The value is 0 when there are no relevant ids
- MRR: `1 / position` of the first relevant id. The value is 0 when none appears

[retrieval_lab/evaluate.py](retrieval_lab/evaluate.py) calls the same query function as the API and prints all three strategies. The report shows the numbers. It does not name a winning strategy.

The dataset is JSON with `k`, `rrf_k`, and `cases[].relevant_chunk_ids`. Those ids are human labels taken from the `/ingest` response.

## What the tests cover

`pytest` uses `MemoryStore` and `FakeEmbedder`. It covers chunking, RRF, metrics, an invalid PDF, an empty PDF, a duplicate upload, `replace=true`, the `/query` response shape, and the absence of `DELETE` or `TRUNCATE` in the migration. It does not measure live latency or retrieval quality.
