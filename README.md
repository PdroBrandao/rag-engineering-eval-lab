# Retrieval Lab

Local API for ingesting PDFs and comparing vector, lexical, and hybrid retrieval. The service does not generate answers.

## Requirements

- Python 3.11 and the `.venv` virtualenv
- `.env` variables: `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`
- [sql/001_retrieval.sql](sql/001_retrieval.sql), already applied to the configured Supabase project

`.env` stays out of Git. The Supabase key is the server secret key. It does not appear in code, logs, Postman, or this documentation.

## Start the API

From the project directory:

```bash
source .venv/bin/activate
uvicorn retrieval_lab.main:app --host 127.0.0.1 --port 8000
```

Equivalent command:

```bash
.venv/bin/python -m retrieval_lab.main
```

Both listen only on `127.0.0.1:8000`. Interactive documentation is at `http://127.0.0.1:8000/docs`.

## Endpoints

`GET /health` checks the process and runs `select` on `documents`. It does not call OpenAI.

`POST /ingest` accepts `multipart/form-data`:

- `file`: PDF, up to 20 MB
- `chunk_size`: default 500 tokens
- `chunk_overlap`: default 50 tokens
- `replace`: `true` returns 400. v1 does not delete chunks.

Uploading the same PDF bytes again returns 409 and does not generate embeddings again.

`POST /query`:

```json
{
  "question": "What are the main risks of LLM-as-a-Judge?",
  "top_k": 5
}
```

The response includes the vector and lexical lists used for fusion, the hybrid list cut to `top_k`, latencies, and the documents found. `similarity`, `lexical_score`, and `rrf_score` are not on the same scale.

Optional `candidate_k` sets the size of each list before fusion. The default is `max(top_k, 20)`, capped at 200. It must be greater than or equal to `top_k`. The default `rrf_k` is 60.

## Evaluation

After ingesting a document, replace the ids in [eval/dataset.example.json](eval/dataset.example.json) with the relevant `chunk_ids` and run:

```bash
.venv/bin/python -m retrieval_lab.evaluate eval/dataset.example.json
```

The command prints Recall@K, Precision@K, and MRR for all three strategies. It calls the database and OpenAI. The example file does not yet contain labeled ids.

## Tests

```bash
.venv/bin/pytest
```

The tests do not call Supabase or OpenAI. On the first run, `tiktoken` may download the `cl100k_base` encoding into the local cache.

## Postman

Import [postman/retrieval-lab.postman_collection.json](postman/retrieval-lab.postman_collection.json). The `base_url` variable is `http://127.0.0.1:8000`.

1. Send **Health**.
2. In **Ingest PDF**, choose a PDF for the `file` field and send it. Leave `replace` as `false`.
3. Copy a passage from the ingested text, or a question about it, into **Query**.
4. Compare `vector`, `lexical`, and `hybrid` by rank. Do not compare the scores with each other.

Sending the same file again should return 409. A file that is not a PDF should return 400.

## Limitations of this version

- `replace=true` is disabled. Reprocessing the same PDF at another chunk size would require deleting the current chunks, and that swap is not atomic.
- Hybrid search only sees the first `candidate_k` results from each strategy.
- Chunking does not cross page boundaries.
- The API has no authentication. Keep it on `127.0.0.1`.
