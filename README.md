# RAG Engineering & Evaluation Lab

Reusable engineering lab for building and measuring retrieval-augmented generation. The retrieval service ingests PDFs and searches them. The evaluation framework measures whether the right evidence was retrieved and, separately, whether the generated answer is correct.

Suggested repository: `rag-engineering-eval-lab`.

## Capabilities

**Retrieval API.** Ingest a PDF, chunk it with `cl100k_base`, embed it with `text-embedding-3-small` (1536 dimensions), and store the chunks in PostgreSQL (Supabase, pgvector). Query with vector search, lexical search, or hybrid fusion (reciprocal rank fusion). `POST /ingest` and `POST /query` require a JWT. Collection access is RBAC loaded from PostgreSQL, not from token claims. `GET /health` is unauthenticated. The API itself does not generate answers.

**Evaluation framework.** Compare 27 retrieval configurations (chunk size 256, 512, or 1024; overlap 26, 51, or 102; `top_k` 3, 5, or 10; strategy vector, lexical, or hybrid). Ground truth carries required evidence spans. A Development set is used to choose a configuration. A Frozen set is held out. Scoring covers confirmed evidence, an LLM-as-a-Judge for answer correctness, paired differences, Wilson intervals, latency, and cost. Metrics contract v0.4 is the active definition. The manifest policy records every verdict as `inconclusive`.

## Architecture

```text
PDF → extract pages → token chunks → embeddings → documents
                                              ↓
question → vector RPC ──┐
         → lexical RPC ─┴→ RRF → top_k

evaluation: ground truth → evidence matcher → retrieval metrics
                         → generator → LLM judge → answer metrics
```

Chunking does not cross page boundaries. Hybrid fusion sees only the first `candidate_k` rows from each list (default `max(top_k, 20)`, cap 200). `similarity`, `lexical_score`, and `rrf_score` are not on the same scale.

Details: [ARCHITECTURE.md](ARCHITECTURE.md). Authentication and the limits of application-enforced isolation: [SECURITY_ARCHITECTURE.md](SECURITY_ARCHITECTURE.md).

## Features

- PDF ingestion with duplicate detection by content hash, per collection
- Vector, English full-text, and Portuguese OR lexical search (`sql/004_lexical_pt_or.sql`)
- Hybrid RRF (`rrf_k` default 60)
- JWT validation (JWKS or a local public key) and collection RBAC
- Experiment runner with checkpoint and resume
- Evidence matching, including `needs_review` for ambiguous tables
- Answer grading: span containment, numeric tolerance, and LLM-as-a-Judge
- Uncertainty, latency (measured shared execution vs. standalone estimate), and cost split (generation, embedding, judge)
- Published Development and Frozen artifacts for the case study below

## Case study: Itaú 2025

The Itaú Unibanco Integrated Annual Report 2025 is one corpus used to exercise the lab. It is not the product. Question text and evidence quotes stay in Portuguese because that is the source language.

Canonical inputs:

- Ground truth v1.3: [eval/ground_truth/itau_50_v1.3.jsonl](eval/ground_truth/itau_50_v1.3.jsonl), [development_v1.3.jsonl](eval/ground_truth/development_v1.3.jsonl), [frozen_benchmark_v1.3.jsonl](eval/ground_truth/frozen_benchmark_v1.3.jsonl)
- Split record: [eval/ground_truth/dataset_split_v1.3.json](eval/ground_truth/dataset_split_v1.3.json)
- Metrics: [eval/metrics_contract.v0.4.md](eval/metrics_contract.v0.4.md)
- Parameters and verdict policy: [eval/experiment_manifest.v0.3.json](eval/experiment_manifest.v0.3.json)

Published runs. Directory names are the public layout. The `run_id` stored inside each run is unchanged.

| Public name | Directory | What it contains |
| --- | --- | --- |
| Development Run | `runs/development` | 30 questions, all 27 configurations. Recorded `run_id`: `development-v13-ptor`. |
| Frozen Benchmark | `runs/frozen` | 20 held-out questions, baseline and the selected configuration. Recorded `run_id`: `frozen-v13-ptor`. |

Confirmed evidence is retrieval quality: the matcher found every required span. Answer correctness is a separate judgment of the generated text. An answer can be correct while the required evidence was not confirmed (`correct_unsupported`).

| | Development Run | Frozen Benchmark |
| --- | --- | --- |
| Configurations | 27 | baseline `cs512_top5_hybrid` and `cs512_top10_lexical` |
| Correct answers | selection input; see the report | 18/20 for both configurations |
| Confirmed evidence | selection input; see the report | 15/20 lexical, 13/20 baseline |
| Statistical verdict | `inconclusive` | `inconclusive` |

The Frozen Wilson intervals overlap on both answer correctness and confirmed evidence. There is no statistically proven winner. Full method, paired differences, costs, and limitations: [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md). Run layout: [runs/README.md](runs/README.md).

## Prerequisites

- Python 3.11
- A PostgreSQL database with `pgvector` (Supabase is what this lab uses)
- An OpenAI API key for embeddings and, only if you re-run generation, for chat and judging
- An identity provider that signs RS256 JWTs, or a local key pair kept outside Git

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`. Do not commit it. Apply these SQL files in order on the database, manually:

1. [sql/001_retrieval.sql](sql/001_retrieval.sql)
2. [sql/002_auth_rbac.sql](sql/002_auth_rbac.sql)
3. [sql/003_experiment_index_sets.sql](sql/003_experiment_index_sets.sql)
4. [sql/004_lexical_pt_or.sql](sql/004_lexical_pt_or.sql)

[sql/002_seed_dev_identities.example.sql](sql/002_seed_dev_identities.example.sql) is an optional local identity. Edit the issuer and subject so they match the token you actually send.

For local JWT checks, set `LOCAL_JWT_PUBLIC_KEY_FILE` to a public key. Keep the private key under `.local-auth/` or another ignored path. That directory is gitignored.

Obtain the Itaú Unibanco *Relatório Anual Integrado 2025* from the company's own investor-relations publications. Place it only on your machine, at `data/Relatório Anual Integrado 2025.pdf`. `data/` is gitignored. The PDF is a third-party document and is not part of this repository. One dataset regression test reads that local file; the rest of the suite does not.

## API usage

```bash
.venv/bin/uvicorn retrieval_lab.main:app --host 127.0.0.1 --port 8000
```

Interactive docs: `http://127.0.0.1:8000/docs`. The process listens on localhost only.

`GET /health` checks the process and a `select` against `documents`. It does not call OpenAI.

`POST /ingest` is `multipart/form-data`: `file` (PDF, up to 20 MB), `collection`, optional `chunk_size` (default 500) and `chunk_overlap` (default 50). `replace=true` returns 400. The caller needs `documents:write` on that collection. A duplicate file in the same collection returns 409.

`POST /query` requires `documents:read`:

```json
{
  "question": "What are the main risks of LLM-as-a-Judge?",
  "top_k": 5,
  "collections": ["legacy"]
}
```

`collections` is optional. Omitting it searches every collection the caller may read. Naming a forbidden collection returns 403. The response includes the vector list, the lexical list, the hybrid list cut to `top_k`, and latencies.

A Postman collection is at [postman/retrieval-lab.postman_collection.json](postman/retrieval-lab.postman_collection.json). Set `base_url` to `http://127.0.0.1:8000` and supply `access_token`. Do not store real tokens in the collection.

The older labeled-id evaluator, which does not use the v1.3 ground truth, is:

```bash
.venv/bin/python -m retrieval_lab.evaluate eval/dataset.example.json
```

That command calls the database and OpenAI. The example file has no labeled chunk ids yet. It is a separate, older path from the v1.3 ground truth.

## Evaluation workflow

Ingest a corpus into the three index sets (this calls the embeddings API):

```bash
.venv/bin/python -m retrieval_lab.experiments.ingest_corpus \
  --pdf "data/Relatório Anual Integrado 2025.pdf" \
  --collection itau_2025
```

Run retrieval only, or retrieval plus generation, against a ground-truth file. Generation and judging are paid calls. The published results are already in `runs/development` and `runs/frozen`. A new run writes to `runs/<run-id>` unless you pass `--out-dir`. Use a new run id so you do not overwrite those directories.

```bash
.venv/bin/python -m retrieval_lab.experiments.runner \
  --run-id my-run \
  --manifest eval/experiment_manifest.v0.3.json \
  --ground-truth eval/ground_truth/development_v1.3.jsonl \
  --layers retrieval \
  --lexical-version portuguese_or \
  --read-collections itau_2025
```

`--layers retrieval,end_to_end` adds generation and judging. `--config-ids` limits the run to named configurations. `--question-ids` limits the questions. A completed run can be resumed from its checkpoint.

## Tests

```bash
.venv/bin/pytest
```

The suite does not call Supabase or OpenAI. On a cold machine, `tiktoken` may download the `cl100k_base` encoding into the local cache. The Q041 regression reads the local annual-report PDF described above.

## Publication notes

Do not commit `.env`, `.local-auth/`, `data/`, virtualenvs, caches, zip archives, or local credential helpers. Historical runs under `runs/archive/` stay on disk for reproducibility and are not part of the published tree.
