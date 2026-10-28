# Future experiments

This lab does not yet have quality or latency measurements. The values below are conditions to measure, not results.

Each experiment needs a fixed set of questions with hand-labeled `relevant_chunk_ids`, the same PDF, and the same parameters outside the factor being changed. Run:

```bash
.venv/bin/python -m retrieval_lab.evaluate path/to/dataset.json
```

The metrics are Recall@K, Precision@K, and MRR, calculated separately for vector, lexical, and hybrid retrieval. `similarity`, `ts_rank`, and the RRF score are not compared across strategies.

## Retrieval strategy

Keep 500-token chunks, overlap 50, `top_k` 5, `candidate_k` 20, and `rrf_k` 60. Compare the three columns of the report on the same dataset.

Questions that separate the behaviors:

- rare terms that appear literally in the PDF
- paraphrases that do not repeat those terms
- questions made mostly of English stop words, because `websearch_to_tsquery('english', ...)` removes those words

Also record how many rows each list returned. An empty lexical list changes the hybrid result, because RRF only adds a score from lists that contain the chunk.

## Chunk size and overlap

v1 cannot re-ingest the same bytes at a new chunk size: `replace=true` is disabled so the endpoint does not delete chunks. For this experiment, each condition needs a distinct file, or a copy of the corpus loaded into a separate table. Do not reuse the same `chunk_ids` across conditions.

Initial conditions:

- 200 tokens, overlap 0
- 500 tokens, overlap 50
- 800 tokens, overlap 100

Current chunking does not cross page boundaries. A later experiment can compare that split with a window over the whole document, storing the chunk's starting page. That changes the meaning of the `page` column.

## Fusion depth

On the same corpus and the same labels, vary only `candidate_k`: 5, 20, and 50. Hybrid results stay cut to `top_k` 5. The question to measure is whether a chunk outside the top 5 of one list, but inside the top 20 of both, changes Recall@5 and MRR. Measure the change. Do not assume its direction.

Vary `rrf_k` over 10, 60, and 100 with `candidate_k` fixed. The report should record the constant used, because the RRF score changes with the constant while the ranking metrics may stay the same.

## Latency

`POST /query` already separates `embed_ms`, `vector_ms`, `lexical_ms`, `fusion_ms`, and `hybrid_ms`. A later comparison should repeat the same question and record the median and maximum. One sample does not support a conclusion. `hybrid_ms` includes both searches run in sequence and does not include the embedding call.

Use `EXPLAIN` to see whether the vector query uses the cosine HNSW index and whether the lexical query uses the GIN index. The planner can choose a sequential scan while the table is small. That is an observation about the plan, not a performance improvement.
