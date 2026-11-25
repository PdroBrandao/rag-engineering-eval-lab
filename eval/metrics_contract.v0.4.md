# Metrics contract v0.4

This document defines the v1.3 metrics. It does not change the 27 configurations, the Development/Frozen split, or the v1.1 ground-truth cases that v1.3 inherited. Question text stays in Portuguese.

## What changed from v0.3, and why

| v0.3 | v0.4 |
| --- | --- |
| `ndcg_at_k` could exceed 1 | `ndcg_at_k` is normalized inside the retrieved list and stays in `[0, 1]` |
| The `mrr` field was the reciprocal rank of the first prefix that confirms all required evidence | The field is `reciprocal_full_evidence_rank`. Conventional MRR was not implemented |
| `strategy_total_ms` omitted a cached embedding and was presented as a measurement | `measured_execution_latency_ms` is the shared-run clock. `estimated_standalone_latency_ms` adds the cached embedding back for vector and hybrid only |
| Judge cost was missing from the total | `generation_cost_usd`, `embedding_cost_usd`, `judge_cost_usd`, and `evaluation_total_cost_usd` are separate |

Historical runs kept locally at `runs/archive/full-q001-q010-v01` and `runs/archive/full-q001-q010-ptor-v02` keep the v0.3 names and values. They are not rewritten and are not part of the published tree. Cross-contract nDCG deltas stay blank.

## nDCG@K

Grades are unchanged:

- 3: the chunk alone confirms every required span;
- 2: full textual match of at least one span, but not of all of them, and the question is not `table_ambiguous`;
- 1: `table_ambiguous` review, or a partial fragment;
- 0: no overlap with the required evidence.

`DCG = Σ grade_i / log2(i + 1)`, with `i` starting at 1.

`IDCG` is the DCG of the same observed grades sorted descending. If `IDCG = 0`, nDCG is 0. Otherwise `nDCG = DCG / IDCG`.

This normalization answers whether the retrieved list is in its own best order. It does not compare the list with an ideal corpus ranking. A weak list already in its best internal order scores 1.

The v0.3 IDCG used `DCG([3, 0, 0, ...])`. That denominator could be smaller than the observed DCG, so nDCG could exceed 1. Those values are not comparable with v0.4.

## Full-evidence reciprocal rank

`reciprocal_full_evidence_rank = 1 / r`, where `r` is the rank of the first prefix that confirms all required evidence. The value is 0 when no prefix does.

The historical `mrr` column stored this quantity under the old name. A comparison must label it `legacy_field_name_for_reciprocal_full_evidence_rank`. It is not the conventional mean of reciprocal ranks over relevant items.

Conventional MRR was left out. The contract does not choose among four relevance definitions: confirmed evidence, any textual overlap, `needs_review`, or a chunk that carries part of a multi-span answer. Picking one would change the meaning without a recorded decision.

## Latency

`measured_execution_latency_ms` is the time the shared run actually spent. A cached embedding is visible in the metadata and is not added again.

`estimated_standalone_latency_ms` adds that cached embedding back for vector and hybrid retrieval. Lexical retrieval does not embed, so its two clocks match.

`strategy_total_ms` remains an alias of the measured execution time.

End-to-end measured latency is measured retrieval plus measured generation. The standalone end-to-end estimate is the standalone retrieval estimate plus measured generation. Judge latency belongs to evaluation and is excluded from both.

The standalone estimate reuses one shared call. It is not a repeated benchmark of an isolated configuration. The `promising` gate still uses the measured shared-execution end-to-end p95, labeled `measured_shared_execution_excludes_cached_embed_not_a_production_benchmark`.

## Cost

Per question and configuration:

- `generation_cost_usd`: the chat call. Missing token counts stay unknown, not zero.
- `embedding_cost_usd`: counted once per question and index set. It is not summed across configurations.
- `judge_cost_usd`: the evaluation call. A rubric that does not use the judge is `not_applicable` and costs 0. A judge rubric that did not run is `not_evaluated` and stays unknown.
- `rag_operational_cost_usd`: generation plus embedding.
- `evaluation_total_cost_usd`: operational cost plus judge cost. Any unknown part makes the total unknown.

`api_cost_usd_chat_estimated` remains an alias of the generation cost. Prices come from `experiment_manifest.v0.3`, field `generate.pricing_usd_per_1m`.

## What this contract does not change

Recall@K, Evidence Coverage, the grade scale, the `table_ambiguous` rule, the `promising` thresholds, and the `inconclusive` verdict policy are unchanged. Failed, judge-uncertain, and unsupported-rubric rows are not scored as zero.
