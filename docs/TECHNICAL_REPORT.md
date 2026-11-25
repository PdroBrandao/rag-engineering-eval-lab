# RAG Lab v1.3 — Technical Report

## Method

The corpus is the Itaú 2025 integrated annual report, collection `itau_2025`. The ground truth is v1.3: 50 questions split into Development (30) and Frozen (20) with seed `itau-50-v1`. Question text and evidence quotes stay in Portuguese, the language of the source.

Retrieval uses three chunk sizes (256, 512, and 1024 tokens), three `top_k` values (3, 5, and 10), and three strategies (vector, lexical, and hybrid). Lexical search and the lexical leg of hybrid search use `portuguese_or`. The vector leg is unchanged. Hybrid fusion is RRF with `k = 60`. Each question keeps 20 candidates per leg.

The baseline is `cs512_top5_hybrid`. Metrics follow contract v0.4: confirmed Recall@K, full-evidence reciprocal rank, nDCG@K normalized inside the retrieved list, and Evidence Coverage. The manifest policy sets every verdict to `inconclusive`. A `table_ambiguous` match is `needs_review` and is not a confirmed hit. The `promising` gate uses the shared-execution latency and does not count as a benchmark of an isolated configuration.

## Selection on Development

The Development Run is published at `runs/development`. Its recorded `run_id` remains `development-v13-ptor`. It ran all 27 configurations. The selected configuration is `cs512_top10_lexical`: confirmed evidence 0.833 and correct answers 0.900, against 0.733 and 0.767 for the baseline. The paired differences are +3 evidence and +4 answers, with no losses. `cs1024_top10_lexical` reaches 0.900 and 0.933, but it nearly doubles the generation cost and the context. The Wilson intervals overlap, so there is no statistical winner.

Four configurations were marked `promising`. The gate does not declare a winner. The manifest still records `inconclusive`.

The audit found a possible judge false positive on Q011. The original Development rows were preserved and were not recalculated.

## Frozen results

The Frozen Benchmark is published at `runs/frozen`. Its recorded `run_id` remains `frozen-v13-ptor`. It compared only the baseline and `cs512_top10_lexical`.

| Metric | Baseline `cs512_top5_hybrid` | `cs512_top10_lexical` |
| --- | ---: | ---: |
| Correct answers | 0.900 (18/20) | 0.900 (18/20) |
| Wilson 95% | 0.699–0.972 | 0.699–0.972 |
| Confirmed evidence | 0.650 (13/20) | 0.750 (15/20) |
| Wilson 95% | 0.433–0.819 | 0.531–0.888 |
| Correct and unsupported | 0.300 | 0.200 |
| Mean context, tokens | 2,339.85 | 4,776.55 |
| Generation cost | US$ 0.006746 | US$ 0.012864 |
| Measured end-to-end p95 | 6,634.713 ms | 3,897.222 ms |

Both configurations reach 90% correct answers. Confirmed evidence is 75% for lexical search and 65% for the baseline. The intervals overlap. The verdict stays `inconclusive`. The p95 is the shared-execution clock, not an isolated benchmark.

## Paired differences on Frozen

Lexical search gains confirmed evidence on Q034 and Q039, and loses none. On answers, it gains Q029 and loses Q020. The net answer difference is zero.

Q020 is an observed answer-verdict difference. It remains subject to human review of the table evidence: the baseline is `correct_unsupported`, lexical search is `incorrect`, and both rows are `needs_review`.

## Five human-review cases

Each configuration has the same five unconfirmed cases: Q017, Q019, Q020, Q026, and Q037. Q017, Q019, Q026, and Q037 are correct and unsupported on both configurations. Q020 is the only answer-verdict change, and it is still waiting on human review of the table evidence.

## Costs

| | Generation | Judge | Embedding | Operational | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development, 27 configurations | 0.342703 | 0.014317 | 0.000059 | 0.342762 | 0.357079 |
| Frozen, 2 configurations | 0.019611 | 0.000710 | 0.000013 | 0.019623 | 0.020333 |

Values are in US dollars. The embedding is shared by index set. The judge is an evaluation cost, not part of the operational RAG cost.

## Limitations

- The Frozen sample has 20 questions. The Wilson intervals overlap on both answer accuracy and confirmed evidence.
- The policy keeps every verdict `inconclusive`, including the four Development configurations marked `promising`.
- Five table or multi-span cases per configuration stay unconfirmed. Q020, the only answer-verdict change, is one of them.
- The Development audit recorded a possible judge false positive on Q011. The original rows were preserved and were not recalculated. Q011 is not in the Frozen set.
- Latency is the shared-execution clock. A cached embedding is visible, but it is not an isolated measurement of one configuration.
- Chunks of 256 tokens with overlap 26 miss spans longer than the overlap on Q012, Q015, and Q035. The chunker was left unchanged.

## Conclusion

On the untouched Frozen set, both configurations score 90% correct answers. Lexical search confirms 75% of the evidence and the baseline confirms 65%, with no evidence loss in the paired comparison. Q020 remains under human review of the table evidence. There is no statistical winner.
