# Medium-articles RAG — Benchmark Report

Questions evaluated: **300** | Judged: **300**

Hyperparameters at run time (read from `rag/config.py`):
- `chunk_size`: 512
- `overlap_ratio`: 0.2
- `top_k`: 10

---

## Retrieval quality (per category)

| Cat | N | hit@1 | hit@3 | hit@10 | MRR | distinct_articles | n_gt_hits | distinct_gt@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 75 | 62.7% | 74.7% | 86.7% | 0.703 | 10.0 | 0.87 | 0.75 |
| C2 | 75 | 53.3% | 73.3% | 88.0% | 0.649 | 10.0 | 4.52 | 1.48 |
| C3 | 75 | 84.0% | 90.7% | 96.0% | 0.880 | 10.0 | 0.96 | 0.91 |
| C4 | 75 | 61.3% | 78.7% | 86.7% | 0.708 | 10.0 | 4.96 | 1.65 |

Notes:
- **hit@k**: fraction of questions whose ground-truth article appears in top-k.
- **MRR**: mean reciprocal rank of first ground-truth match (0 if none in top-k).
- **distinct_articles**: avg # of distinct articles in returned top-k (max 10).
- **n_gt_hits**: avg count of ground-truth articles found anywhere in top-k.
- **distinct_gt@3**: avg # distinct on-topic articles in top-3 — the SPEC §3 "3 distinct articles" target for C2.

---

## Latency (per category, ms)

| Cat | embed p50 | embed p95 | query p50 | query p95 | generate p50 | generate p95 | total p50 | total p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 524 | 2572 | 204 | 236 | 6463 | 14413 | 7655 | 15379 |
| C2 | 471 | 2538 | 203 | 224 | 5131 | 17293 | 6180 | 17766 |
| C3 | 519 | 2410 | 205 | 232 | 7322 | 10584 | 8500 | 12806 |
| C4 | 579 | 2722 | 199 | 217 | 10573 | 14082 | 11794 | 15452 |

---

## Judge scores (per category)

| Cat | N | faithfulness | correctness | IDK rate |
|---|---:|---:|---:|---:|
| C1 | 75 | 2.67 / 3 | 2.39 / 3 | 9.3% |
| C2 | 75 | 2.85 / 3 | 2.52 / 3 | 5.3% |
| C3 | 75 | 2.59 / 3 | 2.96 / 3 | 4.0% |
| C4 | 75 | 2.64 / 3 | 2.88 / 3 | 0.0% |

Score distributions:

| Cat | f=0 | f=1 | f=2 | f=3 | c=0 | c=1 | c=2 | c=3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 0 | 4 | 17 | 54 | 8 | 7 | 8 | 52 |
| C2 | 1 | 2 | 4 | 68 | 6 | 7 | 4 | 58 |
| C3 | 0 | 2 | 27 | 46 | 0 | 0 | 3 | 72 |
| C4 | 0 | 5 | 17 | 53 | 0 | 2 | 5 | 68 |

---

## Improvement priorities (auto-derived)

- **C2 multi-result diversity is below target.** Avg distinct on-topic articles in top-3 = 1.48 (target ≥ 2.5). Consider over-fetch ratio or MMR-style reranking to spread top-k across articles.
- **C1 generation p95 = 14413 ms.** Long-tail latency from the chat model. Consider streaming for UX, or reducing the per-chunk context length.
