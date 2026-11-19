# Evaluation: Subset strategies vs Plain RAG

Questions completed: **9** of 12 | retrieval k = 20 | clusters = 2 | subsets = 5 | strategies = chain, mmr, centroid, bridge

## 1. Overall results (all questions)

| System | Final acc. | Oracle | Subset hit | Concept recall | Best-subset cov. | Distinct chunks | Subset overlap | Avg sec |
|---|---|---|---|---|---|---|---|---|
| Speculative RAG [chain] | 66.7% | 66.7% | 77.8% | 77.8% | 92.6% | 11.7 | 0.14 | 217.8 |
| Speculative RAG [mmr] | 44.4% | 66.7% | 77.8% | 72.8% | 92.6% | 11.7 | 0.14 | 177.3 |
| Speculative RAG [centroid] | 44.4% | 44.4% | 77.8% | 72.8% | 92.6% | 11.7 | 0.14 | 176.3 |
| Speculative RAG [bridge] | 44.4% | 44.4% | 77.8% | 74.3% | 92.6% | 13.3 | 0.14 | 199.9 |
| Plain RAG (baseline) | 44.4% | - | - | 50.0% | - | - | - | 62.3 |

Retrieval MRR: **0.448**

## 2. Accuracy by difficulty

| Level | Speculative [chain] | Speculative [mmr] | Speculative [centroid] | Speculative [bridge] | Plain RAG |
|---|---|---|---|---|---|
| Easy (4) | 100.0% | 100.0% | 100.0% | 100.0% | 75.0% |
| Medium (1) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Hard (4) | 50.0% | 0.0% | 0.0% | 0.0% | 25.0% |

## 3. Head-to-head vs Plain RAG (final answer correct)

| Strategy | Speculative better | Plain RAG better | Same outcome | Exact sign test p |
|---|---|---|---|---|
| chain | 2 | 0 | 7 | 0.500 |
| mmr | 1 | 1 | 7 | 1.000 |
| centroid | 1 | 1 | 7 | 1.000 |
| bridge | 1 | 1 | 7 | 1.000 |

## 4. Strategy vs strategy (final answer correct)

| Comparison | First better | Second better | Same outcome | Exact sign test p | Concept recall (first / second) | Questions with identical subsets |
|---|---|---|---|---|---|---|
| chain vs mmr | 2 | 0 | 7 | 0.500 | 77.8% / 72.8% | 0 of 9 |
| chain vs centroid | 2 | 0 | 7 | 0.500 | 77.8% / 72.8% | 0 of 9 |
| chain vs bridge | 2 | 0 | 7 | 0.500 | 77.8% / 74.3% | 0 of 9 |
| mmr vs centroid | 0 | 0 | 9 | 1.000 | 72.8% / 72.8% | 0 of 9 |
| mmr vs bridge | 0 | 0 | 9 | 1.000 | 72.8% / 74.3% | 0 of 9 |
| centroid vs bridge | 0 | 0 | 9 | 1.000 | 72.8% / 74.3% | 0 of 9 |

_Questions where two strategies built identical subsets reuse the same drafts, so they can never count as a win for either side._

## 5. Per-question results

| ID | Level | Cluster sizes | chain | mmr | centroid | bridge | Plain RAG |
|---|---|---|---|---|---|---|---|
| E01 | easy | 10/10 | ✓ | ✓ | ✓ | ✓ | ✓ |
| E02 | easy | 16/4 | ✓ | ✓ | ✓ | ✓ | ✓ |
| E03 | easy | 14/6 | ✓ | ✓ | ✓ | ✓ | ✗ |
| E04 | easy | 9/11 | ✓ | ✓ | ✓ | ✓ | ✓ |
| M01 | medium | 15/5 | ✗ | ✗ | ✗ | ✗ | ✗ |
| H01 | hard | 9/11 | ✓ | ✗ | ✗ | ✗ | ✗ |
| H02 | hard | 7/13 | ✗ | ✗ | ✗ | ✗ | ✗ |
| H03 | hard | 13/7 | ✗ | ✗ | ✗ | ✗ | ✗ |
| H04 | hard | 9/11 | ✓ | ✗ | ✗ | ✗ | ✓ |

## 6. Subset structure

| Strategy | Distinct chunks used | Subset overlap (Jaccard) | Intra-subset similarity (lower = less repeated) | Mean chunk distance (lower = closer to question) |
|---|---|---|---|---|
| chain | 11.7 | 0.14 | 0.704 | 0.368 |
| mmr | 11.7 | 0.14 | 0.689 | 0.367 |
| centroid | 11.7 | 0.14 | 0.695 | 0.367 |
| bridge | 13.3 | 0.14 | 0.692 | 0.371 |

## 7. Reading the results

- **chain**: final accuracy 66.7%, concept recall 77.8%.
- **mmr**: final accuracy 44.4%, concept recall 72.8%.
- **centroid**: final accuracy 44.4%, concept recall 72.8%.
- **bridge**: final accuracy 44.4%, concept recall 74.3%.
- Highest final accuracy: **chain**.
- With 9 questions, a gap of one or two questions is within noise (the drafts come from an LLM and are not perfectly repeatable). Trust the sign-test p-values in the strategy-vs-strategy table, not the raw percentages.