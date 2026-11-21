# Evaluation: Speculative RAG vs Plain RAG

Questions completed: **8** of 12 | retrieval k = 20 | clusters = 2 | subsets = 5

## 1. Overall results (all questions)

| System | Final acc. | Oracle | Subset hit | Concept recall | Best-subset cov. | Distinct chunks | Subset overlap | Avg sec |
|---|---|---|---|---|---|---|---|---|
| Speculative RAG | 75.0% | 87.5% | 75.0% | 81.2% | 91.7% | 11.8 | 0.14 | 194.6 |
| Plain RAG (baseline) | 75.0% | - | - | 81.2% | - | - | - | 10.2 |

Retrieval MRR: **0.473**

## 2. Retrieval quality (no LLM calls)

| Cutoff | Hit@n | Concept hit@n | Concept recall@n | Precision@n | nDCG@n |
|---|---|---|---|---|---|
| top-1 | 37.5% | 37.5% | 56.2% | 62.5% | 58.3% |
| top-3 | 50.0% | 50.0% | 68.8% | 54.2% | 56.0% |
| top-5 | 75.0% | 75.0% | 91.7% | 57.5% | 65.1% |
| top-10 | 75.0% | 75.0% | 91.7% | 51.2% | 67.5% |
| top-20 | 75.0% | 75.0% | 91.7% | 50.0% | 77.1% |

- **MRR** (first chunk holding every concept group): **0.473**
- Mean rank of that chunk, when one exists: 2.50 (6 of 8 questions)
- Mean rank to full concept coverage, when reached: 2.50 (6 of 8 questions)
- Questions where some concept is never retrieved: **2** of 8

_Hit@n needs one chunk to hold the whole answer; Concept hit@n and recall@n let the answer be spread over several chunks. Relevance is judged from `required_concepts` (or `ground_truth`), so these are proxies for gold-label metrics. nDCG ranks the retrieved chunks among themselves and cannot see chunks that were not retrieved._

**By difficulty (top-20):**

| Level | Hit@20 | Concept hit@20 | Concept recall@20 | MRR |
|---|---|---|---|---|
| Easy (3) | 100.0% | 100.0% | 100.0% | 0.750 |
| Medium (2) | 100.0% | 100.0% | 100.0% | 0.667 |
| Hard (3) | 33.3% | 33.3% | 77.8% | 0.067 |

**Questions with incomplete concept recall at top-20:**

| ID | Level | Recall@20 | Concept groups never retrieved |
|---|---|---|---|
| H01 | hard | 83.3% | letter and spirit of the Constitution |
| H02 | hard | 50.0% | non-economic activity |

## 3. LLM usage and estimated/charged cost

- Drafting model (**qwen/qwen3.5-9b**): 126,527 input tokens, 88,445 output tokens, **$0.0402750190**.
- Verification model (**qwen/qwen3.7-flash**): 102,285 input tokens, 36,837 output tokens, **$0.0078573600**.
- **Total LLM cost: $0.0481323790 USD** across 8 questions.
- Per-request costs use OpenRouter's reported `usage.cost` when available; otherwise they are estimated from configured per-million-token rates.


## 4. Accuracy by difficulty

| Level | Speculative RAG | Plain RAG |
|---|---|---|
| Easy (3) | 100.0% | 100.0% |
| Medium (2) | 100.0% | 100.0% |
| Hard (3) | 33.3% | 33.3% |

## 5. Head-to-head vs Plain RAG (final answer correct)

| Speculative better | Plain RAG better | Same outcome | Exact sign test p |
|---|---|---|---|
| 0 | 0 | 8 | 1.000 |

## 6. Per-question results

| ID | Level | Cluster sizes | Speculative RAG | Plain RAG |
|---|---|---|---|---|
| E01 | easy | 9/11 | ✓ | ✓ |
| E02 | easy | 16/4 | ✓ | ✓ |
| E03 | easy | 14/6 | ✓ | ✓ |
| M02 | medium | 13/7 | ✓ | ✓ |
| M03 | medium | 10/10 | ✓ | ✓ |
| H01 | hard | 9/11 | ✓ | ✓ |
| H02 | hard | 7/13 | ✗ | ✗ |
| H03 | hard | 11/9 | ✗ | ✗ |

## 7. Subset structure

| Distinct chunks used | Subset overlap (Jaccard) | Intra-subset similarity (lower = less repeated) | Mean chunk distance (lower = closer to question) |
|---|---|---|---|
| 11.8 | 0.14 | 0.688 | 0.389 |

## 8. Reading the results

- **Speculative RAG**: final accuracy 75.0%, concept recall 81.2%.
Final accuracy is tied (75.0%); compare concept recall (81.2% vs 81.2%) above.