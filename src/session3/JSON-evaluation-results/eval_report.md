# Evaluation: Speculative RAG vs Plain RAG

Questions completed: **8** of 8 | retrieval k = 20 | clusters = 2 | subsets = 5

## 1. Overall results (all questions)

| System | Final acc. | Oracle | Subset hit | Concept recall | Best-subset cov. | Distinct chunks | Subset overlap | Avg sec |
|---|---|---|---|---|---|---|---|---|
| Speculative RAG | 62.5% | 62.5% | 100.0% | 80.2% | 100.0% | 12.0 | 0.13 | 28.7 |
| Plain RAG (baseline) | 50.0% | - | - | 50.0% | - | - | - | 2.4 |

Retrieval MRR: **0.388**

## 2. Accuracy by question language

| Language | Speculative RAG | Plain RAG | Concept recall |
|---|---|---|---|
| English (4) | 25.0% | 50.0% | 60.4% |
| Arabic (4) | 100.0% | 50.0% | 100.0% |

## 2b. Same-language vs cross-language

| Setting | Speculative RAG | Plain RAG | Concept recall |
|---|---|---|---|
| Same language (question = document) (4) | 100.0% | 50.0% | 100.0% |
| Cross-language (question != document) (4) | 25.0% | 50.0% | 60.4% |

## 2. Retrieval quality (no LLM calls)

| Cutoff | Hit@n | Concept hit@n | Concept recall@n | Precision@n | nDCG@n |
|---|---|---|---|---|---|
| top-1 | 12.5% | 12.5% | 12.5% | 12.5% | 12.5% |
| top-3 | 62.5% | 62.5% | 62.5% | 20.8% | 40.8% |
| top-5 | 62.5% | 62.5% | 62.5% | 12.5% | 40.8% |
| top-10 | 100.0% | 100.0% | 100.0% | 10.0% | 53.3% |
| top-20 | 100.0% | 100.0% | 100.0% | 5.0% | 53.3% |

- **MRR** (first chunk holding every concept group): **0.388**
- Mean rank of that chunk, when one exists: 4.12 (8 of 8 questions)
- Mean rank to full concept coverage, when reached: 4.12 (8 of 8 questions)
- Questions where some concept is never retrieved: **0** of 8

_Hit@n needs one chunk to hold the whole answer; Concept hit@n and recall@n let the answer be spread over several chunks. Relevance is judged from `required_concepts` (or `ground_truth`), so these are proxies for gold-label metrics. nDCG ranks the retrieved chunks among themselves and cannot see chunks that were not retrieved._

**By difficulty (top-20):**

| Level | Hit@20 | Concept hit@20 | Concept recall@20 | MRR |
|---|---|---|---|---|
| Medium (6) | 100.0% | 100.0% | 100.0% | 0.472 |
| Hard (2) | 100.0% | 100.0% | 100.0% | 0.133 |

## 3. LLM usage and estimated/charged cost

- Drafting model (**qwen/qwen3.5-9b**): 124,434 input tokens, 3,540 output tokens, **$0.0162268200**.
- Verification model (**qwen/qwen3.7-flash**): 57,295 input tokens, 21,306 output tokens, **$0.0044886300**.
- **Total LLM cost: $0.0207154500 USD** across 8 questions.
- Per-request costs use OpenRouter's reported `usage.cost` when available; otherwise they are estimated from configured per-million-token rates.


## 4. Accuracy by difficulty

| Level | Speculative RAG | Plain RAG |
|---|---|---|
| Medium (6) | 66.7% | 66.7% |
| Hard (2) | 50.0% | 0.0% |

## 5. Head-to-head vs Plain RAG (final answer correct)

| Speculative better | Plain RAG better | Same outcome | Exact sign test p |
|---|---|---|---|
| 2 | 1 | 5 | 1.000 |

## 6. Per-question results

| ID | Level | Cluster sizes | Speculative RAG | Plain RAG |
|---|---|---|---|---|
| AR05 | hard | 7/13 | ✓ | ✗ |
| AR06 | medium | 7/13 | ✓ | ✗ |
| AR07 | medium | 13/7 | ✓ | ✓ |
| AR08 | medium | 13/7 | ✓ | ✓ |
| AR05-EN | hard | 6/14 | ✗ | ✗ |
| AR06-EN | medium | 9/11 | ✗ | ✗ |
| AR07-EN | medium | 14/6 | ✓ | ✓ |
| AR08-EN | medium | 12/8 | ✗ | ✓ |

## 7. Subset structure

| Distinct chunks used | Subset overlap (Jaccard) | Intra-subset similarity (lower = less repeated) | Mean chunk distance (lower = closer to question) |
|---|---|---|---|
| 12.0 | 0.13 | 0.519 | 0.569 |

## 8. Reading the results

- **Speculative RAG**: final accuracy 62.5%, concept recall 80.2%.
Speculative RAG scored higher on final accuracy (62.5% vs 50.0%).