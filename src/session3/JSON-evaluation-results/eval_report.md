# Evaluation: Speculative RAG vs Plain RAG

Questions completed: **8** of 8 | retrieval k = 20 | clusters = 2 | subsets = 5

_Subsets-only run: no drafts or verification were executed, so no accuracy metrics were produced. The table below describes the subsets only._

## Subset structure

| Distinct chunks used | Subset overlap (Jaccard) | Intra-subset similarity (lower = less repeated) | Mean chunk distance (lower = closer to question) |
|---|---|---|---|
| 11.1 | 0.17 | 0.430 | 0.706 |

## 2. Retrieval quality (no LLM calls)

| Cutoff | Hit@n | Concept hit@n | Concept recall@n | Precision@n | nDCG@n |
|---|---|---|---|---|---|
| top-1 | 25.0% | 25.0% | 25.0% | 25.0% | 25.0% |
| top-3 | 50.0% | 50.0% | 50.0% | 25.0% | 34.4% |
| top-5 | 75.0% | 75.0% | 75.0% | 22.5% | 45.4% |
| top-10 | 87.5% | 87.5% | 87.5% | 13.8% | 50.3% |
| top-20 | 87.5% | 87.5% | 87.5% | 8.1% | 52.6% |

- **MRR** (first chunk holding every concept group): **0.438**
- Mean rank of that chunk, when one exists: 3.71 (7 of 8 questions)
- Mean rank to full concept coverage, when reached: 3.71 (7 of 8 questions)
- Questions where some concept is never retrieved: **1** of 8

_Hit@n needs one chunk to hold the whole answer; Concept hit@n and recall@n let the answer be spread over several chunks. Relevance is judged from `required_concepts` (or `ground_truth`), so these are proxies for gold-label metrics. nDCG ranks the retrieved chunks among themselves and cannot see chunks that were not retrieved._

**By difficulty (top-20):**

| Level | Hit@20 | Concept hit@20 | Concept recall@20 | MRR |
|---|---|---|---|---|
| Easy (2) | 100.0% | 100.0% | 100.0% | 0.300 |
| Medium (5) | 80.0% | 80.0% | 80.0% | 0.540 |
| Hard (1) | 100.0% | 100.0% | 100.0% | 0.200 |

**Questions with incomplete concept recall at top-20:**

| ID | Level | Recall@20 | Concept groups never retrieved |
|---|---|---|---|
| AR07 | medium | 0.0% | ٢١ صفر ١٤٣٤ |