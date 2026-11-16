# Rebalance comparison: Speculative RAG with vs without `rebalance_clusters()`

Questions completed: **1** of 1 | retrieval k = 8 | clusters = 2 | min cluster size = 4 | subsets = 5

## 1. How often does rebalancing matter?

- Questions where rebalancing changed the K-Means labels: **1 / 1**
- Questions where it changed the subsets given to the drafts: **1 / 1**
- On the remaining questions both variants get identical evidence, so they cannot differ.

| ID | Level | Raw sizes | Rebalanced sizes | Subsets changed |
|---|---|---|---|---|
| H03 | hard | 3/5 | 4/4 | yes |

## 2. Overall results (all questions)

| System | Final acc. | Oracle | Subset hit | Concept recall | Best-subset cov. | Distinct chunks | Subset overlap | Avg sec |
|---|---|---|---|---|---|---|---|---|
| Without rebalance | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% | 8.0 | 0.36 | 149.6 |
| With rebalance | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 8.0 | 0.36 | 155.0 |
| Plain RAG (reference) | 0.0% | - | - | 0.0% | - | - | - | 67.2 |

Retrieval MRR (identical for both variants): **0.250**

## 3. Accuracy by difficulty

| Level | Without rebalance | With rebalance | Plain RAG |
|---|---|---|---|
| Hard (1) | 0.0% | 0.0% | 0.0% |

## 4. Head-to-head (final answer correct)

| Rebalance better | Without better | Same outcome | Exact sign test p |
|---|---|---|---|
| 0 | 0 | 1 | 1.000 |

### Only the 1 question(s) where the subsets differed

| System | Final acc. | Oracle | Subset hit | Concept recall | Best-subset cov. | Distinct chunks | Subset overlap | Avg sec |
|---|---|---|---|---|---|---|---|---|
| Without rebalance | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% | 8.0 | 0.36 | 149.6 |
| With rebalance | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 8.0 | 0.36 | 155.0 |

## 5. Per-question results

| ID | Level | Raw sizes | Rebal. sizes | Subsets differ | Without | With | Plain RAG |
|---|---|---|---|---|---|---|---|
| H03 | hard | 3/5 | 4/4 | yes | ✗ | ✗ | ✗ |

## 6. Reading the results

Final accuracy is tied (0.0%); compare concept recall, oracle accuracy and subset diversity above for secondary differences.

Caveats: the question set is small, so one question moves accuracy by several percentage points and the sign test rarely reaches significance. Scores use phrase matching, so check the saved answers for correct paraphrases. A difference only appears on questions where K-Means produced a cluster with fewer than 4 chunks.