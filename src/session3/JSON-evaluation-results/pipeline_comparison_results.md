# Subsets vs Clusters-only

Run: 2026-11-21 12:46:22 | questions: 12 of 12 processed | compared on both methods: 6

- **Subsets**: 5 overlapping chain subsets built from the K-Means clusters, one draft per subset.
- **Clusters only**: the 2 whole clusters, one draft per cluster, no subsets.
- Both go through the same verifier and the same phrase matching on required_concepts.

## Summary

| Method | Final accuracy | Oracle (any draft correct) | Concept recall | Avg seconds (incl. subset building) |
|---|---|---|---|---|
| Subsets (5 drafts) | 100.0% | 100.0% | 100.0% | 148.9 |
| Clusters only (2 drafts) | 100.0% | 100.0% | 100.0% | 48.5 |
| Plain RAG (reference) | 100.0% | - | 100.0% | 12.0 |

## Head-to-head (final answer correct)

| Subsets better | Clusters better | Same outcome | Exact sign test p |
|---|---|---|---|
| 0 | 0 | 6 | 1.000 |

## Accuracy by difficulty

| Level | Questions | Subsets | Clusters only | Plain RAG |
|---|---|---|---|---|
| easy | 3 | 100.0% | 100.0% | 100.0% |
| hard | 1 | 100.0% | 100.0% | 100.0% |
| medium | 2 | 100.0% | 100.0% | 100.0% |

## Per question

| ID | Level | Subsets | Clusters only | Plain RAG |
|---|---|---|---|---|
| E01 | easy | correct | correct | correct |
| E02 | easy | correct | correct | correct |
| E03 | easy | correct | correct | correct |
| M02 | medium | correct | correct | correct |
| M03 | medium | correct | correct | correct |
| H01 | hard | correct | correct | correct |

## Not compared (failed or incomplete)

- E04: OpenRouter returned an empty response for Draft 2
- M01: OpenRouter returned an empty response for Draft 4
- M04: OpenRouter returned an empty response for Draft 5
- H02: OpenRouter returned an empty response for Draft 2
- H03: OpenRouter returned an empty response for Draft 2
- H04: OpenRouter returned an empty response for Draft 1

## Details by question

### E01 [easy]: Which three broad governmental powers does the Constitution distribute among the three branches of the federal government?

Reference answer: legislative, executive, and judicial powers

#### Subsets (correct, 63.8s)

- Final answer: The Constitution distributes the legislative, executive, and judicial powers among the three branches of the federal government.
- Chunk positions per subset: [[1, 2, 13, 9], [2, 6, 9, 15], [6, 5, 15, 12], [5, 4, 3, 7], [4, 11, 7, 13]]
- **Draft 1** — verifier score: 10.0
  The Constitution distributes the powers to legislate, to execute, and to adjudicate among the Legislative, Executive, and Judicial branches of the federal government.
- **Draft 2** — verifier score: 10.0
  The Constitution distributes the three broad governmental powers of legislative, executive, and judicial among the three branches of the federal government.
- **Draft 3** — verifier score: 10.0
  The Constitution distributes the legislative, executive, and judicial powers among the three branches of the federal government.
- **Draft 4** — verifier score: 10.0
  The Constitution distributes the three broad governmental powers of legislation, execution, and adjudication among the legislative, executive, and judicial branches of the federal government.
- **Draft 5** — verifier score: 10.0
  The Constitution distributes the three broad governmental powers of legislative, executive, and judicial authority among the three branches of the federal government.

#### Clusters only (correct, 68.2s)

- Final answer: The Constitution distributes the three broad governmental powers of legislative, executive, and judicial among the three branches of the federal government.
- Chunk positions per cluster: [[1, 2, 4, 5, 6, 8, 10, 11, 17], [3, 7, 9, 12, 13, 14, 15, 16, 18, 19, 20]]
- **Draft 1** — verifier score: 10.0
  The Constitution distributes the three broad governmental powers of legislative, executive, and judicial among the three branches of the federal government.
- **Draft 2** — verifier score: 10.0
  The Constitution distributes the powers to legislate, to execute, and to adjudicate among the three branches of the federal government.

### E02 [easy]: Which two chambers make up the United States Congress under Article I?

Reference answer: Senate and House of Representatives

#### Subsets (correct, 65.1s)

- Final answer: Under Article I, the United States Congress is composed of the Senate and the House of Representatives.
- Chunk positions per subset: [[1, 2, 19, 20], [2, 3, 8, 16], [3, 5, 16, 19], [5, 6, 8, 16], [6, 4, 16, 19]]
- **Draft 1** — verifier score: 10.0
  Under Article I, the United States Congress is composed of the Senate and the House of Representatives.
- **Draft 2** — verifier score: 10.0
  Under Article I, the United States Congress is composed of the Senate and the House of Representatives.
- **Draft 3** — verifier score: 0.0
  
- **Draft 4** — verifier score: 10.0
  The United States Congress under Article I is composed of the House of Representatives and the Senate.
- **Draft 5** — verifier score: 10.0
  Under Article I, the United States Congress is composed of the Senate and the House of Representatives.

#### Clusters only (correct, 56.4s)

- Final answer: The United States Congress under Article I consists of a Senate and a House of Representatives.
- Chunk positions per cluster: [[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18], [8, 16, 19, 20]]
- **Draft 1** — verifier score: 10.0
  The United States Congress under Article I consists of a Senate and a House of Representatives.
- **Draft 2** — verifier score: 10.0
  Under Article I, the United States Congress consists of the House of Representatives and the Senate.

### E03 [easy]: Under Section 1 of the Fourteenth Amendment, what does the first sentence establish about people born or naturalized in the United States and subject to its jurisdiction?

Reference answer: citizens of the United States and of the State wherein they reside

#### Subsets (correct, 50.9s)

- Final answer: The first sentence of Section 1 of the Fourteenth Amendment establishes that all persons born or naturalized in the United States and subject to its jurisdiction are citizens of the United States and of the State wherein they reside.
- Chunk positions per subset: [[2, 9, 6, 5], [9, 4, 5, 18], [4, 13, 18, 19], [13, 8, 1, 3], [8, 7, 3, 6]]
- **Draft 1** — verifier score: 10.0
  The first sentence of Section 1 of the Fourteenth Amendment establishes that all persons born or naturalized in the United States and subject to its jurisdiction are citizens of the United States and of the State wherein they reside.
- **Draft 2** — verifier score: 10.0
  The first sentence of Section 1 of the Fourteenth Amendment establishes that all persons born or naturalized in the United States and subject to its jurisdiction are citizens of the United States and of the State wherein they reside.
- **Draft 3** — verifier score: 10.0
  Under Section 1 of the Fourteenth Amendment, the first sentence establishes that all persons born or naturalized in the United States and subject to its jurisdiction are citizens of the United States and of the State wherein they reside.
- **Draft 4** — verifier score: 10.0
  Under Section 1 of the Fourteenth Amendment, the first sentence establishes that all persons born or naturalized in the United States and subject to its jurisdiction are citizens of the United States and of the State wherein they reside.
- **Draft 5** — verifier score: 10.0
  Under Section 1 of the Fourteenth Amendment, the first sentence establishes that all persons born or naturalized in the United States and subject to its jurisdiction are citizens of the United States and of the State wherein they reside.

#### Clusters only (correct, 56.0s)

- Final answer: Under Section 1 of the Fourteenth Amendment, the first sentence establishes that all persons born or naturalized in the United States and subject to its jurisdiction are citizens of the United States and of the State wherein they reside.
- Chunk positions per cluster: [[2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20], [1, 3, 5, 6, 18, 19]]
- **Draft 1** — verifier score: 10.0
  Under Section 1 of the Fourteenth Amendment, the first sentence establishes that all persons born or naturalized in the United States and subject to its jurisdiction are citizens of the United States and of the State wherein they reside.
- **Draft 2** — verifier score: 10.0
  The first sentence of Section 1 of the Fourteenth Amendment establishes that all persons born or naturalized in the United States and subject to its jurisdiction are citizens of the United States and of the State wherein they reside.

### M02 [medium]: What are the three broad categories of activity that Congress may regulate under its Commerce Clause power?

Reference answer: channels of interstate commerce; instrumentalities of interstate commerce or persons or things in interstate commerce; and activities that substantially affect interstate commerce

#### Subsets (correct, 337.3s)

- Final answer: The three broad categories of activity that Congress may regulate under its Commerce Clause power are channels of interstate commerce, instrumentalities of interstate commerce or persons or things in interstate commerce, and activities having a substantial relation to interstate commerce.
- Chunk positions per subset: [[4, 6, 2, 5], [6, 7, 5, 9], [7, 10, 9, 13], [10, 8, 1, 3], [8, 11, 3, 2]]
- **Draft 1** — verifier score: 0.0
  The provided documents do not identify three broad categories of activity that Congress may regulate under its Commerce Clause power.
- **Draft 2** — verifier score: 0.0
  The provided documents do not contain the answer to the question regarding the three broad categories of activity Congress may regulate under its Commerce Clause power.
- **Draft 3** — verifier score: 0.0
  The provided documents do not list the three broad categories of activity that Congress may regulate under its Commerce Clause power.
- **Draft 4** — verifier score: 10.0
  The three broad categories of activity that Congress may regulate under its Commerce Clause power are channels of interstate commerce, instrumentalities of interstate commerce or persons or things in interstate commerce, and activities having a substantial relation to interstate commerce.
- **Draft 5** — verifier score: 10.0
  The three broad categories of activity that Congress may regulate under its Commerce Clause power are channels of interstate commerce, instrumentalities of interstate commerce (or persons or things in interstate commerce), and activities having a substantial relation to interstate commerce.

#### Clusters only (correct, 41.6s)

- Final answer: The three broad categories of activity that Congress may regulate under its Commerce Clause power are channels of interstate commerce, instrumentalities of interstate commerce or persons or things in interstate commerce, and activities having a substantial relation to interstate commerce.
- Chunk positions per cluster: [[4, 6, 7, 8, 10, 11, 14, 15, 16, 17, 18, 19, 20], [1, 2, 3, 5, 9, 12, 13]]
- **Draft 1** — verifier score: 0.0
  Under its Commerce Clause power, Congress may regulate activity by restraining or prohibiting it at all times for the welfare of the public.
- **Draft 2** — verifier score: 10.0
  The three broad categories of activity that Congress may regulate under its Commerce Clause power are channels of interstate commerce, instrumentalities of interstate commerce or persons or things in interstate commerce, and activities having a substantial relation to interstate commerce.

### M03 [medium]: According to the Tenth Amendment, where do powers go when they are not delegated to the United States by the Constitution and are not prohibited to the States?

Reference answer: reserved to the States respectively, or to the people

#### Subsets (correct, 53.3s)

- Final answer: According to the Tenth Amendment, powers not delegated to the United States by the Constitution and not prohibited to the States are reserved to the States respectively, or to the people.
- Chunk positions per subset: [[1, 2, 8, 11], [2, 5, 11, 9], [5, 7, 9, 15], [7, 10, 3, 4], [10, 6, 4, 8]]
- **Draft 1** — verifier score: 10.0
  
- **Draft 2** — verifier score: 10.0
  According to the Tenth Amendment, powers that are not delegated to the United States by the Constitution and are not prohibited to the States are reserved to the States respectively or to the people.
- **Draft 3** — verifier score: 10.0
  According to the Tenth Amendment, powers that are not delegated to the United States by the Constitution and are not prohibited to the States are reserved to the States respectively or to the people.
- **Draft 4** — verifier score: 10.0
  According to the Tenth Amendment, powers not delegated to the United States by the Constitution and not prohibited to the States are reserved to the States respectively, or to the people.
- **Draft 5** — verifier score: 10.0
  According to the Tenth Amendment, powers that are not delegated to the United States by the Constitution and are not prohibited to the States are reserved to the States respectively or to the people.

#### Clusters only (correct, 22.2s)

- Final answer: According to the Tenth Amendment, powers that are not delegated to the United States by the Constitution and are not prohibited to the States are reserved to the States respectively or to the people.
- Chunk positions per cluster: [[1, 2, 5, 6, 7, 10, 12, 13, 14, 19], [3, 4, 8, 9, 11, 15, 16, 17, 18, 20]]
- **Draft 1** — verifier score: 10.0
  According to the Tenth Amendment, powers that are not delegated to the United States by the Constitution and are not prohibited to the States are reserved to the States respectively or to the people.
- **Draft 2** — verifier score: 10.0
  According to the Tenth Amendment, powers that are not delegated to the United States by the Constitution and are not prohibited to the States are reserved to the States respectively or to the people.

### H01 [hard]: State the test from McCulloch v. Maryland for deciding whether Congress may use an implied means to carry out a constitutional power. Include the requirements concerning the end and the means.

Reference answer: the end must be legitimate and within the scope of the Constitution, and the means must be appropriate, plainly adapted to that end, not prohibited, and consistent with the letter and spirit of the Constitution

#### Subsets (correct, 323.3s)

- Final answer: The test from McCulloch v. Maryland requires that the end be legitimate and within the scope of the Constitution, while the means must be appropriate, plainly adapted to that end, not prohibited, and consistent with the letter and spirit of the Constitution.
- Chunk positions per subset: [[3, 7, 6, 11], [7, 8, 11, 12], [8, 5, 12, 4], [5, 10, 1, 2], [10, 13, 2, 6]]
- **Draft 1** — verifier score: 5.0
  The test from McCulloch v. Maryland for deciding whether Congress may use an implied means to carry out a constitutional power is that the means must be appropriate and plainly adapted to the permitted end.
- **Draft 2** — verifier score: 4.0
  The test from McCulloch v. Maryland for deciding whether Congress may use an implied means to carry out a constitutional power requires that the means be in fact conducive to the exercise of a power granted by the Constitution.
- **Draft 3** — verifier score: 8.0
  The test from McCulloch v. Maryland for deciding whether Congress may use an implied means to carry out a constitutional power requires that the end be legitimate and that the means be appropriate and plainly adapted to that end.
- **Draft 4** — verifier score: 10.0
  The test from McCulloch v. Maryland requires that the end be legitimate and within the scope of the Constitution, while the means must be appropriate, plainly adapted to that end, not prohibited, and consistent with the letter and spirit of the Constitution.
- **Draft 5** — verifier score: 2.0
  The test from McCulloch v. Maryland for deciding whether Congress may use an implied means to carry out a constitutional power requires that the means be appropriate, bona fide, and plainly adapted to the end, and not prohibited by the Constitution or by any just rules of international law.

#### Clusters only (correct, 46.8s)

- Final answer: The test from McCulloch v. Maryland requires that the end be legitimate and within the scope of the Constitution, and the means be appropriate, plainly adapted to that end, not prohibited, and consistent with the letter and spirit of the Constitution.
- Chunk positions per cluster: [[3, 5, 7, 8, 10, 13, 14, 18, 20], [1, 2, 4, 6, 9, 11, 12, 15, 16, 17, 19]]
- **Draft 1** — verifier score: 6.0
  The test from McCulloch v. Maryland for deciding whether Congress may use an implied means to carry out a constitutional power is that the means must be appropriate and plainly adapted to the permitted end.
- **Draft 2** — verifier score: 10.0
  The test from McCulloch v. Maryland requires that the end be legitimate and within the scope of the Constitution, and the means be appropriate, plainly adapted to that end, not prohibited, and consistent with the letter and spirit of the Constitution.
