"""
Evaluation of Speculative RAG against a plain-RAG baseline.
One run produces eval_results.json, eval_report.md, eval_trace.md and
eval_retrieval_trace.md.

For every question in the ground-truth file the script:
  1. retrieves the top-k chunks and runs K-Means clustering,
  2. builds overlapping subsets from the K-Means clusters,
  3. writes one draft per subset with the OpenRouter drafting model and lets the
     OpenRouter verifier score the drafts (a second call breaks a tie for the top
     score),
  4. runs a plain-RAG baseline once, using all retrieved chunks in one prompt.

Metrics
  Final accuracy ........ final answer matches every required concept group
  Oracle accuracy ....... at least one of the drafts is correct
  Subset hit ............ at least one subset kept an answer-bearing chunk
  Concept recall ........ average share of required concept groups found in the
                          final answer (partial credit)
  Best-subset coverage .. share of concept groups present in the best subset
  Distinct chunks used .. how many different chunks the subsets cover
  Subset overlap ........ mean pairwise Jaccard overlap between subsets
  Avg seconds ........... subset construction + drafts + verification
Plus accuracy by difficulty and a paired comparison (wins / losses / ties and an
exact sign test) against Plain RAG.

Retrieval metrics (no LLM, no gold-chunk labels; relevance comes from required_concepts,
or from ground_truth when a question has no required_concepts):
  Hit@n ................. one of the top-n chunks holds ALL required concept groups
  Concept hit@n ......... the top-n chunks TOGETHER cover all concept groups
  Concept recall@n ...... share of concept groups covered by the union of the top-n chunks
  Precision@n ........... share of the top-n chunks that contain at least one concept group
  nDCG@n ................ ranking quality inside the retrieved set (gain = share of groups
                          a chunk holds); 1.0 = the most useful chunks are ranked first
  MRR ................... mean 1 / rank of the first chunk that holds ALL concept groups
  Rank to full coverage . how deep you must read to have every concept group

Outputs (in JSON-evaluation-results/)
  eval_results.json ... every answer, score and subset position
  eval_report.md ...... the readable comparison tables
  eval_trace.md ....... step-by-step trace with the FULL text of every chunk:
                        retrieved chunks, clusters, subsets, drafts, final answers.
                        APPENDED to on every run (a "Run started" header separates
                        runs); an existing file is never truncated or overwritten.
  eval_retrieval_trace.md  retrieval evaluation only, also appended to run after run:
                        per question the concept groups each retrieved chunk holds,
                        hit/recall/precision/nDCG per cutoff, and a final summary. The
                        same text is written as INFO to the app log and to
                        `docker compose logs` (see TRACE_RETRIEVAL_EVAL below).
  (eval_results.json and eval_report.md are regenerated on every run.)

Usage
  python evaluate.py                   # full evaluation
  python evaluate.py --subsets-only    # no LLM calls: inspect clusters/subsets
  python evaluate.py --ground-truth PATH  # use another ground-truth json (default: the
                                       # file in JSON-Test-cases/, ALL its questions)
  python evaluate.py --k 8             # fewer chunks
  python evaluate.py --ids E01 H02     # only some questions
  python evaluate.py --levels hard     # only some levels
  python evaluate.py --limit 3         # first N selected questions
  python evaluate.py --quiet           # no step-by-step trace in the terminal
  python evaluate.py --chunk-chars 300 # clip chunks in the terminal (default: show
                                       # everything; eval_trace.md is always full)

Environment
  TRACE_RETRIEVAL_EVAL=0   switch off eval_retrieval_trace.md and the retrieval log records
  MIRROR_TO_DOCKER_LOGS=0  do not copy the retrieval log records to the container's own
                           log (only matters for `docker compose exec ... evaluate.py`)
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from math import comb
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = Path(__file__).resolve()
SESSION_DIR = HERE.parents[2]  # keep the same project layout as the original evaluator
SRC_DIR = HERE.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logging_config import setup_logging  # noqa: E402
setup_logging(SESSION_DIR)
logger = logging.getLogger("evaluate")

from Speculative_RAG import (  # noqa: E402
    build_index,
    list_documents,
    retrieve,
    cluster,
    build_subsets,
    draft,
    verify,
    count_question_tokens,
    count_output_tokens,
    MIN_INTERVAL,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_N_CLUSTERS,
    DEFAULT_N_SUBSETS,
    DRAFT_MODEL,
    DRAFT_REASONING,
    VERIFY_MODEL,
    detect_language,
)

EVAL_PATH = SESSION_DIR / "JSON-Test-cases/cluster_comparison_ground_truth.json"
RESULTS_PATH = SESSION_DIR / "JSON-evaluation-results/eval_results.json"
REPORT_PATH = RESULTS_PATH.with_name("eval_report.md")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

K = DEFAULT_RETRIEVAL_K
N_SUBSETS = DEFAULT_N_SUBSETS
N_CLUSTERS = DEFAULT_N_CLUSTERS
TRACE = True            # print a human-readable trace of every step (see --quiet)
TRACE_CHARS = 0         # 0 = show every chunk in full; set --chunk-chars N to clip the terminal
TRACE_PATH = RESULTS_PATH.with_name("eval_trace.md")
# Retrieval evaluation gets its own file so eval_trace.md is never overwritten by it.
RETRIEVAL_TRACE_PATH = RESULTS_PATH.with_name("eval_retrieval_trace.md")
RETRIEVAL_SNIPPET_CHARS = 160
TRACE_RETRIEVAL_EVAL = os.getenv("TRACE_RETRIEVAL_EVAL", "1").strip().lower() not in (
    "0", "false", "no", "off")
MIRROR_TO_DOCKER_LOGS = os.getenv("MIRROR_TO_DOCKER_LOGS", "1").strip().lower() not in (
    "0", "false", "no", "off")


def contains(text, truths):
    """Return True if an accepted ground-truth phrase appears in text."""
    if isinstance(truths, str):
        truths = [truths]
    text_norm = _norm(text)
    for truth in truths:
        needle = _norm(truth)
        if needle and needle in text_norm:
            return True
    return False


def matches_item(text, item):
    """
    Require at least one phrase from every required_concepts group.
    This is stricter than a single substring but tolerates common paraphrases.
    Falls back to ground_truth phrase matching for older test-set records.
    """
    groups = item.get("required_concepts")
    if not groups:
        return contains(text, item["ground_truth"])
    normalized = _norm(text)
    for alternatives in groups:
        if not any(
            _norm(phrase) in normalized
            for phrase in alternatives
        ):
            return False
    return True


VERIFIER_DOWN_MARKERS = ("verifier quota is exhausted", "resource_exhausted", "verifier model")


def stage(progress, method, text):
    """Always-visible stage line: which method (SUBSETS / PLAIN RAG) the model is
    working on right now. Goes to the terminal and the app log (so also to
    `docker compose logs`); it is NOT written to eval_trace.md or eval_retrieval_trace.md."""
    line = f">>> {progress}[{method}] {text}"
    print("  " + line, flush=True)
    logger.info(line)


def run_drafts_and_verification(question, subsets, token_tracker=None, method=None, progress=""):
    """Write one draft per evidence group with OpenRouter, then verify them with OpenRouter.
    `method` only controls the stage lines that are printed (e.g. "SUBSETS")."""
    if method:
        stage(progress, method, f"STAGE 1/2 drafting: {len(subsets)} drafts in parallel ...")
    with ThreadPoolExecutor(max_workers=len(subsets)) as pool:
        futures = [
            pool.submit(draft, question, subset, i + 1, token_tracker, detect_language(question))
            for i, subset in enumerate(subsets)
        ]
        drafts = [f.result() for f in futures]
    if method:
        stage(progress, method, f"STAGE 2/2 verifying: scoring {len(drafts)} drafts ...")
    scores, best, tie_reason, final_answer = verify(
        question, drafts, subsets, token_tracker=token_tracker,
        language=detect_language(question),
    )
    return {
        "answer": final_answer,
        "drafts": drafts,
        "scores": scores,
        "best_draft_index": best,
        "verifier_decision": tie_reason,
    }


# --------------------------------------------------------------------------- #
# Small metric helpers
# --------------------------------------------------------------------------- #
def concept_coverage(text, item):
    """Share of required_concepts groups found in text (1.0 = all groups)."""
    groups = item.get("required_concepts")
    if not groups:
        return 1.0 if matches_item(text, item) else 0.0
    found = sum(
        matches_item(text, {"required_concepts": [group], "ground_truth": item["ground_truth"]})
        for group in groups
    )
    return found / len(groups)


def concept_groups(item):
    """required_concepts, or the ground_truth phrases as one group of alternatives."""
    groups = item.get("required_concepts")
    if groups:
        return groups
    truths = item["ground_truth"]
    return [[truths] if isinstance(truths, str) else list(truths)]


def _norm(text):
    return re.sub(r"[^\w\u0600-\u06ff]+", " ", str(text).casefold(), flags=re.UNICODE).strip()


def _has_phrase(norm_text, phrase):
    needle = _norm(phrase)
    return bool(needle) and needle in norm_text


def groups_found(text, groups):
    """One bool per concept group: does text contain at least one alternative of it?"""
    norm = _norm(text)
    return [any(_has_phrase(norm, phrase) for phrase in alternatives) for alternatives in groups]


def retrieval_cutoffs(n_docs):
    """Cutoffs reported for hit@n / recall@n: 1, 3, 5, 10 and the full retrieved set."""
    return sorted({c for c in (1, 3, 5, 10) if c < n_docs} | {n_docs})


def retrieval_metrics(docs, item, cutoffs):
    """
    Retrieval quality of the ranked chunk list, without gold chunk labels.
    A concept group counts as retrieved when any of its phrases appears in a chunk.
    """
    groups = concept_groups(item)
    n_groups = len(groups)
    per_chunk = [groups_found(doc, groups) for doc in docs]

    first_rank_per_group = [
        next((rank for rank, found in enumerate(per_chunk, start=1) if found[g]), None)
        for g in range(n_groups)
    ]
    full_chunk_rank = next(
        (rank for rank, found in enumerate(per_chunk, start=1) if all(found)), None
    )
    gains_all = [sum(found) / n_groups for found in per_chunk]
    ideal_order = sorted(gains_all, reverse=True)

    def dcg(gains):
        return sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    at_k = {}
    for c in cutoffs:
        top = per_chunk[:c]
        covered = [any(found[g] for found in top) for g in range(n_groups)]
        ideal = dcg(ideal_order[:c])
        at_k[str(c)] = {
            "hit": any(all(found) for found in top),
            "concept_hit": all(covered),
            "recall": sum(covered) / n_groups,
            "precision": sum(any(found) for found in top) / len(top) if top else 0.0,
            "ndcg": dcg(gains_all[:c]) / ideal if ideal > 0 else 0.0,
        }

    missing = [groups[g][0] for g in range(n_groups) if first_rank_per_group[g] is None]
    return {
        "group_labels": [alternatives[0] for alternatives in groups],
        # 1-based concept group numbers held by each retrieved chunk, in rank order
        "chunk_groups_found": [[g + 1 for g, hit in enumerate(found) if hit] for found in per_chunk],
        "first_answer_bearing_rank": full_chunk_rank,
        "reciprocal_rank": 1.0 / full_chunk_rank if full_chunk_rank else 0.0,
        "rank_to_full_coverage": (
            max(first_rank_per_group) if all(r is not None for r in first_rank_per_group) else None
        ),
        "chunk_concept_matches": [sum(found) for found in per_chunk],
        "n_concept_groups": n_groups,
        "missing_concepts": missing,
        "at_k": at_k,
    }


def mean_pairwise_jaccard(position_sets):
    """Average overlap between subsets (0 = disjoint, 1 = identical)."""
    sets = [set(s) for s in position_sets]
    values = [
        len(a & b) / len(a | b)
        for i, a in enumerate(sets) for b in sets[i + 1:]
        if a | b
    ]
    return float(np.mean(values)) if values else 1.0


def sign_test_p(wins, losses):
    """Exact two-sided sign test on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def positions_of(docs, subsets):
    return [[docs.index(doc) + 1 for doc in subset] for subset in subsets]


def cluster_sizes(labels):
    return {str(int(l)): int(sum(int(x) == int(l) for x in labels)) for l in sorted(set(labels))}


# --------------------------------------------------------------------------- #
# One question
# --------------------------------------------------------------------------- #
def build_subsets_timed(docs, labels, distances, embeddings):
    """Build chain subsets; return (subsets, seconds, error_text)."""
    started = time.perf_counter()
    try:
        subsets = build_subsets(
            docs, labels, n_subsets=N_SUBSETS, distances=distances, embeddings=embeddings,
        )
        return subsets, time.perf_counter() - started, None
    except Exception as error:
        logger.exception("Chain subset construction failed")
        return None, time.perf_counter() - started, str(error)


def subset_structure(positions, embeddings, distances):
    """Properties of the subsets themselves (no LLM needed), from 1-based chunk positions."""
    vectors = np.asarray(embeddings, dtype=float)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    sims = []
    for pos in positions:
        idx = [p - 1 for p in pos]
        sims += [float(vectors[a] @ vectors[b]) for i, a in enumerate(idx) for b in idx[i + 1:]]
    used = sorted({p - 1 for pos in positions for p in pos})
    return {
        "distinct_chunks_used": len(used),
        "mean_subset_overlap": mean_pairwise_jaccard(positions),
        # mean cosine similarity between chunks inside one subset (lower = less repeated content)
        "intra_subset_similarity": float(np.mean(sims)) if sims else 0.0,
        # mean retrieval distance of the chunks used (lower = closer to the question)
        "mean_chunk_distance": (
            float(np.mean([float(distances[i]) for i in used]))
            if distances is not None and used else 0.0
        ),
    }


def score_speculative(item, subsets, positions, generated, construction_seconds,
                      generation_seconds, structure):
    final_correct = matches_item(generated["answer"], item)
    best = generated["best_draft_index"]
    return {
        "status": "ok",
        "answer": generated["answer"],
        "draft_answers": [d["answer"] for d in generated["drafts"]],
        "verifier_scores": generated["scores"],
        "best_draft_index": best,
        "verifier_decision": generated["verifier_decision"],
        "subset_chunk_positions": positions,
        "final_correct": final_correct,
        "oracle_draft_correct": any(matches_item(d["answer"], item) for d in generated["drafts"]),
        "subset_hit": any(matches_item(doc, item) for subset in subsets for doc in subset),
        "answer_concept_recall": concept_coverage(generated["answer"], item),
        "best_subset_concept_coverage": max(
            concept_coverage(" ".join(subset), item) for subset in subsets
        ),
        **structure,
        "subset_construction_seconds": construction_seconds,
        "draft_and_verification_seconds": generation_seconds,
    }

def failed_speculative(error):
    return {
        "status": "error", "error": error, "final_correct": False,
        "oracle_draft_correct": False, "subset_hit": False,
        "answer_concept_recall": 0.0, "best_subset_concept_coverage": 0.0,
        "distinct_chunks_used": 0, "mean_subset_overlap": 1.0,
        "intra_subset_similarity": 0.0, "mean_chunk_distance": 0.0,
        "subset_construction_seconds": 0.0, "draft_and_verification_seconds": 0.0,
    }


# --------------------------------------------------------------------------- #
# Human-readable trace (terminal = clipped, eval_trace.md = full text)
# --------------------------------------------------------------------------- #
def _clip(text, full):
    text = " ".join(str(text).split())
    if full or TRACE_CHARS <= 0 or len(text) <= TRACE_CHARS:
        return text
    return text[:TRACE_CHARS].rstrip() + " ..."


def emit(build):
    """build(full) returns a list of lines. The terminal gets the clipped version,
    the trace file gets the complete one."""
    if not TRACE:
        return
    print("\n".join(build(False)), flush=True)
    with open(TRACE_PATH, "a", encoding="utf-8") as file:
        file.write("\n".join(build(True)) + "\n")


def _ids(positions):
    return " ".join(f"#{p}" for p in positions)


def trace_retrieval(docs, metas, distances, k):
    def build(full):
        lines = ["", f"  == RETRIEVED CHUNKS ({len(docs)} of k={k}, #1 = most similar) =="]
        for i, doc in enumerate(docs, start=1):
            dist = f"dist {float(distances[i - 1]):.3f}  " if distances is not None else ""
            src = (metas[i - 1] or {}).get("source", "?") if metas else "?"
            lines.append(f"  #{i:<2} {dist}[{src}]")
            lines.append(f"{_clip(doc, full)}")
        return lines
    emit(build)


def _stdout_is_container_log():
    """True when this process writes to the same stream `docker compose logs` reads (PID 1's stdout)."""
    try:
        mine, container = os.fstat(1), os.stat("/proc/1/fd/1")
        return (mine.st_dev, mine.st_ino) == (container.st_dev, container.st_ino)
    except OSError:
        return False


def _mirror_to_container_logs(text):
    """
    `docker compose logs` shows only what PID 1 writes. A script started with
    `docker compose exec backend python evaluate.py` writes to its own terminal instead,
    so copy the record to PID 1's stdout. Skipped when stdout already is the container log.
    """
    if not MIRROR_TO_DOCKER_LOGS or _stdout_is_container_log():
        return
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("/proc/1/fd/1", "a", encoding="utf-8") as stream:
            stream.write(f"{stamp} INFO evaluate: {text}\n")
    except OSError:
        pass  # not in a container, or no permission: the app log still has it


def emit_retrieval(lines):
    """One record per block: eval_retrieval_trace.md + app log (INFO) + docker compose logs."""
    if not TRACE_RETRIEVAL_EVAL:
        return
    text = "\n".join(lines)
    with open(RETRIEVAL_TRACE_PATH, "a", encoding="utf-8") as file:
        file.write(text + "\n\n")
    logger.info("%s", text)
    _mirror_to_container_logs(text)


def retrieval_trace_lines(item, docs, metas, distances, rm):
    labels = rm["group_labels"]
    lines = [
        f"### RETRIEVAL EVAL | {item.get('id', '?')} [{item.get('level', '?')}] {item['question']}",
        f"Required concept groups ({len(labels)}): "
        + "; ".join(f"G{i}={label}" for i, label in enumerate(labels, start=1)),
        "Retrieved chunks (#1 = most similar) and the concept groups each one holds:",
    ]
    for i, doc in enumerate(docs, start=1):
        dist = f"dist {float(distances[i - 1]):.3f}  " if distances is not None else ""
        src = (metas[i - 1] or {}).get("source", "?") if metas else "?"
        held = " ".join(f"G{g}" for g in rm["chunk_groups_found"][i - 1]) or "-"
        snippet = " ".join(str(doc).split())[:RETRIEVAL_SNIPPET_CHARS]
        lines.append(f"  #{i:<2} {dist}[{src}] holds: {held}")
        lines.append(f"      {snippet}")
    for c, m in rm["at_k"].items():
        lines.append(
            f"  @{c:<3} hit={'Y' if m['hit'] else 'n'}  concept_hit={'Y' if m['concept_hit'] else 'n'}  "
            f"recall={m['recall']:.2f}  precision={m['precision']:.2f}  ndcg={m['ndcg']:.2f}"
        )
    first = rm["first_answer_bearing_rank"]
    lines.append(
        f"  First chunk holding ALL groups: {('#' + str(first)) if first else 'none'}"
        f"  | reciprocal rank {rm['reciprocal_rank']:.3f}"
        f"  | rank to full coverage: {rm['rank_to_full_coverage'] or 'never'}"
    )
    if rm["missing_concepts"]:
        lines.append("  Never retrieved: " + "; ".join(rm["missing_concepts"]))
    return lines


def trace_clusters(docs, labels):
    def build(full):
        lines = ["", "  == K-MEANS CLUSTERS =="]
        for label in sorted({int(x) for x in labels}):
            members = [i for i, x in enumerate(labels, start=1) if int(x) == label]
            lines.append(f"  Cluster {label}  ({len(members)} chunks): {_ids(members)}")
            for i in members:
                lines.append(f"      #{i:<2} {_clip(docs[i - 1], full)}")
        return lines
    emit(build)


def trace_subsets(docs, labels, positions):
    def build(full):
        lines = ["", "  == SUBSETS =="]
        for n, pos in enumerate(positions, start=1):
            by_cluster = {}
            for p in pos:
                by_cluster.setdefault(int(labels[p - 1]), []).append(p)
            detail = "; ".join(f"cluster {c}: {_ids(v)}" for c, v in sorted(by_cluster.items()))
            lines.append(f"  Subset {n}: {_ids(pos)}   ({detail})")
            for p in pos:
                lines.append(f"      #{p:<2} {_clip(docs[p - 1], full)}")
        used = {p for pos in positions for p in pos}
        unused = [i for i in range(1, len(docs) + 1) if i not in used]
        lines.append(f"  Chunks used by at least one subset: {_ids(sorted(used))}")
        lines.append("  Chunks NOT given to any draft: " + (_ids(unused) if unused else "none"))
        return lines
    emit(build)


def trace_generation(generated, positions):
    def build(full):
        lines = ["", "  == DRAFTS AND VERIFICATION =="]
        for n, d in enumerate(generated["drafts"]):
            mark = "  <== selected" if n == generated["best_draft_index"] else ""
            lines.append(f"  Draft {n + 1}  (subset {n + 1}: {_ids(positions[n])})  "
                         f"verifier score {generated['scores'][n]}{mark}")
            lines.append(f"      Answer   : {_clip(d['answer'], full)}")
            lines.append(f"      Rationale: {_clip(d['rationale'], full)}")
        lines.append(f"  FINAL ANSWER: {_clip(generated['answer'], full)}")
        return lines
    emit(build)


def begin_trace_file(path, title):
    """Append a run header; create the file (with its title) only if it does not exist yet."""
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8") as file:
        if is_new:
            file.write(f"# {title}\n")
        file.write(f"\n## Run started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")


def score_one_question(item, k, subsets_only, progress=""):
    question = item["question"]
    language = detect_language(question)
    source = item.get("source") or None
    q_tokens = count_question_tokens(question)
    logger.info("Evaluating %s | question_tokens=%d", item.get("id", "?"), q_tokens)

    docs, metas, embeddings, distances = retrieve(
        question, k=k, return_distances=True, source=source
    )
    base = {
        "id": item.get("id"),
        "question": question,
        "level": item.get("level"),
        "topic": item.get("topic"),
        "ground_truth": item["ground_truth"],
        "source": source,
        "language": language,
        "draft_model": DRAFT_MODEL,
        "verifier_model": VERIFY_MODEL,
    }
    if not docs:
        return {**base, "status": "no_retrieval"}

    labels = cluster(embeddings, n_clusters=N_CLUSTERS)

    rm = retrieval_metrics(docs, item, retrieval_cutoffs(len(docs)))
    retrieval_rank = rm["first_answer_bearing_rank"] or 0

    trace_retrieval(docs, metas, distances, k)
    emit_retrieval(retrieval_trace_lines(item, docs, metas, distances, rm))
    trace_clusters(docs, labels)

    if not subsets_only:
        stage(progress, "SUBSETS", f"building {N_SUBSETS} chain subsets from the K-Means clusters ...")
    subsets, construct_sec, error_msg = build_subsets_timed(docs, labels, distances, embeddings)
    positions = positions_of(docs, subsets) if subsets else None
    if subsets:
        trace_subsets(docs, labels, positions)

    record = {
        **base,
        "status": "ok",
        "retrieval": {
            "retrieved_chunks": len(docs),
            "answer_bearing_chunk_found": retrieval_rank > 0,
            "first_answer_bearing_rank": retrieval_rank or None,
            "reciprocal_rank": rm["reciprocal_rank"],
            "rank_to_full_coverage": rm["rank_to_full_coverage"],
            "n_concept_groups": rm["n_concept_groups"],
            "missing_concepts": rm["missing_concepts"],
            "chunk_concept_matches": rm["chunk_concept_matches"],
            "group_labels": rm["group_labels"],
            "chunk_groups_found": rm["chunk_groups_found"],
            "at_k": rm["at_k"],
            "cluster_sizes": cluster_sizes(labels),
        },
    }
    if subsets_only:
        record["speculative_rag"] = (
            {"subset_positions": positions, "subset_error": None,
             **subset_structure(positions, embeddings, distances)}
            if subsets else {"subset_positions": None, "subset_error": error_msg}
        )
        record["subset_positions"] = record["speculative_rag"].get("subset_positions")
        record["subset_error"] = record["speculative_rag"].get("subset_error")
        emit(lambda full: ["  " + "-" * 70])
        return record

    token_tracker = {
        "question_tokens": q_tokens,
        "draft_openrouter_input": 0,
        "draft_openrouter_output": 0,
        "draft_openrouter_cost": 0.0,
        "draft_openrouter_cost_provider_reported_calls": 0,
        "draft_openrouter_cost_estimated_calls": 0,
        "verifier_openrouter_input": 0,
        "verifier_openrouter_output": 0,
        "verifier_openrouter_cost": 0.0,
        "verifier_openrouter_cost_provider_reported_calls": 0,
        "verifier_openrouter_cost_estimated_calls": 0,
    }

    # --- Speculative RAG (chain subsets) --------------------------------- #
    if not subsets:
        record["speculative_rag"] = failed_speculative(error_msg)
    else:
        structure = subset_structure(positions, embeddings, distances)
        try:
            started = time.perf_counter()
            generated = run_drafts_and_verification(
                question, subsets, token_tracker=token_tracker, method="SUBSETS", progress=progress
            )
            seconds_spec = time.perf_counter() - started
            trace_generation(generated, positions)
            record["speculative_rag"] = score_speculative(
                item, subsets, positions, generated, construct_sec, seconds_spec, structure
            )
        except Exception as error:
            # A failed subsets run must not stop the plain-RAG stage.
            if any(t in str(error).lower() for t in VERIFIER_DOWN_MARKERS):
                raise
            logger.exception("Subsets run failed for %s", item.get("id"))
            record["speculative_rag"] = failed_speculative(f"{type(error).__name__}: {error}")
    spec = record["speculative_rag"]
    if spec.get("status") == "ok":
        stage(progress, "SUBSETS", f"DONE in {construct_sec + spec['draft_and_verification_seconds']:.1f}s "
              f"-> {'CORRECT' if spec['final_correct'] else 'wrong'}")
    else:
        stage(progress, "SUBSETS", f"FAILED: {spec.get('error')}")

    # --- Plain RAG baseline ---------------------------------------------- #
    stage(progress, "PLAIN RAG", "STAGE 1/1 drafting from all retrieved chunks ...")
    started = time.perf_counter()
    try:
        baseline_answer = draft(
            question,
            docs,
            draft_idx="baseline",
            token_tracker=token_tracker,
            language=detect_language(question),
        )["answer"]
        baseline_error = None
    except Exception as error:
        if any(t in str(error).lower() for t in VERIFIER_DOWN_MARKERS):
            raise
        logger.exception("Plain RAG failed for %s", item.get("id"))
        baseline_answer, baseline_error = "", f"{type(error).__name__}: {error}"
        stage(progress, "PLAIN RAG", f"FAILED: {error}")
    emit(lambda full: ["", "  == PLAIN RAG (all retrieved chunks in one prompt) ==",
                       f"  Answer: {_clip(baseline_answer, full)}"])
    record["baseline"] = {
        "answer": baseline_answer,
        "correct": matches_item(baseline_answer, item),
        "answer_concept_recall": concept_coverage(baseline_answer, item),
        "seconds": time.perf_counter() - started,
        **({"status": "error", "error": baseline_error} if baseline_error else {}),
    }

    logger.info(
        "[EVAL TOKEN SUMMARY | %s] question_tokens=%d | final_answer_tokens=%d | "
        "OpenRouter drafting(in=%d, out=%d) | OpenRouter verification(in=%d, out=%d)",
        item.get("id", "?"),
        q_tokens,
        count_output_tokens(spec.get("answer", "")),
        token_tracker["draft_openrouter_input"],
        token_tracker["draft_openrouter_output"],
        token_tracker["verifier_openrouter_input"],
        token_tracker["verifier_openrouter_output"],
    )
    record["llm_usage"] = {
        "draft_model": DRAFT_MODEL,
        "verifier_model": VERIFY_MODEL,
        **token_tracker,
        "total_llm_cost_usd": (token_tracker["draft_openrouter_cost"]
                               + token_tracker["verifier_openrouter_cost"]),
    }
    logger.info(
        "[EVAL COST SUMMARY | %s] drafting=$%.10f (provider-reported=%d, estimated=%d) | "
        "verification=$%.10f (provider-reported=%d, estimated=%d) | total=$%.10f USD",
        item.get("id", "?"),
        token_tracker["draft_openrouter_cost"],
        token_tracker["draft_openrouter_cost_provider_reported_calls"],
        token_tracker["draft_openrouter_cost_estimated_calls"],
        token_tracker["verifier_openrouter_cost"],
        token_tracker["verifier_openrouter_cost_provider_reported_calls"],
        token_tracker["verifier_openrouter_cost_estimated_calls"],
        record["llm_usage"]["total_llm_cost_usd"],
    )

    def final_lines(full):
        selected = spec.get("best_draft_index")
        return [
            "", "  == FINAL ANSWERS ==",
            f"  Speculative RAG: {_clip(spec.get('answer', '(no answer)'), True)}",
            f"      -> selected draft {selected + 1 if selected is not None else '-'}, "
            f"{'CORRECT' if spec['final_correct'] else 'wrong'} by phrase matching",
            f"  Plain RAG      : {_clip(baseline_answer, True)}",
            "", "  == RESULT ==",
            f"  Expected       : {_clip(item['ground_truth'], full)}",
            f"  Speculative RAG: {'CORRECT' if spec['final_correct'] else 'wrong'}",
            f"  Plain RAG      : {'CORRECT' if record['baseline']['correct'] else 'wrong'}",
            "  " + "-" * 70,
        ]
    emit(final_lines)
    return record


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def pct(x):
    return f"{x:.1%}"


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def summary_row(rows):
    v = [r["speculative_rag"] for r in rows]
    return {
        "accuracy": mean(x["final_correct"] for x in v),
        "oracle": mean(x["oracle_draft_correct"] for x in v),
        "subset_hit": mean(x["subset_hit"] for x in v),
        "concept_recall": mean(x["answer_concept_recall"] for x in v),
        "best_subset_cov": mean(x["best_subset_concept_coverage"] for x in v),
        "distinct_chunks": mean(x["distinct_chunks_used"] for x in v),
        "overlap": mean(x["mean_subset_overlap"] for x in v),
        "intra_sim": mean(x["intra_subset_similarity"] for x in v),
        "chunk_distance": mean(x["mean_chunk_distance"] for x in v),
        "seconds": mean(
            x["subset_construction_seconds"] + x["draft_and_verification_seconds"] for x in v
        ),
    }


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def structure_table(rows):
    """Properties of the subsets themselves; works without any LLM calls."""
    v = [r["speculative_rag"] for r in rows
         if r["speculative_rag"].get("subset_positions") or r["speculative_rag"].get("subset_chunk_positions")]
    headers = ["Distinct chunks used", "Subset overlap (Jaccard)",
               "Intra-subset similarity (lower = less repeated)",
               "Mean chunk distance (lower = closer to question)"]
    if not v:
        return md_table(headers, [["-", "-", "-", "-"]])
    return md_table(headers, [[
        f"{mean(x['distinct_chunks_used'] for x in v):.1f}",
        f"{mean(x['mean_subset_overlap'] for x in v):.2f}",
        f"{mean(x['intra_subset_similarity'] for x in v):.3f}",
        f"{mean(x['mean_chunk_distance'] for x in v):.3f}",
    ]])


def _at(row, cutoff):
    """Metrics at a cutoff; a question with fewer chunks than the cutoff uses its largest one."""
    table = row["retrieval"]["at_k"]
    keys = sorted(int(c) for c in table)
    usable = [c for c in keys if c <= cutoff]
    return table[str(usable[-1] if usable else keys[0])]


def retrieval_section(ok, number=None):
    rows = [r for r in ok if r.get("retrieval", {}).get("at_k")]
    title = "Retrieval quality (no LLM calls)"
    out = ["", f"## {number}. {title}" if number else f"## {title}", ""]
    if not rows:
        return out + ["No retrieval metrics were recorded."]
    cutoffs = sorted({int(c) for r in rows for c in r["retrieval"]["at_k"]})
    top = cutoffs[-1]

    table = [
        [f"top-{c}"] + [pct(mean(_at(r, c)[key] for r in rows))
                        for key in ("hit", "concept_hit", "recall", "precision", "ndcg")]
        for c in cutoffs
    ]
    out.append(md_table(
        ["Cutoff", "Hit@n", "Concept hit@n", "Concept recall@n", "Precision@n", "nDCG@n"], table))

    full_ranks = [r["retrieval"]["first_answer_bearing_rank"] for r in rows
                  if r["retrieval"]["first_answer_bearing_rank"]]
    cover_ranks = [r["retrieval"]["rank_to_full_coverage"] for r in rows
                   if r["retrieval"]["rank_to_full_coverage"]]
    never = [r for r in rows if not r["retrieval"]["rank_to_full_coverage"]]
    out += [
        "",
        f"- **MRR** (first chunk holding every concept group): "
        f"**{mean(r['retrieval']['reciprocal_rank'] for r in rows):.3f}**",
        f"- Mean rank of that chunk, when one exists: "
        f"{mean(full_ranks):.2f} ({len(full_ranks)} of {len(rows)} questions)",
        f"- Mean rank to full concept coverage, when reached: "
        f"{mean(cover_ranks):.2f} ({len(cover_ranks)} of {len(rows)} questions)",
        f"- Questions where some concept is never retrieved: **{len(never)}** of {len(rows)}",
        "",
        "_Hit@n needs one chunk to hold the whole answer; Concept hit@n and recall@n let the "
        "answer be spread over several chunks. Relevance is judged from `required_concepts` "
        "(or `ground_truth`), so these are proxies for gold-label metrics. nDCG ranks the "
        "retrieved chunks among themselves and cannot see chunks that were not retrieved._",
    ]

    level_rows = []
    for level in ("easy", "medium", "hard"):
        group = [r for r in rows if r.get("level") == level]
        if group:
            level_rows.append([
                f"{level.title()} ({len(group)})",
                pct(mean(_at(r, top)["hit"] for r in group)),
                pct(mean(_at(r, top)["concept_hit"] for r in group)),
                pct(mean(_at(r, top)["recall"] for r in group)),
                f"{mean(r['retrieval']['reciprocal_rank'] for r in group):.3f}",
            ])
    if level_rows:
        out += ["", f"**By difficulty (top-{top}):**", "", md_table(
            ["Level", f"Hit@{top}", f"Concept hit@{top}", f"Concept recall@{top}", "MRR"], level_rows)]

    failures = [r for r in rows if _at(r, top)["recall"] < 1.0]
    if failures:
        out += ["", f"**Questions with incomplete concept recall at top-{top}:**", "", md_table(
            ["ID", "Level", f"Recall@{top}", "Concept groups never retrieved"],
            [[r["id"], r.get("level", "?"), pct(_at(r, top)["recall"]),
              "; ".join(r["retrieval"]["missing_concepts"]) or "-"] for r in failures])]
    return out


def build_report(results, k, subsets_only):
    ok = [r for r in results if r.get("status") == "ok"]
    out = ["# Evaluation: Speculative RAG vs Plain RAG", ""]
    out.append(
        f"Questions completed: **{len(ok)}** of {len(results)} | retrieval k = {k} | "
        f"clusters = {N_CLUSTERS} | subsets = {N_SUBSETS}"
    )
    if not ok:
        out.append("\nNo completed questions, so no metrics can be calculated.")
        return "\n".join(out)

    if subsets_only:
        out += ["", "_Subsets-only run: no drafts or verification were executed, so no "
                "accuracy metrics were produced. The table below describes the subsets only._",
                "", "## Subset structure", "", structure_table(ok)]
        out += retrieval_section(ok, 2)
        return "\n".join(out)

    base_acc = mean(r["baseline"]["correct"] for r in ok)
    base_recall = mean(r["baseline"]["answer_concept_recall"] for r in ok)
    base_sec = mean(r["baseline"]["seconds"] for r in ok)
    mrr = mean(r["retrieval"]["reciprocal_rank"] for r in ok)
    spec = summary_row(ok)

    rows = [
        ["Speculative RAG", pct(spec["accuracy"]), pct(spec["oracle"]), pct(spec["subset_hit"]),
         pct(spec["concept_recall"]), pct(spec["best_subset_cov"]),
         f"{spec['distinct_chunks']:.1f}", f"{spec['overlap']:.2f}", f"{spec['seconds']:.1f}"],
        ["Plain RAG (baseline)", pct(base_acc), "-", "-", pct(base_recall), "-", "-", "-",
         f"{base_sec:.1f}"],
    ]
    out += ["", "## 1. Overall results (all questions)", "", md_table(
        ["System", "Final acc.", "Oracle", "Subset hit", "Concept recall",
         "Best-subset cov.", "Distinct chunks", "Subset overlap", "Avg sec"], rows,
    ), "", f"Retrieval MRR: **{mrr:.3f}**"]
    language_rows = []
    for language, label in (("en", "English"), ("ar", "Arabic")):
        group = [r for r in ok if r.get("language") == language]
        if group:
            language_rows.append([
                f"{label} ({len(group)})",
                pct(summary_row(group)["accuracy"]),
                pct(mean(r["baseline"]["correct"] for r in group)),
                pct(summary_row(group)["concept_recall"]),
            ])
    if language_rows:
        out += ["", "## 2. Accuracy by language", "",
                md_table(["Language", "Speculative RAG", "Plain RAG", "Concept recall"], language_rows)]
    out += retrieval_section(ok, 2)

    usage_rows = [r.get("llm_usage", {}) for r in ok if r.get("llm_usage")]
    if usage_rows:
        draft_cost = sum(float(u.get("draft_openrouter_cost", 0.0)) for u in usage_rows)
        verify_cost = sum(float(u.get("verifier_openrouter_cost", 0.0)) for u in usage_rows)
        total_cost = sum(float(u.get("total_llm_cost_usd", 0.0)) for u in usage_rows)
        draft_in = sum(int(u.get("draft_openrouter_input", 0)) for u in usage_rows)
        draft_out = sum(int(u.get("draft_openrouter_output", 0)) for u in usage_rows)
        verify_in = sum(int(u.get("verifier_openrouter_input", 0)) for u in usage_rows)
        verify_out = sum(int(u.get("verifier_openrouter_output", 0)) for u in usage_rows)
        out += ["", "## 3. LLM usage and estimated/charged cost", "",
                f"- Drafting model (**{DRAFT_MODEL}**): {draft_in:,} input tokens, "
                f"{draft_out:,} output tokens, **${draft_cost:.10f}**.",
                f"- Verification model (**{VERIFY_MODEL}**): {verify_in:,} input tokens, "
                f"{verify_out:,} output tokens, **${verify_cost:.10f}**.",
                f"- **Total LLM cost: ${total_cost:.10f} USD** across {len(usage_rows)} questions.",
                "- Per-request costs use OpenRouter's reported `usage.cost` when available; "
                "otherwise they are estimated from configured per-million-token rates."]
        out += [""]
        difficulty_section = 4
    else:
        difficulty_section = 3

    out += ["", f"## {difficulty_section}. Accuracy by difficulty", ""]
    level_rows = []
    for level in ("easy", "medium", "hard"):
        group = [r for r in ok if r.get("level") == level]
        if group:
            level_rows.append([
                f"{level.title()} ({len(group)})",
                pct(summary_row(group)["accuracy"]),
                pct(mean(r["baseline"]["correct"] for r in group)),
            ])
    out.append(md_table(["Level", "Speculative RAG", "Plain RAG"], level_rows))

    wins = sum(r["speculative_rag"]["final_correct"] and not r["baseline"]["correct"] for r in ok)
    losses = sum(r["baseline"]["correct"] and not r["speculative_rag"]["final_correct"] for r in ok)
    out += ["", f"## {difficulty_section + 1}. Head-to-head vs Plain RAG (final answer correct)", "",
            md_table(["Speculative better", "Plain RAG better", "Same outcome", "Exact sign test p"],
                     [[wins, losses, len(ok) - wins - losses, f"{sign_test_p(wins, losses):.3f}"]])]

    section = difficulty_section + 2
    out += ["", f"## {section}. Per-question results", "", md_table(
        ["ID", "Level", "Cluster sizes", "Speculative RAG", "Plain RAG"],
        [[r["id"], r["level"], "/".join(map(str, r["retrieval"]["cluster_sizes"].values())),
          "✓" if r["speculative_rag"]["final_correct"] else "✗",
          "✓" if r["baseline"]["correct"] else "✗"] for r in ok],
    )]
    section += 1

    out += ["", f"## {section}. Subset structure", "", structure_table(ok)]
    section += 1

    out += ["", f"## {section}. Reading the results", "",
            f"- **Speculative RAG**: final accuracy {pct(spec['accuracy'])}, "
            f"concept recall {pct(spec['concept_recall'])}."]
    if spec["accuracy"] > base_acc:
        out.append(f"Speculative RAG scored higher on final accuracy ({pct(spec['accuracy'])} vs "
                   f"{pct(base_acc)}).")
    elif spec["accuracy"] < base_acc:
        out.append(f"Plain RAG scored higher on final accuracy ({pct(base_acc)} vs "
                   f"{pct(spec['accuracy'])}).")
    else:
        out.append(f"Final accuracy is tied ({pct(spec['accuracy'])}); compare concept recall "
                   f"({pct(spec['concept_recall'])} vs {pct(base_recall)}) above.")
    return "\n".join(out)



# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def evaluate():
    global N_CLUSTERS, TRACE, TRACE_CHARS, EVAL_PATH
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--k", type=int, default=K, help=f"chunks to retrieve (default {K})")
    parser.add_argument("--clusters", type=int, default=N_CLUSTERS,
                        help=f"number of K-Means clusters (default {N_CLUSTERS})")
    parser.add_argument("--quiet", action="store_true",
                        help="turn off the step-by-step trace (chunks, clusters, subsets, drafts)")
    parser.add_argument("--chunk-chars", type=int, default=TRACE_CHARS,
                        help="clip each chunk to N characters in the terminal (default 0 = show everything). "
                             "The saved trace file always has the full text.")
    parser.add_argument("--ground-truth", default=str(EVAL_PATH),
                        help="ground-truth json; every question in it is run unless "
                             "--ids/--levels/--limit narrow it (default: %(default)s)")
    parser.add_argument("--subsets-only", action="store_true",
                        help="build clusters/subsets but skip all LLM calls")
    parser.add_argument("--ids", nargs="*", help="only these question ids, e.g. E01 H02")
    parser.add_argument("--levels", nargs="*", help="only these levels: easy medium hard")
    parser.add_argument("--limit", type=int, help="only the first N selected questions")
    args = parser.parse_args()
    if args.clusters != 2:
        raise SystemExit(
            f"--clusters {args.clusters} is not supported: build_subsets() takes chunks from "
            "only the first TWO clusters, so any extra cluster would be silently dropped "
            "(together with whatever chunks it holds). Use --clusters 2, or change "
            "build_subsets() first."
        )
    N_CLUSTERS = args.clusters
    EVAL_PATH = Path(args.ground_truth)
    TRACE, TRACE_CHARS = not args.quiet, args.chunk_chars
    # Traces are appended to, never truncated: earlier runs stay in the files.
    if TRACE:
        begin_trace_file(TRACE_PATH, "Evaluation trace (full chunk text)")
    if TRACE_RETRIEVAL_EVAL:
        begin_trace_file(RETRIEVAL_TRACE_PATH, "Retrieval evaluation trace")

    with open(EVAL_PATH, encoding="utf-8") as file:
        items = json.load(file)
    if args.ids:
        items = [i for i in items if i.get("id") in set(args.ids)]
    if args.levels:
        items = [i for i in items if i.get("level") in set(args.levels)]
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise SystemExit("No questions selected.")

    total_in_file = len(json.loads(EVAL_PATH.read_text(encoding="utf-8")))
    print(f"Loaded {len(items)} of {total_in_file} questions from {EVAL_PATH}")
    print(f"Retrieval k={args.k}; clusters={N_CLUSTERS}; subsets={N_SUBSETS}; "
          f"draft model={DRAFT_MODEL}; draft reasoning={DRAFT_REASONING}")
    if not args.subsets_only and MIN_INTERVAL > 0:
        print(f"Verifier pacing: {MIN_INTERVAL:g}s minimum interval between calls.")

    build_index()
    indexed = list_documents()
    unknown = sorted({i["source"] for i in items if i.get("source") and i["source"] not in indexed})
    if unknown:
        raise ValueError(
            f"The following source PDFs are not indexed: {unknown}. "
            f"Indexed documents: {sorted(indexed)}. Place them in the project's pdfs folder and rerun."
        )

    results = []
    for index, item in enumerate(items, start=1):
        print(f"\n[{index}/{len(items)}] [{item.get('level', '?')}] {item.get('id', '')} "
              f"{item['question']}", flush=True)
        progress = f"[{index}/{len(items)}] {item.get('id', '')} "
        try:
            results.append(score_one_question(item, args.k, args.subsets_only, progress))
        except Exception as error:
            logger.exception("Question %s failed", item.get("id"))
            results.append({"id": item.get("id"), "question": item["question"],
                            "level": item.get("level"), "status": "error",
                            "evaluation_error": str(error)})
            text = str(error).lower()
            if any(t in text for t in VERIFIER_DOWN_MARKERS):
                print("Stopping: verifier is unavailable or credits are exhausted.")
                break
        with open(RESULTS_PATH, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False, default=str)

    done = [r for r in results if r.get("status") == "ok"]
    emit_retrieval([f"### RETRIEVAL EVAL SUMMARY | {len(done)} of {len(results)} questions | k={args.k}"]
                   + retrieval_section(done)[1:])

    report = build_report(results, args.k, args.subsets_only)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nDetailed results: {RESULTS_PATH}\nReadable report : {REPORT_PATH}")
    if TRACE:
        print(f"Full trace      : {TRACE_PATH}")
    if TRACE_RETRIEVAL_EVAL:
        print(f"Retrieval trace : {RETRIEVAL_TRACE_PATH}")


if __name__ == "__main__":
    evaluate()