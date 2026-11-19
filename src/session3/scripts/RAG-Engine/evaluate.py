"""
Evaluation of Speculative RAG against a plain-RAG baseline.
One run produces eval_results.json, eval_report.md, and eval_trace.md.

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
Plus retrieval MRR, accuracy by difficulty, and a paired comparison (wins / losses /
ties and an exact sign test) against Plain RAG.

Outputs (in JSON-evaluation-results/)
  eval_results.json ... every answer, score and subset position
  eval_report.md ...... the readable comparison tables
  eval_trace.md ....... step-by-step trace with the FULL text of every chunk:
                        retrieved chunks, clusters, subsets, drafts, final answers

Usage
  python evaluate.py                   # full evaluation
  python evaluate.py --subsets-only    # no LLM calls: inspect clusters/subsets
  python evaluate.py --k 8             # fewer chunks
  python evaluate.py --ids E01 H02     # only some questions
  python evaluate.py --levels hard     # only some levels
  python evaluate.py --limit 3         # first N selected questions
  python evaluate.py --quiet           # no step-by-step trace in the terminal
  python evaluate.py --chunk-chars 300 # clip chunks in the terminal (default: show
                                       # everything; eval_trace.md is always full)
"""
import argparse
import json
import logging
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
    VERIFY_MODEL,
)

EVAL_PATH = SESSION_DIR / "JSON-Test-cases/ground_truth.json"
RESULTS_PATH = SESSION_DIR / "JSON-evaluation-results/eval_results.json"
REPORT_PATH = RESULTS_PATH.with_name("eval_report.md")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

K = DEFAULT_RETRIEVAL_K
N_SUBSETS = DEFAULT_N_SUBSETS
N_CLUSTERS = DEFAULT_N_CLUSTERS
STRATEGIES = ["chain"]  # the only enabled subset strategy
TRACE = True            # print a human-readable trace of every step (see --quiet)
TRACE_CHARS = 0         # 0 = show every chunk in full; set --chunk-chars N to clip the terminal
TRACE_PATH = RESULTS_PATH.with_name("eval_trace.md")


def contains(text, truths):
    """Return True if an accepted ground-truth phrase appears in text."""
    if isinstance(truths, str):
        truths = [truths]
    text_norm = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    for truth in truths:
        needle = re.sub(r"[^a-z0-9]+", " ", str(truth).lower()).strip()
        if needle and re.search(
            rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])",
            text_norm,
        ):
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
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    for alternatives in groups:
        if not any(
            re.search(
                rf"(?<![a-z0-9]){re.escape(re.sub(r'[^a-z0-9]+', ' ', phrase.lower()).strip())}(?![a-z0-9])",
                normalized,
            )
            for phrase in alternatives
        ):
            return False
    return True


def run_drafts_and_verification(question, subsets, token_tracker=None):
    """Write one draft per subset with OpenRouter, then verify them with OpenRouter."""
    with ThreadPoolExecutor(max_workers=len(subsets)) as pool:
        futures = [
            pool.submit(draft, question, subset, i + 1, token_tracker)
            for i, subset in enumerate(subsets)
        ]
        drafts = [f.result() for f in futures]
    scores, best, tie_reason, final_answer = verify(
        question, drafts, subsets, token_tracker=token_tracker
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


def strategy_header(strategy, note=""):
    emit(lambda full: ["", f"  ######## SUBSET STRATEGY: {strategy}{note} ########"])


def score_one_question(item, k, subsets_only, strategies):
    question = item["question"]
    source = item.get("source") or None
    q_tokens = count_question_tokens(question)
    logger.info("Evaluating %s | question_tokens=%d | strategies=%s",
                item.get("id", "?"), q_tokens, ",".join(strategies))

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
    }
    if not docs:
        return {**base, "status": "no_retrieval"}

    labels = cluster(embeddings, n_clusters=N_CLUSTERS)

    retrieval_rank = next(
        (rank for rank, doc in enumerate(docs, start=1) if matches_item(doc, item)), 0
    )

    trace_retrieval(docs, metas, distances, k)
    trace_clusters(docs, labels)

    # Same retrieval and clusters for every strategy; only the subsets differ.
    built = {}
    for strategy in strategies:
        subsets, construct_sec, error_msg = build_subsets_timed(
            docs, labels, distances, embeddings
        )
        positions = positions_of(docs, subsets) if subsets else None
        built[strategy] = (subsets, positions, construct_sec, error_msg)
        strategy_header(strategy)
        if subsets:
            trace_subsets(docs, labels, positions)

    record = {
        **base,
        "status": "ok",
        "retrieval": {
            "retrieved_chunks": len(docs),
            "answer_bearing_chunk_found": retrieval_rank > 0,
            "first_answer_bearing_rank": retrieval_rank or None,
            "reciprocal_rank": 1.0 / retrieval_rank if retrieval_rank else 0.0,
            "cluster_sizes": cluster_sizes(labels),
        },
    }
    if subsets_only:
        record["strategies"] = {}
        for strategy in strategies:
            subsets, positions, _, error_msg = built[strategy]
            record["strategies"][strategy] = (
                {"subset_positions": positions, "subset_error": None,
                 **subset_structure(positions, embeddings, distances)}
                if subsets else {"subset_positions": None, "subset_error": error_msg}
            )
        first = record["strategies"][strategies[0]]
        record["subset_positions"] = first.get("subset_positions")
        record["subset_error"] = first.get("subset_error")
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

    # --- Speculative RAG, once per strategy ------------------------------ #
    record["strategies"] = {}
    generated_by_strategy = {}
    seen = {}   # identical subsets -> reuse the drafts instead of paying (and varying) twice
    for strategy in strategies:
        subsets, positions, construct_sec, error_msg = built[strategy]
        if not subsets:
            record["strategies"][strategy] = failed_speculative(error_msg)
            continue
        signature = tuple(tuple(pos) for pos in positions)
        if signature in seen:
            twin = seen[signature]
            result = dict(record["strategies"][twin])
            result["reused_from"] = twin
            result["subset_construction_seconds"] = construct_sec
            generated_by_strategy[strategy] = generated_by_strategy[twin]
            strategy_header(strategy, f" (identical subsets to {twin}: drafts reused)")
        else:
            structure = subset_structure(positions, embeddings, distances)
            strategy_header(strategy, " (drafting and verification)")
            started = time.perf_counter()
            generated = run_drafts_and_verification(question, subsets, token_tracker=token_tracker)
            seconds_spec = time.perf_counter() - started
            trace_generation(generated, positions)
            result = score_speculative(
                item, subsets, positions, generated, construct_sec, seconds_spec, structure
            )
            generated_by_strategy[strategy] = generated
            seen[signature] = strategy
        record["strategies"][strategy] = result
    # first strategy, kept under the old key so older tooling still finds it
    record["speculative_rag"] = record["strategies"][strategies[0]]

    # --- Plain RAG baseline (identical for every strategy, so run once) --- #
    started = time.perf_counter()
    baseline_answer = draft(question, docs, draft_idx="baseline", token_tracker=token_tracker)["answer"]
    emit(lambda full: ["", "  == PLAIN RAG (all retrieved chunks in one prompt) ==",
                       f"  Answer: {_clip(baseline_answer, full)}"])
    record["baseline"] = {
        "answer": baseline_answer,
        "correct": matches_item(baseline_answer, item),
        "answer_concept_recall": concept_coverage(baseline_answer, item),
        "seconds": time.perf_counter() - started,
    }

    final_out_tokens = {
        s: count_output_tokens(record["strategies"][s].get("answer", "")) for s in strategies
    }
    logger.info(
        "[EVAL TOKEN SUMMARY | %s] question_tokens=%d | final_answer_tokens=%s | "
        "OpenRouter drafting(in=%d, out=%d) | OpenRouter verification(in=%d, out=%d)",
        item.get("id", "?"),
        q_tokens,
        final_out_tokens,
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
        lines = ["", "  == FINAL ANSWERS =="]
        for strategy in strategies:
            r = record["strategies"][strategy]
            lines += [
                f"  Speculative RAG [{strategy}]: {_clip(r.get('answer', '(no answer)'), True)}",
                f"      -> selected draft {r.get('best_draft_index', -1) + 1}, "
                f"{'CORRECT' if r['final_correct'] else 'wrong'} by phrase matching",
            ]
        lines += [
            f"  Plain RAG      : {_clip(baseline_answer, True)}",
            "", "  == RESULT ==",
            f"  Expected       : {_clip(item['ground_truth'], full)}",
        ]
        for strategy in strategies:
            ok = record["strategies"][strategy]["final_correct"]
            lines.append(f"  {strategy:<9}: {'CORRECT' if ok else 'wrong'}")
        lines += [f"  Plain RAG: {'CORRECT' if record['baseline']['correct'] else 'wrong'}",
                  "  " + "-" * 70]
        return lines
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


def summary_row(rows, strategy):
    v = [r["strategies"][strategy] for r in rows]
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


def structure_table(rows, strategies):
    """Properties of the subsets themselves; works without any LLM calls."""
    table = []
    for strategy in strategies:
        v = [r["strategies"][strategy] for r in rows if r["strategies"][strategy].get("subset_positions")
             or r["strategies"][strategy].get("subset_chunk_positions")]
        if not v:
            table.append([strategy, "-", "-", "-", "-"])
            continue
        table.append([
            strategy,
            f"{mean(x['distinct_chunks_used'] for x in v):.1f}",
            f"{mean(x['mean_subset_overlap'] for x in v):.2f}",
            f"{mean(x['intra_subset_similarity'] for x in v):.3f}",
            f"{mean(x['mean_chunk_distance'] for x in v):.3f}",
        ])
    return md_table(
        ["Strategy", "Distinct chunks used", "Subset overlap (Jaccard)",
         "Intra-subset similarity (lower = less repeated)", "Mean chunk distance (lower = closer to question)"],
        table,
    )


def positions_for(record, strategy):
    r = record["strategies"][strategy]
    return r.get("subset_positions") or r.get("subset_chunk_positions")


def build_report(results, k, subsets_only):
    ok = [r for r in results if r.get("status") == "ok"]
    strategies = list(ok[0]["strategies"]) if ok else list(STRATEGIES)
    multi = len(strategies) > 1
    title = "Subset strategies vs Plain RAG" if multi else "Speculative RAG vs Plain RAG"
    out = [f"# Evaluation: {title}", ""]
    out.append(
        f"Questions completed: **{len(ok)}** of {len(results)} | retrieval k = {k} | "
        f"clusters = {N_CLUSTERS} | subsets = {N_SUBSETS} | strategies = {', '.join(strategies)}"
    )
    if not ok:
        out.append("\nNo completed questions, so no metrics can be calculated.")
        return "\n".join(out)

    if subsets_only:
        out += ["", "_Subsets-only run: no drafts or verification were executed, so no "
                "accuracy metrics were produced. The table below describes the subsets only._",
                "", "## Subset structure", "", structure_table(ok, strategies)]
        return "\n".join(out)

    base_acc = mean(r["baseline"]["correct"] for r in ok)
    base_recall = mean(r["baseline"]["answer_concept_recall"] for r in ok)
    base_sec = mean(r["baseline"]["seconds"] for r in ok)
    mrr = mean(r["retrieval"]["reciprocal_rank"] for r in ok)
    spec = {s: summary_row(ok, s) for s in strategies}

    rows = [
        [f"Speculative RAG [{s}]", pct(m["accuracy"]), pct(m["oracle"]), pct(m["subset_hit"]),
         pct(m["concept_recall"]), pct(m["best_subset_cov"]),
         f"{m['distinct_chunks']:.1f}", f"{m['overlap']:.2f}", f"{m['seconds']:.1f}"]
        for s, m in spec.items()
    ]
    rows.append(["Plain RAG (baseline)", pct(base_acc), "-", "-", pct(base_recall), "-", "-", "-",
                 f"{base_sec:.1f}"])
    out += ["", "## 1. Overall results (all questions)", "", md_table(
        ["System", "Final acc.", "Oracle", "Subset hit", "Concept recall",
         "Best-subset cov.", "Distinct chunks", "Subset overlap", "Avg sec"], rows,
    ), "", f"Retrieval MRR: **{mrr:.3f}**"]

    usage_rows = [r.get("llm_usage", {}) for r in ok if r.get("llm_usage")]
    if usage_rows:
        draft_cost = sum(float(u.get("draft_openrouter_cost", 0.0)) for u in usage_rows)
        verify_cost = sum(float(u.get("verifier_openrouter_cost", 0.0)) for u in usage_rows)
        total_cost = sum(float(u.get("total_llm_cost_usd", 0.0)) for u in usage_rows)
        draft_in = sum(int(u.get("draft_openrouter_input", 0)) for u in usage_rows)
        draft_out = sum(int(u.get("draft_openrouter_output", 0)) for u in usage_rows)
        verify_in = sum(int(u.get("verifier_openrouter_input", 0)) for u in usage_rows)
        verify_out = sum(int(u.get("verifier_openrouter_output", 0)) for u in usage_rows)
        out += ["", "## 2. LLM usage and estimated/charged cost", "",
                f"- Drafting model (**{DRAFT_MODEL}**): {draft_in:,} input tokens, "
                f"{draft_out:,} output tokens, **${draft_cost:.10f}**.",
                f"- Verification model (**{VERIFY_MODEL}**): {verify_in:,} input tokens, "
                f"{verify_out:,} output tokens, **${verify_cost:.10f}**.",
                f"- **Total LLM cost: ${total_cost:.10f} USD** across {len(usage_rows)} questions.",
                "- Per-request costs use OpenRouter's reported `usage.cost` when available; "
                "otherwise they are estimated from configured per-million-token rates."]
        out += [""]
        difficulty_section = 3
    else:
        difficulty_section = 2

    out += ["", f"## {difficulty_section}. Accuracy by difficulty", ""]
    level_rows = []
    for level in ("easy", "medium", "hard"):
        group = [r for r in ok if r.get("level") == level]
        if group:
            level_rows.append(
                [f"{level.title()} ({len(group)})"]
                + [pct(summary_row(group, s)["accuracy"]) for s in strategies]
                + [pct(mean(r["baseline"]["correct"] for r in group))]
            )
    out.append(md_table(["Level"] + [f"Speculative [{s}]" for s in strategies] + ["Plain RAG"],
                        level_rows))

    h2h = []
    for s in strategies:
        wins = sum(r["strategies"][s]["final_correct"] and not r["baseline"]["correct"] for r in ok)
        losses = sum(r["baseline"]["correct"] and not r["strategies"][s]["final_correct"] for r in ok)
        h2h.append([s, wins, losses, len(ok) - wins - losses, f"{sign_test_p(wins, losses):.3f}"])
    out += ["", f"## {difficulty_section + 1}. Head-to-head vs Plain RAG (final answer correct)", "", md_table(
        ["Strategy", "Speculative better", "Plain RAG better", "Same outcome", "Exact sign test p"], h2h,
    )]

    section = difficulty_section + 2
    if multi:
        pair_rows = []
        for i, a in enumerate(strategies):
            for b in strategies[i + 1:]:
                wins = sum(r["strategies"][a]["final_correct"] and not r["strategies"][b]["final_correct"]
                           for r in ok)
                losses = sum(r["strategies"][b]["final_correct"] and not r["strategies"][a]["final_correct"]
                             for r in ok)
                identical = sum(
                    positions_for(r, a) is not None and positions_for(r, a) == positions_for(r, b)
                    for r in ok
                )
                pair_rows.append([
                    f"{a} vs {b}", wins, losses, len(ok) - wins - losses,
                    f"{sign_test_p(wins, losses):.3f}",
                    f"{pct(spec[a]['concept_recall'])} / {pct(spec[b]['concept_recall'])}",
                    f"{identical} of {len(ok)}",
                ])
        out += ["", f"## {section}. Strategy vs strategy (final answer correct)", "", md_table(
            ["Comparison", "First better", "Second better", "Same outcome", "Exact sign test p",
             "Concept recall (first / second)", "Questions with identical subsets"], pair_rows,
        ), "", "_Questions where two strategies built identical subsets reuse the same drafts, so "
                "they can never count as a win for either side._"]
        section += 1

    def mark(r, s):
        return "✓" if r["strategies"][s]["final_correct"] else "✗"

    out += ["", f"## {section}. Per-question results", "", md_table(
        ["ID", "Level", "Cluster sizes"] + strategies + ["Plain RAG"],
        [[r["id"], r["level"], "/".join(map(str, r["retrieval"]["cluster_sizes"].values()))]
         + [mark(r, s) for s in strategies]
         + ["✓" if r["baseline"]["correct"] else "✗"] for r in ok],
    )]
    section += 1

    out += ["", f"## {section}. Subset structure", "", structure_table(ok, strategies)]
    section += 1

    out += ["", f"## {section}. Reading the results", ""]
    best = max(strategies, key=lambda s: (spec[s]["accuracy"], spec[s]["concept_recall"]))
    for s in strategies:
        out.append(f"- **{s}**: final accuracy {pct(spec[s]['accuracy'])}, "
                   f"concept recall {pct(spec[s]['concept_recall'])}.")
    if multi:
        tied = [s for s in strategies if spec[s]["accuracy"] == spec[best]["accuracy"]]
        if len(tied) > 1:
            out.append(f"- Final accuracy is tied between {', '.join(tied)}; "
                       "compare concept recall and the structure table.")
        else:
            out.append(f"- Highest final accuracy: **{best}**.")
        out.append(f"- With {len(ok)} questions, a gap of one or two questions is within noise "
                   "(the drafts come from an LLM and are not perfectly repeatable). "
                   "Trust the sign-test p-values in the strategy-vs-strategy table, "
                   "not the raw percentages.")
    elif spec[best]["accuracy"] > base_acc:
        out.append(f"Speculative RAG scored higher on final accuracy ({pct(spec[best]['accuracy'])} vs "
                   f"{pct(base_acc)}).")
    elif spec[best]["accuracy"] < base_acc:
        out.append(f"Plain RAG scored higher on final accuracy ({pct(base_acc)} vs "
                   f"{pct(spec[best]['accuracy'])}).")
    else:
        out.append(f"Final accuracy is tied ({pct(spec[best]['accuracy'])}); compare concept recall "
                   f"({pct(spec[best]['concept_recall'])} vs {pct(base_recall)}) above.")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def evaluate():
    global N_CLUSTERS, TRACE, TRACE_CHARS, STRATEGIES
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--k", type=int, default=K, help=f"chunks to retrieve (default {K})")
    parser.add_argument("--clusters", type=int, default=N_CLUSTERS,
                        help=f"number of K-Means clusters (default {N_CLUSTERS})")
    parser.add_argument("--quiet", action="store_true",
                        help="turn off the step-by-step trace (chunks, clusters, subsets, drafts)")
    parser.add_argument("--chunk-chars", type=int, default=TRACE_CHARS,
                        help="clip each chunk to N characters in the terminal (default 0 = show everything). "
                             "The saved trace file always has the full text.")
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
    STRATEGIES = ["chain"]
    TRACE, TRACE_CHARS = not args.quiet, args.chunk_chars
    if TRACE:
        TRACE_PATH.write_text("# Evaluation trace (full chunk text)\n", encoding="utf-8")

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

    print(f"Loaded {len(items)} questions from {EVAL_PATH}")
    print(f"Retrieval k={args.k}; clusters={N_CLUSTERS}; subsets={N_SUBSETS}; "
          f"strategies={', '.join(STRATEGIES)}")
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
        try:
            results.append(score_one_question(item, args.k, args.subsets_only, STRATEGIES))
        except Exception as error:
            logger.exception("Question %s failed", item.get("id"))
            results.append({"id": item.get("id"), "question": item["question"],
                            "level": item.get("level"), "status": "error",
                            "evaluation_error": str(error)})
            text = str(error).lower()
            if any(t in text for t in ("verifier quota is exhausted", "resource_exhausted",
                                       "verifier model")):
                print("Stopping: verifier is unavailable or credits are exhausted.")
                break
        with open(RESULTS_PATH, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False, default=str)

    report = build_report(results, args.k, args.subsets_only)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nDetailed results: {RESULTS_PATH}\nReadable report : {REPORT_PATH}")
    if TRACE:
        print(f"Full trace      : {TRACE_PATH}")


if __name__ == "__main__":
    evaluate()