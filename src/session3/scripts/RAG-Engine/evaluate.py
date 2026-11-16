"""
Evaluation of Speculative RAG against a plain-RAG baseline.
One run produces eval_results.json, eval_report.md, and eval_trace.md.

For every question in the ground-truth file the script:
  1. retrieves the top-k chunks and runs K-Means clustering,
  2. builds overlapping subsets from the K-Means clusters,
  3. writes one draft per subset with the local Ollama model and lets the
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
)

EVAL_PATH = SESSION_DIR / "JSON-Test-cases/ground_truth.json"
RESULTS_PATH = SESSION_DIR / "JSON-evaluation-results/eval_results.json"
REPORT_PATH = RESULTS_PATH.with_name("eval_report.md")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

K = DEFAULT_RETRIEVAL_K
N_SUBSETS = DEFAULT_N_SUBSETS
N_CLUSTERS = DEFAULT_N_CLUSTERS
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
    """Write one draft per subset with Ollama, then verify them with the OpenRouter verifier."""
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
    """Build subsets; return (subsets, seconds, error_text)."""
    started = time.perf_counter()
    try:
        subsets = build_subsets(
            docs, labels, n_subsets=N_SUBSETS, distances=distances, embeddings=embeddings
        )
        return subsets, time.perf_counter() - started, None
    except Exception as error:
        logger.exception("Subset construction failed")
        return None, time.perf_counter() - started, str(error)


def score_speculative(item, subsets, positions, generated, construction_seconds, generation_seconds):
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
        "distinct_chunks_used": len({p for pos in positions for p in pos}),
        "mean_subset_overlap": mean_pairwise_jaccard(positions),
        "subset_construction_seconds": construction_seconds,
        "draft_and_verification_seconds": generation_seconds,
    }


def failed_speculative(error):
    return {
        "status": "error", "error": error, "final_correct": False,
        "oracle_draft_correct": False, "subset_hit": False,
        "answer_concept_recall": 0.0, "best_subset_concept_coverage": 0.0,
        "distinct_chunks_used": 0, "mean_subset_overlap": 1.0,
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


def score_one_question(item, k, subsets_only):
    question = item["question"]
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
    }
    if not docs:
        return {**base, "status": "no_retrieval"}

    labels = cluster(embeddings, n_clusters=N_CLUSTERS)

    retrieval_rank = next(
        (rank for rank, doc in enumerate(docs, start=1) if matches_item(doc, item)), 0
    )

    subsets, construct_sec, error_msg = build_subsets_timed(docs, labels, distances, embeddings)
    positions = positions_of(docs, subsets) if subsets else None

    trace_retrieval(docs, metas, distances, k)
    trace_clusters(docs, labels)
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
        record["subset_positions"] = positions
        record["subset_error"] = error_msg
        emit(lambda full: ["  " + "-" * 70])
        return record

    token_tracker = {
        "question_tokens": q_tokens,
        "ollama_input": 0,
        "ollama_output": 0,
        "openrouter_input": 0,
        "openrouter_output": 0,
    }

    # --- Speculative RAG ------------------------------------------------ #
    if subsets:
        started = time.perf_counter()
        generated = run_drafts_and_verification(question, subsets, token_tracker=token_tracker)
        seconds_spec = time.perf_counter() - started
        trace_generation(generated, positions)
        record["speculative_rag"] = score_speculative(
            item, subsets, positions, generated, construct_sec, seconds_spec
        )
    else:
        record["speculative_rag"] = failed_speculative(error_msg)

    # --- Plain RAG baseline --------------------------------------------- #
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

    final_out_tokens = count_output_tokens(record["speculative_rag"].get("answer", ""))
    logger.info(
        "[EVAL TOKEN SUMMARY | %s] question_tokens=%d | final_answer_tokens=%d | "
        "Ollama(in=%d, out=%d) | OpenRouter(in=%d, out=%d)",
        item.get("id", "?"),
        q_tokens,
        final_out_tokens,
        token_tracker["ollama_input"],
        token_tracker["ollama_output"],
        token_tracker["openrouter_input"],
        token_tracker["openrouter_output"],
    )

    spec_v = record["speculative_rag"]
    emit(lambda full: [
        "", "  == FINAL ANSWER ==",
        f"  Speculative RAG: {_clip(spec_v.get('answer', '(no answer)'), True)}",
        f"      -> selected draft {spec_v.get('best_draft_index', -1) + 1}, "
        f"{'CORRECT' if spec_v['final_correct'] else 'wrong'} by phrase matching",
        f"  Plain RAG      : {_clip(baseline_answer, True)}",
        "", "  == RESULT ==",
        f"  Expected       : {_clip(item['ground_truth'], full)}",
        f"  Speculative RAG: {'CORRECT' if spec_v['final_correct'] else 'wrong'}"
        f"   Plain RAG: {'CORRECT' if record['baseline']['correct'] else 'wrong'}",
        "  " + "-" * 70,
    ])
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
        "seconds": mean(
            x["subset_construction_seconds"] + x["draft_and_verification_seconds"] for x in v
        ),
    }


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


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
                "accuracy metrics were produced._"]
        return "\n".join(out)

    spec = summary_row(ok)
    base_acc = mean(r["baseline"]["correct"] for r in ok)
    base_recall = mean(r["baseline"]["answer_concept_recall"] for r in ok)
    base_sec = mean(r["baseline"]["seconds"] for r in ok)
    mrr = mean(r["retrieval"]["reciprocal_rank"] for r in ok)

    out += ["", "## 1. Overall results (all questions)", "", md_table(
        ["System", "Final acc.", "Oracle", "Subset hit", "Concept recall",
         "Best-subset cov.", "Distinct chunks", "Subset overlap", "Avg sec"],
        [
            ["Speculative RAG", pct(spec["accuracy"]), pct(spec["oracle"]), pct(spec["subset_hit"]),
             pct(spec["concept_recall"]), pct(spec["best_subset_cov"]),
             f"{spec['distinct_chunks']:.1f}", f"{spec['overlap']:.2f}", f"{spec['seconds']:.1f}"],
            ["Plain RAG (baseline)", pct(base_acc), "-", "-", pct(base_recall), "-", "-", "-",
             f"{base_sec:.1f}"],
        ],
    ), "", f"Retrieval MRR: **{mrr:.3f}**"]

    out += ["", "## 2. Accuracy by difficulty", ""]
    level_rows = []
    for level in ("easy", "medium", "hard"):
        rows = [r for r in ok if r.get("level") == level]
        if rows:
            ls = summary_row(rows)
            level_rows.append([
                f"{level.title()} ({len(rows)})",
                pct(ls["accuracy"]),
                pct(mean(r["baseline"]["correct"] for r in rows)),
            ])
    out.append(md_table(["Level", "Speculative RAG", "Plain RAG"], level_rows))

    wins = sum(r["speculative_rag"]["final_correct"] and not r["baseline"]["correct"] for r in ok)
    losses = sum(r["baseline"]["correct"] and not r["speculative_rag"]["final_correct"] for r in ok)
    ties = len(ok) - wins - losses
    out += ["", "## 3. Head-to-head vs Plain RAG (final answer correct)", "", md_table(
        ["Speculative better", "Plain RAG better", "Same outcome", "Exact sign test p"],
        [[wins, losses, ties, f"{sign_test_p(wins, losses):.3f}"]],
    )]

    out += ["", "## 4. Per-question results", "", md_table(
        ["ID", "Level", "Cluster sizes", "Speculative RAG", "Plain RAG"],
        [[r["id"], r["level"],
          "/".join(map(str, r["retrieval"]["cluster_sizes"].values())),
          "✓" if r["speculative_rag"]["final_correct"] else "✗",
          "✓" if r["baseline"]["correct"] else "✗"] for r in ok],
    )]

    out += ["", "## 5. Reading the results", ""]
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
    global N_CLUSTERS, TRACE, TRACE_CHARS
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
    print(f"Retrieval k={args.k}; clusters={N_CLUSTERS}; subsets={N_SUBSETS}")
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
            results.append(score_one_question(item, args.k, args.subsets_only))
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