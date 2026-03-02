"""
Exercise 9: Retrieval Score Analysis
======================================
For 10 different queries, retrieve the top-10 chunks and record the full
similarity score distribution. Then experiment with a score threshold to
filter out low-relevance chunks.

Metrics recorded per query:
  - All 10 scores
  - Score range (max - min), gap between rank-1 and rank-2
  - Number of chunks above threshold
  - Answer produced by standard RAG (k=10)
  - Answer produced by threshold-filtered RAG

Corpus  : Model T Ford service manual  (default)
Output  : exercise_9_results.txt

Usage:
    python exercise_9_score_analysis.py
"""

# ===========================================================================
# 0. IMPORTS
# ===========================================================================
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from exercise_1_rag_comparison import (
    load_embed_model, build_index, chunk_all, load_folder,
    load_llm, retrieve, generate,
    MODEL_T_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
    RAG_PROMPT,
)

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

TOP_K_ANALYSIS  = 10          # retrieve 10 chunks for score analysis
SCORE_THRESHOLD = 0.5         # filter threshold experiment
CORPUS_DIR      = MODEL_T_DIR
CORPUS_LABEL    = "Model T Manual"
RESULTS_FILE    = Path(__file__).parent / "exercise_9_results.txt"

QUERIES_10: List[str] = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
    "How do I start a Model T Ford?",
    "How do I adjust the valve clearances on a Model T?",
    "What is the cooling system capacity of the Model T?",
    "How do I remove the cylinder head on a Model T?",
    "What are the tire specifications for a Model T Ford?",
    "How do I adjust the brakes on a Model T?",
]

# ===========================================================================
# 2. HELPERS
# ===========================================================================

def build_context_from_pairs(pairs: List[Tuple]) -> str:
    parts = []
    for chunk, score in pairs:
        parts.append(
            f"[Source: {chunk.source_file} | Score: {score:.3f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def score_stats(scores: List[float]) -> Dict:
    if not scores:
        return {}
    sorted_s = sorted(scores, reverse=True)
    return {
        "max":       round(sorted_s[0], 4),
        "min":       round(sorted_s[-1], 4),
        "range":     round(sorted_s[0] - sorted_s[-1], 4),
        "gap_1_2":   round(sorted_s[0] - sorted_s[1], 4) if len(sorted_s) > 1 else 0,
        "mean":      round(sum(scores) / len(scores), 4),
        "above_thr": sum(1 for s in scores if s >= SCORE_THRESHOLD),
    }

# ===========================================================================
# 3. EXPERIMENT
# ===========================================================================

def run_score_analysis(queries: List[str],
                       embed_model,
                       index,
                       chunks,
                       tokenizer,
                       llm) -> List[Dict]:
    results: List[Dict] = []

    for q_idx, question in enumerate(queries, start=1):
        print(f"\n  Q{q_idx}: {question[:65]}…")

        # Retrieve top-10
        retrieved = retrieve(question, embed_model, index, chunks,
                             top_k=TOP_K_ANALYSIS)
        scores  = [s for _, s in retrieved]
        sources = [c.source_file for c, _ in retrieved]
        stats   = score_stats(scores)
        print(f"    scores: {[round(s,4) for s in scores]}")
        print(f"    stats : {stats}")

        # Answer A: standard RAG with all top-10 chunks
        context_full = build_context_from_pairs(retrieved)
        prompt_full  = RAG_PROMPT.format(context=context_full, question=question)
        t0 = time.time()
        answer_full  = generate(prompt_full, tokenizer, llm)
        t_full = round(time.time() - t0, 2)

        # Answer B: threshold-filtered RAG (only chunks with score >= threshold)
        filtered  = [(c, s) for c, s in retrieved if s >= SCORE_THRESHOLD]
        n_filtered = len(filtered)
        if filtered:
            context_thr  = build_context_from_pairs(filtered)
            prompt_thr   = RAG_PROMPT.format(context=context_thr, question=question)
            t0 = time.time()
            answer_thr   = generate(prompt_thr, tokenizer, llm)
            t_thr = round(time.time() - t0, 2)
        else:
            answer_thr = "[No chunks above threshold — no answer generated]"
            t_thr = 0.0

        print(f"    full RAG: {t_full}s  |  threshold RAG ({n_filtered} chunks): {t_thr}s")

        results.append({
            "question":    question,
            "scores":      [round(s, 4) for s in scores],
            "sources":     sources,
            "stats":       stats,
            "n_filtered":  n_filtered,
            "answer_full": answer_full,
            "t_full":      t_full,
            "answer_thr":  answer_thr,
            "t_thr":       t_thr,
        })

    return results

# ===========================================================================
# 4. SAVE TO TXT
# ===========================================================================

def save_results(results: List[Dict], corpus_label: str) -> None:
    W = 80
    lines: List[str] = []

    lines.append("Exercise 9: Retrieval Score Analysis — Results")
    lines.append(f"Generated   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Corpus      : {corpus_label}")
    lines.append(f"Chunk size  : {CHUNK_SIZE}  |  Overlap: {CHUNK_OVERLAP}")
    lines.append(f"Top-K       : {TOP_K_ANALYSIS}")
    lines.append(f"Threshold   : {SCORE_THRESHOLD}")
    lines.append("")

    for idx, r in enumerate(results, start=1):
        lines.append("=" * W)
        lines.append(f"  Q{idx}: {r['question']}")
        lines.append("=" * W)
        lines.append("")
        lines.append(f"  Scores (rank 1→10) : {r['scores']}")
        lines.append(f"  Sources            : {r['sources']}")
        lines.append(f"  Max score          : {r['stats'].get('max')}")
        lines.append(f"  Min score          : {r['stats'].get('min')}")
        lines.append(f"  Score range        : {r['stats'].get('range')}")
        lines.append(f"  Gap rank-1 to 2    : {r['stats'].get('gap_1_2')}")
        lines.append(f"  Mean score         : {r['stats'].get('mean')}")
        lines.append(f"  Above threshold    : {r['n_filtered']} / {TOP_K_ANALYSIS}  (threshold={SCORE_THRESHOLD})")
        lines.append("")

        lines.append(f"  ── A: Full RAG (all {TOP_K_ANALYSIS} chunks, {r['t_full']}s) ──")
        for ln in textwrap.wrap(r["answer_full"], width=72):
            lines.append(f"    {ln}")
        lines.append("")

        lines.append(f"  ── B: Threshold RAG ({r['n_filtered']} chunks ≥ {SCORE_THRESHOLD}, {r['t_thr']}s) ──")
        for ln in textwrap.wrap(r["answer_thr"], width=72):
            lines.append(f"    {ln}")
        lines.append("")

    # Score distribution summary across all queries
    lines.append("=" * W)
    lines.append("  SCORE DISTRIBUTION SUMMARY (across all 10 queries)")
    lines.append("=" * W)
    lines.append(f"  {'Q':>3}  {'Max':>7}  {'Gap1-2':>8}  {'Mean':>7}  {'Range':>7}  {'Above thr':>10}")
    lines.append(f"  {'-'*50}")
    for idx, r in enumerate(results, start=1):
        st = r["stats"]
        lines.append(
            f"  {idx:>3}  {st.get('max','n/a'):>7}  {st.get('gap_1_2','n/a'):>8}  "
            f"{st.get('mean','n/a'):>7}  {st.get('range','n/a'):>7}  "
            f"{r['n_filtered']:>10}"
        )
    lines.append("")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[save] Results written to {RESULTS_FILE}")

# ===========================================================================
# 5. MAIN
# ===========================================================================

def main() -> None:
    print("=" * 80)
    print("  Exercise 9: Retrieval Score Analysis")
    print(f"  Corpus      : {CORPUS_LABEL}")
    print(f"  Queries     : {len(QUERIES_10)}")
    print(f"  Top-K       : {TOP_K_ANALYSIS}")
    print(f"  Threshold   : {SCORE_THRESHOLD}")
    print("=" * 80)

    print("\n[setup] Loading embedding model…")
    embed_model = load_embed_model()

    print("[setup] Loading and indexing documents…")
    documents = load_folder(CORPUS_DIR)
    chunks    = chunk_all(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    index     = build_index(chunks, embed_model)

    print("[setup] Loading LLM…")
    tokenizer, llm = load_llm()

    results = run_score_analysis(
        QUERIES_10, embed_model, index, chunks, tokenizer, llm
    )

    save_results(results, CORPUS_LABEL)


if __name__ == "__main__":
    main()
