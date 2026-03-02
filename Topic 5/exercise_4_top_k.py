"""
Exercise 4: Effect of Top-K Retrieval Count
============================================
Vary the number of chunks retrieved (k = 1, 3, 5, 10, 20) and observe how
it affects answer quality, completeness, accuracy, and response latency.

Corpus  : Model T Ford service manual  (default)
Queries : 5 queries defined in QUERIES below
Output  : exercise_4_results.txt

Usage:
    python exercise_4_top_k.py
"""

# ===========================================================================
# 0. IMPORTS
# ===========================================================================
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from exercise_1_rag_comparison import (
    load_embed_model, build_index, chunk_all, load_folder,
    load_llm, rag_query,
    MODEL_T_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

K_VALUES: List[int] = [1, 3, 5, 10, 20]

QUERIES: List[str] = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
    "How do I start a Model T Ford?",
]

CORPUS_DIR  = MODEL_T_DIR          # swap to CR_DIR to use Congressional Record
CORPUS_LABEL = "Model T Manual"

RESULTS_FILE = Path(__file__).parent / "exercise_4_results.txt"

# ===========================================================================
# 2. EXPERIMENT
# ===========================================================================

def run_top_k_experiment(queries: List[str],
                         k_values: List[int],
                         embed_model,
                         index,
                         chunks,
                         tokenizer,
                         llm) -> List[Dict]:
    """
    For every (k, query) combination run the RAG pipeline and record:
      k, question, answer, elapsed_seconds, n_chunks_retrieved,
      top_chunk_scores (list), top_chunk_sources (list)
    """
    results: List[Dict] = []

    for k in k_values:
        print(f"\n[k={k}] Running {len(queries)} queries…")
        for q_idx, question in enumerate(queries, start=1):
            print(f"  Q{q_idx}: {question[:60]}…")
            t0 = time.time()
            answer, retrieved = rag_query(
                question, tokenizer, llm, embed_model, index, chunks,
                top_k=k
            )
            elapsed = round(time.time() - t0, 2)

            scores  = [round(s, 4) for _, s in retrieved]
            sources = [c.source_file for c, _ in retrieved]

            results.append({
                "k":           k,
                "question":    question,
                "answer":      answer,
                "elapsed":     elapsed,
                "n_retrieved": len(retrieved),
                "scores":      scores,
                "sources":     sources,
            })
            print(f"    done in {elapsed}s  |  scores: {scores}")

    return results

# ===========================================================================
# 3. SAVE TO TXT
# ===========================================================================

def save_results(results: List[Dict], corpus_label: str) -> None:
    W = 80
    lines: List[str] = []

    lines.append("Exercise 4: Effect of Top-K Retrieval Count — Results")
    lines.append(f"Generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Corpus     : {corpus_label}")
    lines.append(f"Chunk size : {CHUNK_SIZE}  |  Overlap: {CHUNK_OVERLAP}")
    lines.append(f"K values   : {K_VALUES}")
    lines.append("")

    current_k = None
    q_num = 0

    for r in results:
        if r["k"] != current_k:
            current_k = r["k"]
            q_num = 0
            lines.append("")
            lines.append("=" * W)
            lines.append(f"  K = {current_k}")
            lines.append("=" * W)

        q_num += 1
        lines.append("")
        lines.append(f"  Q{q_num}: {r['question']}")
        lines.append(f"  Elapsed  : {r['elapsed']}s")
        lines.append(f"  Retrieved: {r['n_retrieved']} chunks")
        lines.append(f"  Scores   : {r['scores']}")
        lines.append(f"  Sources  : {r['sources']}")
        lines.append(f"  Answer:")
        for ln in textwrap.wrap(r["answer"], width=72):
            lines.append(f"    {ln}")
        lines.append("")

    # Latency summary table
    lines.append("=" * W)
    lines.append("  LATENCY SUMMARY (seconds per query)")
    lines.append("=" * W)
    lines.append(f"  {'K':<6} {'Q1':>7} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'Q5':>7} {'Avg':>8}")
    lines.append(f"  {'-'*50}")

    by_k: Dict[int, List[float]] = {}
    for r in results:
        by_k.setdefault(r["k"], []).append(r["elapsed"])

    for k in K_VALUES:
        times = by_k.get(k, [])
        avg   = round(sum(times) / len(times), 2) if times else 0
        row   = f"  {k:<6}"
        for t in times:
            row += f" {t:>7.2f}"
        row += f" {avg:>8.2f}"
        lines.append(row)

    lines.append("")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[save] Results written to {RESULTS_FILE}")

# ===========================================================================
# 4. MAIN
# ===========================================================================

def main() -> None:
    print("=" * 80)
    print("  Exercise 4: Effect of Top-K Retrieval Count")
    print(f"  Corpus : {CORPUS_LABEL}")
    print(f"  K values: {K_VALUES}")
    print("=" * 80)

    # ── Build pipeline ────────────────────────────────────────────────────────
    print("\n[setup] Loading embedding model…")
    embed_model = load_embed_model()

    print("[setup] Loading and indexing documents…")
    documents = load_folder(CORPUS_DIR)
    chunks    = chunk_all(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    index     = build_index(chunks, embed_model)

    print("[setup] Loading LLM…")
    tokenizer, llm = load_llm()

    # ── Run experiment ────────────────────────────────────────────────────────
    results = run_top_k_experiment(
        QUERIES, K_VALUES, embed_model, index, chunks, tokenizer, llm
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(results, CORPUS_LABEL)


if __name__ == "__main__":
    main()
