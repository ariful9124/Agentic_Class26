"""
Exercise 11: Cross-Document Synthesis
========================================
Test questions that require combining information from multiple chunks.
For each synthesis query, run with k = 3, 5, and 10 to see if more chunks
improve the model's ability to synthesize a complete answer.

Synthesis queries are designed so that:
  - The full answer is spread across several non-adjacent sections
  - A single retrieved chunk is not sufficient on its own
  - The model must combine and reconcile multiple passages

Corpus  : Model T Ford service manual  (default)
Output  : exercise_11_results.txt

Usage:
    python exercise_11_cross_doc_synthesis.py
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
    load_llm, rag_query,
    MODEL_T_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

K_VALUES: List[int] = [3, 5, 10]
CORPUS_DIR   = MODEL_T_DIR
CORPUS_LABEL = "Model T Manual"
RESULTS_FILE = Path(__file__).parent / "exercise_11_results.txt"

# Queries requiring synthesis across multiple chunks/sections
SYNTHESIS_QUERIES: List[str] = [
    "What are ALL the maintenance tasks mentioned throughout the manual?",
    "What tools and equipment are needed for a complete engine tune-up?",
    "Summarize all lubrication points and recommended lubricants in the manual.",
    "What are all the adjustment procedures mentioned for the transmission?",
    "What safety warnings or precautions are mentioned throughout the manual?",
]

# ===========================================================================
# 2. EXPERIMENT
# ===========================================================================

def run_synthesis_experiment(queries: List[str],
                             k_values: List[int],
                             embed_model,
                             index,
                             chunks,
                             tokenizer,
                             llm) -> List[Dict]:
    """
    For every (query, k) combination, run RAG and record:
      - k used
      - answer text
      - number of unique source files in retrieved chunks
      - all retrieved scores and sources
      - elapsed time
    """
    results: List[Dict] = []

    for question in queries:
        print(f"\n  Query: {question[:70]}…")

        for k in k_values:
            print(f"    [k={k}]", end=" ", flush=True)
            t0 = time.time()
            answer, retrieved = rag_query(
                question, tokenizer, llm, embed_model, index, chunks,
                top_k=k
            )
            elapsed = round(time.time() - t0, 2)

            scores         = [round(s, 4) for _, s in retrieved]
            sources        = [c.source_file for c, _ in retrieved]
            unique_sources = list(dict.fromkeys(sources))   # preserve order, deduplicate

            print(f"{elapsed}s | {len(unique_sources)} unique source(s)")

            results.append({
                "question":       question,
                "k":              k,
                "answer":         answer,
                "elapsed":        elapsed,
                "scores":         scores,
                "sources":        sources,
                "unique_sources": unique_sources,
                "n_unique_src":   len(unique_sources),
            })

    return results

# ===========================================================================
# 3. SAVE TO TXT
# ===========================================================================

def save_results(results: List[Dict], corpus_label: str) -> None:
    W = 80
    lines: List[str] = []

    lines.append("Exercise 11: Cross-Document Synthesis — Results")
    lines.append(f"Generated   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Corpus      : {corpus_label}")
    lines.append(f"Chunk size  : {CHUNK_SIZE}  |  Overlap: {CHUNK_OVERLAP}")
    lines.append(f"K values    : {K_VALUES}")
    lines.append("")

    # Group by query
    questions = list(dict.fromkeys(r["question"] for r in results))

    for q_idx, question in enumerate(questions, start=1):
        lines.append("")
        lines.append("=" * W)
        lines.append(f"  Q{q_idx}: {question}")
        lines.append("=" * W)
        lines.append("")

        q_results = [r for r in results if r["question"] == question]
        for r in q_results:
            lines.append(f"  ── k = {r['k']}  ({r['elapsed']}s) ──────────────────────────────────")
            lines.append(f"  Scores         : {r['scores']}")
            lines.append(f"  Sources        : {r['sources']}")
            lines.append(f"  Unique sources : {r['unique_sources']}")
            lines.append(f"  Answer:")
            for ln in textwrap.wrap(r["answer"], width=72):
                lines.append(f"    {ln}")
            lines.append("")

    # Summary: unique sources and answer length vs k
    lines.append("=" * W)
    lines.append("  SUMMARY: Unique sources retrieved and answer length vs k")
    lines.append("=" * W)
    lines.append(f"  {'Q':>3}  {'k':>4}  {'Unique src':>11}  {'Ans words':>10}  {'Elapsed':>9}")
    lines.append(f"  {'-'*45}")

    for r in results:
        q_num = questions.index(r["question"]) + 1
        n_words = len(r["answer"].split())
        lines.append(
            f"  {q_num:>3}  {r['k']:>4}  {r['n_unique_src']:>11}  "
            f"{n_words:>10}  {r['elapsed']:>8.2f}s"
        )
    lines.append("")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[save] Results written to {RESULTS_FILE}")

# ===========================================================================
# 4. MAIN
# ===========================================================================

def main() -> None:
    print("=" * 80)
    print("  Exercise 11: Cross-Document Synthesis")
    print(f"  Corpus   : {CORPUS_LABEL}")
    print(f"  K values : {K_VALUES}")
    print(f"  Queries  : {len(SYNTHESIS_QUERIES)} synthesis queries")
    print("=" * 80)

    print("\n[setup] Loading embedding model…")
    embed_model = load_embed_model()

    print("[setup] Loading and indexing documents…")
    documents = load_folder(CORPUS_DIR)
    chunks    = chunk_all(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    index     = build_index(chunks, embed_model)

    print("[setup] Loading LLM…")
    tokenizer, llm = load_llm()

    results = run_synthesis_experiment(
        SYNTHESIS_QUERIES, K_VALUES, embed_model, index, chunks, tokenizer, llm
    )

    save_results(results, CORPUS_LABEL)


if __name__ == "__main__":
    main()
