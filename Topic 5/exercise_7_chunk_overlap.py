"""
Exercise 7: Chunk Overlap Experiment
======================================
Re-chunk the corpus with four overlap values while keeping chunk size fixed
at 512 characters. For each configuration, rebuild the FAISS index and run
the test queries.

Overlap values : 0, 64, 128, 256  (chunk size fixed at 512)
Test queries   : 3 queries whose answers may span chunk boundaries

NOTE: This exercise rebuilds the index four times and is slow.
      Run on Colab with a T4 GPU or better.

Corpus  : Model T Ford service manual  (default)
Output  : exercise_7_results.txt

Usage:
    python exercise_7_chunk_overlap.py
"""

# ===========================================================================
# 0. IMPORTS
# ===========================================================================
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from exercise_1_rag_comparison import (
    load_embed_model, build_index, chunk_all, load_folder,
    load_llm, rag_query,
    MODEL_T_DIR,
)

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

CHUNK_SIZE_FIXED = 512
OVERLAP_VALUES: List[int] = [0, 64, 128, 256]
TOP_K = 5

CORPUS_DIR   = MODEL_T_DIR
CORPUS_LABEL = "Model T Manual"
RESULTS_FILE = Path(__file__).parent / "exercise_7_results.txt"

# Queries whose answers are likely to span chunk boundaries
TEST_QUERIES: List[str] = [
    "How do I adjust the carburetor on a Model T?",
    "What oil should I use in a Model T engine?",
    "How do I fix a slipping transmission band?",
]

# ===========================================================================
# 2. EXPERIMENT
# ===========================================================================

def run_overlap_experiment(documents,
                           queries: List[str],
                           overlap_values: List[int],
                           embed_model,
                           tokenizer,
                           llm) -> List[Dict]:
    """
    For each overlap value: rebuild chunks + index, run all queries,
    record n_chunks, index_size, scores, answers, and timing.
    """
    results: List[Dict] = []

    for overlap in overlap_values:
        print(f"\n[overlap={overlap}] Re-chunking with size={CHUNK_SIZE_FIXED}, overlap={overlap}…")
        t_chunk_start = time.time()
        chunks = chunk_all(documents, CHUNK_SIZE_FIXED, overlap)
        t_chunk = round(time.time() - t_chunk_start, 2)

        print(f"  {len(chunks)} chunks created in {t_chunk}s — rebuilding index…")
        t_idx_start = time.time()
        index = build_index(chunks, embed_model)
        t_idx = round(time.time() - t_idx_start, 2)

        index_size = index.ntotal
        print(f"  Index: {index_size} vectors in {t_idx}s")

        for q_idx, question in enumerate(queries, start=1):
            print(f"  Q{q_idx}: {question[:60]}…")
            t0 = time.time()
            answer, retrieved = rag_query(
                question, tokenizer, llm, embed_model, index, chunks,
                top_k=TOP_K
            )
            elapsed = round(time.time() - t0, 2)

            scores  = [round(s, 4) for _, s in retrieved]
            sources = [c.source_file for c, _ in retrieved]

            results.append({
                "overlap":     overlap,
                "n_chunks":    len(chunks),
                "index_size":  index_size,
                "t_chunk":     t_chunk,
                "t_index":     t_idx,
                "question":    question,
                "answer":      answer,
                "elapsed":     elapsed,
                "scores":      scores,
                "sources":     sources,
            })
            print(f"    answered in {elapsed}s  |  scores: {scores}")

    return results

# ===========================================================================
# 3. SAVE TO TXT
# ===========================================================================

def save_results(results: List[Dict], corpus_label: str) -> None:
    W = 80
    lines: List[str] = []

    lines.append("Exercise 7: Chunk Overlap Experiment — Results")
    lines.append(f"Generated   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Corpus      : {corpus_label}")
    lines.append(f"Chunk size  : {CHUNK_SIZE_FIXED} (fixed)")
    lines.append(f"Overlaps    : {OVERLAP_VALUES}")
    lines.append(f"Top-K       : {TOP_K}")
    lines.append("")

    current_overlap = None
    q_num = 0

    for r in results:
        if r["overlap"] != current_overlap:
            current_overlap = r["overlap"]
            q_num = 0

            # Grab index metadata from first result of this overlap
            lines.append("")
            lines.append("=" * W)
            lines.append(f"  OVERLAP = {current_overlap}")
            lines.append(f"  Total chunks : {r['n_chunks']}")
            lines.append(f"  Index vectors: {r['index_size']}")
            lines.append(f"  Chunk time   : {r['t_chunk']}s  |  Index time: {r['t_index']}s")
            lines.append("=" * W)

        q_num += 1
        lines.append("")
        lines.append(f"  Q{q_num}: {r['question']}")
        lines.append(f"  Elapsed : {r['elapsed']}s")
        lines.append(f"  Scores  : {r['scores']}")
        lines.append(f"  Sources : {r['sources']}")
        lines.append(f"  Answer  :")
        for ln in textwrap.wrap(r["answer"], width=72):
            lines.append(f"    {ln}")
        lines.append("")

    # Summary table: n_chunks and avg latency per overlap
    lines.append("=" * W)
    lines.append("  SUMMARY TABLE")
    lines.append("=" * W)
    lines.append(f"  {'Overlap':>8} {'N Chunks':>10} {'Avg Latency':>13}")
    lines.append(f"  {'-'*35}")

    by_overlap: Dict[int, List] = {}
    for r in results:
        by_overlap.setdefault(r["overlap"], []).append(r)

    for ov in OVERLAP_VALUES:
        rows = by_overlap.get(ov, [])
        n_chunks = rows[0]["n_chunks"] if rows else "n/a"
        avg_lat  = round(sum(r["elapsed"] for r in rows) / len(rows), 2) if rows else 0
        lines.append(f"  {ov:>8} {str(n_chunks):>10} {avg_lat:>12.2f}s")

    lines.append("")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[save] Results written to {RESULTS_FILE}")

# ===========================================================================
# 4. MAIN
# ===========================================================================

def main() -> None:
    print("=" * 80)
    print("  Exercise 7: Chunk Overlap Experiment")
    print(f"  Corpus     : {CORPUS_LABEL}")
    print(f"  Chunk size : {CHUNK_SIZE_FIXED} (fixed)")
    print(f"  Overlaps   : {OVERLAP_VALUES}")
    print("=" * 80)

    print("\n[setup] Loading embedding model…")
    embed_model = load_embed_model()

    print("[setup] Loading documents (once, shared across all overlap configs)…")
    documents = load_folder(CORPUS_DIR)

    print("[setup] Loading LLM…")
    tokenizer, llm = load_llm()

    results = run_overlap_experiment(
        documents, TEST_QUERIES, OVERLAP_VALUES, embed_model, tokenizer, llm
    )

    save_results(results, CORPUS_LABEL)


if __name__ == "__main__":
    main()
