"""
Exercise 8: Chunk Size Experiment
===================================
Test how chunk size affects retrieval precision and answer quality.
Re-chunk at three sizes (128, 512, 2048) with overlap fixed at 0,
rebuild the index each time, and run the same 5 queries.

Chunk sizes : 128, 512, 2048  (overlap fixed at 0)
Test queries: 5 queries

NOTE: This exercise rebuilds the index three times and is slow.
      Run on Colab with a T4 GPU or better.

Corpus  : Model T Ford service manual  (default)
Output  : exercise_8_results.txt

Usage:
    python exercise_8_chunk_size.py
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

CHUNK_SIZES: List[int] = [128, 512, 2048]
OVERLAP_FIXED = 0     # keep overlap constant so only chunk size varies
TOP_K = 5

CORPUS_DIR   = MODEL_T_DIR
CORPUS_LABEL = "Model T Manual"
RESULTS_FILE = Path(__file__).parent / "exercise_8_results.txt"

TEST_QUERIES: List[str] = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
    "How do I start a Model T Ford?",
]

# ===========================================================================
# 2. EXPERIMENT
# ===========================================================================

def run_chunk_size_experiment(documents,
                              queries: List[str],
                              chunk_sizes: List[int],
                              embed_model,
                              tokenizer,
                              llm) -> List[Dict]:
    results: List[Dict] = []

    for size in chunk_sizes:
        print(f"\n[chunk_size={size}] Re-chunking with overlap={OVERLAP_FIXED}…")
        t0 = time.time()
        chunks = chunk_all(documents, size, OVERLAP_FIXED)
        t_chunk = round(time.time() - t0, 2)

        print(f"  {len(chunks)} chunks in {t_chunk}s — rebuilding index…")
        t0 = time.time()
        index = build_index(chunks, embed_model)
        t_idx = round(time.time() - t0, 2)
        print(f"  Index: {index.ntotal} vectors in {t_idx}s")

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
            # snippet lengths show how much context each chunk contributed
            snippet_lens = [len(c.text) for c, _ in retrieved]

            results.append({
                "chunk_size":   size,
                "n_chunks":     len(chunks),
                "index_size":   index.ntotal,
                "t_chunk":      t_chunk,
                "t_index":      t_idx,
                "question":     question,
                "answer":       answer,
                "elapsed":      elapsed,
                "scores":       scores,
                "sources":      sources,
                "snippet_lens": snippet_lens,
            })
            print(f"    answered in {elapsed}s  |  scores: {scores}")

    return results

# ===========================================================================
# 3. SAVE TO TXT
# ===========================================================================

def save_results(results: List[Dict], corpus_label: str) -> None:
    W = 80
    lines: List[str] = []

    lines.append("Exercise 8: Chunk Size Experiment — Results")
    lines.append(f"Generated   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Corpus      : {corpus_label}")
    lines.append(f"Chunk sizes : {CHUNK_SIZES}  (overlap fixed at {OVERLAP_FIXED})")
    lines.append(f"Top-K       : {TOP_K}")
    lines.append("")

    current_size = None
    q_num = 0

    for r in results:
        if r["chunk_size"] != current_size:
            current_size = r["chunk_size"]
            q_num = 0
            lines.append("")
            lines.append("=" * W)
            lines.append(f"  CHUNK SIZE = {current_size}")
            lines.append(f"  Total chunks : {r['n_chunks']}")
            lines.append(f"  Index vectors: {r['index_size']}")
            lines.append(f"  Chunk time   : {r['t_chunk']}s  |  Index time: {r['t_index']}s")
            lines.append("=" * W)

        q_num += 1
        lines.append("")
        lines.append(f"  Q{q_num}: {r['question']}")
        lines.append(f"  Elapsed       : {r['elapsed']}s")
        lines.append(f"  Scores        : {r['scores']}")
        lines.append(f"  Sources       : {r['sources']}")
        lines.append(f"  Snippet lens  : {r['snippet_lens']}")
        lines.append(f"  Answer        :")
        for ln in textwrap.wrap(r["answer"], width=72):
            lines.append(f"    {ln}")
        lines.append("")

    # Summary table
    lines.append("=" * W)
    lines.append("  SUMMARY TABLE")
    lines.append("=" * W)
    lines.append(f"  {'Size':>8} {'N Chunks':>10} {'Avg Score':>11} {'Avg Latency':>13}")
    lines.append(f"  {'-'*46}")

    by_size: Dict[int, List] = {}
    for r in results:
        by_size.setdefault(r["chunk_size"], []).append(r)

    for sz in CHUNK_SIZES:
        rows = by_size.get(sz, [])
        n_ch   = rows[0]["n_chunks"] if rows else "n/a"
        all_sc = [s for r in rows for s in r["scores"]]
        avg_sc = round(sum(all_sc) / len(all_sc), 4) if all_sc else 0
        avg_lt = round(sum(r["elapsed"] for r in rows) / len(rows), 2) if rows else 0
        lines.append(f"  {sz:>8} {str(n_ch):>10} {avg_sc:>11.4f} {avg_lt:>12.2f}s")

    lines.append("")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[save] Results written to {RESULTS_FILE}")

# ===========================================================================
# 4. MAIN
# ===========================================================================

def main() -> None:
    print("=" * 80)
    print("  Exercise 8: Chunk Size Experiment")
    print(f"  Corpus     : {CORPUS_LABEL}")
    print(f"  Sizes      : {CHUNK_SIZES}")
    print(f"  Overlap    : {OVERLAP_FIXED} (fixed)")
    print("=" * 80)

    print("\n[setup] Loading embedding model…")
    embed_model = load_embed_model()

    print("[setup] Loading documents (shared across all size configs)…")
    documents = load_folder(CORPUS_DIR)

    print("[setup] Loading LLM…")
    tokenizer, llm = load_llm()

    results = run_chunk_size_experiment(
        documents, TEST_QUERIES, CHUNK_SIZES, embed_model, tokenizer, llm
    )

    save_results(results, CORPUS_LABEL)


if __name__ == "__main__":
    main()
