"""
Exercise 6: Query Phrasing Sensitivity
========================================
Test how different phrasings of the same underlying question affect retrieval.
For each phrasing, record the top-5 retrieved chunks, their similarity scores,
and compute the overlap between result sets across phrasings.

Underlying question: How to adjust the ignition timing on a Model T?
Phrasings: 6 variations (formal, casual, keywords, question, indirect, technical)

Corpus  : Model T Ford service manual  (default)
Output  : exercise_6_results.txt

Usage:
    python exercise_6_query_phrasing.py
"""

# ===========================================================================
# 0. IMPORTS
# ===========================================================================
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Dict, Tuple, Set

sys.path.insert(0, str(Path(__file__).parent))

from exercise_1_rag_comparison import (
    load_embed_model, build_index, chunk_all, load_folder,
    retrieve,
    MODEL_T_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

TOP_K = 5
CORPUS_DIR   = MODEL_T_DIR
CORPUS_LABEL = "Model T Manual"
RESULTS_FILE = Path(__file__).parent / "exercise_6_results.txt"

# Each tuple: (label, phrasing)
PHRASINGS: List[Tuple[str, str]] = [
    ("formal",    "What is the recommended procedure for adjusting the ignition timing on a Model T Ford engine?"),
    ("casual",    "How do I set the timing on my Model T?"),
    ("keywords",  "timing adjustment Model T ignition"),
    ("question",  "When should I adjust the spark timing?"),
    ("indirect",  "Ignition advance settings procedure"),
    ("technical", "Magneto timing specifications Model T Ford"),
]

# ===========================================================================
# 2. EXPERIMENT
# ===========================================================================

def chunk_id(chunk) -> str:
    """Unique identifier for a chunk: source file + chunk index."""
    return f"{chunk.source_file}::{chunk.chunk_index}"


def run_phrasing_experiment(phrasings: List[Tuple[str, str]],
                            embed_model,
                            index,
                            chunks) -> List[Dict]:
    results: List[Dict] = []

    for label, query in phrasings:
        print(f"  [{label}]: {query[:70]}…")
        t0 = time.time()
        retrieved = retrieve(query, embed_model, index, chunks, top_k=TOP_K)
        elapsed   = round(time.time() - t0, 3)

        chunk_ids = [chunk_id(c) for c, _ in retrieved]
        scores    = [round(s, 4) for _, s in retrieved]
        sources   = [c.source_file for c, _ in retrieved]
        snippets  = [c.text[:120] for c, _ in retrieved]

        results.append({
            "label":     label,
            "query":     query,
            "elapsed":   elapsed,
            "chunk_ids": chunk_ids,
            "scores":    scores,
            "sources":   sources,
            "snippets":  snippets,
        })
        print(f"    scores: {scores}  ({elapsed}s)")

    return results


def compute_overlap_matrix(results: List[Dict]) -> List[List[int]]:
    """
    Build an n×n matrix where cell [i][j] = number of chunks shared
    between phrasing i and phrasing j.
    """
    n    = len(results)
    sets = [set(r["chunk_ids"]) for r in results]
    matrix: List[List[int]] = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(len(sets[i] & sets[j]))
        matrix.append(row)
    return matrix

# ===========================================================================
# 3. SAVE TO TXT
# ===========================================================================

def save_results(results: List[Dict],
                 overlap: List[List[int]],
                 corpus_label: str) -> None:
    W = 80
    lines: List[str] = []

    lines.append("Exercise 6: Query Phrasing Sensitivity — Results")
    lines.append(f"Generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Corpus     : {corpus_label}")
    lines.append(f"Chunk size : {CHUNK_SIZE}  |  Overlap: {CHUNK_OVERLAP}  |  Top-K: {TOP_K}")
    lines.append("")

    # Per-phrasing results
    for idx, r in enumerate(results, start=1):
        lines.append("=" * W)
        lines.append(f"  Phrasing {idx}: [{r['label']}]")
        lines.append(f"  Query   : {r['query']}")
        lines.append(f"  Elapsed : {r['elapsed']}s")
        lines.append("=" * W)
        lines.append("")

        for rank, (cid, score, source, snippet) in enumerate(
            zip(r["chunk_ids"], r["scores"], r["sources"], r["snippets"]), start=1
        ):
            lines.append(f"  Rank {rank} | Score: {score} | {source}")
            lines.append(f"  ID     : {cid}")
            for ln in textwrap.wrap(snippet, width=72):
                lines.append(f"    {ln}")
            lines.append("")

        lines.append("")

    # Overlap matrix
    lines.append("=" * W)
    lines.append("  OVERLAP MATRIX  (shared chunks between phrasings)")
    lines.append("  Rows/cols: " + "  ".join(
        f"[{r['label'][:5]}]" for r in results
    ))
    lines.append("=" * W)

    labels = [r["label"][:8] for r in results]
    header = "  " + "".join(f"{lb:>10}" for lb in labels)
    lines.append(header)
    for i, row in enumerate(overlap):
        row_str = f"  {labels[i]:<8}" + "".join(f"{v:>10}" for v in row)
        lines.append(row_str)

    lines.append("")

    # Unique chunk tally
    lines.append("-" * W)
    lines.append("  UNIQUE CHUNKS PER PHRASING")
    lines.append("-" * W)
    all_ids: Set[str] = set()
    for r in results:
        all_ids.update(r["chunk_ids"])
    lines.append(f"  Total unique chunks across all phrasings: {len(all_ids)}")
    for r in results:
        lines.append(f"  [{r['label']}] unique: {len(set(r['chunk_ids']))}  "
                     f"ids: {r['chunk_ids']}")
    lines.append("")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[save] Results written to {RESULTS_FILE}")

# ===========================================================================
# 4. MAIN
# ===========================================================================

def main() -> None:
    print("=" * 80)
    print("  Exercise 6: Query Phrasing Sensitivity")
    print(f"  Corpus: {CORPUS_LABEL}")
    print(f"  {len(PHRASINGS)} phrasings of the same underlying question")
    print("=" * 80)

    print("\n[setup] Loading embedding model…")
    embed_model = load_embed_model()

    print("[setup] Loading and indexing documents…")
    documents = load_folder(CORPUS_DIR)
    chunks    = chunk_all(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    index     = build_index(chunks, embed_model)

    # No LLM needed — this exercise is retrieval-only
    print(f"\n[run] Retrieving top-{TOP_K} chunks for each phrasing…")
    results = run_phrasing_experiment(PHRASINGS, embed_model, index, chunks)

    overlap = compute_overlap_matrix(results)

    save_results(results, overlap, CORPUS_LABEL)


if __name__ == "__main__":
    main()
