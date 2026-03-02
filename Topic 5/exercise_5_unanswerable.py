"""
Exercise 5: Handling Unanswerable Questions
============================================
Test how well the RAG system handles three categories of questions that
cannot be answered from the corpus:
  1. Completely off-topic
  2. Related but not in the corpus
  3. False premise

Each question is run twice:
  A) Standard RAG prompt
  B) Hardened RAG prompt: adds "If the context doesn't contain the answer,
     say 'I cannot answer this from the available documents.'"

Corpus  : Model T Ford service manual  (default)
Output  : exercise_5_results.txt

Usage:
    python exercise_5_unanswerable.py
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

TOP_K = 5
CORPUS_DIR   = MODEL_T_DIR
CORPUS_LABEL = "Model T Manual"
RESULTS_FILE = Path(__file__).parent / "exercise_5_results.txt"

# (question, category)
UNANSWERABLE: List[Tuple[str, str]] = [
    # Completely off-topic
    ("What is the capital of France?",
     "off-topic"),
    ("What is the boiling point of ethanol in degrees Celsius?",
     "off-topic"),
    # Related but not in corpus
    ("What is the horsepower rating of the 1925 Model T engine?",
     "related-not-in-corpus"),
    ("What was the original retail price of a 1920 Model T Runabout?",
     "related-not-in-corpus"),
    # False premise
    ("Why does the manual recommend synthetic oil for the Model T?",
     "false-premise"),
    ("What is the fuel injection system specification for the Model T?",
     "false-premise"),
]

# Standard prompt (from Exercise 1)
STANDARD_PROMPT = RAG_PROMPT   # already imported from exercise_1

# Hardened prompt — adds the explicit "cannot answer" instruction
HARDENED_PROMPT = """\
You are a helpful assistant. Answer the question using ONLY the information \
in the provided context. If the context does not contain enough information \
to answer, say exactly: "I cannot answer this from the available documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# ===========================================================================
# 2. EXPERIMENT
# ===========================================================================

def build_context(retrieved: list) -> str:
    parts = []
    for chunk, score in retrieved:
        parts.append(
            f"[Source: {chunk.source_file} | Score: {score:.3f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def run_unanswerable_experiment(questions: List[Tuple[str, str]],
                                embed_model,
                                index,
                                chunks,
                                tokenizer,
                                llm) -> List[Dict]:
    results: List[Dict] = []

    for q_idx, (question, category) in enumerate(questions, start=1):
        print(f"\n  Q{q_idx} [{category}]: {question[:65]}…")

        retrieved = retrieve(question, embed_model, index, chunks, top_k=TOP_K)
        context   = build_context(retrieved)
        sources   = [c.source_file for c, _ in retrieved]
        scores    = [round(s, 4) for _, s in retrieved]

        # ── Standard prompt ────────────────────────────────────────────────
        prompt_std = STANDARD_PROMPT.format(context=context, question=question)
        t0 = time.time()
        answer_std = generate(prompt_std, tokenizer, llm)
        t_std = round(time.time() - t0, 2)
        print(f"    Standard  ({t_std}s): {answer_std[:80]}…")

        # ── Hardened prompt ────────────────────────────────────────────────
        prompt_hard = HARDENED_PROMPT.format(context=context, question=question)
        t0 = time.time()
        answer_hard = generate(prompt_hard, tokenizer, llm)
        t_hard = round(time.time() - t0, 2)
        print(f"    Hardened  ({t_hard}s): {answer_hard[:80]}…")

        results.append({
            "question":      question,
            "category":      category,
            "sources":       sources,
            "scores":        scores,
            "answer_std":    answer_std,
            "elapsed_std":   t_std,
            "answer_hard":   answer_hard,
            "elapsed_hard":  t_hard,
        })

    return results

# ===========================================================================
# 3. SAVE TO TXT
# ===========================================================================

def save_results(results: List[Dict], corpus_label: str) -> None:
    W = 80
    lines: List[str] = []

    lines.append("Exercise 5: Handling Unanswerable Questions — Results")
    lines.append(f"Generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Corpus     : {corpus_label}")
    lines.append(f"Chunk size : {CHUNK_SIZE}  |  Overlap: {CHUNK_OVERLAP}  |  Top-K: {TOP_K}")
    lines.append("")
    lines.append("Prompts tested:")
    lines.append("  A (standard) : standard RAG prompt from Exercise 1")
    lines.append("  B (hardened) : adds explicit 'I cannot answer' instruction")
    lines.append("")

    categories = ["off-topic", "related-not-in-corpus", "false-premise"]

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue

        lines.append("")
        lines.append("=" * W)
        lines.append(f"  CATEGORY: {cat.upper()}")
        lines.append("=" * W)

        for idx, r in enumerate(cat_results, start=1):
            lines.append("")
            lines.append(f"  Q{idx}: {r['question']}")
            lines.append(f"  Retrieval scores : {r['scores']}")
            lines.append(f"  Top sources      : {r['sources'][:3]}")
            lines.append("")
            lines.append(f"  ── A: Standard prompt ({r['elapsed_std']}s) ──")
            for ln in textwrap.wrap(r["answer_std"], width=72):
                lines.append(f"    {ln}")
            lines.append("")
            lines.append(f"  ── B: Hardened prompt ({r['elapsed_hard']}s) ──")
            for ln in textwrap.wrap(r["answer_hard"], width=72):
                lines.append(f"    {ln}")
            lines.append("")
            lines.append("-" * W)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[save] Results written to {RESULTS_FILE}")

# ===========================================================================
# 4. MAIN
# ===========================================================================

def main() -> None:
    print("=" * 80)
    print("  Exercise 5: Handling Unanswerable Questions")
    print(f"  Corpus: {CORPUS_LABEL}")
    print("=" * 80)

    print("\n[setup] Loading embedding model…")
    embed_model = load_embed_model()

    print("[setup] Loading and indexing documents…")
    documents = load_folder(CORPUS_DIR)
    chunks    = chunk_all(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    index     = build_index(chunks, embed_model)

    print("[setup] Loading LLM…")
    tokenizer, llm = load_llm()

    results = run_unanswerable_experiment(
        UNANSWERABLE, embed_model, index, chunks, tokenizer, llm
    )

    save_results(results, CORPUS_LABEL)


if __name__ == "__main__":
    main()
