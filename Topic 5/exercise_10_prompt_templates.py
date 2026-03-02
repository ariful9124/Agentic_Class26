"""
Exercise 10: Prompt Template Variations
=========================================
Test five different prompt templates on the same 5 queries and compare how
the template affects accuracy, groundedness, helpfulness, and citation quality.

Templates:
  1. minimal    — just context and question, no instructions
  2. strict     — answer ONLY from context, explicit "cannot answer" fallback
  3. citation   — quote exact passages supporting the answer
  4. permissive — may supplement context with own knowledge
  5. structured — list facts first, then synthesize

Corpus  : Model T Ford service manual  (default; last query uses CR corpus)
Output  : exercise_10_results.txt

Usage:
    python exercise_10_prompt_templates.py
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
    MODEL_T_DIR, CR_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

TOP_K        = 5
CORPUS_DIR   = MODEL_T_DIR
CORPUS_LABEL = "Model T Manual"
RESULTS_FILE = Path(__file__).parent / "exercise_10_results.txt"

# Each entry: (label, template_string)
# Both {context} and {question} must appear in the template.
PROMPT_VARIANTS: List[Tuple[str, str]] = [
    (
        "minimal",
        "{context}\n\nQuestion: {question}\nAnswer:",
    ),
    (
        "strict",
        "Answer ONLY based on the context below. "
        "If the answer is not in the context, say exactly: "
        "'I cannot answer this from the available documents.'\n\n"
        "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
    ),
    (
        "citation",
        "You are a helpful assistant. Quote the exact passages from the context "
        "that support your answer, then give your answer.\n\n"
        "CONTEXT:\n{context}\n\nQUESTION: {question}\n\n"
        "RELEVANT QUOTES:\nANSWER:",
    ),
    (
        "permissive",
        "You are a helpful assistant. Use the context below to help answer "
        "the question. You may also draw on your own knowledge where helpful.\n\n"
        "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
    ),
    (
        "structured",
        "You are a helpful assistant. Follow this format exactly:\n"
        "FACTS FROM CONTEXT: (bullet list of relevant facts found in the context)\n"
        "SYNTHESIS: (your complete answer based on the facts above)\n\n"
        "CONTEXT:\n{context}\n\nQUESTION: {question}\n\n"
        "FACTS FROM CONTEXT:\nSYNTHESIS:",
    ),
]

TEST_QUERIES: List[str] = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
    # Deliberately unanswerable from the corpus — tests strict vs permissive
    "What is the recommended tire pressure for a Model T Ford?",
]

# ===========================================================================
# 2. EXPERIMENT
# ===========================================================================

def build_context(retrieved) -> str:
    parts = []
    for chunk, score in retrieved:
        parts.append(
            f"[Source: {chunk.source_file} | Score: {score:.3f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def run_template_experiment(queries: List[str],
                            prompt_variants: List[Tuple[str, str]],
                            embed_model,
                            index,
                            chunks,
                            tokenizer,
                            llm) -> List[Dict]:
    """
    For every (template, query) combination, retrieve context once (shared
    across templates for the same query), then generate with each template.
    """
    # Pre-retrieve context for each query (same context for all templates)
    print("[retrieve] Pre-fetching context for all queries…")
    query_contexts: Dict[str, Tuple] = {}
    for question in queries:
        retrieved = retrieve(question, embed_model, index, chunks, top_k=TOP_K)
        context   = build_context(retrieved)
        sources   = [c.source_file for c, _ in retrieved]
        scores    = [round(s, 4) for _, s in retrieved]
        query_contexts[question] = (context, sources, scores)

    results: List[Dict] = []

    for q_idx, question in enumerate(queries, start=1):
        context, sources, scores = query_contexts[question]
        print(f"\n  Q{q_idx}: {question[:65]}…")

        for label, template in prompt_variants:
            prompt = template.format(context=context, question=question)
            t0     = time.time()
            answer = generate(prompt, tokenizer, llm)
            elapsed = round(time.time() - t0, 2)
            print(f"    [{label}] {elapsed}s : {answer[:80]}…")

            results.append({
                "question":       question,
                "template_label": label,
                "template":       template,
                "sources":        sources,
                "scores":         scores,
                "answer":         answer,
                "elapsed":        elapsed,
            })

    return results

# ===========================================================================
# 3. SAVE TO TXT
# ===========================================================================

def save_results(results: List[Dict], corpus_label: str) -> None:
    W = 80
    lines: List[str] = []

    lines.append("Exercise 10: Prompt Template Variations — Results")
    lines.append(f"Generated   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Corpus      : {corpus_label}")
    lines.append(f"Chunk size  : {CHUNK_SIZE}  |  Overlap: {CHUNK_OVERLAP}  |  Top-K: {TOP_K}")
    lines.append(f"Templates   : {[lbl for lbl, _ in PROMPT_VARIANTS]}")
    lines.append("")

    # Group by question
    questions = list(dict.fromkeys(r["question"] for r in results))

    for q_idx, question in enumerate(questions, start=1):
        lines.append("")
        lines.append("=" * W)
        lines.append(f"  Q{q_idx}: {question}")
        lines.append("=" * W)

        q_results = [r for r in results if r["question"] == question]
        if q_results:
            lines.append(f"  Retrieval scores : {q_results[0]['scores']}")
            lines.append(f"  Sources          : {q_results[0]['sources']}")
        lines.append("")

        for r in q_results:
            lines.append(f"  ── [{r['template_label']}] ({r['elapsed']}s) ──")
            for ln in textwrap.wrap(r["answer"], width=72):
                lines.append(f"    {ln}")
            lines.append("")

    # Timing table
    lines.append("=" * W)
    lines.append("  LATENCY TABLE (seconds) — rows=queries, cols=templates")
    lines.append("=" * W)
    template_labels = [lbl for lbl, _ in PROMPT_VARIANTS]
    header = f"  {'':40}" + "".join(f"{lbl:>12}" for lbl in template_labels)
    lines.append(header)

    for q_idx, question in enumerate(questions, start=1):
        row = f"  Q{q_idx} {question[:37]:<37}"
        for lbl in template_labels:
            match = next(
                (r for r in results
                 if r["question"] == question and r["template_label"] == lbl),
                None
            )
            row += f"{match['elapsed']:>12.2f}" if match else f"{'n/a':>12}"
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
    print("  Exercise 10: Prompt Template Variations")
    print(f"  Corpus    : {CORPUS_LABEL}")
    print(f"  Templates : {[lbl for lbl, _ in PROMPT_VARIANTS]}")
    print("=" * 80)

    print("\n[setup] Loading embedding model…")
    embed_model = load_embed_model()

    print("[setup] Loading and indexing documents…")
    documents = load_folder(CORPUS_DIR)
    chunks    = chunk_all(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    index     = build_index(chunks, embed_model)

    print("[setup] Loading LLM…")
    tokenizer, llm = load_llm()

    results = run_template_experiment(
        TEST_QUERIES, PROMPT_VARIANTS, embed_model, index, chunks, tokenizer, llm
    )

    save_results(results, CORPUS_LABEL)


if __name__ == "__main__":
    main()
