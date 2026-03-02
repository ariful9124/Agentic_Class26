"""
Exercise 2: Open Model + RAG vs. Large Model Comparison
========================================================
Perhaps a larger model without RAG could be competitive with a small model
with RAG.

This script calls GPT-4o Mini with NO tools on every query from Exercise 1,
then places those answers side-by-side with the Exercise 1 results so you can
compare three conditions:

  ┌─────────────────────────────────────────────────────────────────┐
  │  A  │  Qwen 2.5 1.5B — direct (no RAG, from Exercise 1)        │
  │  B  │  Qwen 2.5 1.5B + RAG     (from Exercise 1)               │
  │  C  │  GPT-4o Mini — direct    (no tools, this exercise)        │
  └─────────────────────────────────────────────────────────────────┘

Modes
-----
  GPT-ONLY (default, RUN_QWEN_COMPARISON = False):
      Only condition C is run here.  Open your Exercise 1 terminal output
      alongside this output to compare manually — no extra GPU time needed.

  FULL (RUN_QWEN_COMPARISON = True):
      All three conditions are run in one shot.  Requires a GPU (or patience).
      Qwen model loads are reused across both corpora.

Key research questions
----------------------
  1. Does GPT-4o Mini hallucinate less than Qwen 2.5 1.5B without RAG?
  2. Which queries does GPT-4o Mini answer correctly from its own knowledge?
  3. How does GPT-4o Mini's training cutoff affect answers about each corpus?
       • Model T Ford manual  → 1909–1919 content  → well within training data
       • Congressional Record Jan 2026 → after GPT-4o Mini's cutoff (~Oct 2023)

Training cutoff reference
-------------------------
  GPT-4o Mini knowledge cutoff : October 2023
  Model T Ford production era  : 1908–1927  (well-known historical content)
  Congressional Record corpus  : January 2026  (beyond cutoff)

Usage
-----
  pip install openai python-dotenv
  python exercise_2_gpt_comparison.py

  # For the three-way Qwen comparison (needs GPU):
  # Set RUN_QWEN_COMPARISON = True at the top of this file
"""

# ===========================================================================
# 0. IMPORTS
# ===========================================================================
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Load OPENAI_API_KEY from the project-level .env before importing openai
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"   # Topic 5/../.env
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
        print(f"[env] Loaded API key from {_env_path}")
    else:
        load_dotenv()   # fall back to cwd .env or existing env vars
except ImportError:
    pass   # dotenv not installed — rely on the environment variable being set

from openai import OpenAI

# ===========================================================================
# 1. CONFIGURATION — edit these to change behaviour
# ===========================================================================

# Set to True to also run Qwen 2.5 1.5B (needs GPU; builds the RAG pipeline)
RUN_QWEN_COMPARISON: bool = True

# GPT model to test — exercise specifies GPT-4o Mini
GPT_MODEL: str = "gpt-4o-mini"

# GPT generation settings
GPT_MAX_TOKENS: int  = 512
GPT_TEMPERATURE: float = 0.3   # Low = focused, deterministic

# Results cache — saved after running so you can reload without re-inference
RESULTS_CACHE: Path = Path(__file__).parent / "exercise_2_results.txt"

# Top-K chunks for RAG (only used when RUN_QWEN_COMPARISON = True)
TOP_K: int = 5

# ---------------------------------------------------------------------------
# Query lists (identical to Exercise 1)
# ---------------------------------------------------------------------------
QUERIES_MODEL_T: List[str] = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
]

QUERIES_CR: List[str] = [
    "What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?",
    "What mistake did Elise Stefanik make in Congress on January 23, 2026?",
    "What is the purpose of the Main Street Parity Act?",
    "Who in Congress has spoken for and against funding of pregnancy centers?",
]

# Corpus folders (mirrors Exercise 1 paths)
BASE_DIR   = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "Corpora"
MODEL_T_DIR = CORPUS_DIR / "ModelTService"          / "pdf_embedded"
CR_DIR      = CORPUS_DIR / "Congressional_Record_Jan_2026" / "pdf_embedded"

# ===========================================================================
# 2. GPT-4o MINI — DIRECT QUERY (no tools)
# ===========================================================================

# Shared OpenAI client (picks up OPENAI_API_KEY from environment)
_openai_client: Optional[OpenAI] = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set.\n"
                "Set it in your shell or in the .env file at the project root."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def gpt_direct_query(question: str,
                     model: str = GPT_MODEL,
                     max_tokens: int = GPT_MAX_TOKENS,
                     temperature: float = GPT_TEMPERATURE) -> str:
    """
    Send a single question to GPT-4o Mini with NO tools and NO extra context.
    The model must answer from its pre-training knowledge alone.

    This deliberately mirrors the 'direct_query' baseline from Exercise 1 so
    that the two models are tested under equivalent conditions.
    """
    client = get_openai_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "as accurately as possible based on your training knowledge. "
                    "Do not use any external tools or look anything up."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        # No 'tools' parameter → model has no function-calling capability here
    )

    return response.choices[0].message.content.strip()


def run_gpt_queries(queries: List[str], corpus_label: str) -> List[Dict]:
    """Run every query in *queries* through GPT-4o Mini and return result dicts."""
    results = []
    for i, question in enumerate(queries, start=1):
        print(f"  [GPT] Q{i}/{len(queries)}: {question[:70]}…")
        t0 = time.time()
        answer = gpt_direct_query(question)
        elapsed = time.time() - t0
        print(f"         answered in {elapsed:.1f}s")
        results.append({
            "corpus":      corpus_label,
            "question":    question,
            "gpt_answer":  answer,
            "gpt_elapsed": round(elapsed, 2),
        })
    return results

# ===========================================================================
# 3. QWEN + RAG PIPELINE (optional — imported from Exercise 1)
# ===========================================================================

def _load_qwen_pipeline():
    """
    Lazily import Exercise 1's pipeline functions.
    Returns (embed_model, tokenizer, model) or raises ImportError.
    """
    # exercise_1 must be in the same directory
    sys.path.insert(0, str(BASE_DIR))
    from exercise_1_rag_comparison import (   # type: ignore
        load_embed_model, build_index, chunk_all, load_folder,
        load_llm, rag_query, direct_query as qwen_direct,
        CHUNK_SIZE, CHUNK_OVERLAP,
    )
    return (load_embed_model, build_index, chunk_all, load_folder,
            load_llm, rag_query, qwen_direct, CHUNK_SIZE, CHUNK_OVERLAP)


def run_qwen_queries(queries: List[str], corpus_dir: Path,
                     corpus_label: str) -> List[Dict]:
    """Build the RAG index for *corpus_dir* and run all three Qwen conditions."""
    (load_embed_model, build_index, chunk_all, load_folder,
     load_llm, rag_query_fn, qwen_direct_fn,
     CHUNK_SIZE, CHUNK_OVERLAP) = _load_qwen_pipeline()

    print(f"\n  [Qwen] Loading embedding model…")
    embed_model = load_embed_model()

    print(f"  [Qwen] Loading documents from {corpus_dir}…")
    documents = load_folder(corpus_dir)
    chunks    = chunk_all(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    index     = build_index(chunks, embed_model)

    print(f"  [Qwen] Loading LLM…")
    tokenizer, model = load_llm()

    results = []
    for i, question in enumerate(queries, start=1):
        print(f"  [Qwen] Q{i}/{len(queries)}: {question[:70]}…")

        t0 = time.time()
        q_direct = qwen_direct_fn(question, tokenizer, model)
        t_direct = time.time() - t0

        t0 = time.time()
        q_rag, _ = rag_query_fn(question, tokenizer, model, embed_model,
                                 index, chunks, top_k=TOP_K)
        t_rag = time.time() - t0

        results.append({
            "corpus":         corpus_label,
            "question":       question,
            "qwen_direct":    q_direct,
            "qwen_rag":       q_rag,
            "qwen_t_direct":  round(t_direct, 2),
            "qwen_t_rag":     round(t_rag, 2),
        })

    return results

# ===========================================================================
# 4. OUTPUT FORMATTING
# ===========================================================================

_WIDTH = 80

def _bar(char: str = "=") -> str:
    return char * _WIDTH

def _box(title: str) -> str:
    return f"\n{_bar()}\n  {title}\n{_bar()}"

def _wrap(text: str, indent: int = 4) -> str:
    pad = " " * indent
    return textwrap.fill(
        text, width=_WIDTH,
        initial_indent=pad, subsequent_indent=pad
    )


def print_gpt_result(result: Dict, q_num: int) -> None:
    """Print a single GPT-4o Mini result entry."""
    corpus = result["corpus"]
    q      = result["question"]
    ans    = result["gpt_answer"]

    print(f"\n{'─'*_WIDTH}")
    print(f"  Q{q_num} [{corpus}]")
    print(f"{'─'*_WIDTH}")
    print(f"  QUESTION:")
    print(_wrap(q))
    print(f"\n  GPT-4o Mini answer ({result['gpt_elapsed']}s):")
    print(_wrap(ans))


def print_three_way(gpt_r: Dict, qwen_r: Dict, q_num: int) -> None:
    """Print the full three-way comparison for one query."""
    corpus = gpt_r["corpus"]
    q      = gpt_r["question"]

    print(f"\n{'─'*_WIDTH}")
    print(f"  Q{q_num} [{corpus}]")
    print(f"{'─'*_WIDTH}")
    print(f"  QUESTION:")
    print(_wrap(q))

    print(f"\n  ── A: Qwen 2.5 1.5B  (no RAG, {qwen_r['qwen_t_direct']}s) ──────────────")
    print(_wrap(qwen_r["qwen_direct"]))

    print(f"\n  ── B: Qwen 2.5 1.5B + RAG ({qwen_r['qwen_t_rag']}s) ──────────────────")
    print(_wrap(qwen_r["qwen_rag"]))

    print(f"\n  ── C: GPT-4o Mini (no tools, {gpt_r['gpt_elapsed']}s) ─────────────────")
    print(_wrap(gpt_r["gpt_answer"]))


def print_analysis(results_gpt: List[Dict],
                   results_qwen: Optional[List[Dict]] = None) -> None:
    """Print the training-cutoff analysis and per-query assessment guide."""

    print(_box("TRAINING CUTOFF ANALYSIS"))
    print("""
  GPT-4o Mini knowledge cutoff : ~October 2023
  ─────────────────────────────────────────────────────────────────
  Corpus                   Period          Within GPT-4o Mini's knowledge?
  ─────────────────────────────────────────────────────────────────
  Model T Ford manual      1908–1927       YES — well-documented history
  Congressional Record     Jan 2026        NO  — 2+ years after cutoff
  ─────────────────────────────────────────────────────────────────

  Expected pattern
  ────────────────
  • Model T queries  : GPT-4o Mini likely knows the factual answers
    (spark plug gap, carburetor adjustment etc. are widely documented online).
    Compare against Qwen 1.5B no-RAG — does the larger model give more
    precise / correct values?

  • Congressional Record queries : GPT-4o Mini CANNOT know the Jan 2026
    sessions.  It will either say "I don't know", hallucinate plausible-
    sounding but false statements, or give generic procedural descriptions.
    This is where Qwen + RAG should dramatically outperform GPT-4o Mini.

  Qwen 2.5 1.5B cutoff : ~September 2024
  ─────────────────────────────────────────────────────────────────
  Still before January 2026, so Qwen-no-RAG has the same gap for the
  Congressional Record corpus.  Both models need RAG for those queries.
""")

    print(_box("DOCUMENTATION GUIDE — What to record for each query"))
    print("""
  For each query, note:

  MODEL T FORD QUERIES (historical, ~1919)
  ─────────────────────────────────────────
  1. Does GPT-4o Mini give correct / precise values? (e.g., actual plug gap)
  2. Does Qwen 1.5B no-RAG give a plausible but vague answer?
  3. Does Qwen + RAG cite the manual and give the verbatim spec?
  4. Where GPT-4o Mini is correct, does it match what the manual says?

  CONGRESSIONAL RECORD QUERIES (Jan 2026 — beyond all model cutoffs)
  ──────────────────────────────────────────────────────────────────
  1. Does GPT-4o Mini say "I don't know" or does it hallucinate names/content?
  2. Does Qwen 1.5B no-RAG invent a plausible-sounding congressional speech?
  3. Does Qwen + RAG quote the actual record and identify the correct speaker?
  4. If GPT-4o Mini hallucinates, is the hallucination internally consistent
     (hard to detect without RAG) or obviously wrong?

  HALLUCINATION TAXONOMY
  ──────────────────────
  Type 1  – Confident confabulation: model invents specific values/quotes
  Type 2  – Plausible generalisation: technically true but not from the text
  Type 3  – Honest uncertainty: "I'm not certain / I don't have that info"
  Type 4  – Correct from training: answer matches the source without RAG

  The goal is to show that:
    • RAG reduces Type 1 hallucinations
    • Larger models (GPT-4o Mini) show fewer Type 1 errors than Qwen 1.5B
      on historical content, but cannot overcome the cutoff gap for Jan 2026
    • RAG + even a small model can match or beat a large model with no retrieval
""")

    # Quick per-query flag
    print(_bar("-"))
    print("  PER-QUERY GPT-4o MINI ASSESSMENT")
    print(_bar("-"))

    cutoff_note = {
        "Model T Manual":       "(historical — GPT-4o Mini likely has training data)",
        "Congressional Record": "(Jan 2026 — BEYOND GPT-4o Mini cutoff, expect gaps)",
    }

    for i, r in enumerate(results_gpt, start=1):
        corpus = r["corpus"]
        q      = r["question"][:65]
        ans    = r["gpt_answer"].lower()

        uncertainty_words = ["i don't know", "i'm not sure", "i cannot",
                             "i do not have", "my knowledge", "cutoff",
                             "as of my", "after my"]
        hallucination_flags = ["approximately", "typically", "generally",
                               "usually", "may vary", "estimated"]
        confident_specifics = any(c.isdigit() for c in r["gpt_answer"])

        note_parts = []
        if any(w in ans for w in uncertainty_words):
            note_parts.append("→ model admits uncertainty")
        elif any(w in ans for w in hallucination_flags):
            note_parts.append("→ vague/hedged (possible Type 2)")
        elif confident_specifics:
            note_parts.append("→ gives specific values (verify against source)")

        note_str = "  ".join(note_parts) if note_parts else "→ confident answer"
        label    = cutoff_note.get(corpus, "")

        print(f"\n  Q{i}: {q}…")
        print(f"       {label}")
        print(f"       Assessment: {note_str}")

    print()

# ===========================================================================
# 5. RESULTS PERSISTENCE
# ===========================================================================

def _section(title: str, width: int = 80) -> str:
    bar = "=" * width
    return f"\n{bar}\n  {title}\n{bar}\n"


def save_results(gpt_results: List[Dict],
                 qwen_results: Optional[List[Dict]] = None) -> None:
    """
    Save all results to a human-readable plain-text file.
    Each query gets its own block showing: corpus, question, timings,
    and every answer condition that was run.
    """
    lines: List[str] = []
    lines.append("Exercise 2: Open Model + RAG vs. Large Model — Results")
    lines.append(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"GPT model : {GPT_MODEL}  (no tools)")
    lines.append(f"Qwen run  : {'YES' if qwen_results else 'NO (GPT-only mode)'}")
    lines.append("")

    # Build a lookup from question → qwen result (if available)
    qwen_by_q: Dict[str, Dict] = {}
    if qwen_results:
        for r in qwen_results:
            qwen_by_q[r["question"]] = r

    for idx, gpt_r in enumerate(gpt_results, start=1):
        corpus   = gpt_r["corpus"]
        question = gpt_r["question"]
        qwen_r   = qwen_by_q.get(question)

        lines.append("=" * 80)
        lines.append(f"  Q{idx}  [{corpus}]")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"  QUESTION: {question}")
        lines.append("")

        # GPT-4o Mini
        lines.append(f"  ── GPT-4o Mini (no tools, {gpt_r['gpt_elapsed']}s) ──")
        for line in textwrap.wrap(gpt_r["gpt_answer"], width=76):
            lines.append(f"    {line}")
        lines.append("")

        # Qwen conditions (only present in full-comparison mode)
        if qwen_r:
            lines.append(f"  ── Qwen 2.5 1.5B  no RAG ({qwen_r['qwen_t_direct']}s) ──")
            for line in textwrap.wrap(qwen_r["qwen_direct"], width=76):
                lines.append(f"    {line}")
            lines.append("")

            lines.append(f"  ── Qwen 2.5 1.5B + RAG ({qwen_r['qwen_t_rag']}s) ──")
            for line in textwrap.wrap(qwen_r["qwen_rag"], width=76):
                lines.append(f"    {line}")
            lines.append("")

        lines.append("")

    with open(RESULTS_CACHE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[save] Results written to {RESULTS_CACHE}")

# ===========================================================================
# 6. MAIN
# ===========================================================================

def main() -> None:
    print(_bar())
    print("  Exercise 2: Open Model + RAG vs. Large Model Comparison")
    print(f"  GPT model : {GPT_MODEL}  (no tools)")
    print(f"  Qwen run  : {'YES — building RAG pipeline' if RUN_QWEN_COMPARISON else 'NO  — GPT-only mode (set RUN_QWEN_COMPARISON=True to enable)'}")
    print(_bar())

    # ── Verify API key early ────────────────────────────────────────────────
    try:
        client = get_openai_client()
        print(f"[ok] OpenAI client initialised (model: {GPT_MODEL})")
    except EnvironmentError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    all_gpt_results: List[Dict] = []
    all_qwen_results: List[Dict] = []

    # ── GPT-4o Mini — Model T queries ───────────────────────────────────────
    print(_box("GPT-4o Mini — Model T Ford Manual queries"))
    gpt_mt = run_gpt_queries(QUERIES_MODEL_T, "Model T Manual")
    all_gpt_results.extend(gpt_mt)

    for i, r in enumerate(gpt_mt, start=1):
        print_gpt_result(r, i)

    # ── GPT-4o Mini — Congressional Record queries ──────────────────────────
    print(_box("GPT-4o Mini — Congressional Record Jan 2026 queries"))
    gpt_cr = run_gpt_queries(QUERIES_CR, "Congressional Record")
    all_gpt_results.extend(gpt_cr)

    for i, r in enumerate(gpt_cr, start=1):
        print_gpt_result(r, len(QUERIES_MODEL_T) + i)

    # ── Optional: Qwen comparison ────────────────────────────────────────────
    if RUN_QWEN_COMPARISON:
        print(_box("Qwen 2.5 1.5B — Model T (no RAG + RAG)"))
        qwen_mt = run_qwen_queries(QUERIES_MODEL_T, MODEL_T_DIR, "Model T Manual")
        all_qwen_results.extend(qwen_mt)

        print(_box("Qwen 2.5 1.5B — Congressional Record (no RAG + RAG)"))
        qwen_cr = run_qwen_queries(QUERIES_CR, CR_DIR, "Congressional Record")
        all_qwen_results.extend(qwen_cr)

        # ── Three-way comparison ─────────────────────────────────────────────
        print(_box("THREE-WAY COMPARISON: Qwen no-RAG | Qwen+RAG | GPT-4o Mini"))
        print("  (A = Qwen direct, B = Qwen+RAG, C = GPT-4o Mini direct)\n")

        combined_qwen = qwen_mt + qwen_cr
        for idx, (g, q) in enumerate(zip(all_gpt_results, combined_qwen), start=1):
            print_three_way(g, q, idx)

    else:
        # GPT-only mode: remind user how to compare with Exercise 1 output
        print(_box("HOW TO COMPARE WITH EXERCISE 1"))
        print("""
  You are running in GPT-ONLY mode.  To do a full three-way comparison:

  Option A — side-by-side terminals:
      Run exercise_1_rag_comparison.py in one terminal and keep this output
      in another.  Match questions by number to compare answers.

  Option B — full run (GPU required):
      Set  RUN_QWEN_COMPARISON = True  at the top of this file and re-run.
      The Qwen model will be loaded, the RAG index built, and all three
      conditions printed in a single comparison block.

  Option C — load saved Exercise 1 results:
      After Exercise 1 completes, its output is in your terminal history.
      Exercise 2 saves its own results to:
          exercise_2_results.json
      You can write a small analysis script that loads both JSON files.
""")

    # ── Save results ─────────────────────────────────────────────────────────
    save_results(all_gpt_results, all_qwen_results or None)

    # ── Analysis ─────────────────────────────────────────────────────────────
    print_analysis(all_gpt_results,
                   all_qwen_results if RUN_QWEN_COMPARISON else None)

    # ── Quick timing summary ─────────────────────────────────────────────────
    print(_bar("-"))
    print("  TIMING SUMMARY")
    print(_bar("-"))
    total_gpt = sum(r["gpt_elapsed"] for r in all_gpt_results)
    print(f"  GPT-4o Mini: {len(all_gpt_results)} queries in {total_gpt:.1f}s "
          f"({total_gpt/len(all_gpt_results):.1f}s avg per query)")
    if all_qwen_results:
        total_qd = sum(r["qwen_t_direct"] for r in all_qwen_results)
        total_qr = sum(r["qwen_t_rag"]    for r in all_qwen_results)
        n        = len(all_qwen_results)
        print(f"  Qwen direct : {n} queries in {total_qd:.1f}s ({total_qd/n:.1f}s avg)")
        print(f"  Qwen + RAG  : {n} queries in {total_qr:.1f}s ({total_qr/n:.1f}s avg)")
        print(f"  Note: Qwen times include model-load amortised across {n} queries.")
    print()
    print("  Results saved to:", RESULTS_CACHE)
    print(_bar())


if __name__ == "__main__":
    main()
