"""
Exercise 3: Open Model + RAG vs. State-of-the-Art Chat Model
=============================================================
Compare the local RAG pipeline (Qwen 2.5 1.5B) against a frontier cloud model
that receives ONLY the raw question — no file upload, no retrieval context.

  ┌─────────────────────────────────────────────────────────────────┐
  │  A  │  Qwen 2.5 1.5B + RAG    (local, Exercise 1 pipeline)      │
  │  B  │  Frontier model, no RAG  (cloud API, question-only)        │
  └─────────────────────────────────────────────────────────────────┘

This replicates what a user would experience if they:
  A — ran the Exercise 1 pipeline on their laptop
  B — copy-pasted the same question into GPT-4o / Claude web chat with
      no file attachment and no retrieval

Supported frontier providers
-----------------------------
  openai    → gpt-4o  (or any other OpenAI chat model)
  anthropic → claude-sonnet-4-6  (or claude-opus-4-6)

Set FRONTIER_PROVIDER and FRONTIER_MODEL below to switch between them.
You can also set RUN_BOTH_PROVIDERS = True to query both APIs and print
a three-way A / B(OpenAI) / B(Anthropic) comparison.

Corpora used (same as Exercise 1)
-----------------------------------
  Model T Ford service manual        (Corpora/ModelTService/pdf_embedded/)
  Congressional Record Jan 2026      (Corpora/Congressional_Record_Jan_2026/pdf_embedded/)

Key research questions
-----------------------
  1. Can a frontier model match Qwen + RAG on well-documented historical
     content (Model T, ~1919)?
  2. Where the frontier model is past its training cutoff (Jan 2026 CR),
     does RAG give the small model a decisive advantage?
  3. Is the frontier model's reasoning / prose quality noticeably better,
     even when its factual grounding is weaker?
  4. At what point does RAG + small model stop being "good enough"?

Training cutoff reference
--------------------------
  GPT-4o             : ~October 2023
  Claude Sonnet 4.6  : ~early 2025  (still before Jan 2026 CR corpus)
  Qwen 2.5 1.5B      : ~September 2024
  Model T manual     : 1908–1927  → all models should know this
  Congressional Rec. : January 2026 → beyond every model's cutoff

Usage
-----
  pip install openai anthropic python-dotenv
  python exercise_3_frontier_comparison.py

  # To query both OpenAI and Anthropic in one run:
  # Set RUN_BOTH_PROVIDERS = True below
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

# Load API keys from project-root .env before importing client libraries
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
        print(f"[env] Loaded keys from {_env_path}")
    else:
        load_dotenv()
except ImportError:
    pass  # rely on environment variables being already set

# ===========================================================================
# 1. CONFIGURATION — edit these to change behaviour
# ===========================================================================

# ── Frontier model selection ────────────────────────────────────────────────
# Set FRONTIER_PROVIDER to "openai" or "anthropic"
FRONTIER_PROVIDER: str = "openai"

# OpenAI model options  : "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"
# Anthropic model options: "claude-opus-4-6", "claude-sonnet-4-6"
FRONTIER_MODEL: str = "gpt-4o"

# Set True to run BOTH providers and show a three-way comparison
RUN_BOTH_PROVIDERS: bool = False
SECOND_PROVIDER: str  = "anthropic"
SECOND_MODEL: str     = "claude-sonnet-4-6"

# ── Qwen + RAG ──────────────────────────────────────────────────────────────
# Set False to skip local Qwen inference (only cloud answers printed)
RUN_QWEN_RAG: bool = True

# ── Generation settings ─────────────────────────────────────────────────────
FRONTIER_MAX_TOKENS: int   = 512
FRONTIER_TEMPERATURE: float = 0.3   # low = focused / deterministic

# ── RAG settings (Qwen pipeline) ────────────────────────────────────────────
TOP_K: int = 5

# ── Output ──────────────────────────────────────────────────────────────────
RESULTS_FILE: Path = Path(__file__).parent / "exercise_3_results.txt"

# ── Corpus paths (mirrors Exercise 1) ───────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CORPUS_DIR  = BASE_DIR / "Corpora"
MODEL_T_DIR = CORPUS_DIR / "ModelTService"                   / "pdf_embedded"
CR_DIR      = CORPUS_DIR / "Congressional_Record_Jan_2026"   / "pdf_embedded"

# ---------------------------------------------------------------------------
# Query lists (identical to Exercises 1 & 2)
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

# ===========================================================================
# 2. FRONTIER MODEL CLIENTS
# ===========================================================================

# ── OpenAI ──────────────────────────────────────────────────────────────────

def _openai_client():
    """Return an OpenAI client, raising clearly if the key is missing."""
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env or shell environment."
        )
    return OpenAI(api_key=key)


def query_openai(question: str,
                 model: str = FRONTIER_MODEL,
                 max_tokens: int = FRONTIER_MAX_TOKENS,
                 temperature: float = FRONTIER_TEMPERATURE) -> str:
    """
    Send *question* to an OpenAI chat model with NO tools and NO document
    context — replicates pasting the question into the web chat interface.
    """
    client = _openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "as accurately as you can from your training knowledge. "
                    "Do not use any tools or external resources."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        # Deliberately NO 'tools' parameter
    )
    return response.choices[0].message.content.strip()


# ── Anthropic ───────────────────────────────────────────────────────────────

def _anthropic_client():
    """Return an Anthropic client, raising clearly if the key is missing."""
    try:
        import anthropic as _anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is not installed.\n"
            "Run:  pip install anthropic"
        )
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env or shell environment."
        )
    return _anthropic.Anthropic(api_key=key)


def query_anthropic(question: str,
                    model: str = "claude-sonnet-4-6",
                    max_tokens: int = FRONTIER_MAX_TOKENS,
                    temperature: float = FRONTIER_TEMPERATURE) -> str:
    """
    Send *question* to a Claude model with no system context beyond a brief
    instruction — replicates pasting the question into Claude.ai with no
    attached files.
    """
    client = _anthropic_client()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=(
            "You are a helpful assistant. Answer the user's question as "
            "accurately as you can from your training knowledge. "
            "Do not use any tools or external resources."
        ),
        messages=[{"role": "user", "content": question}],
    )
    # content is a list of ContentBlock objects; extract text
    return "".join(
        block.text for block in message.content
        if hasattr(block, "text")
    ).strip()


# ── Dispatcher ───────────────────────────────────────────────────────────────

def query_frontier(question: str,
                   provider: str,
                   model: str,
                   max_tokens: int = FRONTIER_MAX_TOKENS,
                   temperature: float = FRONTIER_TEMPERATURE) -> str:
    """Route to the correct provider and return the answer string."""
    provider = provider.lower()
    if provider == "openai":
        return query_openai(question, model, max_tokens, temperature)
    if provider == "anthropic":
        return query_anthropic(question, model, max_tokens, temperature)
    raise ValueError(f"Unknown provider '{provider}'. Use 'openai' or 'anthropic'.")


def run_frontier_queries(queries: List[str],
                         corpus_label: str,
                         provider: str,
                         model: str) -> List[Dict]:
    """Run every query through the frontier model and return result dicts."""
    results: List[Dict] = []
    label = f"{provider}/{model}"
    for i, question in enumerate(queries, start=1):
        print(f"  [frontier:{label}] Q{i}/{len(queries)}: {question[:65]}…")
        t0 = time.time()
        answer = query_frontier(question, provider, model)
        elapsed = round(time.time() - t0, 2)
        print(f"    answered in {elapsed}s")
        results.append({
            "corpus":           corpus_label,
            "question":         question,
            "frontier_answer":  answer,
            "frontier_elapsed": elapsed,
            "frontier_label":   label,
        })
    return results

# ===========================================================================
# 3. QWEN + RAG PIPELINE (lazy import from Exercise 1)
# ===========================================================================

def _import_ex1():
    """
    Import Exercise 1's pipeline.  Done lazily so that users who only want
    to run the cloud-only mode don't need torch/transformers installed.
    """
    sys.path.insert(0, str(BASE_DIR))
    from exercise_1_rag_comparison import (   # type: ignore
        load_embed_model, build_index, chunk_all, load_folder,
        load_llm, rag_query,
        CHUNK_SIZE, CHUNK_OVERLAP,
    )
    return (load_embed_model, build_index, chunk_all, load_folder,
            load_llm, rag_query, CHUNK_SIZE, CHUNK_OVERLAP)


def run_qwen_rag_queries(queries: List[str],
                         corpus_dir: Path,
                         corpus_label: str,
                         embed_model=None,
                         tokenizer=None,
                         llm=None,
                         index=None,
                         chunks=None) -> Tuple[List[Dict], object, object, object, object, object]:
    """
    Build the RAG index for *corpus_dir* (unless pre-built objects are passed)
    then run every query.  Returns results + reusable pipeline objects so they
    don't need to be rebuilt for each corpus.
    """
    (load_embed_model, build_index, chunk_all, load_folder,
     load_llm, rag_query_fn, CHUNK_SIZE, CHUNK_OVERLAP) = _import_ex1()

    if embed_model is None:
        print("  [Qwen] Loading embedding model…")
        embed_model = load_embed_model()

    if tokenizer is None or llm is None:
        print("  [Qwen] Loading LLM…")
        tokenizer, llm = load_llm()

    print(f"  [Qwen] Loading and indexing {corpus_label}…")
    documents = load_folder(corpus_dir)
    chunks    = chunk_all(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    index     = build_index(chunks, embed_model)

    results: List[Dict] = []
    for i, question in enumerate(queries, start=1):
        print(f"  [Qwen+RAG] Q{i}/{len(queries)}: {question[:65]}…")
        t0 = time.time()
        answer, retrieved = rag_query_fn(
            question, tokenizer, llm, embed_model, index, chunks, top_k=TOP_K
        )
        elapsed = round(time.time() - t0, 2)
        top_sources = [
            f"{c.source_file} (score={s:.3f})" for c, s in retrieved[:3]
        ]
        results.append({
            "corpus":        corpus_label,
            "question":      question,
            "qwen_answer":   answer,
            "qwen_elapsed":  elapsed,
            "top_sources":   top_sources,
        })
        print(f"    answered in {elapsed}s")

    return results, embed_model, tokenizer, llm, index, chunks

# ===========================================================================
# 4. OUTPUT FORMATTING
# ===========================================================================

_W = 80

def _bar(ch: str = "=") -> str:
    return ch * _W

def _box(title: str) -> str:
    return f"\n{_bar()}\n  {title}\n{_bar()}"

def _wrap(text: str, indent: int = 4) -> str:
    pad = " " * indent
    return textwrap.fill(text, width=_W,
                         initial_indent=pad, subsequent_indent=pad)


def print_comparison(q_num: int,
                     corpus: str,
                     question: str,
                     qwen_r: Optional[Dict],
                     frontier_results: List[Dict]) -> None:
    """Print a single query's side-by-side comparison block."""
    print(f"\n{'─'*_W}")
    print(f"  Q{q_num}  [{corpus}]")
    print(f"{'─'*_W}")
    print(f"  QUESTION: {question}\n")

    if qwen_r:
        srcs = "  |  ".join(qwen_r["top_sources"]) if qwen_r["top_sources"] else "n/a"
        print(f"  ── A: Qwen 2.5 1.5B + RAG ({qwen_r['qwen_elapsed']}s) ──────────────────────")
        print(f"  Sources: {srcs}")
        print(_wrap(qwen_r["qwen_answer"]))
        print()

    for fr in frontier_results:
        label = fr["frontier_label"]
        print(f"  ── B: {label}  (no RAG, {fr['frontier_elapsed']}s) ──────────────────")
        print(_wrap(fr["frontier_answer"]))
        print()


# ===========================================================================
# 5. RESULTS FILE
# ===========================================================================

def save_results(all_qwen: List[Dict],
                 all_frontier_primary: List[Dict],
                 all_frontier_secondary: Optional[List[Dict]]) -> None:
    """Write a human-readable .txt file with every answer."""
    lines: List[str] = []

    lines.append("Exercise 3: Open Model + RAG vs. State-of-the-Art Chat Model — Results")
    lines.append(f"Generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Condition A: Qwen 2.5 1.5B + RAG  (local)")
    lines.append(f"Condition B: {FRONTIER_PROVIDER}/{FRONTIER_MODEL}  (no RAG, no file upload)")
    if all_frontier_secondary:
        lines.append(f"Condition C: {SECOND_PROVIDER}/{SECOND_MODEL}  (no RAG, no file upload)")
    lines.append("")

    # Index secondary frontier by question for easy lookup
    sec_by_q: Dict[str, Dict] = {}
    if all_frontier_secondary:
        for r in all_frontier_secondary:
            sec_by_q[r["question"]] = r

    # Pair Qwen and primary frontier results by question
    qwen_by_q: Dict[str, Dict] = {r["question"]: r for r in all_qwen}

    seen_corpora: List[str] = []
    for idx, fr in enumerate(all_frontier_primary, start=1):
        corpus   = fr["corpus"]
        question = fr["question"]
        qwen_r   = qwen_by_q.get(question)
        sec_r    = sec_by_q.get(question)

        if corpus not in seen_corpora:
            seen_corpora.append(corpus)
            lines.append("")
            lines.append("=" * _W)
            lines.append(f"  CORPUS: {corpus}")
            lines.append("=" * _W)

        lines.append("")
        lines.append(f"  Q{idx}  [{corpus}]")
        lines.append(f"  QUESTION: {question}")
        lines.append("")

        # Condition A — Qwen + RAG
        if qwen_r:
            srcs = "  |  ".join(qwen_r["top_sources"]) if qwen_r["top_sources"] else "n/a"
            lines.append(f"  ── A: Qwen 2.5 1.5B + RAG ({qwen_r['qwen_elapsed']}s) ──")
            lines.append(f"  Sources: {srcs}")
            for ln in textwrap.wrap(qwen_r["qwen_answer"], width=76):
                lines.append(f"    {ln}")
            lines.append("")
        else:
            lines.append("  ── A: Qwen + RAG  [not run] ──")
            lines.append("")

        # Condition B — primary frontier
        lines.append(f"  ── B: {fr['frontier_label']}  (no RAG, {fr['frontier_elapsed']}s) ──")
        for ln in textwrap.wrap(fr["frontier_answer"], width=76):
            lines.append(f"    {ln}")
        lines.append("")

        # Condition C — secondary frontier (optional)
        if sec_r:
            lines.append(f"  ── C: {sec_r['frontier_label']}  (no RAG, {sec_r['frontier_elapsed']}s) ──")
            for ln in textwrap.wrap(sec_r["frontier_answer"], width=76):
                lines.append(f"    {ln}")
            lines.append("")

        lines.append("-" * _W)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[save] Results written to {RESULTS_FILE}")


# ===========================================================================
# 6. ANALYSIS
# ===========================================================================

def print_analysis(all_qwen: List[Dict],
                   all_frontier: List[Dict]) -> None:
    """Print the documentation guide and auto-flag each query."""

    print(_box("TRAINING CUTOFF REFERENCE"))
    print(f"""
  Model              Cutoff           Model T (1919)    CR Jan 2026
  ─────────────────────────────────────────────────────────────────
  GPT-4o             ~Oct 2023        In knowledge      BEYOND cutoff
  Claude Sonnet 4.6  ~early 2025      In knowledge      BEYOND cutoff
  Qwen 2.5 1.5B      ~Sep 2024        In knowledge      BEYOND cutoff
  ─────────────────────────────────────────────────────────────────

  For ALL three models, the Congressional Record Jan 2026 content is
  beyond the training cutoff.  RAG is the only way to answer those
  queries correctly — the frontier model's larger size gives no advantage
  there.

  For Model T queries the frontier model likely has more reliable training
  data than Qwen 1.5B, but Qwen + RAG can quote verbatim specs from the
  actual service manual, which the frontier cannot (without file upload).
""")

    print(_box("DOCUMENTATION GUIDE"))
    print("""
  For each query, record:

  MODEL T FORD QUERIES
  ─────────────────────
  • Is Qwen + RAG answer grounded in the manual text (cites a page/source)?
  • Does the frontier model give a correct answer from general knowledge?
  • When the frontier model is correct, is it as precise as the manual?
  • Are there cases where the frontier model gives a WRONG specific value?
    (e.g., incorrect spark plug gap, wrong oil weight)

  CONGRESSIONAL RECORD QUERIES (Jan 2026 — beyond all cutoffs)
  ──────────────────────────────────────────────────────────────
  • Does Qwen + RAG successfully retrieve and cite the correct passage?
  • Does the frontier model admit ignorance, or does it hallucinate?
  • If the frontier model hallucinates, is the fabrication plausible enough
    to fool someone who hasn't read the original CR?
  • Does the frontier model's prose quality hide its factual inaccuracy?

  COMPARATIVE SCORING RUBRIC
  ───────────────────────────
  Score each answer 0–3 on each dimension:
    Factual accuracy   : 0 wrong / 1 partially correct / 2 correct / 3 cited
    Completeness       : 0 missing / 1 partial / 2 adequate / 3 comprehensive
    Grounding evidence : 0 none / 1 vague / 2 referenced / 3 verbatim quote
    Honest uncertainty : 0 confident+wrong / 1 hedged / 2 says "I don't know"

  Expected outcome
  ─────────────────
  Model T  : Frontier model ≈ Qwen+RAG on factual accuracy.
             Qwen+RAG wins on grounding (cites the manual).
  CR 2026  : Qwen+RAG wins decisively.  Frontier model fails on factual
             accuracy because the events are beyond its training cutoff.
             RAG + even a small model can beat a large frontier model on
             domain-specific or temporally-recent content.
""")

    # Per-query auto-assessment
    print(_bar("-"))
    print("  AUTO-ASSESSMENT: FRONTIER MODEL RESPONSES")
    print(_bar("-"))

    cutoff_note = {
        "Model T Manual":       "(historical — frontier model likely has training data)",
        "Congressional Record": "(Jan 2026 — BEYOND all model cutoffs, expect gaps)",
    }
    uncertainty_words = [
        "i don't know", "i'm not sure", "i cannot", "i do not have",
        "my knowledge", "cutoff", "as of my", "after my", "beyond my"
    ]
    hallucination_flags = [
        "approximately", "typically", "generally", "usually",
        "may vary", "estimated", "i believe", "i think",
    ]

    frontier_by_q = {r["question"]: r for r in all_frontier}
    qwen_by_q     = {r["question"]: r for r in all_qwen}

    for idx, question in enumerate(QUERIES_MODEL_T + QUERIES_CR, start=1):
        corpus = "Model T Manual" if idx <= len(QUERIES_MODEL_T) else "Congressional Record"
        fr_r   = frontier_by_q.get(question)
        qw_r   = qwen_by_q.get(question)

        if not fr_r:
            continue

        ans_lower = fr_r["frontier_answer"].lower()
        notes: List[str] = []

        if any(w in ans_lower for w in uncertainty_words):
            notes.append("frontier admits uncertainty / knowledge cutoff")
        elif any(w in ans_lower for w in hallucination_flags):
            notes.append("frontier hedges — possible generalisation (Type 2)")
        else:
            notes.append("frontier gives confident answer — verify against source")

        if qw_r:
            q_has_source = bool(qw_r.get("top_sources"))
            notes.append(f"Qwen+RAG retrieved {len(qw_r.get('top_sources', []))} source(s)")

        note_label = cutoff_note.get(corpus, "")
        print(f"\n  Q{idx}: {question[:65]}…")
        print(f"       {note_label}")
        for n in notes:
            print(f"       → {n}")

    print()


# ===========================================================================
# 7. MAIN
# ===========================================================================

def main() -> None:
    print(_bar())
    print("  Exercise 3: Open Model + RAG vs. State-of-the-Art Chat Model")
    print(f"  Condition A  : Qwen 2.5 1.5B + RAG  ({'enabled' if RUN_QWEN_RAG else 'DISABLED'})")
    print(f"  Condition B  : {FRONTIER_PROVIDER}/{FRONTIER_MODEL}  (no RAG)")
    if RUN_BOTH_PROVIDERS:
        print(f"  Condition C  : {SECOND_PROVIDER}/{SECOND_MODEL}  (no RAG)")
    print(_bar())

    all_qwen:      List[Dict] = []
    all_primary:   List[Dict] = []
    all_secondary: List[Dict] = []

    # Shared Qwen pipeline objects (reused across corpora to avoid re-loading)
    embed_model = tokenizer = llm = None

    # ── Model T corpus ────────────────────────────────────────────────────────
    print(_box("Corpus: Model T Ford Service Manual"))

    # — Condition B: primary frontier —
    print(f"\n[frontier] Running {FRONTIER_PROVIDER}/{FRONTIER_MODEL} on Model T queries…")
    primary_mt = run_frontier_queries(
        QUERIES_MODEL_T, "Model T Manual", FRONTIER_PROVIDER, FRONTIER_MODEL
    )
    all_primary.extend(primary_mt)

    # — Condition C: secondary frontier (optional) —
    secondary_mt: List[Dict] = []
    if RUN_BOTH_PROVIDERS:
        print(f"\n[frontier] Running {SECOND_PROVIDER}/{SECOND_MODEL} on Model T queries…")
        secondary_mt = run_frontier_queries(
            QUERIES_MODEL_T, "Model T Manual", SECOND_PROVIDER, SECOND_MODEL
        )
        all_secondary.extend(secondary_mt)

    # — Condition A: Qwen + RAG —
    qwen_mt: List[Dict] = []
    if RUN_QWEN_RAG:
        print("\n[Qwen+RAG] Running local pipeline on Model T queries…")
        qwen_mt, embed_model, tokenizer, llm, _, _ = run_qwen_rag_queries(
            QUERIES_MODEL_T, MODEL_T_DIR, "Model T Manual",
            embed_model=embed_model, tokenizer=tokenizer, llm=llm
        )
        all_qwen.extend(qwen_mt)

    # Print Model T comparisons
    qwen_mt_by_q = {r["question"]: r for r in qwen_mt}
    sec_mt_by_q  = {r["question"]: r for r in secondary_mt}
    for i, fr in enumerate(primary_mt, start=1):
        q = fr["question"]
        print_comparison(
            q_num           = i,
            corpus          = "Model T Manual",
            question        = q,
            qwen_r          = qwen_mt_by_q.get(q),
            frontier_results= [fr] + ([sec_mt_by_q[q]] if q in sec_mt_by_q else []),
        )

    # ── Congressional Record corpus ───────────────────────────────────────────
    print(_box("Corpus: Congressional Record — January 2026"))

    # — Condition B —
    print(f"\n[frontier] Running {FRONTIER_PROVIDER}/{FRONTIER_MODEL} on CR queries…")
    primary_cr = run_frontier_queries(
        QUERIES_CR, "Congressional Record", FRONTIER_PROVIDER, FRONTIER_MODEL
    )
    all_primary.extend(primary_cr)

    # — Condition C —
    secondary_cr: List[Dict] = []
    if RUN_BOTH_PROVIDERS:
        print(f"\n[frontier] Running {SECOND_PROVIDER}/{SECOND_MODEL} on CR queries…")
        secondary_cr = run_frontier_queries(
            QUERIES_CR, "Congressional Record", SECOND_PROVIDER, SECOND_MODEL
        )
        all_secondary.extend(secondary_cr)

    # — Condition A —
    qwen_cr: List[Dict] = []
    if RUN_QWEN_RAG:
        print("\n[Qwen+RAG] Running local pipeline on Congressional Record queries…")
        qwen_cr, embed_model, tokenizer, llm, _, _ = run_qwen_rag_queries(
            QUERIES_CR, CR_DIR, "Congressional Record",
            embed_model=embed_model, tokenizer=tokenizer, llm=llm
        )
        all_qwen.extend(qwen_cr)

    # Print CR comparisons
    qwen_cr_by_q = {r["question"]: r for r in qwen_cr}
    sec_cr_by_q  = {r["question"]: r for r in secondary_cr}
    for i, fr in enumerate(primary_cr, start=1):
        q = fr["question"]
        print_comparison(
            q_num           = len(QUERIES_MODEL_T) + i,
            corpus          = "Congressional Record",
            question        = q,
            qwen_r          = qwen_cr_by_q.get(q),
            frontier_results= [fr] + ([sec_cr_by_q[q]] if q in sec_cr_by_q else []),
        )

    # ── Save results ──────────────────────────────────────────────────────────
    save_results(
        all_qwen       = all_qwen,
        all_frontier_primary   = all_primary,
        all_frontier_secondary = all_secondary or None,
    )

    # ── Analysis ──────────────────────────────────────────────────────────────
    print_analysis(all_qwen, all_primary)

    # ── Timing summary ────────────────────────────────────────────────────────
    print(_bar("-"))
    print("  TIMING SUMMARY")
    print(_bar("-"))

    def _avg(lst: List[Dict], key: str) -> str:
        vals = [r[key] for r in lst if key in r]
        return f"{sum(vals):.1f}s total, {sum(vals)/len(vals):.1f}s avg" if vals else "n/a"

    print(f"  Frontier ({FRONTIER_PROVIDER}/{FRONTIER_MODEL}): {_avg(all_primary, 'frontier_elapsed')}")
    if all_secondary:
        print(f"  Frontier ({SECOND_PROVIDER}/{SECOND_MODEL}): {_avg(all_secondary, 'frontier_elapsed')}")
    if all_qwen:
        print(f"  Qwen 2.5 1.5B + RAG           : {_avg(all_qwen, 'qwen_elapsed')}")
    print(f"\n  Results saved to: {RESULTS_FILE}")
    print(_bar())


if __name__ == "__main__":
    main()
