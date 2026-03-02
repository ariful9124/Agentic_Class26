"""
Exercise 1: Open Model RAG vs. No RAG Comparison
=================================================
Compare Qwen 2.5 1.5B Instruct answers with and without Retrieval-Augmented
Generation (RAG) on two corpora:
  1. Model T Ford service manual  (Corpora/ModelTService/pdf_embedded/)
  2. Congressional Record Jan 2026 (Corpora/Congressional_Record_Jan_2026/pdf_embedded/)

Optional: also builds a combined index with both corpora and re-runs queries.

Pipeline:
  Documents → Chunking → Embedding (all-MiniLM-L6-v2) → FAISS index
  Query → Embed → Similarity search → Top-K chunks → Prompt → Qwen → Answer

Usage:
  python exercise_1_rag_comparison.py

Environment:
  pip install torch transformers sentence-transformers faiss-cpu pymupdf accelerate
"""

# ===========================================================================
# 0. IMPORTS AND ENVIRONMENT SETUP
# ===========================================================================
import os
import sys
import pickle
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# Must be set before importing torch so MPS ops fall back gracefully on Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> Tuple[str, torch.dtype]:
    """Return (device_string, dtype) for the best available hardware."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] CUDA GPU: {name} ({mem:.1f} GB)")
        return "cuda", torch.float16

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("[device] Apple Silicon MPS (float32)")
        return "mps", torch.float32

    print("[device] CPU only — inference will be slow")
    return "cpu", torch.float32


DEVICE, DTYPE = get_device()

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

# Paths — adjust if your working directory is different
BASE_DIR   = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "Corpora"

MODEL_T_DIR = CORPUS_DIR / "ModelTService" / "pdf_embedded"
CR_DIR      = CORPUS_DIR / "Congressional_Record_Jan_2026" / "pdf_embedded"

# LLM
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Embedding model
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking parameters
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 128

# Retrieval
TOP_K = 5

# Generation
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.3   # Low → focused/deterministic

# Optional: run the combined-corpus experiment at the end
RUN_COMBINED_EXPERIMENT = True

# ---------------------------------------------------------------------------
# Queries — as specified in the exercise brief
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
# 2. DOCUMENT LOADING
# ===========================================================================

def load_pdf(filepath: Path) -> str:
    """Extract embedded text from a PDF using PyMuPDF."""
    doc = fitz.open(str(filepath))
    parts = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            parts.append(f"\n[Page {page_num}]\n{text}")
    doc.close()
    return "\n".join(parts)


def load_txt(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8", errors="ignore")


def load_folder(folder: Path) -> List[Tuple[str, str]]:
    """
    Recursively load all PDF/TXT/MD files from *folder*.
    Returns list of (filename, content) tuples.
    """
    docs: List[Tuple[str, str]] = []
    if not folder.exists():
        print(f"[warn] Folder not found: {folder}")
        return docs

    for fp in sorted(folder.rglob("*")):
        if not fp.is_file():
            continue
        suffix = fp.suffix.lower()
        try:
            if suffix == ".pdf":
                content = load_pdf(fp)
            elif suffix in (".txt", ".md", ".text"):
                content = load_txt(fp)
            else:
                continue
            if content.strip():
                docs.append((fp.name, content))
                print(f"  [load] {fp.name}  ({len(content):,} chars)")
        except Exception as exc:
            print(f"  [error] Could not load {fp.name}: {exc}")

    return docs

# ===========================================================================
# 3. CHUNKING
# ===========================================================================

@dataclass
class Chunk:
    text:        str
    source_file: str
    chunk_index: int
    start_char:  int
    end_char:    int


def chunk_document(text: str, source: str,
                   chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """
    Sliding-window chunking that tries to break at paragraph or sentence
    boundaries to avoid cutting mid-thought.
    """
    chunks: List[Chunk] = []
    start = 0
    idx   = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Prefer paragraph break
            para = text.rfind("\n\n", start + chunk_size // 2, end)
            if para != -1:
                end = para + 2
            else:
                # Fall back to sentence break
                sent = text.rfind(". ", start + chunk_size // 2, end)
                if sent != -1:
                    end = sent + 2

        snippet = text[start:end].strip()
        if snippet:
            chunks.append(Chunk(snippet, source, idx, start, end))
            idx += 1

        start = end - overlap
        # Safety: make sure we always advance
        if chunks and start <= chunks[-1].start_char:
            start = end

    return chunks


def chunk_all(documents: List[Tuple[str, str]],
              chunk_size: int = CHUNK_SIZE,
              overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for filename, content in documents:
        doc_chunks = chunk_document(content, filename, chunk_size, overlap)
        all_chunks.extend(doc_chunks)
        print(f"  [chunk] {filename}: {len(doc_chunks)} chunks")
    print(f"  Total chunks: {len(all_chunks)}")
    return all_chunks

# ===========================================================================
# 4. EMBEDDING + FAISS INDEX
# ===========================================================================

def load_embed_model(model_id: str = EMBED_MODEL_ID) -> SentenceTransformer:
    print(f"[embed] Loading {model_id} on {DEVICE} …")
    return SentenceTransformer(model_id, device=DEVICE)


def build_index(chunks: List[Chunk], embed_model: SentenceTransformer) -> faiss.Index:
    """Embed all chunks and build a cosine-similarity FAISS index."""
    print(f"[index] Embedding {len(chunks)} chunks …")
    texts = [c.text for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner-product = cosine after normalization
    index.add(embeddings)
    print(f"[index] Built with {index.ntotal} vectors (dim={dim})")
    return index


def retrieve(query: str, embed_model: SentenceTransformer,
             index: faiss.Index, chunks: List[Chunk],
             top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
    """Return the top-k most relevant (chunk, score) pairs for *query*."""
    q_emb = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    return [(chunks[i], float(s)) for s, i in zip(scores[0], indices[0]) if i != -1]

# ===========================================================================
# 5. LLM LOADING AND GENERATION
# ===========================================================================

def load_llm(model_id: str = LLM_MODEL_ID):
    """Load Qwen 2.5 Instruct model and tokenizer."""
    print(f"\n[llm] Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    load_kwargs = dict(torch_dtype=DTYPE, trust_remote_code=True)

    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", **load_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model = model.to(DEVICE)

    print(f"[llm] Loaded on {DEVICE}")
    return tokenizer, model


def generate(prompt: str, tokenizer, model,
             max_new_tokens: int = MAX_NEW_TOKENS,
             temperature: float = TEMPERATURE) -> str:
    """Run the LLM and return only the newly generated text."""
    inputs = tokenizer(prompt, return_tensors="pt")
    target = model.device if DEVICE == "cuda" else DEVICE
    inputs = {k: v.to(target) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ===========================================================================
# 6. PROMPT TEMPLATES
# ===========================================================================

DIRECT_PROMPT = """\
Answer the following question as accurately as you can based on your training knowledge.

Question: {question}

Answer:"""

RAG_PROMPT = """\
You are a helpful assistant. Answer the question using ONLY the information in the \
provided context. If the context does not contain enough information, say so clearly. \
Quote or cite specific parts of the context to support your answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# ===========================================================================
# 7. QUERY FUNCTIONS
# ===========================================================================

def direct_query(question: str, tokenizer, model) -> str:
    """Ask the LLM with no retrieval context."""
    prompt = DIRECT_PROMPT.format(question=question)
    return generate(prompt, tokenizer, model)


def rag_query(question: str, tokenizer, model,
              embed_model: SentenceTransformer,
              index: faiss.Index, chunks: List[Chunk],
              top_k: int = TOP_K,
              show_context: bool = False) -> Tuple[str, List[Tuple[Chunk, float]]]:
    """
    Run the full RAG pipeline and return (answer, retrieved_results).
    retrieved_results is a list of (chunk, score) pairs.
    """
    results = retrieve(question, embed_model, index, chunks, top_k)

    context_parts = []
    for chunk, score in results:
        context_parts.append(
            f"[Source: {chunk.source_file} | Score: {score:.3f}]\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    if show_context:
        print("\n" + "=" * 70)
        print("RETRIEVED CONTEXT")
        print("=" * 70)
        print(context[:3000] + ("\n…[truncated]" if len(context) > 3000 else ""))
        print("=" * 70 + "\n")

    prompt = RAG_PROMPT.format(context=context, question=question)
    answer = generate(prompt, tokenizer, model)
    return answer, results

# ===========================================================================
# 8. OUTPUT FORMATTING
# ===========================================================================

_WIDTH = 80

def _box(title: str) -> str:
    bar = "=" * _WIDTH
    return f"\n{bar}\n  {title}\n{bar}"


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=_WIDTH, initial_indent=prefix,
                         subsequent_indent=prefix)


def print_comparison(question: str, direct: str, rag: str,
                     rag_results: List[Tuple[Chunk, float]],
                     question_num: int, corpus_label: str) -> None:
    print(_box(f"Q{question_num} [{corpus_label}]"))
    print(f"\n  QUESTION:\n{_wrap(question)}\n")

    print("  ── WITHOUT RAG ──────────────────────────────────────────────")
    print(_wrap(direct))

    print("\n  ── WITH RAG ─────────────────────────────────────────────────")
    print(_wrap(rag))

    print("\n  ── TOP RETRIEVED CHUNKS ─────────────────────────────────────")
    for i, (chunk, score) in enumerate(rag_results, 1):
        snippet = chunk.text[:180].replace("\n", " ")
        print(f"  [{i}] score={score:.4f} | {chunk.source_file}")
        print(f"      {snippet}…")

    print()


def print_analysis_prompt(question: str, direct: str, rag: str) -> None:
    """Print a quick self-analysis section to help document findings."""
    print("  ── ANALYSIS NOTES ───────────────────────────────────────────")
    # Simple heuristics to flag potential hallucination patterns
    hallucination_flags = [
        "I'm not sure",
        "I don't have",
        "I cannot find",
        "hallucinate",
        "approximately",
        "generally",
        "typically",
        "usually",
        "may vary",
    ]
    grounding_flags = [
        "Source:",
        "according to",
        "states that",
        "page",
        "chapter",
        "section",
        "the document",
        "the manual",
        "the record",
    ]

    direct_lower = direct.lower()
    rag_lower    = rag.lower()

    d_flags = [f for f in hallucination_flags if f.lower() in direct_lower]
    r_flags = [f for f in grounding_flags     if f.lower() in rag_lower]

    if d_flags:
        print(f"  * Direct answer signals uncertainty/vagueness: {d_flags}")
    else:
        print("  * Direct answer appears confident (check for hallucinated specifics)")

    if r_flags:
        print(f"  * RAG answer uses grounding language: {r_flags}")
    else:
        print("  * RAG answer may not explicitly cite sources")

    # Length comparison
    d_words = len(direct.split())
    r_words = len(rag.split())
    print(f"  * Direct answer length: {d_words} words | RAG answer length: {r_words} words")
    print()

# ===========================================================================
# 9. PIPELINE BUILDER
# ===========================================================================

def build_pipeline(corpus_dir: Path,
                   embed_model: SentenceTransformer,
                   label: str) -> Tuple[List[Chunk], faiss.Index]:
    """Load corpus, chunk it, embed, and return (chunks, index)."""
    print(_box(f"Building pipeline: {label}"))
    docs   = load_folder(corpus_dir)
    chunks = chunk_all(docs)
    index  = build_index(chunks, embed_model)
    return chunks, index

# ===========================================================================
# 10. MAIN EXPERIMENT
# ===========================================================================

def run_experiment(queries: List[str],
                   corpus_label: str,
                   chunks: List[Chunk],
                   index: faiss.Index,
                   embed_model: SentenceTransformer,
                   tokenizer,
                   model,
                   top_k: int = TOP_K,
                   show_context: bool = False) -> List[dict]:
    """
    Run each query with and without RAG.  Returns a list of result dicts for
    optional downstream analysis.
    """
    results_log = []

    for q_num, question in enumerate(queries, start=1):
        print(f"\n[running] Q{q_num}/{len(queries)}: {question[:70]}…")

        # --- No-RAG ---
        t0 = time.time()
        direct_answer = direct_query(question, tokenizer, model)
        t_direct = time.time() - t0

        # --- RAG ---
        t0 = time.time()
        rag_answer, rag_results = rag_query(
            question, tokenizer, model, embed_model, index, chunks,
            top_k=top_k, show_context=show_context
        )
        t_rag = time.time() - t0

        # Print side-by-side comparison
        print_comparison(question, direct_answer, rag_answer, rag_results,
                         q_num, corpus_label)
        print_analysis_prompt(question, direct_answer, rag_answer)

        print(f"  Timing: direct={t_direct:.1f}s | rag={t_rag:.1f}s\n")

        results_log.append({
            "corpus":        corpus_label,
            "question":      question,
            "direct_answer": direct_answer,
            "rag_answer":    rag_answer,
            "rag_results":   rag_results,
        })

    return results_log


def print_summary(all_results: List[dict]) -> None:
    """Print a brief summary table of findings."""
    print(_box("EXPERIMENT SUMMARY"))
    print(f"  {'#':<4} {'Corpus':<22} {'Question (truncated)':<40} {'Notes'}")
    print(f"  {'-'*4} {'-'*22} {'-'*40} {'-'*10}")

    for i, r in enumerate(all_results, 1):
        corpus = r["corpus"][:22]
        q      = r["question"][:40]
        # Very rough notes
        direct_len = len(r["direct_answer"].split())
        rag_len    = len(r["rag_answer"].split())
        note = f"direct={direct_len}w, rag={rag_len}w"
        print(f"  {i:<4} {corpus:<22} {q:<40} {note}")

    print()


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    print("=" * _WIDTH)
    print("  Exercise 1: Open Model RAG vs. No RAG Comparison")
    print(f"  Model : {LLM_MODEL_ID}")
    print(f"  Device: {DEVICE}  |  Dtype: {DTYPE}")
    print("=" * _WIDTH)

    # ------------------------------------------------------------------
    # Load shared embedding model (used for all experiments)
    # ------------------------------------------------------------------
    embed_model = load_embed_model()

    # ------------------------------------------------------------------
    # Load LLM once (shared across all experiments)
    # ------------------------------------------------------------------
    tokenizer, model = load_llm()

    all_results: List[dict] = []

    # ================================================================
    # EXPERIMENT A: Model T Ford Manual
    # ================================================================
    print(_box("EXPERIMENT A — Model T Ford Service Manual"))
    mt_chunks, mt_index = build_pipeline(MODEL_T_DIR, embed_model, "Model T Manual")

    results_a = run_experiment(
        queries        = QUERIES_MODEL_T,
        corpus_label   = "Model T Manual",
        chunks         = mt_chunks,
        index          = mt_index,
        embed_model    = embed_model,
        tokenizer      = tokenizer,
        model          = model,
        top_k          = TOP_K,
        show_context   = False,   # Set True to print full retrieved context
    )
    all_results.extend(results_a)

    # ================================================================
    # EXPERIMENT B: Congressional Record Jan 2026
    # ================================================================
    print(_box("EXPERIMENT B — Congressional Record Jan 2026"))
    cr_chunks, cr_index = build_pipeline(CR_DIR, embed_model, "Congressional Record")

    results_b = run_experiment(
        queries        = QUERIES_CR,
        corpus_label   = "Congressional Record",
        chunks         = cr_chunks,
        index          = cr_index,
        embed_model    = embed_model,
        tokenizer      = tokenizer,
        model          = model,
        top_k          = TOP_K,
        show_context   = False,
    )
    all_results.extend(results_b)

    # ================================================================
    # OPTIONAL EXPERIMENT C: Combined corpus
    # ================================================================
    if RUN_COMBINED_EXPERIMENT:
        print(_box("EXPERIMENT C (Optional) — Combined Corpus: Model T + Congressional Record"))
        print("[info] Merging both corpora into a single FAISS index …")

        combined_chunks = mt_chunks + cr_chunks
        combined_index  = build_index(combined_chunks, embed_model)

        print("\n--- Combined index: Model T queries ---")
        results_c1 = run_experiment(
            queries        = QUERIES_MODEL_T,
            corpus_label   = "Combined→ModelT",
            chunks         = combined_chunks,
            index          = combined_index,
            embed_model    = embed_model,
            tokenizer      = tokenizer,
            model          = model,
            top_k          = TOP_K,
        )

        print("\n--- Combined index: Congressional Record queries ---")
        results_c2 = run_experiment(
            queries        = QUERIES_CR,
            corpus_label   = "Combined→CR",
            chunks         = combined_chunks,
            index          = combined_index,
            embed_model    = embed_model,
            tokenizer      = tokenizer,
            model          = model,
            top_k          = TOP_K,
        )

        print("\n[analysis] Compare results_c1/results_c2 against results_a/results_b")
        print("           to see if mixing corpora degrades or improves retrieval quality.")

        all_results.extend(results_c1)
        all_results.extend(results_c2)



if __name__ == "__main__":
    main()
