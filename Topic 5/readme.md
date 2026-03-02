# Topic 5 — Retrieval-Augmented Generation (RAG) Pipeline

A hands-on exploration of RAG mechanics using **Qwen 2.5 1.5B**, **GPT-4o Mini**, and frontier models across two corpora:

| Corpus | Content | Period |
|--------|---------|--------|
| **Model T Ford Service Manual** | Mechanical repair & maintenance | 1908–1927 |
| **Congressional Record — Jan 2026** | U.S. legislative sessions | January 2026 |

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#exercise-0-setup)
- [Exercise 1 — RAG vs. No-RAG](#exercise-1--open-model-rag-vs-no-rag)
- [Exercise 2 — GPT-4o Mini vs. Qwen + RAG](#exercise-2--open-model--rag-vs-large-model)
- [Exercise 3 — Frontier Model vs. RAG](#exercise-3--open-model--rag-vs-frontier-model)
- [Exercise 4 — Effect of Top-K](#exercise-4--effect-of-top-k-retrieval-count)
- [Exercise 5 — Unanswerable Questions](#exercise-5--handling-unanswerable-questions)
- [Exercise 6 — Query Phrasing Sensitivity](#exercise-6--query-phrasing-sensitivity)
- [Exercise 7 — Chunk Overlap](#exercise-7--chunk-overlap-experiment)
- [Exercise 8 — Chunk Size](#exercise-8--chunk-size-experiment)
- [Exercise 9 — Retrieval Score Analysis](#exercise-9--retrieval-score-analysis)
- [Exercise 10 — Prompt Template Variations](#exercise-10--prompt-template-variations)
- [Exercise 11 — Cross-Document Synthesis](#exercise-11--cross-document-synthesis)

---

## Project Structure

```
Topic 5/
├── Corpora/
│   ├── ModelTService/pdf_embedded/          # Model T Ford service manual (PDFs)
│   ├── Congressional_Record_Jan_2026/       # CR Jan 2026 issues (PDFs + TXT)
│   ├── Learjet/                             # Additional corpus
│   └── EU_AI_Act/                           # Additional corpus
│
├── manual_rag_pipeline_universal.ipynb      # Reference notebook (all mechanics)
│
├── exercise_1_rag_comparison.py             # Qwen + RAG vs no-RAG baseline
├── exercise_2_gpt_comparison.py             # GPT-4o Mini (no tools) comparison
├── exercise_3_frontier_comparison.py        # GPT-4o / Claude vs Qwen + RAG
├── exercise_4_top_k.py                      # k = 1, 3, 5, 10, 20 sweep
├── exercise_5_unanswerable.py               # Off-topic / false-premise handling
├── exercise_6_query_phrasing.py             # Phrasing sensitivity + overlap matrix
├── exercise_7_chunk_overlap.py              # Overlap = 0, 64, 128, 256
├── exercise_8_chunk_size.py                 # Size = 128, 512, 2048
├── exercise_9_score_analysis.py             # Score distribution + threshold filter
├── exercise_10_prompt_templates.py          # 5 prompt variants compared
├── exercise_11_cross_doc_synthesis.py       # Multi-chunk synthesis at k = 3, 5, 10
│
└── exercise_*_results.txt                   # Auto-generated output for each exercise
```

**Pipeline overview:**
```
Documents → Chunking → Embedding (all-MiniLM-L6-v2) → FAISS Index
                                                             ↓
Query → Embed → Similarity Search → Top-K Chunks → Prompt → Qwen → Answer
```

**Dependencies:**
```bash
pip install torch transformers sentence-transformers faiss-cpu pymupdf accelerate
pip install openai anthropic python-dotenv   # for cloud model exercises
```

---

## Exercise 0 — Setup

> **Status:** ✅ Completed

- Notebook verified running locally and on Colab.
- Corpora unzipped; PDFs loaded from `Corpora/<corpus>/pdf_embedded/`.

---

## Exercise 1 — Open Model RAG vs. No-RAG

**Script:** `exercise_1_rag_comparison.py` → `exercise_1_results.txt`

**Setup:** Qwen 2.5 1.5B Instruct with `all-MiniLM-L6-v2` embeddings and a FAISS `IndexFlatIP` index. Chunk size 512, overlap 128, top-K 5.

### Findings

**Does the model hallucinate specific values without RAG?**

Yes — consistently. Without RAG the model invented a wrong spark plug gap (0.5 mm, self-correcting to 3.175 mm), a fabricated oil grade (10W-30), and entirely made-up Congressional events (e.g., Flood doubting a bill, Stefanik voting against her own party's budget). All January 2026 content was pure confabulation since it is past the training cutoff.

**Does RAG ground the answers in the actual manual?**

Strongly, for well-indexed content. The transmission-band fix and the Mayor David Black recognition were answered accurately with page-level citations. The weak spot was the oil question: retrieval scores were low (0.37–0.40) and the model supplemented with brand recommendations (Mobil 1, Castrol) instead of admitting the corpus lacked the answer. The Stefanik question failed similarly even with RAG.

**Are there questions where the model's general knowledge is actually correct?**

Without RAG the model performs best on general mechanical principles and widely known technical concepts (how carburetors function, how mechanical adjustments work). It performs poorly on historical specifications, exact numeric values, domain-specific procedures, and recent events — all of which need RAG for accuracy.

### Optional — Combined Corpus

Combining the Model T manual (~632 chunks) and the Congressional Record (~47,878 chunks) into a single database (~48,510 chunks) had minimal impact on overall quality. Retrieval remained domain-appropriate — Model T questions still retrieved manual passages and Congressional questions retrieved legislative records. Minor cross-domain noise appeared occasionally (e.g., a Congressional passage surfacing in a Model T oil query with a low similarity score), but did not usually affect final answers. Domain-specific indexing or filtering would improve robustness for ambiguous queries.

---

## Exercise 2 — Open Model + RAG vs. Large Model

**Script:** `exercise_2_gpt_comparison.py` → `exercise_2_results.txt`

**Setup:** GPT-4o Mini called directly with no tools and no document context — replicating a plain web-chat paste.

### Findings

**Does GPT-4o Mini hallucinate less than Qwen 2.5 1.5B?**

Yes. For all four Congressional Record questions (January 2026 events), GPT-4o Mini correctly stated that the information was beyond its training cutoff rather than inventing facts. In contrast, Qwen 2.5 1.5B without RAG frequently fabricated details (e.g., Stefanik voting records, infrastructure bills, incorrect policy claims). Even for Model T questions, GPT-4o Mini produced plausible general-knowledge answers while Qwen often generated confident but incorrect specifics.

**Which questions does GPT-4o Mini answer correctly?**

GPT-4o Mini answered the four Model T questions (Q1–Q4) directionally correctly because these topics are widely documented and predate its **October 2023** training cutoff by over a century (Model T manual dates to 1919). It could not answer the January 2026 Congressional Record questions (Q5–Q8), which fall well after its training period. This confirms that GPT-4o Mini's parametric knowledge is reliable for older historical domains but that RAG is required for recent documents.

---

## Exercise 3 — Open Model + RAG vs. Frontier Model

**Script:** `exercise_3_frontier_comparison.py` → `exercise_3_results.txt`

**Setup:** GPT-4o (or Claude Sonnet 4.6) with no file upload vs. Qwen 2.5 1.5B + RAG.

### Findings

**Where does the frontier model's general knowledge succeed?**

It succeeds on timeless, well-known maintenance guidance where broad automotive knowledge is sufficient (e.g., carburetor mixture control concepts, general troubleshooting for slipping bands, a reasonable non-detergent oil recommendation). GPT-4o gave coherent, practically useful answers for the Model T questions without needing the manual (especially Q1, Q3, Q4).

**Did the frontier model appear to use live web search?**

No. For the Congressional Record questions (Q5–Q6), outputs explicitly stated the post-cutoff limitation. For Q7–Q8 it produced plausible-but-generic statements rather than citing specific 2026 passages — a clear signal it relied entirely on pretrained knowledge and not retrieval.

**Where does RAG provide more accurate, specific answers?**

RAG wins whenever the question needs document-grounded, post-cutoff, or citation-level detail:

- **CR Q5:** RAG correctly pulls Flood's praise of Mayor David Black from the Jan 13 2026 issue; GPT-4o refuses due to cutoff.
- **CR Q7:** RAG gives the Main Street Parity Act's specific purpose (10% equity requirement / alignment with 504 programs) from CREC-2026-01-20; GPT-4o confidently describes a different unrelated "online sales tax parity" concept — a hallucination by association.

**When does RAG add value vs. when does a powerful model suffice?**

A powerful model is often enough for general, evergreen questions where approximate procedural knowledge is acceptable. RAG adds most value when correctness depends on the corpus — especially for newer-than-cutoff content, exact attributions ("who said what"), and precise legislative or technical definitions. Without RAG a frontier model may answer confidently using similar-sounding prior knowledge (Q7), so RAG isn't just for recency — it prevents "confidently wrong" answers when the user needs what the documents actually say.

---

## Exercise 4 — Effect of Top-K Retrieval Count

**Script:** `exercise_4_top_k.py` → `exercise_4_results.txt`
**Tested:** k = 1, 3, 5, 10, 20 across 5 queries.

### Findings

**At what point does adding more context stop helping?**

Adding more context improves results up to **k ≈ 3–5**, where the system usually retrieves at least one chunk containing the key information. Beyond this point, additional chunks do not consistently improve accuracy and mainly make answers longer without adding useful detail. k > 5 shows diminishing returns for this corpus and chunking setup.

**When does too much context hurt?**

Too much context begins to hurt when higher k values introduce irrelevant or partially related chunks that the model tries to combine. This leads to confusion or incorrect conclusions — e.g., inconsistent spark-plug gap values at higher k, or noisy fabricated details in the oil recommendation question. The model merges multiple weak signals instead of relying on the most relevant passage.

**How does k interact with chunk size?**

Chunk size determines how much information each retrieved unit contains. With 512-character chunks, a small k is sufficient and larger k mostly adds redundant or distracting information. Smaller chunks would need a larger k to capture complete procedures; larger chunks need a smaller k but may reduce retrieval precision by including extra irrelevant content.

---

## Exercise 5 — Handling Unanswerable Questions

**Script:** `exercise_5_unanswerable.py` → `exercise_5_results.txt`

**Categories tested:**
- Completely off-topic (e.g., "What is the capital of France?")
- Related but not in corpus (e.g., "What is the horsepower of a 1925 Model T?")
- False premise (e.g., "Why does the manual recommend synthetic oil?")

### Findings

**Does the model admit it doesn't know?**

With the **standard prompt**, the model only sometimes admits uncertainty. For off-topic and false-premise questions it often provides partial or irrelevant explanations before acknowledging missing information. With the **hardened prompt** ("If the context doesn't contain the answer, say 'I cannot answer this from the available documents.'"), the behavior improves substantially — the model consistently produces clear refusals. The hardened prompt makes the model much more reliable at explicitly admitting when an answer is not in the corpus.

**Does it hallucinate plausible-sounding but wrong answers?**

Yes, frequently under the standard prompt. The model invented a retail price ($290) for the 1920 Model T Runabout and fabricated a justification for the boiling point of ethanol using unrelated manual text. Even with the hardened prompt the model sometimes still produces correct general-knowledge values (e.g., ethanol boiling point, capital of France) before stating the documents do not contain the answer.

**Does retrieved context help or hurt?**

Retrieved context helps for genuinely answerable questions, but for unanswerable ones it can encourage hallucination under the standard prompt. When irrelevant Model T passages are retrieved (often with low similarity scores), the model tries to connect them to the question and generates incorrect explanations. The hardened prompt reduces this behavior. **Irrelevant context increases hallucination risk; explicit abstain instructions improve reliability.**

---

## Exercise 6 — Query Phrasing Sensitivity

**Script:** `exercise_6_query_phrasing.py` → `exercise_6_results.txt`

**Base question:** How to adjust ignition timing on a Model T?
**6 phrasings:** formal, casual, keywords-only, question, indirect, technical.

### Findings

**Which phrasings retrieve the best chunks?**

The **question-style phrasing** ("When should I adjust the spark timing?") retrieves the most relevant chunks — highest similarity scores (≈ 0.53) and passages directly related to ignition timing behavior and spark advance. The **technical phrasing** also retrieves reasonably relevant magneto-related content. Formal and indirect phrasings retrieve generic passages with lower relevance; casual phrasing retrieves mostly unrelated transmission or clutch passages.

**Do keyword-style queries work better or worse than natural questions?**

Worse. Keyword queries produce lower similarity scores and retrieve fragmented or loosely related passages. Natural questions provide better semantic context, allowing the embedding model to match query intent more accurately.

**What does this tell you about query rewriting strategies?**

Rewriting user queries into clear natural-language questions that explicitly mention the target concept (e.g., "spark timing" or "ignition timing") is likely to improve retrieval. Extremely short keyword queries or overly technical phrasing reduce retrieval quality by providing less semantic context. An effective strategy: automatically rewrite queries into well-formed, descriptive questions while preserving original intent.

---

## Exercise 7 — Chunk Overlap Experiment

**Script:** `exercise_7_chunk_overlap.py` → `exercise_7_results.txt`
**Tested:** overlap = 0, 64, 128, 256 (chunk size fixed at 512).

> ⚠️ This exercise rebuilds the index four times — run on Colab with a T4 GPU or better.

### Findings

**Does higher overlap improve retrieval of complete information?**

Higher overlap provides some improvement, especially going from overlap 0 → 64 or 128, where important instructions are less likely to be split across chunk boundaries. Carburetor and transmission-band procedures become more coherent with moderate overlap. However, improvement beyond overlap ≈ 128 is small, and answers are not consistently more accurate at overlap 256.

**What is the cost?**

Higher overlap significantly increases index size and redundancy. Chunk count grows from 439 (overlap 0) → 999 (overlap 256), nearly doubling the index. This increases storage and indexing cost and introduces more duplicate or near-duplicate chunks in retrieval, leading to repetitive context and longer generation times. Average query time increases to ~16 seconds at overlap 256.

**Is there a point of diminishing returns?**

Yes — around **overlap 64–128**. Moderate overlap improves retrieval robustness; larger overlaps mainly add redundancy without meaningful gains in answer quality. Overlap ≈ 128 is a reasonable trade-off between completeness and computational cost.

---

## Exercise 8 — Chunk Size Experiment

**Script:** `exercise_8_chunk_size.py` → `exercise_8_results.txt`
**Tested:** chunk size = 128, 512, 2048 (overlap fixed at 0, 5 queries).

> ⚠️ This exercise rebuilds the index three times — run on Colab with a T4 GPU or better.

### Findings

| Chunk Size | Retrieval Precision | Answer Completeness | Notes |
|------------|--------------------|--------------------|-------|
| **128** | High precision, small units | Often incomplete — procedures split across chunks | Good for fact lookup |
| **512** | Best balance | Complete and coherent | ✅ Sweet spot for this corpus |
| **2048** | Lower precision, more noise | Full info present but noisy | Adequate for long procedures |

**Does optimal size depend on the type of question?**

Yes. Fact-based questions (e.g., spark plug gap) may work with smaller chunks; procedure-based questions (e.g., band adjustment steps) work better with medium-sized chunks like 512.

---

## Exercise 9 — Retrieval Score Analysis

**Script:** `exercise_9_score_analysis.py` → `exercise_9_results.txt`
**Setup:** 10 queries, top-10 chunks retrieved each; score threshold experiment at 0.5.

### Findings

**When is there a clear "winner"?**

Only Q1 (carburetor) shows a noticeable rank-1 vs. rank-2 gap (0.0629: 0.6404 → 0.5775). That is the only case where retrieval strongly prefers one chunk.

**When are scores tightly clustered (ambiguous)?**

Most queries are tightly clustered (gap ≈ 0): Q3, Q4, Q5, Q6, Q8, Q10 show rank-1 ≈ rank-2. Q7 is the most ambiguous (gap = 0.0003). Tight clustering means retrieval is not strongly preferring any single chunk.

**What score threshold would you use?**

A hard threshold of 0.5 is too aggressive for this corpus — it drops entire queries like Q4, Q7, Q9 (zero chunks above 0.5) even when full-RAG could still produce an answer.

**Practical choice:** use **≈ 0.45** as a soft cutoff with a fallback: if nothing passes the threshold, return "not found in corpus / insufficient evidence" rather than forcing an answer.

**How does score distribution correlate with answer quality?**

| Score Pattern | Behaviour |
|--------------|-----------|
| High mean + many chunks above threshold (Q1) | More stable answers |
| Mixed scores, few chunks above threshold (Q2, Q3, Q8, Q10) | Answers usually okay if top chunks contain the key fact; otherwise partially correct with extra filler |
| Low max / none above threshold (Q4, Q7, Q9) | Highest risk zone — answers become hallucinated or unsupported |

---

## Exercise 10 — Prompt Template Variations

**Script:** `exercise_10_prompt_templates.py` → `exercise_10_results.txt`
**Templates:** minimal, strict, citation, permissive, structured (5 queries each).

### Findings

**Which prompt produces the most accurate answers?**

**Strict** and **Structured** are most accurate overall — they are the only ones that reliably avoid answering when the corpus doesn't support it (e.g., Q5 tire pressure: strict correctly outputs "I cannot answer this from the available documents."). Citation is not automatically accurate: it still hallucinates in Q4 (injects "synthetic oils") and Q5 (invents a nonsense measurement as tire pressure).

**Which produces the most useful answers?**

**Structured** is the best usable middle ground — it gives short, step-like answers when support exists (Q1/Q3) and is less likely than permissive or minimal to go off the rails. Citation is useful only when it actually quotes the right line (e.g., Q2 spark plug gap), but sometimes adds incorrect interpretation.

**Is there a trade-off between strict grounding and helpfulness?**

Yes. Strict grounding reduces hallucinations (especially for unsupported questions) but can feel less helpful because it refuses more often and may omit best-practice advice. Permissive and minimal prompts feel helpful but are the most likely to add unsupported steps, mix in unrelated maintenance, or guess numbers.

> **Practical recommendation:** Use a Strict/Structured default. If extra helpfulness is desired, allow it only as clearly-labelled "general advice, not from the manual" after the grounded answer (or refusal) has already been given.

---

## Exercise 11 — Cross-Document Synthesis

**Script:** `exercise_11_cross_doc_synthesis.py` → `exercise_11_results.txt`
**Setup:** 5 synthesis queries requiring information from multiple sections; k = 3, 5, 10.

### Findings

**Can the model successfully combine information from multiple chunks?**

Sometimes, but only when retrieval already covers the right facets. Going from k = 3 → k = 10 increases unique source files (e.g., Q5 reaches 5 sources) and the answer becomes broader. However, more k does not guarantee better synthesis — Q1's k = 10 answer is shorter and less complete than k = 5, even with a larger retrieved set, suggesting the generation step is not reliably aggregating all available evidence.

**Does it miss information that wasn't retrieved?**

Yes — by definition. Q1 ("ALL maintenance tasks") is a whole-corpus question, but the top chunks have low scores (~0.35–0.42) and only 2–3 unique PDFs even at k = 10. The model can only list tasks from those slices and misses everything not surfaced by retrieval.

**Does contradictory information in different chunks cause problems?**

It can. When chunks are only loosely related (low similarity), the model tends to "fill gaps" with prior knowledge instead of reconciling conflicts. With truly conflicting statements (e.g., two different specs), a plain RAG prompt often picks one at random or blends them unless explicitly instructed to report both and flag the disagreement.

**Bottom line:**

- RAG synthesises across chunks only as well as retrieval coverage allows.
- For "ALL / exhaustive" questions, iterative retrieval (multi-query, section walking, or index-wide scan) is needed — simply raising k is not sufficient.
- To handle contradictions, add an instruction: *"If sources disagree, list both and say 'conflict' rather than merging."*
