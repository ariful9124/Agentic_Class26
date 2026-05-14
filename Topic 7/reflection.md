# Topic 7 — Reflection on MCP and the Asta Tools

Based on my work in this folder:
- `exercise_a_discover_tools.py` — manual `tools/list` interrogation
- `exercise_b_direct_tool_calls.py` — three hand-written drills
- `exercise_c_chatbot.py` — LLM-orchestrated chatbot with dynamic tool discovery
- `exercise_d_citation_explorer.py` — agent with code-controlled tool order

---

## 1. Hand-written schemas vs. dynamic discovery: what does automation buy and cost?

**What it bought me, concretely.**
In Exercise B I had to hand-write the tool name, parameter names, parameter types, and field defaults for `search_papers_by_relevance`, `get_citations`, and `get_paper`. When the exercise spec referenced `search_papers` and `get_references` (which don't exist on this server), I had to probe the real tool list manually and remap. In Exercise C, the entire schema layer collapsed into ~10 lines (`get_asta_tools()` in `exercise_c_chatbot.py`). All 8 tool definitions came from the server itself. If Asta added a new tool tomorrow, I'd change zero lines.

The other thing automation bought me was a single source of truth for tool *descriptions* — GPT-4o mini reads the same text that I would otherwise paraphrase in a Python comment. That alignment matters.

**What it cost me.**
- **A new startup failure mode.** Every run begins with a network round-trip to fetch the schema. If the Asta server is down at boot, the chatbot can't even start; in Exercise B I could at least call cached tools by name.
- **Loss of curation.** The server-provided descriptions are minimal. For example, `get_paper`'s `fields` parameter is documented only as `"Fields"` with default `"title"`. GPT-4o mini repeatedly forgot to pass `fields="title,authors"` because there's no schema signal that those field names even exist. I had to compensate in `SYSTEM_PROMPT`, hard-coding the warning that "Most Asta tools have `fields` defaulting to just 'title'." So schema automation *eliminated* one place where I encode knowledge but *moved it* to a different place — the prompt.
- **A trust surface.** The schema text becomes part of the LLM's system context. A malicious or buggy MCP server can effectively prompt-inject the client by sneaking instructions into a tool description. With hand-written schemas this risk doesn't exist.
- **Schema drift.** If Asta renamed `keyword` to `query` overnight, my Exercise C chatbot would silently start producing wrong tool calls. With hand-written schemas the breakage is loud and local.

Net: dynamic discovery is the right default for any system that talks to more than one server, but it doesn't remove the need to think about what the LLM actually sees.

---

## 2. Rich JSON in the context window: what did I include vs. discard?

In `call_asta_tool()` (Exercise C) I made three choices about what reaches the LLM:

1. **Concatenate all `content[].text` items** into a single tool message. Asta's `search_papers_by_relevance` returns each result as a separate content item — joining them lets the model see all of them at once.
2. **Hard cap at 12,000 characters.** Beyond that I append a `[...truncated]` marker. This prevented one early run from blowing past the 16k context window on `get_paper(fields=references)` for a heavily-cited paper.
3. **Pass through everything else verbatim** — including `openAccessPdf` blocks, `disclaimer` text, and other noise the model doesn't need.

**What I observed in quality.**
When I asked the chatbot "Who wrote *Attention is All You Need* and what else have they published?" the model produced an 8-author wall of text covering every co-author's recent work. The problem wasn't bad reasoning — it was that I handed the model raw output for every author it touched and it dutifully summarized all of it. Quality (in the sense of *concise, focused answers*) improved when I tightened the prompt rather than the data, but the deeper lesson is that **raw rich JSON encourages verbose answers**.

In Exercise D I went the other way: the *agent* (Python) did all filtering and ranking, then handed GPT-4o mini a tight JSON payload containing only the chosen 5 references, the chosen 5 citing papers, and one paper per author. The report was crisp because the model had no choices to make about *what* to include — only about *how* to write it. That convinced me of a general pattern: **filter and rank in code, narrate in the LLM**. The LLM is good at prose; it's average at deciding what to ignore.

---

## 3. What would it take to let the LLM decide order in Exercise D, and what could go wrong?

Exercise D's pipeline (in `exercise_d_citation_explorer.py`) hard-codes the order:

```
Step 1: get_paper(seed)                  ──► gives paperId, authors[]
Step 2: get_paper(refs) ➜ get_paper_batch ──► top 5 refs by citation
Step 3: get_citations(date≥2023)         ──► recent 5
Step 4: get_author_papers × N            ──► one paper per author
Step 5: LLM ─► markdown
```

**To make it LLM-driven** I'd need essentially what I already built in Exercise C: a loop that exposes all 8 Asta tools, plus a system prompt describing the *goal* ("produce a citation-neighborhood report containing exactly these four sections"). I'd also need a small state store outside the LLM, because the model forgets across iterations unless reminded.

**What could go wrong, based on what I already saw in Exercise C:**

- **Wrong-tool moments.** In Exercise C the model once called `get_citations` to "find authors" (citations are papers that cite, not authors of). With deterministic code I never make that mistake.
- **Wrong-ID mistakes.** The model also passed a `paperId` where an `author_id` was expected. The Python code in Exercise D has no opportunity to confuse the two — they're separate variables.
- **Skipped steps.** A "be efficient" model might decide that author profiles aren't necessary and omit Step 4 entirely, breaking the report structure.
- **Sequential when parallel is fine.** GPT-4o mini emits one tool call per turn unless prompted to batch. Exercise D currently runs 4 `get_author_papers` calls strictly sequentially — but a careful agent could parallelize. An LLM-driven version would likely be *slower*, not faster.
- **Recursion-limit exhaustion.** Exercise C has a 7-iteration cap. A complex paper with 8 authors and 60 references could easily exceed that.
- **Non-determinism and cost.** The Python pipeline produces the same report every run for ~$0 in tool costs. An LLM-orchestrated version would cost ~$0.05–0.15 per report and might omit or add sections at random.

Letting the LLM choose order is the right call when the *task* is open-ended ("help me research X"). Exercise D's task is fixed and structured; deterministic code is honestly the better tool here.

---

## 4. What a mature MCP ecosystem would offer that today's doesn't

Speaking only from pain points I actually hit:

- **Output schemas, not just input schemas.** I had to discover by trial that `get_citations` returns `{"citingPaper": {...}}` wrappers — and that `references` nested inside a paper return only `paperId + title`, no year or citation count. A real protocol would describe response shape.
- **Enumerated `fields` values.** Asta's `fields` parameter is a free-form string with no list of valid names. GPT-4o mini failed once with `fields="title,year,author"` (singular). An `enum` in the schema would catch that.
- **Standard error taxonomy.** Today errors arrive as `isError: true` plus a free-text English message. I have to grep strings to distinguish "paper not found" from "rate limited" from "invalid argument." A typed error code would let clients react sensibly.
- **Transport unification.** My Exercise A failed with HTTP 406 until I added `Accept: application/json, text/event-stream` — an undocumented gotcha. Different servers do SSE, HTTP, or stdio; clients shouldn't have to special-case.
- **Cost / rate-limit metadata.** A tool description could say "≈1s, rate-limited to 60/min, returns ~5KB." Clients (and LLMs) could then budget.
- **Pagination primitives.** I chunked `get_paper_batch` at 100 IDs manually. A standard `cursor` + `next_page` field would handle this once.
- **Public server registry.** Today MCP servers are word-of-mouth URLs in tutorials. A discovery directory (signed, versioned) would make MCP feel less like a private API.
- **Deprecation signals in the schema.** When Asta eventually adds `search_papers` and retires `search_papers_by_relevance`, the old tool's description should say so — and the schema should carry a `deprecated: true` flag.

The honest summary: MCP solves the "how do I describe a tool to an LLM" problem cleanly. It hasn't yet solved the broader "how do tools and clients evolve together over years" problem that mature ecosystems (REST + OpenAPI, gRPC + proto buffers) eventually had to.
