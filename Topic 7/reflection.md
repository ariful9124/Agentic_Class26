# Topic 7 — Reflection

Short reflection on my work in this folder (Exercises A–D).

---

## 1. Hand-written schemas vs. dynamic discovery

In Exercise B I had to write the tool name, parameters, and types myself. When the exercise referenced `search_papers` and `get_references` (which don't exist on Asta), I had to probe the server manually and pick the closest tool.

In Exercise C, all that disappeared. `get_asta_tools()` is about 10 lines and fetches everything from the server. If Asta adds a new tool tomorrow, my code does not change.

**What it costs:** the chatbot now depends on the server being up at startup. Also, the schema descriptions are short — for example `fields` is just labeled "Fields" with default "title". GPT-4o mini kept forgetting to pass `fields="title,authors"`, so I had to write that rule into the system prompt. The knowledge didn't go away, it just moved from Python to the prompt.

---

## 2. Rich JSON in context window

In `call_asta_tool()` I do three things: join all content items into one string, cap at 12,000 characters, and pass everything else through as-is.

When I asked the chatbot *"Who wrote Attention is All You Need and what else have they published?"*, it produced an 8-author wall of text because I gave it the full output for every author it touched. Quality only improved after I tightened the prompt.

In Exercise D I did the opposite — the Python code picked the top 5 references, top 5 citing papers, and one paper per author *before* the LLM saw anything. The report was much cleaner. My takeaway: **filter and rank in code, write prose in the LLM**.

---

## 3. Letting the LLM decide order in Exercise D

Exercise D runs in a fixed order: seed metadata → references → citations → author profiles → report. Each step depends on the previous one's output (especially author IDs from step 1).

To let the LLM control the order, I'd need to expose all 8 tools and run a loop like Exercise C. But based on what I already saw in Exercise C:
- The model once used `get_citations` to look up authors (wrong tool — citations are papers, not authors).
- It passed a paper ID where an author ID belonged.
- It would run 4 author lookups sequentially instead of in parallel.
- Each iteration is a paid GPT call, so the report becomes more expensive and slower.

For a structured task with a fixed output, Python is honestly the better choice. LLM control makes more sense when the task is open-ended.

---

## 4. What a mature MCP ecosystem should add

Based on real pain points I hit:
- **Output schemas.** I had to learn by trial that `get_citations` wraps results in `{"citingPaper": {...}}`. Only input schemas are described.
- **Enumerated `fields` values.** Free-form strings caused `fields="title,year,author"` (singular) to fail at runtime instead of validation.
- **Standard error codes.** Today I have to read English error text to tell "not found" from "rate limited".
- **Transport consistency.** Exercise A failed with HTTP 406 until I added `Accept: application/json, text/event-stream` — an undocumented gotcha.
- **Pagination.** I chunked `get_paper_batch` at 100 IDs manually.
- **A public registry.** Right now MCP servers are word-of-mouth URLs.

MCP solves *"how do I describe a tool to an LLM"* nicely. It doesn't yet solve *"how do tools and clients evolve together over years"* — which REST + OpenAPI eventually had to figure out.
