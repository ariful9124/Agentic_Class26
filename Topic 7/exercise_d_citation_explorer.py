"""
Exercise D - Citation Network Explorer Agent
============================================
An autonomous agent (NO LLM tool-choice). The agent makes a fixed
sequence of Asta MCP calls directly to build a "citation neighborhood"
around a seed paper, then hands the collected data to GPT-4o mini
which writes the final markdown report.

Usage:
    python exercise_d_citation_explorer.py [PAPER_ID]

If no PAPER_ID is given, defaults to BERT (ARXIV:1810.04805).
The exercise suggests ARXIV:2210.03629 (ReAct), but that paper is not
indexed in this Asta corpus.
"""

import os
import sys
import json
import requests
from openai import OpenAI

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Configuration ────────────────────────────────────────────────────────────
ASTA_URL    = "https://asta-tools.allen.ai/mcp/v1"
ASTA_API_KEY = os.environ["ASTA_API_KEY"]   # set this in your shell before running
ASTA_HEADERS = {
    "Content-Type": "application/json",
    "Accept":       "application/json, text/event-stream",
    "x-api-key":    ASTA_API_KEY
}

DEFAULT_PAPER_ID = "ARXIV:1810.04805"   # BERT
RECENT_YEAR_CUTOFF = "2023-01-01:"      # last ~3 years from current 2026

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "exercise_d_citation_explorer.txt")

client = OpenAI()
MODEL  = "gpt-4o-mini"


# ─── Low-level MCP helpers (same pattern as Exercises B & C) ──────────────────
def _mcp_post(method, params, call_id=1):
    payload = {"jsonrpc": "2.0", "id": call_id, "method": method, "params": params}
    resp = requests.post(ASTA_URL, headers=ASTA_HEADERS, json=payload)
    resp.raise_for_status()
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise ValueError("No SSE data line in Asta response")


def call_asta(name, arguments):
    """Call an Asta tool, return list of parsed JSON objects from content items.
    Raises on hard failure so the agent halts loudly."""
    response = _mcp_post("tools/call", {"name": name, "arguments": arguments}, call_id=2)
    result = response.get("result", {})
    if result.get("isError"):
        err = result.get("content", [{}])[0].get("text", "unknown")
        raise RuntimeError(f"Asta tool '{name}' failed: {err}")
    items = []
    for content in result.get("content", []):
        if content.get("type") == "text":
            try:
                items.append(json.loads(content["text"]))
            except json.JSONDecodeError:
                items.append({"raw": content["text"]})
    return items


def log(msg):
    """Progress logger (goes to stderr so stdout stays a clean markdown report)."""
    print(f"[agent] {msg}", file=sys.stderr)


# ─── Pipeline steps ───────────────────────────────────────────────────────────
def step1_get_seed_metadata(paper_id):
    log(f"Step 1: fetching seed paper metadata for {paper_id}")
    items = call_asta("get_paper", {
        "paper_id": paper_id,
        "fields":   "title,abstract,year,authors,fieldsOfStudy,citationCount"
    })
    return items[0]


def step2_top_references(paper_id, top_n=5):
    log(f"Step 2: fetching references and finding top {top_n} most-cited")
    # 2a: get bare reference list (paperId + title only — that's what Asta returns)
    paper_with_refs = call_asta("get_paper", {
        "paper_id": paper_id,
        "fields":   "references"
    })[0]
    refs = paper_with_refs.get("references", [])
    log(f"        seed has {len(refs)} references; enriching with citation counts")

    # 2b: enrich references with citation counts via batch lookup
    ref_ids = [r["paperId"] for r in refs if r.get("paperId")]
    if not ref_ids:
        return []

    # get_paper_batch can take many ids; chunk just to be safe
    enriched = []
    BATCH = 100
    for i in range(0, len(ref_ids), BATCH):
        chunk = ref_ids[i:i + BATCH]
        enriched.extend(call_asta("get_paper_batch", {
            "ids":    chunk,
            "fields": "title,abstract,year,authors,citationCount"
        }))

    # 2c: sort by citationCount desc, take top N
    enriched.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)
    return enriched[:top_n]


def step3_recent_citing_papers(paper_id, limit=5):
    log(f"Step 3: fetching recent citing papers (date >= {RECENT_YEAR_CUTOFF})")
    items = call_asta("get_citations", {
        "paper_id": paper_id,
        "fields":   "title,year,authors,abstract,citationCount",
        "limit":    limit,
        "publication_date_range": RECENT_YEAR_CUTOFF
    })
    # Each item is wrapped as {"citingPaper": {...}}
    return [it.get("citingPaper", it) for it in items]


def step4_author_profiles(authors, seed_paper_id, top_per_author=1):
    log(f"Step 4: fetching most-cited other work for {len(authors)} author(s)")
    profiles = []
    for author in authors:
        author_id = author.get("authorId")
        author_name = author.get("name", "Unknown")
        if not author_id:
            profiles.append({"name": author_name, "authorId": None, "papers": []})
            continue
        try:
            papers = call_asta("get_author_papers", {
                "author_id":    author_id,
                "paper_fields": "title,year,citationCount",
                "limit":        50
            })
        except RuntimeError as e:
            log(f"        author {author_name}: lookup failed - {e}")
            papers = []
        # Exclude the seed paper itself, then sort by citation count
        other = [p for p in papers if p.get("paperId") != seed_paper_id]
        other.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)
        profiles.append({
            "name":     author_name,
            "authorId": author_id,
            "papers":   other[:top_per_author]
        })
    return profiles


# ─── LLM markdown generation ──────────────────────────────────────────────────
REPORT_SYSTEM_PROMPT = (
    "You are a scientific writing assistant. You will be given a JSON "
    "object containing data about a seed academic paper and its citation "
    "neighborhood (foundational references, recent citing papers, author "
    "profiles). Your job is to produce a clean, well-formatted Markdown "
    "report based ONLY on the data provided. Do not invent facts. If a "
    "field is missing, omit it gracefully. Keep prose tight and factual.\n\n"
    "Output exactly this Markdown structure:\n"
    "# Citation Neighborhood Report: <seed paper title>\n\n"
    "## Summary\n"
    "One paragraph summarizing the seed paper (use its abstract, year, "
    "authors, and fields of study).\n\n"
    "## Foundational Works\n"
    "Numbered list of the 5 key references. For each: **Title** (year) - "
    "one-sentence relevance derived from the reference's abstract.\n\n"
    "## Recent Developments\n"
    "Numbered list of recent citing papers. For each: **Title** (year, "
    "first author) - one-sentence summary from the abstract.\n\n"
    "## Author Profiles\n"
    "One subsection per author. Heading is the author name. Body lists "
    "their most-cited other work with title, year, and citation count."
)


def generate_markdown_report(payload):
    log("Step 5: handing data to GPT-4o mini for markdown generation")
    user_message = (
        "Here is the citation-neighborhood data. Generate the Markdown "
        "report exactly as specified.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ]
    )
    return response.choices[0].message.content


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    paper_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PAPER_ID
    log(f"=== Citation Network Explorer ===")
    log(f"Seed paper ID: {paper_id}")

    # Step 1: seed metadata (everything else depends on this)
    seed = step1_get_seed_metadata(paper_id)
    real_paper_id = seed["paperId"]   # canonical Semantic Scholar ID

    # Steps 2, 3, 4 run after step 1; they are independent of each other
    top_refs        = step2_top_references(real_paper_id, top_n=5)
    recent_citing   = step3_recent_citing_papers(real_paper_id, limit=5)
    author_profiles = step4_author_profiles(
        seed.get("authors", []), real_paper_id, top_per_author=1
    )

    # Step 5: hand to LLM for markdown synthesis
    payload = {
        "seed_paper":        seed,
        "foundational_works": top_refs,
        "recent_developments": recent_citing,
        "author_profiles":   author_profiles
    }

    report_md = generate_markdown_report(payload)

    # Print to stdout (the deliverable)
    print(report_md)

    # Also save for portfolio
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report_md + "\n")
    log(f"Report also saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
