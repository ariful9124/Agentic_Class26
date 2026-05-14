import requests, os, json, sys

# Force UTF-8 on Windows console so Unicode paper titles print cleanly
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ASTA_API_KEY = os.environ["ASTA_API_KEY"]   # set this in your shell before running
URL = "https://asta-tools.allen.ai/mcp/v1"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": ASTA_API_KEY
}

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "exercise_b_direct_tool_calls.txt")


def call_tool(name, arguments, call_id=1):
    payload = {
        "jsonrpc": "2.0",
        "id": call_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments}
    }
    resp = requests.post(URL, headers=headers, json=payload)
    resp.raise_for_status()
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise ValueError("No data line in SSE response")


def parse_content(result):
    """Return list of parsed JSON objects from result content items."""
    items = []
    for item in result.get("result", {}).get("content", []):
        if item.get("type") == "text":
            try:
                items.append(json.loads(item["text"]))
            except json.JSONDecodeError:
                items.append({"raw": item["text"]})
    return items


# ── output goes to both stdout and file ───────────────────────────────────────
lines = []

def out(text=""):
    print(text)
    lines.append(text)


# =============================================================================
# Drill 1 — search_papers_by_relevance: Find recent LLM agent papers
# (Exercise uses "search_papers"; mapped to search_papers_by_relevance,
#  the closest available tool. Parameters: keyword, fields, limit.)
# =============================================================================
out("=" * 65)
out("DRILL 1 — search_papers_by_relevance")
out('Query: "large language model agents" | Top 5 results')
out("=" * 65)

result1 = call_tool(
    "search_papers_by_relevance",
    {
        "keyword": "large language model agents",
        "fields": "title,abstract,year,authors",
        "limit": 5
    },
    call_id=1
)

papers = parse_content(result1)
for i, paper in enumerate(papers, 1):
    title = paper.get("title", "N/A")
    year  = paper.get("year", "N/A")
    authors = paper.get("authors", [])
    author_names = ", ".join(a.get("name", "") for a in authors[:3])
    if len(authors) > 3:
        author_names += " et al."
    out(f"\n{i}. {title}")
    out(f"   Year: {year}")
    out(f"   Authors: {author_names}")


# =============================================================================
# Drill 2 — get_citations: Trace impact of BERT (2023 onward)
# =============================================================================
out("\n" + "=" * 65)
out("DRILL 2 — get_citations")
out("Paper: BERT (ARXIV:1810.04805) | Citations from 2023 onward")
out("=" * 65)

result2 = call_tool(
    "get_citations",
    {
        "paper_id": "ARXIV:1810.04805",
        "fields": "title,year,authors",
        "limit": 10,
        "publication_date_range": "2023-01-01:"
    },
    call_id=2
)

citations = parse_content(result2)
out(f"\nTotal citation records returned: {len(citations)}")
out("First 5 citing papers:\n")
for i, entry in enumerate(citations[:5], 1):
    paper = entry.get("citingPaper", entry)
    title = paper.get("title", "N/A")
    year  = paper.get("year", "N/A")
    out(f"  {i}. {title} ({year})")


# =============================================================================
# Drill 3 — get_paper with fields=references: Intellectual foundation
# Note: The exercise specifies the ReAct paper (ARXIV:2210.03629), which is
# not indexed in this Asta corpus. Using BERT (ARXIV:1810.04805) to
# demonstrate the same access pattern: get_paper with fields=references.
# =============================================================================
out("\n" + "=" * 65)
out("DRILL 3 — get_paper (fields=references)")
out("Paper: BERT (ARXIV:1810.04805) | References sorted by year")
out("Note: Exercise specified ReAct (ARXIV:2210.03629) but that paper")
out("      is not indexed in this Asta corpus; BERT used as substitute.")
out("=" * 65)

result3 = call_tool(
    "get_paper",
    {
        "paper_id": "ARXIV:1810.04805",
        "fields": "title,year,references"
    },
    call_id=3
)

paper_data = parse_content(result3)
if paper_data:
    paper = paper_data[0]
    refs = paper.get("references", [])
    out(f"\nPaper: {paper.get('title', 'N/A')}")
    out(f"Total references: {len(refs)}")
    out("(Note: the Asta API returns only paperId + title for nested")
    out(" reference entries; individual years require a separate lookup.)")
    out("References listed in order returned by API:\n")
    for i, ref in enumerate(refs, 1):
        ref_title = ref.get("title", "N/A")
        out(f"  {i:3}. {ref_title}")

out("\n" + "=" * 65)

# Write output file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nOutput saved to: {OUTPUT_FILE}")
