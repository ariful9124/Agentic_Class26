import requests, os, json

ASTA_API_KEY = os.environ["ASTA_API_KEY"]   # set this in your shell before running

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": ASTA_API_KEY
}

payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}

resp = requests.post(
    "https://asta-tools.allen.ai/mcp/v1",
    headers=headers,
    json=payload
)

resp.raise_for_status()

# Server returns SSE (text/event-stream); extract the JSON from "data:" lines
result_json = None
for line in resp.text.splitlines():
    if line.startswith("data:"):
        result_json = json.loads(line[len("data:"):].strip())
        break

if result_json is None:
    raise ValueError("No data line found in SSE response")

tools = result_json["result"]["tools"]

print(f"Found {len(tools)} tool(s) on the Asta MCP server:\n")
print("=" * 60)

for tool in tools:
    name = tool.get("name", "N/A")
    description = tool.get("description", "No description available.")
    one_line_desc = description.strip().split("\n")[0].strip()

    print(f"\nTool: {name}")
    print(f"  Description: {one_line_desc}")

    schema = tool.get("inputSchema", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    if required:
        print("  Required parameters:")
        for param in required:
            param_info = properties.get(param, {})
            param_type = param_info.get("type", "unknown")
            param_desc = param_info.get("description", "")
            if param_desc:
                print(f"    - {param} ({param_type}): {param_desc}")
            else:
                print(f"    - {param} ({param_type})")
    else:
        print("  Required parameters: none")

print("\n" + "=" * 60)

print("""
Q&A
============================================================

Q1: Which tool would you use to find all papers about
    "transformer attention mechanisms"?

A:  search_papers_by_relevance
    This tool is designed for broad topic/keyword searches across
    the corpus. Passing keyword="transformer attention mechanisms"
    will rank and return papers by relevance to that topic -
    exactly what you need when exploring a research area rather
    than looking up a specific title.

Q2: Which tool would you use to find who else published in the
    same area as a specific author?

A:  Two-step approach:
    1. get_author_papers (author_id) - retrieve the target
       author's papers to identify the topics they work on.
    2. search_papers_by_relevance (keyword) - search with
       keywords from those papers to surface other researchers
       publishing in the same area.

============================================================""")
