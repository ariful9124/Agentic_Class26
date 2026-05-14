"""
Exercise C - Asta-Powered Research Chatbot
==========================================
A chatbot that:
  1. Discovers Asta tool schemas at startup via MCP tools/list.
  2. Converts those schemas to OpenAI function-calling format.
  3. Uses GPT-4o mini to decide which tools to call.
  4. Dispatches tool calls back to the Asta MCP server.
  5. Loops until the model produces a final text answer.
"""

import os
import sys
import json
import requests
from openai import OpenAI

# Force UTF-8 on the Windows console so Unicode paper titles print cleanly
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

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "exercise_c_chatbot.txt")

client = OpenAI()                # reads OPENAI_API_KEY env var
MODEL  = "gpt-4o-mini"

# Output buffer so everything can be saved to the .txt file at the end
_log = []
def out(text=""):
    print(text)
    _log.append(text)


# ─── Internal helper: low-level MCP POST ──────────────────────────────────────
def _mcp_post(method, params, call_id=1):
    """Send any JSON-RPC method to Asta and return the parsed result dict."""
    payload = {"jsonrpc": "2.0", "id": call_id, "method": method, "params": params}
    resp = requests.post(ASTA_URL, headers=ASTA_HEADERS, json=payload)
    resp.raise_for_status()
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise ValueError("No SSE data line in Asta response")


# ─── Function 1: Discover tools and convert schemas ───────────────────────────
def get_asta_tools():
    """Fetch tool schemas from MCP and convert to OpenAI function-calling format."""
    response = _mcp_post("tools/list", {})
    mcp_tools = response["result"]["tools"]

    openai_tools = []
    for tool in mcp_tools:
        # Asta descriptions are multi-line; collapse to a single sentence
        description = tool.get("description", "").strip().split("\n")[0].strip()
        openai_tools.append({
            "type": "function",
            "function": {
                "name":        tool["name"],
                "description": description,
                # MCP inputSchema is already valid JSON Schema -> direct mapping
                "parameters":  tool["inputSchema"]
            }
        })
    return openai_tools


# ─── Function 2: Execute a tool call against the Asta server ──────────────────
def call_asta_tool(name, arguments):
    """Execute a tools/call request on Asta and return the joined text content.

    Any failure (network, HTTP, malformed response, MCP-reported error) is
    converted to an 'ERROR: <message>' string and returned, so the LLM can
    read it as a tool result and try a different approach.
    """
    try:
        response = _mcp_post(
            "tools/call",
            {"name": name, "arguments": arguments},
            call_id=2
        )
    except requests.HTTPError as e:
        return f"ERROR: HTTP {e.response.status_code} from Asta - {e.response.text[:200]}"
    except requests.RequestException as e:
        return f"ERROR: Network error calling Asta - {e}"
    except (json.JSONDecodeError, ValueError) as e:
        return f"ERROR: Could not parse Asta response - {e}"
    except Exception as e:
        return f"ERROR: Unexpected failure calling Asta - {type(e).__name__}: {e}"

    result = response.get("result", {})
    content_items = result.get("content", [])

    # Asta returns one or many {type:"text", text:"..."} items.
    # Concatenate them so the LLM gets the full payload as one tool message.
    chunks = [item.get("text", "") for item in content_items if item.get("type") == "text"]
    text   = "\n".join(chunks).strip()

    # MCP-level errors (tool ran but failed)
    if result.get("isError"):
        text = "ERROR: " + text

    # Truncate very long results so we don't blow up the context window
    MAX_CHARS = 12_000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"\n\n[...truncated, original length {len(text)} chars]"

    return text


# ─── Function 3: One conversation turn with tool-call handling ────────────────
def chat(user_message, messages, tools):
    """One turn of the chatbot loop. Mutates `messages` in place.
       Returns the final text answer from the assistant."""
    messages.append({"role": "user", "content": user_message})
    out(f"\nUser: {user_message}")

    MAX_ITERS = 7
    for iteration in range(1, MAX_ITERS + 1):
        response = client.chat.completions.create(
            model       = MODEL,
            messages    = messages,
            tools       = tools,
            tool_choice = "auto"
        )
        assistant_msg = response.choices[0].message

        # Case A: model wants to call one or more tools
        if assistant_msg.tool_calls:
            out(f"  [iter {iteration}] LLM requested {len(assistant_msg.tool_calls)} tool call(s)")
            # Re-shape assistant message into a plain dict before appending
            messages.append({
                "role":      "assistant",
                "content":   assistant_msg.content,
                "tool_calls": [tc.model_dump() for tc in assistant_msg.tool_calls]
            })

            for tc in assistant_msg.tool_calls:
                fname = tc.function.name
                try:
                    fargs = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fargs = {}
                out(f"     -> {fname}({json.dumps(fargs)})")

                tool_result = call_asta_tool(fname, fargs)
                preview = tool_result.replace("\n", " ")[:120]
                out(f"        result preview: {preview}...")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         fname,
                    "content":      tool_result
                })
            # loop again so the model can read tool output
            continue

        # Case B: model produced a final text answer
        final_text = assistant_msg.content or ""
        messages.append({"role": "assistant", "content": final_text})
        out(f"\nAssistant: {final_text}")
        return final_text

    out("\n[Max tool-call iterations reached]")
    return "[Max iterations reached]"


# ─── Main driver ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful academic-research assistant. You are backed by the "
    "Asta MCP toolset, which provides access to the Semantic Scholar "
    "academic graph (papers, authors, citations, references). Use these "
    "tools to fetch real data; never fabricate.\n\n"
    "Critical rules:\n"
    "- Most Asta tools have `fields` defaulting to just 'title'. When you "
    "need authors, year, abstract, or references, ALWAYS pass an explicit "
    "`fields` argument listing every field you need "
    "(e.g. fields='title,year,authors').\n"
    "- Paper IDs and author IDs are different. NEVER pass a paperId as the "
    "author_id argument. To get an author's other papers: first call "
    "get_paper with fields='authors' to retrieve authorIds, then pass "
    "one of those authorIds to get_author_papers.\n"
    "- get_citations returns *papers that cite a given paper*, NOT the "
    "authors of that paper. To find authors, use get_paper with "
    "fields including 'authors'.\n"
    "- Prefer search_papers_by_relevance for broad topic queries, "
    "search_paper_by_title for a specific known title, "
    "search_authors_by_name + get_author_papers for author lookups, "
    "and get_paper for full paper details.\n"
    "- If a tool returns ERROR, read it carefully and try a different "
    "approach. Do not give up after one failure.\n"
    "When you have the data, answer concisely in plain English with paper "
    "titles, years, and authors as appropriate."
)


def main():
    out("=" * 70)
    out("Exercise C - Asta-Powered Research Chatbot (GPT-4o mini)")
    out("=" * 70)

    # Phase 0: discover tools
    tools = get_asta_tools()
    out(f"\nDiscovered {len(tools)} tools from Asta MCP server:")
    for t in tools:
        out(f"   - {t['function']['name']}: {t['function']['description']}")

    # Phase 1: build a persistent message thread (conversation history)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    out("\n" + "=" * 70)
    out("Interactive chat ready.")
    out("Type a research question, or 'quit' / 'exit' to end the session.")
    out("=" * 70)

    turn = 0
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            out("\n[Session interrupted by user]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q", ":q"}:
            out("\n[Session ended by user]")
            break

        turn += 1
        out("\n" + "-" * 70)
        out(f"TURN {turn}")
        out("-" * 70)
        chat(user_input, messages, tools)

    out("\n" + "=" * 70)
    out(f"Session complete - {turn} turn(s).")
    out("=" * 70)

    # ── Discussion ───────────────────────────────────────────────────────────
    out("\n" + "=" * 70)
    out("DISCUSSION: What changed compared to Exercise B?")
    out("=" * 70)
    out(
        "In Exercise B, every tool call was hand-written: we hard-coded the "
        "tool name, the parameter names, and the parameter types directly "
        "into the Python source. If Asta renamed a tool or added a new one, "
        "we would have had to rewrite the script.\n\n"
        "In this exercise, we wrote almost NO tool-specific code. The schema "
        "for all 8 tools came from the server itself via tools/list, and "
        "get_asta_tools() converted those MCP schemas straight into OpenAI's "
        "function-calling format (the mapping is one-line because MCP's "
        "inputSchema is already valid JSON Schema). The chatbot would work "
        "identically if Asta added new tools tomorrow - we would not change "
        "a single line of code; the new tool would simply appear in the "
        "tools list at the next startup, and GPT-4o mini would learn about "
        "it from its description and start using it.\n\n"
        "That is the core value of MCP: tool discovery is dynamic, so the "
        "client (this chatbot) is decoupled from the server's evolving "
        "capabilities. The same chatbot code can talk to any MCP server - "
        "Asta today, a different server tomorrow - because the contract is "
        "the protocol, not the tool list."
    )
    out("=" * 70)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")
    print(f"\nTranscript saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
