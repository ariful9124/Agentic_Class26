# %%
import os
import argparse
import math
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ✅ SQLite checkpointer
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.tools import TOOLS

def save_graph_image(graph, filename="Topic3_task5.png"):
    """
    Generate a Mermaid diagram of the graph and save it as a PNG image.
    Uses the graph's built-in Mermaid export functionality.
    """
    try:
        # Get the Mermaid PNG representation of the graph
        # This requires the 'grandalf' package for rendering
        png_data = graph.get_graph(xray=True).draw_mermaid_png()

        # Write the PNG data to file
        with open(filename, "wb") as f:
            f.write(png_data)

        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")

# =========================================================
# STATE
# =========================================================

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    should_exit: bool
    verbose: bool
    is_empty: bool


# =========================================================
# NODE: Get user input
# =========================================================

def get_user_input(state: AgentState) -> dict:
    if state.get("verbose", False):
        print("[TRACE] Entering get_user_input")

    print("\n" + "=" * 60)
    print("Tool-Using Chat Agent (LangGraph) - Single Long Conversation")
    print("=" * 60)
    print("Commands: verbose | quiet | quit/exit/q")
    print()

    raw = input("> ").strip()

    if raw.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        return {"should_exit": True, "is_empty": False}

    if raw == "":
        return {"is_empty": True, "should_exit": False}

    if raw.lower() == "verbose":
        print("Verbose mode ENABLED")
        return {"verbose": True, "is_empty": True, "should_exit": False}

    if raw.lower() == "quiet":
        print("Verbose mode DISABLED")
        return {"verbose": False, "is_empty": True, "should_exit": False}

    return {
        "messages": [HumanMessage(content=raw)],
        "should_exit": False,
        "is_empty": False,
    }


# =========================================================
# GRAPH CREATION (assistant + tools + print + loop)
# =========================================================

def create_graph(checkpointer: SqliteSaver, model: str = "gpt-4o-mini"):
    system_msg = SystemMessage(content=(
        "You are a helpful assistant that can use tools.\n"
        "Rules:\n"
        "- Never guess coordinates or counts; always use tools.\n"
        "- Use count_letter_tool for letter counting.\n"
        "- Use sin_tool for sin(x).\n"
        "- Use calculator_tool for add/sub/mul/div if needed.\n"
        "- Use I-64 tools for city list/coords/distance.\n"
        "- Keep answers brief and show key numbers."
    ))

    llm = ChatOpenAI(model=model, temperature=0).bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        if state.get("is_empty", False):
            return "get_user_input"
        return "assistant"
    
    def route_after_assistant(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "print_response"

    def assistant(state: AgentState) -> dict:
        msgs = state.get("messages", [])
        if not msgs or getattr(msgs[0], "type", None) != "system":
            msgs = [system_msg] + msgs
        ai = llm.invoke(msgs)
        return {"messages": [ai]}

    def print_response(state: AgentState) -> dict:
        last = state["messages"][-1]
        if getattr(last, "type", None) in ("ai", "assistant"):
            print("\n" + "-" * 60)
            print("Assistant:")
            print("-" * 60)
            print(last.content)
        return {}

    g = StateGraph(AgentState)

    g.add_node("get_user_input", get_user_input)
    g.add_node("assistant", assistant)
    g.add_node("tools", tool_node)
    g.add_node("print_response", print_response)

    g.add_edge(START, "get_user_input")

    # Decide whether to exit / ask again / call assistant
    g.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "assistant": "assistant",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    # Tool loop (assistant -> tools? -> assistant)
    g.add_conditional_edges("assistant", tools_condition)
    g.add_edge("tools", "assistant")

    # After the assistant finishes (no more tools), print and return to input
    # g.add_edge("assistant", "print_response")
    
    g.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {
            "tools": "tools",
            "print_response": "print_response",
        },
    )

    g.add_edge("tools", "assistant")
    g.add_edge("print_response", "get_user_input")

    return g.compile(checkpointer=checkpointer)





# %%
# =========================================================
# MAIN (resume / start with crash recovery)
# =========================================================

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--thread",
        default=os.environ.get("LANGGRAPH_THREAD_ID", "chat-1"),
        help="Conversation thread_id (same id => resume from checkpoints)",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("LANGGRAPH_CHECKPOINT_DB", "tool_chat_checkpoints.db"),
        help="SQLite checkpoint DB file",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI chat model name (default: gpt-4o-mini)",
    )
    args, _ = parser.parse_known_args()

    print("=" * 70)
    print("LangGraph: Tool-Using Chat Agent (Single Long Conversation)")
    print("=" * 70)
    print(f"Checkpoint DB: {args.db}")
    print(f"Thread ID:     {args.thread}")
    print(f"Model:         {args.model}")
    print()

    with SqliteSaver.from_conn_string(args.db) as checkpointer:
        graph = create_graph(checkpointer=checkpointer, model=args.model)
        config = {"configurable": {"thread_id": args.thread}}

        save_graph_image(graph, filename="Topic3_task5.png")
        # Resume if there is unfinished work
        current = graph.get_state(config)
        if getattr(current, "next", None):
            print("🔄 Found saved state. Resuming from last checkpoint...\n")
            graph.invoke(None, config=config)
        else:
            print("▶️  Starting a new conversation...\n")
            initial_state: AgentState = {
                "messages": [SystemMessage(content="(system placeholder)")]
                , "should_exit": False
                , "verbose": False
                , "is_empty": False
            }
            graph.invoke(initial_state, config=config)


if __name__ == "__main__":
    main()


