# task4.py
from langchain_core.messages import HumanMessage
from utils.graph_agent import build_graph
# version-safe recursion error import
try:
    from langgraph.errors import GraphRecursionError
except Exception:
    GraphRecursionError = Exception


def run_once(graph, user_text: str, recursion_limit: int = 25) -> str:
    try:
        out = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={"recursion_limit": recursion_limit},
        )
        return out["messages"][-1].content
    except GraphRecursionError:
        return f"[Stopped] Hit recursion limit = {recursion_limit} before finishing."
    except Exception as e:
        return f"[Error] {type(e).__name__}: {e}"


# %%
DEMO_MULTI_TOOL_SAME_TURN = [
    "Are there more i's than s's in 'Mississippi riverboats'? Use tools and give both counts."
]

DEMO_CHAINING = [
    "What is the sin of the difference between the number of i's and the number of s's "
    "in 'Mississippi riverboats'? Use tools."
]

DEMO_USE_ALL_TOOLS = [
    "Using tools only:\n"
    "1) Compute distance (miles) between Richmond, VA and Charlottesville, VA.\n"
    "2) Count i's and s's in 'Mississippi riverboats'.\n"
    "3) Let k = i_count - s_count. Use calculator_tool to compute k.\n"
    "4) Compute sin(k) using sin_tool.\n"
    "Return a short summary with distance, counts, k, and sin(k)."
]

DEMO_TRY_HIT_5_TURNS = [
    "Do this strictly one step at a time, using exactly ONE tool call per step, and wait for results:\n"
    "1) Count letter 'i' in 'Mississippi riverboats'\n"
    "2) Count letter 's' in 'Mississippi riverboats'\n"
    "3) Use calculator_tool to compute k = i_count - s_count\n"
    "4) Compute sin(k) using sin_tool\n"
    "5) List I-64 cities\n"
    "If you get cut off, say where you stopped."
]

if __name__ == "__main__":
    graph_parallel = build_graph(one_tool_per_step=False)
    graph_sequential = build_graph(one_tool_per_step=True)

    print("=" * 80)
    print("DEMO A: Multiple tool calls in a single assistant turn")
    print("=" * 80)
    for q in DEMO_MULTI_TOOL_SAME_TURN:
        print("USER:", q)
        print("ASSISTANT:", run_once(graph_parallel, q, recursion_limit=25))
        print()

    print("=" * 80)
    print("DEMO B: Sequential chaining (count -> count -> (calc) -> sin)")
    print("=" * 80)
    for q in DEMO_CHAINING:
        print("USER:", q)
        print("ASSISTANT:", run_once(graph_parallel, q, recursion_limit=25))
        print()

    print("=" * 80)
    print("DEMO C: Query that uses ALL tools")
    print("=" * 80)
    for q in DEMO_USE_ALL_TOOLS:
        print("USER:", q)
        print("ASSISTANT:", run_once(graph_parallel, q, recursion_limit=25))
        print()

    print("=" * 80)
    print("DEMO D: Try to hit 5-turn outer loop limit (one tool per step + recursion_limit=5)")
    print("=" * 80)
    for q in DEMO_TRY_HIT_5_TURNS:
        print("USER:", q)
        print("ASSISTANT:", run_once(graph_sequential, q, recursion_limit=5))
        print()



