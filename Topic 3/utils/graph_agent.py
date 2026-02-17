# graph_agent.py
from typing import TypedDict, Annotated

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from utils.tools import TOOLS


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def make_system_prompt(one_tool_per_step: bool) -> SystemMessage:
    rules = (
        "You are a tool-using assistant.\n"
        "Rules:\n"
        "- Never guess; use tools.\n"
        "- For letter counts, call count_letter_tool.\n"
        "- For sin(x), call sin_tool.\n"
        "- If asked for sin of difference between two letter counts:\n"
        "  call count_letter_tool twice (one per letter), compute diff, then call sin_tool(diff).\n"
        "- For I-64 city info, use list_i64_cities/get_coords_tool/distance_between_cities_tool.\n"
        "- calculator_tool can be used for add/sub/mul/div if you want a tool for the difference.\n"
    )
    if one_tool_per_step:
        rules += "\nIMPORTANT: Call at most ONE tool per assistant turn. If multiple tools are needed, do it step-by-step."
    return SystemMessage(content=rules)


def build_graph(model: str = "gpt-4o-mini", one_tool_per_step: bool = False):
    system = make_system_prompt(one_tool_per_step)
    llm = ChatOpenAI(model=model, temperature=0).bind_tools(TOOLS)

    def assistant(state: AgentState) -> AgentState:
        msgs = state["messages"]
        if not msgs or msgs[0].type != "system":
            msgs = [system] + msgs
        ai = llm.invoke(msgs)
        return {"messages": [ai]}

    g = StateGraph(AgentState)
    g.add_node("assistant", assistant)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_conditional_edges("assistant", tools_condition)
    g.add_edge("tools", "assistant")
    g.set_entry_point("assistant")

    return g.compile()
