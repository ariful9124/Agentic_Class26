import torch
from typing import Annotated, TypedDict

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# -------------------------
# DEVICE SELECTION
# -------------------------
def get_device() -> str:
    """Priority: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"

# -------------------------
# LLM CREATION (LLAMA ONLY)
# -------------------------
def create_llama_llm(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    max_new_tokens: int = 128,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    device = get_device()

    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text= False
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    print("Model loaded successfully!")
    return llm


# -------------------------
# STATE (Message API)
# -------------------------
class AgentState(TypedDict):
    # Message history with reducer that appends/updates messages correctly.
    messages: Annotated[list[AnyMessage], add_messages]

    # Control flags
    verbose: bool
    should_exit: bool
    is_empty: bool


# -------------------------
# HELPERS
# -------------------------
def messages_to_prompt(messages: list[AnyMessage]) -> str:
    """
    Convert Message API objects -> a single text prompt for HF text-generation models.
    """
    lines = []
    for m in messages:
        role = getattr(m, "type", "unknown")  # "system", "human", "ai", "tool"
        content = getattr(m, "content", "")

        if role == "system":
            lines.append(f"System: {content}")
        elif role in ("human", "user"):
            lines.append(f"User: {content}")
        elif role in ("ai", "assistant"):
            lines.append(f"Assistant: {content}")
        elif role in ("tool", "function"):
            lines.append(f"Tool: {content}")
        else:
            lines.append(f"{role}: {content}")

    lines.append("Assistant:")
    return "\n".join(lines)


def clean_llm_text(raw: str) -> str:
    """
    Clean completion so we don't print echoed prompts or fake multi-turn "User:" blocks.
    """
    text = raw

    # If model echoed the prompt and included "Assistant:", keep only what follows.
    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[-1].strip()

    # If it starts inventing new user turns, cut at first "\nUser:"
    if "\nUser:" in text:
        text = text.split("\nUser:", 1)[0].strip()

    return text.strip()


# -------------------------
# GRAPH
# -------------------------
def create_graph(llm):
    # =========================
    # NODE 1: get_user_input
    # =========================
    def get_user_input(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering get_user_input")

        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input().strip()

        # Exit
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            return {"should_exit": True, "is_empty": False}

        # Empty input -> self-loop, no LLM call
        if user_input == "":
            return {"is_empty": True, "should_exit": False}

        # Verbose toggles (no LLM call)
        if user_input.lower() == "verbose":
            print("Verbose mode ENABLED")
            return {"verbose": True, "is_empty": True, "should_exit": False}

        if user_input.lower() == "quiet":
            print("Verbose mode DISABLED")
            return {"verbose": False, "is_empty": True, "should_exit": False}

        # Normal input: append HumanMessage to history
        return {
            "messages": [HumanMessage(content=user_input)],
            "is_empty": False,
            "should_exit": False,
        }

    # =========================
    # ROUTER: 3-way branch + GUARD
    # =========================
    def route_after_input(state: AgentState) -> str:
        if state.get("verbose", False):
            print("[TRACE] Routing after input")

        if state.get("should_exit", False):
            if state.get("verbose", False):
                print("[TRACE] Routing to END")
            return END

        if state.get("is_empty", False):
            if state.get("verbose", False):
                print("[TRACE] Empty input / command handled. Routing back to get_user_input.")
            return "get_user_input"

        # ✅ GUARD: only call LLM if the newest message is HumanMessage
        msgs = state.get("messages", [])
        if not msgs or getattr(msgs[-1], "type", None) != "human":
            if state.get("verbose", False):
                last_type = getattr(msgs[-1], "type", None) if msgs else None
                print(f"[TRACE] No new HumanMessage (last={last_type}). Routing back to get_user_input.")
            return "get_user_input"

        return "call_llm"

    # =========================
    # NODE 2: call_llm
    # =========================
    def call_llm(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering call_llm")
            print(f"[TRACE] messages in history = {len(state.get('messages', []))}")

        prompt = messages_to_prompt(state["messages"])

        if state.get("verbose", False):
            print("[TRACE] Prompt built from message history (first 300 chars):")
            print(prompt[:300])

        raw = llm.invoke(prompt)
        reply = clean_llm_text(raw)

        if state.get("verbose", False):
            print("[TRACE] LLM returned response")

        # Append AI response into message history
        return {"messages": [AIMessage(content=reply)]}

    # =========================
    # NODE 3: print_response
    # =========================
    def print_response(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering print_response")

        last = state["messages"][-1]
        text = getattr(last, "content", "")

        print("\n" + "-" * 50)
        print("LLM Response:")
        print("-" * 50)
        print(text)

        return {}

    # =========================
    # GRAPH BUILDING
    # =========================
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llm": "call_llm",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    graph_builder.add_edge("call_llm", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()


# -------------------------
# MAIN
# -------------------------
def main():
    print("=" * 60)
    print("LangGraph Simple Agent (Chat History via Message API) - Llama only")
    print("=" * 60)
    print("Commands: verbose | quiet | quit/exit/q")
    print()

    llm = create_llama_llm()
    graph = create_graph(llm)

    initial_state: AgentState = {
        "messages": [SystemMessage(content="You are a helpful assistant. Keep answers concise.")],
        "verbose": False,
        "should_exit": False,
        "is_empty": False,
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
