import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

def save_graph_image(graph, filename="lg_graph.png"):
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

# -------------------------
# STATE
# -------------------------
class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    is_empty: bool
    verbose: bool
    active_model: str  # "llama" or "qwen"
    llm_response: str


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
        

def create_hf_llm(model_id: str):
    """
    Generic helper to load any HF causal LM + tokenizer and wrap it as a LangChain LLM.
    """
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
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print("Model loaded successfully!")
    return llm

def get_user_input(state: AgentState) -> dict:
    if state.get("verbose", False):
        print("[TRACE] Entering get_user_input")

    print("\n" + "=" * 50)
    print("Enter your text (or 'quit' to exit):")
    print("=" * 50)

    print("\n> ", end="")
    raw = input().strip()

    # Exit
    if raw.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        return {
            "user_input": raw,
            "should_exit": True,
            "is_empty": False,
        }

    # Empty -> self-loop via router
    if raw == "":
        return {
            "user_input": "",
            "should_exit": False,
            "is_empty": True,
        }

    # Verbose toggles (no LLM call)
    if raw.lower() == "verbose":
        print("Verbose mode ENABLED")
        return {
            "user_input": "",
            "should_exit": False,
            "is_empty": True,   # ensures router loops back without calling LLM
            "verbose": True,
        }

    if raw.lower() == "quiet":
        print("Verbose mode DISABLED")
        return {
            "user_input": "",
            "should_exit": False,
            "is_empty": True,   # ensures router loops back without calling LLM
            "verbose": False,
        }

    # Persistent model switching commands (no LLM call)
    if raw.lower() == "hey qwen":
        print("Switched active model to Qwen")
        return {
            "user_input": "",
            "should_exit": False,
            "is_empty": True,   # loop back to prompt
            "active_model": "qwen",
        }

    if raw.lower() == "hey llama":
        print("Switched active model to Llama")
        return {
            "user_input": "",
            "should_exit": False,
            "is_empty": True,   # loop back to prompt
            "active_model": "llama",
        }

    # Normal input -> proceed to active model
    return {
        "user_input": raw,
        "should_exit": False,
        "is_empty": False,
    }

# -------------------------
# GRAPH
# -------------------------
def create_graph(llama_llm, qwen_llm):
    # =========================
    # ROUTER: 4-way branch
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

        active = state.get("active_model", "llama")
        if active == "qwen":
            if state.get("verbose", False):
                print("[TRACE] Active model = qwen -> call_qwen")
            return "call_qwen"

        if state.get("verbose", False):
            print("[TRACE] Active model = llama -> call_llama")
        return "call_llama"

    # =========================
    # MODEL NODE: Llama
    # =========================
    def call_llama(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering call_llama")
            print(f"[TRACE] user_input = {state['user_input']!r}")

        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"

        raw = llama_llm.invoke(prompt)
        # text = clean_llm_text(raw)

        if state.get("verbose", False):
            print("[TRACE] Llama returned response")

        return {"llm_response": raw}

    # =========================
    # MODEL NODE: Qwen
    # =========================
    def call_qwen(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering call_qwen")
            print(f"[TRACE] user_input = {state['user_input']!r}")

        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"

        raw = qwen_llm.invoke(prompt)
        # text = clean_llm_text(raw)

        if state.get("verbose", False):
            print("[TRACE] Qwen returned response")

        return {"llm_response": raw}

    # =========================
    # NODE: print_response
    # =========================
    def print_response(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering print_response")

        print("\n" + "-" * 50)
        print("LLM Response:")
        print("-" * 50)
        print(state.get("llm_response", ""))

        if state.get("verbose", False):
            print("[TRACE] Returning to get_user_input")

        return {}

    # =========================
    # GRAPH BUILDING
    # =========================
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("call_qwen", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()



# -------------------------
# MAIN
# -------------------------
def main():
    print("=" * 50)
    print("LangGraph Simple Agent (Persistent Switch: Llama <-> Qwen)")
    print("=" * 50)
    print()
    print("Commands:")
    print("  - hey qwen   : switch active model to Qwen (persists)")
    print("  - hey llama  : switch active model to Llama (persists)")
    print("  - verbose    : enable tracing")
    print("  - quiet      : disable tracing")
    print("  - quit/exit/q: exit")
    print()

    # Load both models once (may take time / memory)
    # llama_llm = create_hf_llm
    # qwen_llm = create_qwen()
        # Step 1: Create and configure the LLM
    llama_llm = create_hf_llm(model_id="meta-llama/llama-3.2-1b-instruct")
    qwen_llm= create_hf_llm(model_id="qwen/qwen2.5-0.5b")

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph, filename="Graph_task4_modified.png")

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "is_empty": False,
        "verbose": False,
        "active_model": "llama",  # default
        "llm_response": "",
    }

    # Single invoke; graph loops internally
    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
