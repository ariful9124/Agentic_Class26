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
# UPDATED STATE
# -------------------------
class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    is_empty: bool
    verbose: bool

    llama_response: str
    qwen_response: str



def get_device():
    """
    Detect and return the best available compute device.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
        
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"

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
    
    # Exit command
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        return {
            "user_input": user_input,
            "should_exit": True
        }

    # Empty input -> mark as empty and do NOT call LLM
    if user_input == "":
        return {
            "user_input": "",
            "should_exit": False,
            "is_empty": True
        }
    # Turn verbose ON
    if user_input.lower() == "verbose":
        print("Verbose mode ENABLED")
        return {
            "user_input": "",
            "should_exit": False,
            "verbose": True
        }

    # Turn verbose OFF
    if user_input.lower() == "quiet":
        print("Verbose mode DISABLED")
        return {
            "user_input": "",
            "should_exit": False,
            "verbose": False
        }

    return {
        "user_input": user_input,
        "should_exit": False,
        "is_empty": False
    }




def create_graph(llama_llm, qwen_llm):

    # =========================
    # ROUTER: 3-way branch
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
                print("[TRACE] Empty input detected. Routing back to get_user_input.")
            return "get_user_input"

        # Non-empty input -> dispatch to BOTH models
        if state.get("verbose", False):
            print("[TRACE] Routing to dispatch_to_models")
        return "dispatch_to_models"

    # =========================
    # DISPATCH NODE (fan-out)
    # =========================
    def dispatch_to_models(state: AgentState) -> dict:
        # Optional tracing
        if state.get("verbose", False):
            print("[TRACE] Dispatching to llama + qwen in parallel")
        # No state update needed; edges control fan-out
        return {}

    # =========================
    # MODEL NODE A: Llama
    # =========================
    def call_llama(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering call_llama")

        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"

        resp = llama_llm.invoke(prompt)
        return {"llama_response": resp}

    # =========================
    # MODEL NODE B: Qwen
    # =========================
    def call_qwen(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering call_qwen")

        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"

        resp = qwen_llm.invoke(prompt)
        return {"qwen_response": resp}

    # =========================
    # JOIN + PRINT NODE
    # =========================
    def print_both(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering print_both")

        print("\n" + "-" * 60)
        print("Llama Response:")
        print("-" * 60)
        print(state.get("llama_response", ""))

        print("\n" + "-" * 60)
        print("Qwen Response:")
        print("-" * 60)
        print(state.get("qwen_response", ""))

        return {}

    # =========================
    # GRAPH BUILDING
    # =========================
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("dispatch_to_models", dispatch_to_models)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_both", print_both)

    graph_builder.add_edge(START, "get_user_input")

    # 3-way branch after input
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "dispatch_to_models": "dispatch_to_models",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    # Fan-out: dispatcher sends to BOTH model nodes (parallel)
    graph_builder.add_edge("dispatch_to_models", "call_llama")
    graph_builder.add_edge("dispatch_to_models", "call_qwen")

    # Join: both model nodes go to the print node
    graph_builder.add_edge("call_llama", "print_both")
    graph_builder.add_edge("call_qwen", "print_both")

    # Loop back for next user input
    graph_builder.add_edge("print_both", "get_user_input")

    graph = graph_builder.compile()
    return graph

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



def create_graph(llama_llm, qwen_llm):

    # =========================
    # ROUTER: 3-way branch
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
                print("[TRACE] Empty input detected. Routing back to get_user_input.")
            return "get_user_input"

        # Non-empty input -> dispatch to BOTH models
        if state.get("verbose", False):
            print("[TRACE] Routing to dispatch_to_models")
        return "dispatch_to_models"

    # =========================
    # DISPATCH NODE (fan-out)
    # =========================
    def dispatch_to_models(state: AgentState) -> dict:
        # Optional tracing
        if state.get("verbose", False):
            print("[TRACE] Dispatching to llama + qwen in parallel")
        # No state update needed; edges control fan-out
        return {}

    # =========================
    # MODEL NODE A: Llama
    # =========================
    def call_llama(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering call_llama")

        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"

        resp = llama_llm.invoke(prompt)
        return {"llama_response": resp}

    # =========================
    # MODEL NODE B: Qwen
    # =========================
    def call_qwen(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering call_qwen")

        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"

        resp = qwen_llm.invoke(prompt)
        return {"qwen_response": resp}

    # =========================
    # JOIN + PRINT NODE
    # =========================
    def print_both(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering print_both")

        print("\n" + "-" * 60)
        print("Llama Response:")
        print("-" * 60)
        print(state.get("llama_response", ""))

        print("\n" + "-" * 60)
        print("Qwen Response:")
        print("-" * 60)
        print(state.get("qwen_response", ""))

        return {}

    # =========================
    # GRAPH BUILDING
    # =========================
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("dispatch_to_models", dispatch_to_models)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_both", print_both)

    graph_builder.add_edge(START, "get_user_input")

    # 3-way branch after input
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "dispatch_to_models": "dispatch_to_models",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    # Fan-out: dispatcher sends to BOTH model nodes (parallel)
    graph_builder.add_edge("dispatch_to_models", "call_llama")
    graph_builder.add_edge("dispatch_to_models", "call_qwen")

    # Join: both model nodes go to the print node
    graph_builder.add_edge("call_llama", "print_both")
    graph_builder.add_edge("call_qwen", "print_both")

    # Loop back for next user input
    graph_builder.add_edge("print_both", "get_user_input")

    graph = graph_builder.compile()
    return graph

def main():
    """
    Main function that orchestrates the simple agent workflow:
    1. Initialize the LLM
    2. Create the LangGraph
    3. Save the graph visualization
    4. Run the graph once (it loops internally until user quits)

    The graph handles all looping internally through its edge structure:
    - get_user_input: Prompts and reads from stdin
    - call_llm: Processes input through the LLM
    - print_response: Outputs the response, then loops back to get_user_input

    The graph only terminates when the user types 'quit', 'exit', or 'q'.
    """
    print("=" * 50)
    print("LangGraph Simple Agent with Llama and Qwen")
    print("=" * 50)
    print()

    # Step 1: Create and configure the LLM
    llama_llm = create_hf_llm(model_id="meta-llama/llama-3.2-1b-instruct")
    qwen_llm= create_hf_llm(model_id="qwen/qwen2.5-0.5b")

    # Step 2: Build the LangGraph with the LLM
    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")

    # Step 3: Save a visual representation of the graph before execution
    # This happens BEFORE any graph execution, showing the graph structure
    print("\nSaving graph visualization...")
    # save_graph_image(graph, filename="Graph_task3.png")

    # Step 4: Run the graph - it will loop internally until user quits
    # Create initial state with empty/default values
    # The graph will loop continuously, updating state as it goes:
    #   - get_user_input displays banner, populates user_input and should_exit
    #   - call_llm populates llm_response
    #   - print_response displays output, then loops back to get_user_input
    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "llama_response": "",
        "qwen_response":"",
        "is_empty": False,
        "verbose": False
    }

    # Single invocation - the graph loops internally via print_response -> get_user_input
    # The graph only exits when route_after_input returns END (user typed quit/exit/q)
    graph.invoke(initial_state)

# Entry point - only run main() if this script is executed directly
if __name__ == "__main__":
    main()