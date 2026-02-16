"""
LangGraph: Shared chat history + switch between Llama and Qwen
Assignment behavior:
- 3 entities: Human, Llama, Qwen
- API roles: system, user/human, assistant/ai, tool
- For TARGET model:
    * TARGET's own prior turns => role "assistant"
    * Human + other model turns => role "user"
  and content includes speaker name:
    "Human: ...", "Llama: ...", "Qwen: ..."

Switch rule:
- If user input starts with "Hey Qwen" -> call Qwen
- Otherwise -> call Llama
"""

import torch
from typing import Annotated, TypedDict, List

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# =========================
# STATE
# =========================
class AgentState(TypedDict):
    # ✅ correct reducer to accumulate chat history
    messages: Annotated[List[AnyMessage], add_messages]

    # control flags
    should_exit: bool
    verbose: bool
    is_empty: bool
    use_qwen: bool


# =========================
# DEVICE SELECTION
# =========================
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


# =========================
# HF LLM LOADER
# =========================
def create_hf_llm(
    model_id: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
):
    device = get_device()
    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Ensure pad token exists
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    # ✅ CRITICAL: return only completion, not prompt+completion
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,  # <-- now configurable
        temperature=temperature,  # <-- now configurable
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print("Model loaded successfully!")
    return llm


def system_prompt_for(target: str) -> str:
    other = "Qwen" if target == "Llama" else "Llama"
    return (
        f"You are {target}, an assistant in a conversation involving Human, {other}, and yourself.\n"
        f"Only write {target}'s single response for THIS turn.\n"
        f"Answer ONLY the latest Human message.\n"
        f"Do NOT summarize or comment on the conversation (no meta text like 'the human suggests' or 'this aligns').\n"
        f"Use at most 2 sentences unless Human asks for more."
    )



# =========================
# HISTORY REMAPPING (core requirement)
# =========================
def remap_history_for_target(messages: List[AnyMessage], target: str) -> List[dict]:
    """
    Build chat history in the format the assignment describes.

    Output is a list of dicts: {role: "user"/"assistant"/"system", content: "..."}
    Rule:
      - target's own prior turns => role "assistant"
      - Human + other model turns => role "user"
    content is always "SpeakerName: <text>"
    """
    out: List[dict] = [{"role": "system", "content": system_prompt_for(target)}]

    for m in messages:
        mtype = getattr(m, "type", None)
        if mtype == "system":
            # We generate system prompt per target, so skip stored system messages
            continue

        speaker = getattr(m, "name", None)
        content = getattr(m, "content", "")

        # Default naming if name missing
        if not speaker:
            if mtype in ("human", "user"):
                speaker = "Human"
            elif mtype in ("ai", "assistant"):
                speaker = "Assistant"
            else:
                speaker = "Tool"

        # Decide role for TARGET
        role = "assistant" if speaker == target else "user"
        out.append({"role": role, "content": f"{speaker}: {content}"})

    return out


def history_dicts_to_text_prompt(history: List[dict], target: str) -> str:
    """
    Convert the remapped list[{role,content}] to a single transcript string for HF text-generation.
    This keeps the assignment semantics but still works with non-chat HF pipeline.
    """
    lines = []
    for item in history:
        r = item["role"]
        c = item["content"]
        if r == "system":
            lines.append(f"System: {c}")
        elif r == "assistant":
            lines.append(f"Assistant: {c}")
        else:
            lines.append(f"User: {c}")

    # Model continues as target
    lines.append(f"Assistant: {target}:")
    return "\n".join(lines)


# =========================
# OUTPUT CLEANING
# =========================
def enforce_single_speaker(reply: str, target: str) -> str:
    """
    If model tries to roleplay others, cut it off at first occurrence of other speaker tags.
    """
    other = "Qwen" if target == "Llama" else "Llama"
    cut_markers = [f"{other}:", "Human:", "User:", "Assistant:"]
    # If model repeats "Llama:" at start, that's okay; but cut if other appears later
    for marker in cut_markers:
        idx = reply.find(marker)
        if idx != -1 and not (marker == f"{target}:" and idx == 0):
            reply = reply[:idx].strip()
            break
    return reply.strip()


def clean_llm_text(raw: str, target: str) -> str:
    text = str(raw).strip()

    # Sometimes the model begins with "Llama:"/"Qwen:"; remove it
    prefix = f"{target}:"
    if text.startswith(prefix):
        text = text[len(prefix):].lstrip()

    # Prevent it from speaking for others
    text = enforce_single_speaker(text, target=target)

    # Prevent fake multi-turn
    if "\nUser:" in text:
        text = text.split("\nUser:", 1)[0].strip()

    return text.strip()


def debug_print_history(history: List[dict], title: str, max_items: int = 6):
    print(f"[TRACE] {title} (showing last {max_items} items):")
    tail = history[-max_items:] if len(history) > max_items else history
    for item in tail:
        print(f"  - {item['role']:9s} | {item['content']}")


# =========================
# NODE 1: GET USER INPUT
# =========================
def get_user_input(state: AgentState) -> dict:
    if state.get("verbose", False):
        print("[TRACE] Entering get_user_input")

    print("\n" + "=" * 50)
    print("Enter your text (or 'quit' to exit):")
    print("=" * 50)

    print("\n> ", end="")
    raw = input().strip()

    if raw.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        return {"should_exit": True, "is_empty": False, "use_qwen": False}

    if raw == "":
        return {"is_empty": True, "should_exit": False, "use_qwen": False}

    if raw.lower() == "verbose":
        print("Verbose mode ENABLED")
        return {"verbose": True, "is_empty": True, "should_exit": False, "use_qwen": False}

    if raw.lower() == "quiet":
        print("Verbose mode DISABLED")
        return {"verbose": False, "is_empty": True, "should_exit": False, "use_qwen": False}

    use_qwen = raw.lower().startswith("hey qwen")

    # Strip prefix only for the message content being asked of Qwen
    content = raw
    if use_qwen:
        content = raw[len("hey qwen"):].strip()
        content = content.lstrip(" ,.:;-")
        if content.strip() == "":
            # user typed just "Hey Qwen" with no question
            return {"is_empty": True, "should_exit": False, "use_qwen": True}

    # Store Human message (speaker name is essential)
    return {
        "messages": [HumanMessage(content=content, name="Human")],
        "should_exit": False,
        "is_empty": False,
        "use_qwen": use_qwen,
    }


# =========================
# GRAPH
# =========================
def create_graph(llama_llm, qwen_llm):
    def route_after_input(state: AgentState) -> str:
        if state.get("verbose", False):
            print("[TRACE] Routing after input")

        if state.get("should_exit", False):
            return END

        if state.get("is_empty", False):
            return "get_user_input"

        msgs = state.get("messages", [])
        if not msgs or getattr(msgs[-1], "type", None) != "human":
            return "get_user_input"

        return "call_qwen" if state.get("use_qwen", False) else "call_llama"

    def call_llama(state: AgentState) -> dict:
        target = "Llama"
        history = remap_history_for_target(state.get("messages", []), target=target)
        # print(history)
        prompt = history_dicts_to_text_prompt(history, target=target)
        # print(prompt)
        if state.get("verbose", False):
            debug_print_history(history, "History passed to LLAMA")

        raw = llama_llm.invoke(prompt)
        reply = clean_llm_text(raw, target=target)

        # Store with name so later remapping works
        return {"messages": [AIMessage(content=reply, name="Llama")]}

    def call_qwen(state: AgentState) -> dict:
        target = "Qwen"
        history = remap_history_for_target(state.get("messages", []), target=target)
        # print("Qwen history:", history)
        prompt = history_dicts_to_text_prompt(history, target=target)
        # print(prompt)
        if state.get("verbose", False):
            debug_print_history(history, "History passed to QWEN")

        raw = qwen_llm.invoke(prompt)
        reply = clean_llm_text(raw, target=target)

        return {"messages": [AIMessage(content=reply, name="Qwen")]}

    def print_response(state: AgentState) -> dict:
        last = state["messages"][-1]
        speaker = getattr(last, "name", "Assistant")
        text = getattr(last, "content", "")

        print("\n" + "-" * 50)
        print(f"{speaker} Response:")
        print("-" * 50)
        print(text)
        return {}

    g = StateGraph(AgentState)
    g.add_node("get_user_input", get_user_input)
    g.add_node("call_llama", call_llama)
    g.add_node("call_qwen", call_qwen)
    g.add_node("print_response", print_response)

    g.add_edge(START, "get_user_input")
    g.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
            "get_user_input": "get_user_input",
            END: END,
        },
    )
    g.add_edge("call_llama", "print_response")
    g.add_edge("call_qwen", "print_response")
    g.add_edge("print_response", "get_user_input")
    return g.compile()


# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("LangGraph Agent: Integrated History + Llama/Qwen Switching")
    print("=" * 60)
    print("Commands: verbose | quiet | quit/exit/q")
    print("To talk to Qwen, start your message with: Hey Qwen ...")
    print()

    llama_llm = create_hf_llm("meta-llama/Llama-3.2-1B-Instruct", temperature=0.7, do_sample=True)
    qwen_llm = create_hf_llm("Qwen/Qwen2.5-0.5B-Instruct", temperature=0.4, do_sample=True)

    graph = create_graph(llama_llm, qwen_llm)

    initial_state: AgentState = {
        "messages": [SystemMessage(content="(placeholder; per-target system prompt is injected)")],
        "should_exit": False,
        "is_empty": False,
        "verbose": False,
        "use_qwen": False,
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
