"""
LangGraph: Shared chat history + switch between Llama and Qwen
+ ✅ SQLite checkpointing for crash recovery (resume after restart)

How crash recovery works here:
- We compile the graph with a SqliteSaver checkpointer.
- LangGraph automatically checkpoints state AFTER EVERY NODE completes.
- On restart, we call graph.get_state(config) using the same thread_id.
  - If state.next exists => resume with graph.invoke(None, config=config)
  - Else => start new with graph.invoke(initial_state, config=config)

Notes:
- LLM objects are NOT checkpointed (they are recreated on every run).
- If you kill the process while it's inside input() (before pressing Enter),
  that typed text isn't in state yet, so it can't be recovered.
"""

import os
import argparse
import torch
from typing import Annotated, TypedDict, List

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ✅ Checkpointer (your install exposes SqliteSaver, not SQLiteSaver)
from langgraph.checkpoint.sqlite import SqliteSaver


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

    # ✅ return only completion, not prompt+completion
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
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
        f"Do NOT summarize or comment on the conversation.\n"
        f"Use at most 2 sentences unless Human asks for more."
    )


# =========================
# HISTORY REMAPPING
# =========================
def remap_history_for_target(messages: List[AnyMessage], target: str) -> List[dict]:
    """
    Build chat history in the assignment format.

    Output list of dicts: {role: "user"/"assistant"/"system", content: "..."}
      - target's own prior turns => role "assistant"
      - Human + other model turns => role "user"
    content always "SpeakerName: <text>"
    """
    out: List[dict] = [{"role": "system", "content": system_prompt_for(target)}]

    for m in messages:
        mtype = getattr(m, "type", None)
        if mtype == "system":
            # We generate system prompt per target
            continue

        speaker = getattr(m, "name", None)
        content = getattr(m, "content", "")

        if not speaker:
            if mtype in ("human", "user"):
                speaker = "Human"
            elif mtype in ("ai", "assistant"):
                speaker = "Assistant"
            else:
                speaker = "Tool"

        role = "assistant" if speaker == target else "user"
        out.append({"role": role, "content": f"{speaker}: {content}"})

    return out


def history_dicts_to_text_prompt(history: List[dict], target: str) -> str:
    """
    Convert remapped list[{role,content}] to a single transcript string for HF text-generation.
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

    lines.append(f"Assistant: {target}:")
    return "\n".join(lines)


# =========================
# OUTPUT CLEANING
# =========================
def enforce_single_speaker(reply: str, target: str) -> str:
    other = "Qwen" if target == "Llama" else "Llama"
    cut_markers = [f"{other}:", "Human:", "User:", "Assistant:"]
    for marker in cut_markers:
        idx = reply.find(marker)
        if idx != -1 and not (marker == f"{target}:" and idx == 0):
            reply = reply[:idx].strip()
            break
    return reply.strip()


def clean_llm_text(raw: str, target: str) -> str:
    text = str(raw).strip()

    prefix = f"{target}:"
    if text.startswith(prefix):
        text = text[len(prefix):].lstrip()

    text = enforce_single_speaker(text, target=target)

    if "\nUser:" in text:
        text = text.split("\nUser:", 1)[0].strip()

    return text.strip()


def debug_print_history(history: List[dict], title: str, max_items: int = 6):
    print(f"[TRACE] {title} (showing last {max_items} items):")
    tail = history[-max_items:] if len(history) > max_items else history
    for item in tail:
        # avoid huge console spam; show up to 220 chars
        snippet = item["content"]
        if len(snippet) > 220:
            snippet = snippet[:220] + " ..."
        print(f"  - {item['role']:9s} | {snippet}")


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

    content = raw
    if use_qwen:
        content = raw[len("hey qwen"):].strip()
        content = content.lstrip(" ,.:;-")
        if content.strip() == "":
            return {"is_empty": True, "should_exit": False, "use_qwen": True}

    return {
        "messages": [HumanMessage(content=content, name="Human")],
        "should_exit": False,
        "is_empty": False,
        "use_qwen": use_qwen,
    }


# =========================
# GRAPH (compiled with checkpointer)
# =========================
def create_graph(llama_llm, qwen_llm, checkpointer: SqliteSaver):
    def route_after_input(state: AgentState) -> str:
        if state.get("verbose", False):
            print("[TRACE] Routing after input")

        if state.get("should_exit", False):
            return END

        if state.get("is_empty", False):
            return "get_user_input"

        # Guard: only call LLM if we just got a HumanMessage
        msgs = state.get("messages", [])
        if not msgs or getattr(msgs[-1], "type", None) != "human":
            return "get_user_input"

        return "call_qwen" if state.get("use_qwen", False) else "call_llama"

    def call_llama(state: AgentState) -> dict:
        target = "Llama"
        history = remap_history_for_target(state.get("messages", []), target=target)
        prompt = history_dicts_to_text_prompt(history, target=target)

        if state.get("verbose", False):
            debug_print_history(history, "History passed to LLAMA")

        raw = llama_llm.invoke(prompt)
        reply = clean_llm_text(raw, target=target)
        return {"messages": [AIMessage(content=reply, name="Llama")]}

    def call_qwen(state: AgentState) -> dict:
        target = "Qwen"
        history = remap_history_for_target(state.get("messages", []), target=target)
        prompt = history_dicts_to_text_prompt(history, target=target)

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

    # ✅ compile with checkpointing
    # ---- FIX START: Use context manager for checkpointer ----
    # If SqliteSaver.from_conn_string() returns a context manager,
    # we must use it with "with ... as checkpointer:"
    # To keep the call signature clean for caller, move context management out of here!
    return g.compile(checkpointer=checkpointer)
    # ---- FIX END ----


# =========================
# MAIN (resume / start)
# =========================
def main():
    # Create an ArgumentParser to handle command line arguments (and show --help usage)
    parser = argparse.ArgumentParser(add_help=True)

    # Add --thread argument: Used to specify a conversation thread_id.
    # If not provided, tries LANGGRAPH_THREAD_ID env var, otherwise defaults to 'chat-1'.
    # Using the same thread_id allows the user to resume previous chat sessions (state is checkpointed).
    parser.add_argument(
        "--thread",
        default=os.environ.get("LANGGRAPH_THREAD_ID", "chat-1"),
        help="Conversation thread_id (same id => resume from checkpoints)",
    )

    # Add --db argument: Used to specify the filename for the SQLite checkpoint database.
    # If not provided, tries LANGGRAPH_CHECKPOINT_DB env var, otherwise defaults to 'chat_checkpoints.db'.
    # All checkpoints of chat sessions are stored in this DB file.
    parser.add_argument(
        "--db",
        default=os.environ.get("LANGGRAPH_CHECKPOINT_DB", "task7_chat_checkpoints.db"),
        help="SQLite checkpoint DB file",
    )

    # NOTE: In some environments (such as Jupyter), extra args may be injected (e.g., --f=...json).
    # We use parse_known_args() to ignore unknown arguments and only extract those we care about.
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("LangGraph Agent: Resumable Multi-Model Agent (Llama/Qwen) with Crash Recovery")
    print("=" * 60)
    print("Commands: verbose | quiet | quit/exit/q")
    print("To talk to Qwen, start your message with: Hey Qwen ...")
    print(f"Checkpoint DB: {args.db}")
    print(f"Thread ID: {args.thread}")
    print()

    # Recreate LLMs every run (NOT checkpointed)
    llama_llm = create_hf_llm("meta-llama/Llama-3.2-1B-Instruct", temperature=0.7, do_sample=True)
    qwen_llm = create_hf_llm("Qwen/Qwen2.5-0.5B-Instruct", temperature=0.4, do_sample=True)

    # ✅ persistent checkpointer
    # ------ FIX START: Use context manager for SqliteSaver ------
    with SqliteSaver.from_conn_string(args.db) as checkpointer:
        graph = create_graph(llama_llm, qwen_llm, checkpointer=checkpointer)

        config = {"configurable": {"thread_id": args.thread}}

        # Detect incomplete run and resume
        current = graph.get_state(config)
        if getattr(current, "next", None):
            print("\n🔄 Found saved state. Resuming from last checkpoint...")
            if current.values and current.values.get("verbose", False):
                print(f"[TRACE] Next nodes: {current.next}")
            # Resume from where we left off
            graph.invoke(None, config=config)
        else:
            print("\n▶️  Starting a new conversation...")
            initial_state: AgentState = {
                "messages": [SystemMessage(content="(placeholder; per-target system prompt is injected)")],
                "should_exit": False,
                "is_empty": False,
                "verbose": False,
                "use_qwen": False,
            }
            graph.invoke(initial_state, config=config)
    # ------ FIX END ------


if __name__ == "__main__":
    main()
