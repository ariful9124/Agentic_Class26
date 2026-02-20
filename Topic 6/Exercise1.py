# %%
"""
vl_langgraph_agent.py
Exercise 1: Vision-Language LangGraph Chat Agent (Gradio + Ollama LLaVA)
  
What this does
- Upload ONE image
- Multi-turn chat about that image
- LangGraph manages state (image + messages)
- Gradio provides a polished web UI (works locally + notebooks; Colab with share=True)
- Auto-downscales image if large (helps speed)

Prereqs
1) Install Ollama: https://ollama.com
2) Pull a vision model, e.g.:
     ollama pull llava

Python deps
  pip install gradio langgraph pillow ollama

Run
  python vl_langgraph_agent.py

Notes
- If slow: reduce MAX_IMAGE_SIDE (e.g., 512) and/or JPEG_QUALITY (e.g., 70).
"""

from __future__ import annotations

import io
import base64
from typing import TypedDict, List, Dict, Any, Optional, Tuple

import gradio as gr
from PIL import Image

import ollama  # pip install ollama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# -----------------------------
# Configuration
# -----------------------------
OLLAMA_MODEL = "llava"     # change if needed
MAX_IMAGE_SIDE = 768       # try 512 if slow
JPEG_QUALITY = 85          # try 70 if slow / large uploads

# ✅ NEW: Lines to append after every agent response
POST_RESPONSE_LINES = "\n\n---\n"


# -----------------------------
# Helpers
# -----------------------------
def downscale_image(img: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Downscale so max(width,height) <= max_side, preserving aspect ratio."""
    img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.BICUBIC)


def image_to_b64_jpeg(img: Image.Image, quality: int = JPEG_QUALITY) -> str:
    """Encode PIL image as base64 JPEG (for Ollama vision models)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def msg_role_content(m: Any) -> Tuple[str, str]:
    """
    Accepts either:
      - dict like {"role":"user","content":"..."}
      - LangChain message objects: SystemMessage / HumanMessage / AIMessage / ToolMessage
    Returns (role, content) using OpenAI-ish roles: system|user|assistant|tool
    """
    # dict case
    if isinstance(m, dict):
        role = m.get("role", "") or ""
        content = m.get("content", "") or ""
        return role, content

    # object case
    # sometimes .role exists
    role = getattr(m, "role", None)
    if role:
        content = getattr(m, "content", "") or ""
        return str(role), str(content)

    # LangChain BaseMessage uses .type (system/human/ai/tool) and .content
    mtype = getattr(m, "type", "") or ""
    content = getattr(m, "content", "") or ""

    type_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    role = type_map.get(str(mtype), str(mtype) if mtype else "assistant")
    return role, str(content)


def to_gradio_chat(messages: List[Any]):
    """
    Return Gradio Chatbot history in "messages format" using ChatMessage objects.
    This is the most compatible across Gradio versions when type="messages".
    """
    chat = []
    for m in messages:
        role, content = msg_role_content(m)

        # Normalize role
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"

        # Only show user/assistant turns in the UI
        if role not in ("user", "assistant"):
            continue

        chat.append(gr.ChatMessage(role=role, content=str(content)))
    return chat


def merge_state(prev: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge LangGraph node output into persistent UI state.
    Special case: messages are appended using add_messages reducer.
    """
    out: Dict[str, Any] = dict(prev)

    if "messages" in delta:
        out["messages"] = add_messages(out.get("messages", []), delta["messages"])

    for k, v in delta.items():
        if k == "messages":
            continue
        out[k] = v

    return out


# -----------------------------
# LangGraph State
# -----------------------------
class AgentState(TypedDict, total=False):
    # image payload
    image_b64: str
    has_image: bool

    # conversation (mixed dict / LC message objects)
    messages: List[Any]

    # controls
    user_text: str
    should_reset: bool

    # output
    assistant_text: str


SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. "
    "Answer based ONLY on the uploaded image and the conversation context. "
    "If something is not visible, say so."
)


# -----------------------------
# Graph Nodes
# -----------------------------
def node_set_image(state: AgentState) -> Dict[str, Any]:
    """Ensure an image exists before chatting."""
    if not state.get("image_b64"):
        msg = "Please upload an image first, then ask a question about it."
        return {
            "has_image": False,
            "assistant_text": msg,
            "messages": [{"role": "assistant", "content": msg}],
        }
    return {"has_image": True}


def node_maybe_reset(state: AgentState) -> Dict[str, Any]:
    """Reset chat history (keep image). Also seeds system message on first run."""
    if state.get("should_reset", False):
        return {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "assistant_text": "",
            "user_text": "",
            "should_reset": False,
        }

    if not state.get("messages"):
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}]}

    return {}


def node_add_user_message(state: AgentState) -> Dict[str, Any]:
    """Append user message."""
    txt = (state.get("user_text") or "").strip()
    if not txt:
        return {"assistant_text": "Type a question to continue."}
    return {"messages": [{"role": "user", "content": txt}]}


def node_call_vlm(state: AgentState) -> Dict[str, Any]:
    """Call Ollama vision model with full conversation; attach image to latest user turn."""
    if not state.get("has_image", False):
        return {"assistant_text": state.get("assistant_text", "Please upload an image first.")}

    messages = state.get("messages", [])
    if not messages:
        return {"assistant_text": "Internal error: missing messages."}

    img_b64 = state["image_b64"]

    # Find last user message index
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        role, _ = msg_role_content(messages[i])
        if role == "user":
            last_user_idx = i
            break

    # Build Ollama messages as dicts
    ollama_msgs: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        role, content = msg_role_content(m)
        om: Dict[str, Any] = {"role": role, "content": content}
        if i == last_user_idx:
            om["images"] = [img_b64]
        ollama_msgs.append(om)

    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=ollama_msgs)
        assistant_text = resp["message"]["content"]

        # ✅ NEW: append lines after response
        assistant_text = assistant_text.rstrip() + POST_RESPONSE_LINES

    except Exception as e:
        assistant_text = f"Error calling Ollama model '{OLLAMA_MODEL}': {e}"

    return {
        "assistant_text": assistant_text,
        "messages": [{"role": "assistant", "content": assistant_text}],
        "user_text": "",
    }


def route_after_image_check(state: AgentState) -> str:
    if not state.get("has_image", False):
        return END
    return "maybe_reset"


def route_after_maybe_reset(state: AgentState) -> str:
    txt = (state.get("user_text") or "").strip()
    if not txt:
        return END
    return "add_user"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("set_image", node_set_image)
    g.add_node("maybe_reset", node_maybe_reset)
    g.add_node("add_user", node_add_user_message)
    g.add_node("call_vlm", node_call_vlm)

    g.add_edge(START, "set_image")

    g.add_conditional_edges(
        "set_image",
        route_after_image_check,
        {"maybe_reset": "maybe_reset", END: END},
    )

    g.add_conditional_edges(
        "maybe_reset",
        route_after_maybe_reset,
        {"add_user": "add_user", END: END},
    )

    g.add_edge("add_user", "call_vlm")
    g.add_edge("call_vlm", END)

    return g.compile()


GRAPH = build_graph()


# -----------------------------
# Gradio callbacks
# -----------------------------
def on_image_change(img: Image.Image, st: Optional[Dict[str, Any]]):
    st = st or {}

    if img is None:
        st.pop("image_b64", None)
        st["has_image"] = False
        return st, to_gradio_chat(st.get("messages", [])), "Image cleared."

    img_small = downscale_image(img, MAX_IMAGE_SIDE)
    st["image_b64"] = image_to_b64_jpeg(img_small, JPEG_QUALITY)
    st["has_image"] = True

    # Seed system message if needed
    if not st.get("messages"):
        st["should_reset"] = True
        delta = GRAPH.invoke(st)
        st = merge_state(st, delta)

    return st, to_gradio_chat(st.get("messages", [])), "Image loaded. Ask a question!"


def on_reset(st: Optional[Dict[str, Any]]):
    st = st or {}
    st["should_reset"] = True
    delta = GRAPH.invoke(st)
    st = merge_state(st, delta)
    return st, to_gradio_chat(st.get("messages", [])), "Chat reset."


def on_submit(user_text: str, st: Optional[Dict[str, Any]]):
    st = st or {}
    st["user_text"] = user_text
    st["should_reset"] = False

    delta = GRAPH.invoke(st)
    st = merge_state(st, delta)

    if not st.get("has_image", False):
        status = "Upload an image first."
    elif st.get("assistant_text"):
        status = "Answered."
    else:
        status = "Ready."

    return "", st, to_gradio_chat(st.get("messages", [])), status


# -----------------------------
# Gradio UI
# -----------------------------
def build_ui():
    with gr.Blocks(title="Vision-Language LangGraph Agent (Ollama)") as demo:
        gr.Markdown(
            """
# 🖼️ Vision-Language LangGraph Chat Agent (Ollama + Gradio)

1) Upload an image  
2) Ask multi-turn questions about the image  
3) State + context are managed by **LangGraph**

**Speed tips:** lower `MAX_IMAGE_SIDE` (e.g., 512) and/or `JPEG_QUALITY` (e.g., 70).
            """.strip()
        )

        st = gr.State({})  # persistent state (AgentState)

        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload image")
            chatbot = gr.Chatbot(label="Chat", height=420)

        status = gr.Textbox(label="Status", value="Upload an image to begin.", interactive=False)

        with gr.Row():
            txt = gr.Textbox(label="Your message", placeholder="Ask something about the image...", scale=5)
            send = gr.Button("Send", scale=1)
            reset = gr.Button("Reset Chat", scale=1)

        img_in.change(fn=on_image_change, inputs=[img_in, st], outputs=[st, chatbot, status])
        reset.click(fn=on_reset, inputs=[st], outputs=[st, chatbot, status])

        send.click(fn=on_submit, inputs=[txt, st], outputs=[txt, st, chatbot, status])
        txt.submit(fn=on_submit, inputs=[txt, st], outputs=[txt, st, chatbot, status])

    return demo


if __name__ == "__main__":
    app = build_ui()
    # Local: http://127.0.0.1:7860
    # Colab: app.launch(share=True)
    # Robust port selection: use requested port if possible, but fall back on auto if in use
    preferred_port = 7860
    server_name = "127.0.0.1"
    share = False  # Change True if in Colab

    def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    try:
        if not is_port_in_use(preferred_port, server_name):
            app.launch(server_name=server_name, server_port=preferred_port, share=share)
        else:
            print(f"Port {preferred_port} is in use. Launching on a random open port (set by Gradio).")
            app.launch(server_name=server_name, server_port=None, share=share)
    except OSError as e:
        print(f"Gradio launch error: {e}")
        print("Trying to launch on a random port.")
        app.launch(server_name=server_name, server_port=None, share=share)



