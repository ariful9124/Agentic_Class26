import torch  # added for summarize_text
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------- Hierarchical memory state ----------
# 
# This section maintains the multi-level buffer structure for managing growing chat history:
# - raw_buffer: newest, unsummarized recent messages (the 'working' buffer)
# - summary_l1: an intermediate summary, updated by summarizing oldest raw chunks
# - summary_l2: long-term, highly compressed summary, updated by summarizing previous L1 and L2
#
raw_buffer = []     # L0: freshest segment of messages (not summarized yet)
summary_l1 = ""     # L1: summary of older messages, keeping more information than L2
summary_l2 = ""     # L2: most compressed, long-term (last-resort) summary



def token_len(messages, tokenizer):
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    return enc["input_ids"].shape[1]


def summarize_text(text, model, tokenizer, max_tokens=180):
    """
    Summarizes the given text into a compressed summary under the specified max_tokens.
    Uses the model and tokenizer in deterministic (do_sample=False) mode for repeatable output.

    Args:
        text (str): The text to summarize.
        max_tokens (int): Maximum tokens to try to fit the summary within.
        
    Returns:
        str: A succinct summary covering essential facts, decisions, constraints, and open questions.
    """
    # Prepare a chat-style prompt to guide the model: system for summarization, user with instruction and raw text.
    msgs = [
        {"role": "system", "content": "You compress chat history faithfully."},
        {"role": "user", "content": f"Summarize the following in <= {max_tokens} tokens. Keep only key facts, decisions, constraints, and open questions:\n\n{text}"}
    ]
    # Convert the message list to model input tokens (apply template; add gen prompt).
    enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # Inference-time generation, never training or updating weights.
    with torch.no_grad():
        out = model.generate(
            enc,
            max_new_tokens=max_tokens+30,   # Some margin to avoid abrupt cutoff
            do_sample=False,                # Deterministic output each time
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    # Strip prompt tokens; decode only the newly generated summary
    return tokenizer.decode(out[0][enc.shape[1]:], skip_special_tokens=True).strip()

def msgs_to_text(msgs):
    """
    Converts a sequence of chat messages to a simple plain-text transcript.
    
    Args:
        msgs (list of dict): Each message has "role" (user/assistant) and "content".
    
    Returns:
        str: Transcript with "ROLE: content" lines.
    """
    # Format: USER: message text, ASSISTANT: reply text, etc.
    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs])



def build_working_history(system_msg, summary_l2, summary_l1, raw_buffer, current_user):
    """
    Assemble the current "working" chat history for input to the model, layering memory as:
      - system prompt
      - (optional) long-term summary (L2)
      - (optional) intermediate summary (L1)
      - unsummarized recent buffer
      - current user request

    Args:
        system_msg (dict): The persistent system message/prompt.
        summary_l2 (str): Highly compressed long-term summary (can be blank).
        summary_l1 (str): Intermediate summary (older, not yet in L2; can be blank).
        raw_buffer (list): Unsummarized recent messages.
        current_user (str): The latest user message, not yet appended to buffer.

    Returns:
        list: Structured chat message history to supply to the model.
    """
    working = [system_msg]  # Always begin with system prompt/special directive.
    if summary_l2.strip():
        # Add L2 summary if available, labeled for clarity.
        working.append({"role": "user", "content": f"[Long-term memory]\n{summary_l2}"})
    if summary_l1.strip():
        # Add L1 summary if available.
        working.append({"role": "user", "content": f"[Earlier summary]\n{summary_l1}"})
    # Append the recent unsummarized chat log.
    working.extend(raw_buffer)
    # Add the current, not-yet-processed user input.
    working.append({"role": "user", "content": current_user})
    return working

def hierarchical_manage(system_msg, current_user, model, tokenizer):
    """
    Main context management for hierarchical memory:
      - Summarizes old chat history as it grows (in-place, gradual and layered/“hierarchical”)
      - Keeps as much recent detail as fits in budget (raw_buffer)
      - Pushes older content to more and more compressed memory (L1, then L2)
      - As a final safety, drops the oldest raw messages if still too long

    Args:
        system_msg (dict): System prompt for the session.
        current_user (str): Most recent user turn (not yet committed to buffer).

    Returns:
        list: The best chat history sequence that fits in available model context.
    """
    # Constants grouped at the top of the function
    CHUNK = 4                 # Number of messages to summarize at once (tradeoff: detail vs. cost)
    L1_MAX_TOKENS = 200       # Max tokens for intermediate summary (L1)
    L2_MAX_TOKENS = 160       # Max tokens for long-term summary (L2)
    RAW_DROP = 2              # Number of oldest unsummarized messages to drop if still over budget

    # MAX_CTX: context length the model can process; fallback 4096 if not specified
    # MAX_CTX = getattr(model.config, "max_position_embeddings", 4096)
    MAX_CTX = 1024
    MAX_NEW = 256       # Tokens reserved for new completions (assistant's next reply)
    BUDGET = MAX_CTX - MAX_NEW    # Context "budget" for prompt/history/custom content
    global raw_buffer, summary_l1, summary_l2

    # Step 0: Assemble a candidate history/input sequence with all memory layers and new user input.
    working = build_working_history(system_msg, summary_l2, summary_l1, raw_buffer, current_user)

    # Step 1: If history is too long and raw_buffer is large, gradually summarize old chunks into L1.
    while token_len(working, tokenizer) > BUDGET and len(raw_buffer) > CHUNK:
        # Take out the oldest CHUNK messages to compress them
        old_chunk = raw_buffer[:CHUNK]
        raw_buffer = raw_buffer[CHUNK:]   # Advance the buffer by removing oldest chunk

        # Convert the messages chunk to plain text for summarization
        chunk_text = msgs_to_text(old_chunk)
        # Merge chunk into previous L1 summary (if any) before summarizing
        merged = (summary_l1 + "\n" + chunk_text).strip() if summary_l1 else chunk_text
        # Recompute the intermediate summary (L1) using the model
        summary_l1 = summarize_text(merged, model, tokenizer, max_tokens=L1_MAX_TOKENS)

        # Rebuild candidate input with new summary and trimmed buffer
        working = build_working_history(system_msg, summary_l2, summary_l1, raw_buffer, current_user)

    # Step 2: If history is still too long, compress L1 (and previous L2) into a new L2 summary
    # This is a rare/final-resort step; L2 should be terse so we use lower token limit
    if token_len(working, tokenizer) > BUDGET and summary_l1.strip():
        # Merge previous L2 (if any) and current L1 for summary input
        merged_l2 = (summary_l2 + "\n" + summary_l1).strip() if summary_l2 else summary_l1
        summary_l2 = summarize_text(merged_l2, model, tokenizer, max_tokens=L2_MAX_TOKENS)
        summary_l1 = ""  # After absorption, L1 can be cleared
        # Recalculate the current working input after L2 update
        working = build_working_history(system_msg, summary_l2, summary_l1, raw_buffer, current_user)

    # Step 3: If still too long, drop oldest RAW_DROP messages in raw_buffer until it fits (to guarantee bounded size)
    # This is the last defense—oldest detail is lost if summarying wasn't enough
    while token_len(working, tokenizer) > BUDGET and len(raw_buffer) > RAW_DROP:
        raw_buffer = raw_buffer[RAW_DROP:]  # Discard oldest unsummarized messages
        working = build_working_history(system_msg, summary_l2, summary_l1, raw_buffer, current_user)

    # Return final, budget-constrained working history for use as prompt to model
    return working, summary_l1, summary_l2, raw_buffer, BUDGET
