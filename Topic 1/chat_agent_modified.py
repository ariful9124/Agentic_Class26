# %% [markdown]
# # Library

# %%
# The waved yellow underlines in Python editors (such as VSCode, Jupyter, PyCharm) typically indicate warnings from linters or static analyzers,
# such as unused imports, unresolved references, or style issues.
# While you can't remove them with code alone, you can address them by removing unused imports, correct errors, or configure your linter to ignore certain warnings.
# Below is a cleaned version with only commonly-used imports retained. 
# Adjust as necessary based on the actual usage in your notebook!

import json
import time
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm.auto import tqdm
import os
import sys
import platform
import pandas as pd
from utils.contx_logger import ContextMetricsLogger
from utils.contx_management import hierarchical_manage

# %% [markdown]
# # Configurations 

# %%
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HISTORY_ON = False

# %% [markdown]
# ## System Prompts

# %%
# System prompt - This sets the chatbot's behavior and personality
# Change this to customize how the chatbot responds
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."


# %% [markdown]
# # Load Model 

# %%
# ============================================================================
# LOAD MODEL (NO QUANTIZATION)
# ============================================================================

print("Loading model (this takes 1-2 minutes)...")

# Load tokenizer (converts text to numbers and vice versa)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in half precision (float16) for efficiency
# Use float16 on GPU, or float32 on CPU if needed
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,                        # Use FP16 for efficiency
    device_map="auto",                          # Automatically choose GPU/CPU
    low_cpu_mem_usage=True
)

model.eval()  # Set to evaluation mode (no training)
print(f"âœ“ Model loaded! Using device: {model.device}")
# print(f"âœ“ Memory usage: ~2.5 GB (FP16)\n")

# %% [markdown]
# # Chat History

# %%
metrics_logger = ContextMetricsLogger()

# %%
# ========================================================================
# CHAT LOOP (with hierarchical/hybrid context management)
# ========================================================================

chat_history = []
chat_history.append({"role": "system", "content": SYSTEM_PROMPT})

print("="*70)
print("Chat started! Type 'quit' or 'exit' to end the conversation.")
print("="*70 + "\n")

while True:

    user_input = input("You: ").strip()

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    # --------------------------------------------------------------------
    # NEW: Build the "working_history" to feed into the model
    # - If HISTORY_ON: use hierarchical_manage (L0/L1/L2)
    # - If HISTORY_OFF: only system + current user message
    # --------------------------------------------------------------------
    if HISTORY_ON:
        system_msg = chat_history[0]  # always preserve system prompt
        working_history, summary_l1, summary_l2, raw_buffer, BUDGET = hierarchical_manage(system_msg, user_input, model, tokenizer)
    else:
        working_history = [chat_history[0], {"role": "user", "content": user_input}]

    # --------------------------------------------------------------------
    # Tokenize working_history (NOT chat_history)
    # --------------------------------------------------------------------
    encoded = tokenizer.apply_chat_template(
        working_history,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    print("Assistant: ", end="", flush=True)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][input_ids.shape[1]:]
    assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(assistant_response)
    print()
    
    
    # Collect metrics
    latency = time.time() - t0
    input_tokens = input_ids.shape[1]
    output_tokens = new_tokens.numel()
    # Estimate L1/L2 token sizes
    l1_tokens = len(tokenizer.encode(summary_l1)) if summary_l1 else 0
    l2_tokens = len(tokenizer.encode(summary_l2)) if summary_l2 else 0
    # Detect summarization trigger (simple heuristic)
    summarization_triggered = (l1_tokens > 0 or l2_tokens > 0)

    metrics_logger.log(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_budget=BUDGET,
        summarization_triggered=summarization_triggered,
        l1_tokens=l1_tokens,
        l2_tokens=l2_tokens,
        raw_buffer_len=len(raw_buffer),
        latency_seconds=latency
    )
    # --------------------------------------------------------------------
    # NEW: Store conversation
    # - chat_history can still store full log (optional)
    # - raw_buffer is what hierarchical_manage uses as "recent verbatim memory"
    # --------------------------------------------------------------------
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": assistant_response})

    if HISTORY_ON:
        raw_buffer.append({"role": "user", "content": user_input})
        raw_buffer.append({"role": "assistant", "content": assistant_response})

        # Optional: keep raw_buffer from growing too large even before summarization triggers
        if len(raw_buffer) > 40:  # keep last 40 messages (~20 turns)
            raw_buffer = raw_buffer[-40:]




