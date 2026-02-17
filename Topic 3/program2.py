"""
Llama 3.2-1B MMLU Evaluation Script (Laptop Optimized with Quantization)

This script evaluates Llama 3.2-1B on the MMLU benchmark.
Optimized for laptops with 4-bit or 8-bit quantization to reduce memory usage.

Quantization options:
- 4-bit: ~1.5 GB VRAM/RAM (default for laptop)
- 8-bit: ~2.5 GB VRAM/RAM
- No quantization: ~5 GB VRAM/RAM

Usage:
1. Install: pip install transformers torch datasets accelerate tqdm bitsandbytes
2. Login: huggingface-cli login
3. Run: python llama_mmlu_eval_quantized.py

Set QUANTIZATION_BITS below to choose quantization level.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import json
from tqdm.auto import tqdm
import os
from datetime import datetime
import sys
import platform
import time
import pandas as pd

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

MODEL_NAMES = [
    "meta-llama/Llama-3.2-1B-Instruct"
    # "allenai/OLMo-2-0425-1B",
    # "Qwen/Qwen2.5-0.5B"
]

USE_GPU = True  # Set to False to force CPU-only execution

MAX_NEW_TOKENS = 1

PRINT_QUESTIONS = False  # Set to True to see detailed output for each question

# Quantization settings: 4, 8, None
QUANTIZATION_BITS = None  # Change to 4 or 8 to enable quantization

MMLU_SUBJECTS = [
    "college_computer_science"
]

def detect_device():
    """Detect the best available device (CUDA, MPS, or CPU)"""
    if not USE_GPU:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        is_apple_arm = platform.system() == "Darwin" and platform.processor() == "arm"
        if is_apple_arm:
            if QUANTIZATION_BITS is not None:
                print("\n" + "="*70)
                print("ERROR: Metal and Quantization Conflict")
                print("="*70)
                print("Metal Performance Shaders (MPS) is incompatible with quantization.")
                print(f"You have USE_GPU = True and QUANTIZATION_BITS = {QUANTIZATION_BITS}")
                print("")
                print("Please choose one of the following options:")
                print("  1. Set USE_GPU = False to use CPU with quantization")
                print("  2. Set QUANTIZATION_BITS = None to use Metal without quantization")
                print("="*70 + "\n")
                sys.exit(1)
            return "mps"
    return "cpu"

def check_environment():
    global QUANTIZATION_BITS
    """Check environment and dependencies"""
    print("="*70)
    print("Environment Check")
    print("="*70)

    # Check if in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
        in_colab = True
    except:
        print("✓ Running locally (not in Colab)")
        in_colab = False

    # Check system info
    print(f"✓ Platform: {platform.system()} ({platform.machine()})")
    if platform.system() == "Darwin":
        print(f"✓ Processor: {platform.processor()}")

    # Detect and set device
    device = detect_device()

    # Check device
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("✓ Apple Metal (MPS) Available")
        print("✓ Using Metal Performance Shaders for GPU acceleration")
    else:
        print("⚠️  No GPU detected - running on CPU")

    # Check quantization support
    if QUANTIZATION_BITS is not None:
        try:
            import bitsandbytes
            print(f"✓ bitsandbytes installed - {QUANTIZATION_BITS}-bit quantization available")
        except ImportError:
            print(f"❌ bitsandbytes NOT installed - cannot use quantization")
            sys.exit(1)
        if device == 'mps':
            print(f"❌ Apple METAL is incompatible with quantization")
            print("✓ Quantization disabled - loading full precision model")
            QUANTIZATION_BITS = None
            sys.exit(1)
    else:
        print("✓ Quantization disabled - loading full precision model")
    
    # Check HF authentication
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ Hugging Face authenticated")
        else:
            print("⚠️  No Hugging Face token found")
            print("Run: huggingface-cli login")
    except:
        print("⚠️  Could not check Hugging Face authentication")
    
    # Print configuration
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(f"Device: {device}")
    if QUANTIZATION_BITS is not None:
        print(f"Quantization: {QUANTIZATION_BITS}-bit")
        if QUANTIZATION_BITS == 4:
            print(f"Expected memory: ~1.5 GB")
        elif QUANTIZATION_BITS == 8:
            print(f"Expected memory: ~2.5 GB")
    else:
        print(f"Quantization: None (full precision)")
        if device == "cuda":
            print(f"Expected memory: ~5 GB (FP32)")
        elif device == "mps":
            print(f"Expected memory: ~2.5 GB (FP16)")
        else:
            print(f"Expected memory: ~5 GB (FP32)")
    print(f"Number of subjects: {len(MMLU_SUBJECTS)}")

    print("="*70 + "\n")
    return in_colab, device

def get_quantization_config():
    """Create quantization config based on settings"""
    if QUANTIZATION_BITS is None:
        return None

    if QUANTIZATION_BITS == 4:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("Using 4-bit quantization (NF4 + double quant)")
        print("Memory usage: ~1.5 GB")
    elif QUANTIZATION_BITS == 8:
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        print("Using 8-bit quantization")
        print("Memory usage: ~2.5 GB")
    else:
        raise ValueError(f"Invalid QUANTIZATION_BITS: {QUANTIZATION_BITS}. Use 4, 8, or None")
    return config

def load_model_and_tokenizer(model_name, device):
    """Load model with optional quantization"""
    print(f"\nLoading model {model_name}...")
    print(f"Device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded")

        quant_config = get_quantization_config()

        print("Loading model (this may take 2-3 minutes)...")
        if quant_config is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            elif device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)

        model.eval()
        print("✓ Model loaded successfully!")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
            if quant_config is not None:
                print(f"  Quantization: {QUANTIZATION_BITS}-bit active")
        elif device == "mps":
            print(f"  Running on Apple Metal (MPS)")
        return model, tokenizer

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nPossible causes:")
        print("1. No Hugging Face token - Run: huggingface-cli login")
        print("2. Llama license not accepted - Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        print("3. bitsandbytes not installed - Run: pip install bitsandbytes")
        print("4. Out of memory - Try 4-bit quantization or smaller model")
        raise

def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt

def get_model_prediction(model, tokenizer, prompt):
    """Get model's prediction for multiple-choice question.
    Returns:
        answer (str): 'A'/'B'/'C'/'D'
        gpu_compute_s (float|None): CUDA time spent inside model.generate for this sample
    """
    device = model.device
    use_cuda_timing = (device.type == "cuda") and torch.cuda.is_available()

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    gpu_compute_s = None

    if use_cuda_timing:
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0
        )

    if use_cuda_timing:
        end_evt.record()
        torch.cuda.synchronize()
        gpu_compute_s = start_evt.elapsed_time(end_evt) / 1000.0  # ms -> s

    new_tokens = outputs[0, inputs["input_ids"].shape[1]:].detach().cpu()
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    answer = generated_text.strip()[:1].upper()
    if answer not in ["A", "B", "C", "D"]:
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
        else:
            answer = "A"

    return answer, gpu_compute_s

def evaluate_subject(model, tokenizer, subject, print_examples=False):
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'='*70}")

    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"❌ Error loading subject {subject}: {e}")
        return None

    correct = 0
    total = 0
    question_details = []

    gpu_compute_total_s = 0.0

    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]

        prompt = format_mmlu_prompt(question, choices)

        predicted_answer, gpu_s = get_model_prediction(model, tokenizer, prompt)

        if gpu_s is not None:
            gpu_compute_total_s += gpu_s

        is_correct = predicted_answer == correct_answer

        question_details.append({
            "question": question,
            "choices": choices,
            "choice_a": choices[0] if len(choices) > 0 else "",
            "choice_b": choices[1] if len(choices) > 1 else "",
            "choice_c": choices[2] if len(choices) > 2 else "",
            "choice_d": choices[3] if len(choices) > 3 else "",
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "gpu_compute_s": gpu_s,
        })

        if print_examples:
            print("="*40)
            print("Q:", question)
            for label, choice in zip(["A", "B", "C", "D"], choices):
                print(f"  {label}. {choice}")
            print(f"→ Model answer: {predicted_answer} (Correct: {correct_answer})")
            print("Result:", "RIGHT ✓" if is_correct else "WRONG ✗")

        if is_correct:
            correct += 1
        total += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")

    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "question_details": question_details,
        "gpu_compute_total_s": gpu_compute_total_s,
    }