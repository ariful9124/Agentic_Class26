"""
Ollama MMLU Evaluation Script (Ollama API, Laptop Friendly)

This script evaluates an Ollama model on the MMLU benchmark using the Ollama API.
Optimized for laptops. No Huggingface configuration or quantization needed.

Usage:
1. Install dependencies: pip install datasets tqdm requests pandas
2. Install Ollama and pull your model locally: https://ollama.com/
   Example:     ollama pull llama3
3. Run: python llama_mmlu_eval_ollama.py

Set OLLAMA_MODEL_NAME below to choose a model (e.g., "llama3", "phi3", "mistral", etc).
"""

import requests
from datasets import load_dataset
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

OLLAMA_MODEL_NAME = "llama3.2:latest"  # Must be present locally in Ollama ("ollama pull llama3" etc)
OLLAMA_HOST = "http://localhost:11434"  # Default Ollama REST host

MAX_NEW_TOKENS = 1

PRINT_QUESTIONS = False  # Set to True to see detailed output for each question

MMLU_SUBJECTS = [
    "machine_learning"
]

def check_environment():
    """Check environment and dependencies"""
    print("="*70)
    print("Environment Check")
    print("="*70)

    print(f"✓ Platform: {platform.system()} ({platform.machine()})")
    if platform.system() == "Darwin":
        print(f"✓ Processor: {platform.processor()}")

    # Check Ollama API is running
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if resp.status_code == 200:
            print(f"✓ Ollama API detected at {OLLAMA_HOST}")
            models = [m["name"] for m in resp.json().get("models",[])]
            if OLLAMA_MODEL_NAME in models:
                print(f"✓ Ollama model '{OLLAMA_MODEL_NAME}' installed.")
            else:
                print(f"⚠️  Ollama model '{OLLAMA_MODEL_NAME}' is NOT pulled - run: ollama pull {OLLAMA_MODEL_NAME}")
                sys.exit(1)
        else:
            print(f"❌ Could not contact Ollama at {OLLAMA_HOST}: Status code {resp.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Could not contact Ollama at {OLLAMA_HOST}: {e}")
        print("Is Ollama running? https://ollama.com/download")
        sys.exit(1)

    print("="*70 + "\n")
    in_colab = False
    device = "ollama"
    return in_colab, device

def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt

def get_ollama_prediction(model_name, prompt, max_tokens=1):
    """
    Call Ollama API for a multiple choice question.
    Returns:
        answer (str): 'A'/'B'/'C'/'D'
        latency_s (float): Time spent in ollama API for this sample
    """
    url = f"{OLLAMA_HOST}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 1.0,
            "stop": ["\n"]
        }
    }
    t0 = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        t1 = time.time()
        latency_s = t1 - t0
        if response.status_code == 200:
            data = response.json()
            generated_text = data["response"]
            answer = generated_text.strip()[:1].upper()
            # Fallback: look for first A-D in output
            if answer not in ["A","B","C","D"]:
                for char in generated_text.upper():
                    if char in ["A","B","C","D"]:
                        answer = char
                        break
                else:
                    answer = "A"
            return answer, latency_s
        else:
            print(f"Ollama API error {response.status_code}: {response.text}")
            return "A", None
    except Exception as e:
        print(f"Error contacting Ollama: {e}")
        return "A", None

def evaluate_subject(model_name, subject, print_examples=False):
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
    api_total_s = 0.0

    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]

        prompt = format_mmlu_prompt(question, choices)

        predicted_answer, latency_s = get_ollama_prediction(model_name, prompt, max_tokens=MAX_NEW_TOKENS)
        if latency_s is not None:
            api_total_s += latency_s

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
            "ollama_latency_s": latency_s,
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
        "ollama_latency_total_s": api_total_s,
    }