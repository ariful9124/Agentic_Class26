"""
Text-to-SQL LoRA Fine-Tuning with Tinker

Step-by-step implementation following the sql_finetuning_lesson_plan:
  Step 0  - Setup (packages, API key, dataset)
  Step 1  - Load and explore the data
  Step 2  - Define the prompt format
  Step 3  - Evaluate the base model
  Step 4  - Prepare training data (tokenize + weight masks)
  Step 5  - Train for one epoch
  Step 6  - Evaluate the fine-tuned model
  Step 7  - Test on five novel out-of-distribution schemas
  Step 8  - Discussion prompts

Assumptions:
- Your dataset file is `sql_create_context_v4.json`
- You have access to `tinker` and TINKER_API_KEY is set
- You have `sql_matches.py` available in the same folder
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tinker
from tinker import types
from sql_matches import sql_matches


class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


# =========================
# 0. Config  (Step 0 — Setup)
# =========================
# Before running:
#   pip install tinker transformers python-dotenv
#   export TINKER_API_KEY=your_key_here   (or set in .env)
# =========================
# 1. Config
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(r"C:\Users\afsha\OneDrive\Desktop\Agentic_Classmaterials\sql_create_context_v4.json")
NUM_TEST_EXAMPLES = 200
BASE_MODEL = "meta-llama/Llama-3.2-1B"
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
NUM_EPOCHS = 1


# =========================
# 2. Data helpers
# =========================
def load_data(data_path: Path) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_data(data: list[dict], num_test_examples: int = 200) -> tuple[list[dict], list[dict]]:
    data = data.copy()
    random.shuffle(data)
    test_data = data[:num_test_examples]
    train_data = data[num_test_examples:]
    return train_data, test_data


def format_prompt(example: dict) -> tuple[str, str]:
    prompt = (
        f"Table schema:\n"
        f"{example['context']}\n"
        f"Question: {example['question']}\n"
        f"SQL: "
    )
    completion = example["answer"]
    return prompt, completion


def skim_examples(data: list[dict], n: int = 5) -> None:
    """Step 1 — skim several examples to observe SQL complexity range."""
    print(f"\nSkimming {n} random examples:\n")
    sample = random.sample(data, n)
    for i, ex in enumerate(sample, start=1):
        print(f"--- Example {i} ---")
        print(f"  Question : {ex['question']}")
        print(f"  Schema   : {ex['context'][:100]}{'...' if len(ex['context']) > 100 else ''}")
        print(f"  Answer   : {ex['answer']}")
        print()


# =========================
# 3. Weight-mask explanation
# =========================
def show_weight_mask(example: dict, tokenizer, max_items: int = 80) -> None:
    prompt, completion = format_prompt(example)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_str = f" {completion}\n\n"
    completion_tokens = tokenizer.encode(completion_str, add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

    shifted_input = tokens[:-1]
    shifted_target = tokens[1:]
    shifted_weights = weights[1:]

    print(f"Prompt token count:      {len(prompt_tokens)}")
    print(f"Completion token count:  {len(completion_tokens)}")
    print(f"Total token count:       {len(tokens)}")
    print(f"Shifted sequence length: {len(shifted_input)}")
    print("\nFirst few shifted training positions:\n")

    for i in range(min(max_items, len(shifted_input))):
        inp = tokenizer.decode([shifted_input[i]])
        tgt = tokenizer.decode([shifted_target[i]])
        wt = shifted_weights[i]
        print(f"{i:03d} | input={inp!r:20} -> target={tgt!r:20} | weight={wt}")


# =========================
# 4. Convert to Tinker Datum
# =========================
def process_example(example: dict, tokenizer) -> types.Datum:
    prompt, completion = format_prompt(example)

    # Prompt is visible to the model, but not trained on
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0.0] * len(prompt_tokens)

    # Completion is what the model must learn to generate
    completion_str = f" {completion}\n\n"
    completion_tokens = tokenizer.encode(completion_str, add_special_tokens=False)
    completion_weights = [1.0] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    # Next-token prediction shift
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    shifted_weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": np.array(target_tokens, dtype=np.int64),
            "weights": np.array(shifted_weights, dtype=np.float32),
        },
    )


# =========================
# 5. Sampling / evaluation
# =========================
def sample_from_model(sampling_client, tokenizer, context: str, question: str) -> str:
    prompt = (
        f"Table schema:\n{context}\n"
        f"Question: {question}\n"
        f"SQL: "
    )
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

    params = types.SamplingParams(
        max_tokens=150,
        temperature=0.0,
        stop=["\n\n", "Question:"],
    )

    result = sampling_client.sample(
        prompt=model_input,
        sampling_params=params,
        num_samples=1,
    ).result()

    if result.sequences:
        return tokenizer.decode(result.sequences[0].tokens).strip()
    return ""


def eval_one(sampling_client, tokenizer, ex: dict, verbose: bool = False) -> bool:
    generated_sql = sample_from_model(
        sampling_client,
        tokenizer,
        ex["context"],
        ex["question"],
    )
    match = sql_matches(generated_sql, ex["answer"], schema=ex["context"])

    if verbose:
        print("Question:")
        print(ex["question"])
        print("\nGenerated SQL:")
        print(generated_sql)
        print("\nExpected SQL:")
        print(ex["answer"])
        print(f"\nMatch: {match}")

    return match


def evaluate_test_set(
    sampling_client,
    tokenizer,
    test_data: list[dict],
    limit: int | None = None,
    verbose_every: int | None = None,
) -> float:
    subset = test_data if limit is None else test_data[:limit]
    correct = 0

    for i, ex in enumerate(subset, start=1):
        is_correct = eval_one(
            sampling_client,
            tokenizer,
            ex,
            verbose=(verbose_every is not None and i % verbose_every == 0),
        )
        correct += int(is_correct)

        if i % 25 == 0 or i == len(subset):
            print(f"Evaluated {i}/{len(subset)} examples... current accuracy={correct / i:.2%}")

    return correct / len(subset)


# =========================
# 6. Training
# =========================
def train_one_epoch(training_client, processed_train, batch_size: int, learning_rate: float):
    """
    A simple one-epoch training loop using the Tinker API:
    - forward_backward() accumulates gradients for a batch
    - optim_step() applies the Adam optimizer update
    """
    num_batches = (len(processed_train) + batch_size - 1) // batch_size
    losses = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(processed_train))
        batch = processed_train[start:end]

        # Compute gradients
        fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")

        # Apply optimizer update
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=learning_rate)
        )

        # Wait for results
        fwdbwd_result = fwdbwd_future.result()
        optim_future.result()

        loss = getattr(fwdbwd_result, "loss", None)
        if loss is not None:
            losses.append(loss)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            if losses:
                print(
                    f"Batch {batch_idx + 1}/{num_batches} | "
                    f"recent avg loss = {np.mean(losses[-10:]):.4f}"
                )
            else:
                print(f"Batch {batch_idx + 1}/{num_batches} processed.")

    return losses


# =========================
# 7. Novel Schema Tests  (Step 7)
# =========================
NOVEL_SCHEMA_TESTS = [
    # --- Easy: single table, simple WHERE ---
    {
        "difficulty": "Easy",
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)",
        "question": "What are the names of employees in the engineering department?",
        "answer": "SELECT name FROM employees WHERE department = 'engineering'",
    },
    {
        "difficulty": "Easy",
        "context": "CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)",
        "question": "How many products cost more than 50 dollars?",
        "answer": "SELECT COUNT(*) FROM products WHERE price > 50",
    },
    # --- Medium: aggregation / ORDER BY ---
    {
        "difficulty": "Medium",
        "context": "CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)",
        "question": "What is the highest score in the science class?",
        "answer": "SELECT MAX(score) FROM students WHERE class = 'science'",
    },
    {
        "difficulty": "Medium",
        "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)",
        "question": "List the top 3 customers by total order amount.",
        "answer": "SELECT customer, SUM(amount) FROM orders GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3",
    },
    # --- Hard: JOIN + GROUP BY ---
    {
        "difficulty": "Hard",
        "context": (
            "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); "
            "CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)"
        ),
        "question": "How many students are enrolled in each department?",
        "answer": (
            "SELECT courses.department, COUNT(DISTINCT enrollments.student_id) "
            "FROM courses JOIN enrollments ON courses.id = enrollments.course_id "
            "GROUP BY courses.department"
        ),
    },
]


def test_novel_schemas(sampling_client, tokenizer) -> None:
    """Step 7 — test fine-tuned model on five out-of-distribution schemas."""
    print("\n" + "=" * 60)
    print("STEP 7: Novel Schema Tests (out-of-distribution)")
    print("=" * 60)

    for i, case in enumerate(NOVEL_SCHEMA_TESTS, start=1):
        generated = sample_from_model(
            sampling_client, tokenizer, case["context"], case["question"]
        )
        match = sql_matches(generated, case["answer"], schema=case["context"])

        print(f"\n[{i}] Difficulty : {case['difficulty']}")
        print(f"    Question  : {case['question']}")
        print(f"    Generated : {generated}")
        print(f"    Expected  : {case['answer']}")
        print(f"    Match     : {'PASS' if match else 'FAIL'}")


# =========================
# 8. Main
# =========================
def main():
    # ------------------------------------------------------------------
    # Output logging — tee all print() output to a timestamped text file
    # ------------------------------------------------------------------
    log_path = Path(__file__).parent / f"run_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(log_file)
    print(f"Output also saved to: {log_path}\n")

    # ------------------------------------------------------------------
    # STEP 1 — Load and Explore the Data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Load and Explore the Data")
    print("=" * 60)
    print("\nLoading dataset...")
    data = load_data(DATA_PATH)
    print(f"Total examples: {len(data)}")

    # Skim several examples to observe SQL complexity range
    skim_examples(data, n=5)

    print("Creating train/test split (200 held-out test, rest for training)...")
    train_data, test_data = split_data(data, NUM_TEST_EXAMPLES)
    print(f"Training examples: {len(train_data)}")
    print(f"Test examples:     {len(test_data)}")

    # ------------------------------------------------------------------
    # STEP 2 — Define the Prompt Format
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Prompt Format")
    print("=" * 60)
    ex = train_data[0]
    prompt, completion = format_prompt(ex)
    print("\nTemplate: Table schema / Question / SQL:")
    print("\nPROMPT:")
    print(prompt[:500])
    print("\nCOMPLETION (what the model must predict):")
    print(completion)

    # ------------------------------------------------------------------
    # STEP 3 — Evaluate the Base Model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Evaluate the Base Model")
    print("=" * 60)
    print("\nCreating Tinker LoRA training client...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
    tokenizer = training_client.get_tokenizer()
    print("Tokenizer loaded.")

    print("\nInspecting weight mask (prompt=0, completion=1)...")
    show_weight_mask(ex, tokenizer, max_items=60)

    print("\nEvaluating base model on 200 test questions...")
    base_sampling_client = training_client.save_weights_and_get_sampling_client(
        name="base-model"
    )
    base_accuracy = evaluate_test_set(
        base_sampling_client, tokenizer, test_data, limit=None
    )
    print(
        f"\nBase model accuracy: {base_accuracy:.2%} "
        f"({int(base_accuracy * len(test_data))}/{len(test_data)})"
    )

    # ------------------------------------------------------------------
    # STEP 4 — Prepare Training Data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Prepare Training Data")
    print("=" * 60)
    print("\nTokenizing examples and applying weight masks...")
    processed_train = [process_example(ex, tokenizer) for ex in train_data]
    random.shuffle(processed_train)
    print(f"Prepared and shuffled {len(processed_train)} training examples.")

    # ------------------------------------------------------------------
    # STEP 5 — Train
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Train (1 epoch, batch size 256, lr 5e-4)")
    print("=" * 60)
    all_losses = []
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        losses = train_one_epoch(
            training_client=training_client,
            processed_train=processed_train,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
        )
        all_losses.extend(losses)

    if all_losses:
        print(f"\nFinal mean training loss: {np.mean(all_losses):.4f}")

    # ------------------------------------------------------------------
    # STEP 6 — Evaluate the Fine-Tuned Model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Evaluate the Fine-Tuned Model")
    print("=" * 60)
    print("\nSaving fine-tuned weights and creating sampling client...")
    finetuned_sampling_client = training_client.save_weights_and_get_sampling_client(
        name="text-to-sql-finetuned"
    )
    finetuned_accuracy = evaluate_test_set(
        finetuned_sampling_client, tokenizer, test_data, limit=None
    )
    print(
        f"\nFine-tuned model accuracy: {finetuned_accuracy:.2%} "
        f"({int(finetuned_accuracy * len(test_data))}/{len(test_data)})"
    )
    print(
        f"Improvement over base:     "
        f"{(finetuned_accuracy - base_accuracy):.2%} "
        f"({base_accuracy:.2%} -> {finetuned_accuracy:.2%})"
    )

    # ------------------------------------------------------------------
    # STEP 7 — Test on Novel Out-of-Distribution Schemas
    # ------------------------------------------------------------------
    test_novel_schemas(finetuned_sampling_client, tokenizer)

    # ------------------------------------------------------------------
    # STEP 8 — Discussion
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: Discussion Prompts")
    print("=" * 60)
    print("""
Reflect on these questions after reviewing the results:

1. ACCURACY GAINS
   - What was the base model accuracy vs fine-tuned accuracy?
   - Which types of SQL queries improved most (SELECT, WHERE, GROUP BY, JOIN)?

2. FINE-TUNING vs RAG
   - Why is fine-tuning better suited than RAG for learning SQL syntax?
   - When might RAG outperform fine-tuning (e.g., new schema not in training)?

3. ERROR ANALYSIS — categorize any failures into:
   - Wrong column/table names
   - Syntax errors (malformed SQL)
   - Logic errors (correct syntax, wrong semantics)

4. NOVEL SCHEMAS (Step 7 results)
   - Did accuracy drop for out-of-distribution schemas?
   - Which difficulty level failed first — Easy, Medium, or Hard?
""")

    print("Done.")
    log_file.close()
    sys.stdout = sys.stdout._stdout  # restore original stdout


if __name__ == "__main__":
    main()
