Sure — here’s what **each function does**, in the order it matters, and how they connect.

---

## `get_device()`

**Goal:** pick where the model should run.

* Checks **CUDA** first (`torch.cuda.is_available()`): NVIDIA GPU
* Else checks **MPS** (`torch.backends.mps.is_available()`): Apple Silicon GPU
* Else uses **CPU**
* Prints which one it picked, returns `"cuda"` / `"mps"` / `"cpu"`

Why: you want fastest device automatically.

---

## `create_llm()`

**Goal:** download/load Llama 3.2 1B and wrap it so LangChain can call it.

Steps:

1. Calls `get_device()` to decide device.
2. Sets `model_id = "meta-llama/Llama-3.2-1B-Instruct"`.
3. Loads tokenizer:

   * `AutoTokenizer.from_pretrained(model_id)`
4. Loads model:

   * `AutoModelForCausalLM.from_pretrained(...)`
   * Uses `float16` if not CPU, else `float32`
   * Uses `device_map="cuda"` only if CUDA (so HF places model on GPU automatically)
5. If device is `"mps"`, manually does `model.to("mps")` (since device_map isn’t used there)
6. Builds a `transformers.pipeline("text-generation", ...)` with sampling params:

   * `max_new_tokens=256`
   * `temperature=0.7`, `top_p=0.95`
7. Wraps pipeline into LangChain LLM object:

   * `HuggingFacePipeline(pipeline=pipe)`
8. Returns `llm`

So after this, you can do: `llm.invoke(prompt)` and get text back.

---

## `create_graph(llm)`

**Goal:** build the LangGraph workflow (the loop: input → LLM → print → input…).

This function defines **3 node functions** + **1 routing function**, then wires them into a graph.

### Node 1: `get_user_input(state)`

**Goal:** ask user for text, update state.

* Prints prompt banner
* Reads from keyboard: `user_input = input()`
* If user typed `quit/exit/q`:

  * returns `{"user_input": ..., "should_exit": True}`
* else:

  * returns `{"user_input": ..., "should_exit": False}`

Important: it returns only partial updates; LangGraph merges them into state.

---

### Node 2: `call_llm(state)`

**Goal:** send user input to the LLM and store response in state.

* Reads: `state["user_input"]`
* Formats prompt:

  ```python
  prompt = f"User: {user_input}\nAssistant:"
  ```
* Calls model:

  ```python
  response = llm.invoke(prompt)
  ```
* Returns update:

  ```python
  {"llm_response": response}
  ```

---

### Node 3: `print_response(state)`

**Goal:** print the model output.

* Reads: `state["llm_response"]`
* Prints it
* Returns `{}` (no state update)

---

### Routing: `route_after_input(state)`

**Goal:** decide where to go after `get_user_input`.

* If `state["should_exit"] == True` → return `END`
* Else → return `"call_llm"`

This is what creates the “quit exits, otherwise continue” behavior.

---

### Graph wiring inside `create_graph`

This is the actual control flow:

1. `START -> get_user_input`
2. `get_user_input -> (conditional)`

   * if quit → `END`
   * else → `call_llm`
3. `call_llm -> print_response`
4. `print_response -> get_user_input`  ✅ this makes it loop forever

Finally it does:

```python
graph = graph_builder.compile()
```

and returns the compiled graph.

---

## `save_graph_image(graph, filename="lg_graph.png")`

**Goal:** save a picture of the graph before running it.

* Tries:

  ```python
  png_data = graph.get_graph(xray=True).draw_mermaid_png()
  ```
* Writes bytes to `lg_graph.png`
* If it fails, prints error and suggests:

  ```bash
  pip install grandalf
  ```

So: this function is “nice to have” visualization, not required for the agent to run.

---

## `main()`

**Goal:** orchestrate everything.

Steps:

1. Prints header
2. `llm = create_llm()`
3. `graph = create_graph(llm)`
4. `save_graph_image(graph)`
5. Creates `initial_state`:

   ```python
   {"user_input":"", "should_exit":False, "llm_response":""}
   ```
6. Runs the graph once:

   ```python
   graph.invoke(initial_state)
   ```

Even though it “invokes once”, the graph **loops internally** due to the edge:
`print_response -> get_user_input`.

Graph stops only when routing returns `END`.

---

## `if __name__ == "__main__":`

**Goal:** run `main()` only when you execute the file directly.

So if you import this file from another Python script, it won’t auto-run.

---

### The big picture in one line

This script builds a **3-node LangGraph loop**:

**Input (stdin)** → **LLM call** → **Print** → back to **Input**, until quit.

If you want, I can also explain what `StateGraph` is doing during merging (why returning partial dicts works) in ~5 lines.
