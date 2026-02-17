# ⏱ Task 1 — Sequential vs Parallel Execution

All timings were measured using `time` in Git Bash (Windows, RTX 3060 GPU).

---

## 🔹 Hugging Face Models
### Sequential

```bash
time bash -c 'python "Topic 3/program1.py"; python "Topic 3/program2.py"'
```
```
real    0m11.439s
```

### Parallel

```bash
time { python "Topic 3/program1.py" & python "Topic 3/program2.py" & wait; }
```
```
real    0m6.097s
```

---

## 🦙 Using Ollama

### Sequential

```bash
time bash -c 'python "Topic 3/program1_ollama.py"; python "Topic 3/program2_ollama.py"'
```

```
real    0m6.117s
```

### Parallel

```bash
time { python "Topic 3/program1_ollama.py" & python "Topic 3/program2_ollama.py" & wait; }
```

```
real    0m3.265s
```

---

## 📊 Summary

| Setup     | Sequential | Parallel |
| --------- | ---------- | -------- |
| HF Models | 11.439s    | 6.097s   |
| Ollama    | 6.117s     | 3.265s   |

---
## 🔎 Key Findings
* Parallel execution nearly halves runtime.
* Ollama significantly reduces execution time compared to direct HF loading.
* Fastest configuration: **Ollama (Parallel) — 3.265s**.
---

# ⏱ Task 2 
---
## Subsection 1, 2, and 3 done. 
---

# ⏱ Task 3
---
* Tool-Based LLM Agent: Implemented function-calling to retrieve city coordinates and compute distances instead of generating fabricated data.
* Multi-Step Reasoning Loop: Enabled iterative tool usage (lookup → compute → respond) to support structured problem solving.
* Accurate I-64 Distance Computation: Integrated a predefined route dataset with the Haversine formula and handled invalid city queries robustly.
---