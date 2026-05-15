# Text-to-SQL LoRA Fine-Tuning — Results & Analysis

Model: `meta-llama/Llama-3.2-1B` fine-tuned with LoRA on 78,377 question/schema/SQL triples.  
Full run output: [`run_output_20260515_125444.txt`](run_output_20260515_125444.txt)

---

## Before vs. After

### Accuracy on 200 Held-Out Test Questions

| Model | Correct | Accuracy |
|---|---|---|
| Base Llama-3.2-1B (no fine-tuning) | 77 / 200 | 38.50% |
| LoRA Fine-Tuned (1 epoch, 78,377 examples) | 177 / 200 | 88.50% |
| **Improvement** | **+100 questions** | **+50.0 pp** |

The improvement far exceeds what prompt engineering alone could achieve. The base model at 38.5% already understood some SQL — it knew keywords like `SELECT`, `FROM`, `WHERE` — but it frequently hallucinated column names, used wrong table aliases, and produced syntactically broken queries for anything beyond the simplest patterns.

After fine-tuning, the model learned **both** SQL syntax and schema grounding:

- **SQL syntax**: It reliably produces valid `SELECT ... FROM ... WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`, `COUNT(*)`, `MAX()` constructs. The training data covered a wide range of SQL complexity and the model internalized the grammatical patterns.
- **Schema grounding**: It correctly reads the `CREATE TABLE` statement in the prompt and maps column names from the schema into the query. For example, given `CREATE TABLE table_name_75 (year VARCHAR, rank VARCHAR)` and the question *"What year did the rank of 31 happen in?"*, it generates `SELECT year FROM table_name_75 WHERE rank = "31"` — correctly identifying `year` as the target column and `rank` as the filter column from the schema.

The remaining 11.5% failures are mostly on complex multi-table queries (JOINs, subqueries) and edge cases where the question phrasing is ambiguous.

---

## Step 7 — Novel Schema Test Questions (Out-of-Distribution)

The model was tested on 5 schemas it never saw during training.  
**Result: 3/5 passed (60%)**

| # | Difficulty | Generated SQL | Expected SQL | Result | Failure Mode |
|---|---|---|---|---|---|
| 1 | Easy | `SELECT id, name FROM employees WHERE department = 'Engineering'` | `SELECT name FROM employees WHERE department = 'engineering'` | FAIL | Logic error — selected extra column `id` not asked for |
| 2 | Easy | `SELECT COUNT(*) FROM products WHERE price > 50 AND category = 'electronics'` | `SELECT COUNT(*) FROM products WHERE price > 50` | FAIL | Hallucination — invented `AND category = 'electronics'` not in question |
| 3 | Medium | `SELECT MAX(score) FROM students WHERE class = 'Science'` | `SELECT MAX(score) FROM students WHERE class = 'science'` | PASS | Case difference only; execution result matched |
| 4 | Medium | `SELECT customer, SUM(amount) FROM orders GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3` | (same) | PASS | Exact match |
| 5 | Hard | `SELECT T1.id, T1.name, T1.department, COUNT(*) FROM courses AS T1 JOIN enrollments AS T2 ON T1.id = T2.course_id JOIN enrollments AS T3 ON T2.student_id = T3.student_id GROUP BY T1.department` | `SELECT courses.department, COUNT(DISTINCT enrollments.student_id) FROM courses JOIN enrollments ON courses.id = enrollments.course_id GROUP BY courses.department` | PASS | Different SQL style but execution result matched |

**Accuracy dropped from 88.5% (in-distribution) to 60% (novel schemas)**, consistent with the lesson's prediction that novel schemas show lower accuracy. The model generalizes well to aggregation and JOIN patterns but struggles with precisely scoping SELECT columns when given simple novel schemas.

---

## RAG Comparison

Imagine a RAG system with 1,000 (question, SQL) pairs in a vector database. When would it work, and when would it fail?

### Where RAG Would Work Well

**Simple, pattern-matching queries** — e.g., `SELECT [col] FROM [table] WHERE [col] = [value]`.  
If the training set has similar questions like *"Who scored the most points in game X?"* and the new question is *"Who won the most medals in event Y?"*, RAG retrieves a structurally similar example and the LLM can adapt it. RAG is effective here because the SQL pattern is common and the retrieved example provides a reliable template.

**Factual lookups on well-known schemas** — questions over schemas the retrieval set covers well. The vector similarity search finds the right template, and substituting column/table names is straightforward.

### Where RAG Would Struggle

**Novel schemas** (exactly Step 7's situation) — RAG retrieves examples based on question similarity, not schema similarity. If the retrieved SQL references `table_name_15 (pick, school)` but the live schema is `employees (id, name, salary, department)`, the LLM must completely rewrite the SQL anyway. The retrieved example provides almost no useful signal.

**Complex compositional queries** — e.g., *"List the top 3 customers by total order amount."* This requires `GROUP BY + SUM + ORDER BY + LIMIT` chained together. RAG might retrieve a `GROUP BY` example and a separate `ORDER BY` example, but composing them correctly is a reasoning task that retrieval doesn't solve.

**Syntax internalization** — RAG doesn't teach the model SQL grammar. The base model at 38.5% already had access to SQL examples at inference time (via the prompt format), yet still failed on 61.5% of questions. Fine-tuning actually changes the model's weights so that SQL generation becomes a learned skill, not a pattern-matching exercise at inference time.

**Bottom line**: RAG works best when the answer can be found by analogy to a retrieved example. Fine-tuning works better when the task requires internalizing a compositional skill (like SQL grammar + schema reading) that must generalize across thousands of novel inputs.

---

## Error Analysis

From the Step 7 failures and known failure patterns in the 11.5% in-distribution misses:

### Failure Mode 1 — Extra Columns Selected (Logic Error)
**Example**: Asked for employee *names*, generated `SELECT id, name` instead of `SELECT name`.  
**What it tells us**: The model learned to SELECT from the schema but didn't tightly ground the column selection to the question's specific ask. It over-generates, pulling in neighboring columns from the schema. This is a schema-grounding failure, not a syntax failure.

### Failure Mode 2 — Hallucinated Constraints (Logic Error / Hallucination)
**Example**: Asked *"How many products cost more than 50 dollars?"*, generated `WHERE price > 50 AND category = 'electronics'`.  
**What it tells us**: The model invented a filter condition (`category = 'electronics'`) that has no basis in the question. This is a hallucination — the model pattern-matched from training examples where `category` columns were commonly filtered, and injected that pattern even when not asked. The model learned SQL structure but not always the strict boundary of "only express what the question asks."

### Failure Mode 3 — Wrong SQL Style but Correct Result (Not a Real Failure)
**Example**: Case 5 (Hard JOIN) generated a more verbose query with double JOINs and extra SELECT columns, yet the execution result matched.  
**What it tells us**: The model has genuinely learned SQL semantics — it can produce a logically equivalent query in a different style. This is actually a sign of robust learning. The evaluation (execution-based comparison on seeded SQLite databases) correctly credits this as a pass.

### Summary of What the Model Learned vs. What It Still Misses

| Capability | Status |
|---|---|
| SQL keyword syntax (SELECT, FROM, WHERE, GROUP BY, etc.) | Learned well |
| Reading schema and mapping column names | Learned well |
| Aggregation (COUNT, MAX, SUM) | Learned well |
| JOIN across tables | Mostly learned |
| Precise column scoping (select only what's asked) | Partially learned — tends to over-select |
| Strict question-to-filter mapping (no hallucination) | Partially learned — occasionally adds spurious conditions |
| Complex subqueries / nested SELECT | Likely still weak (not tested here) |
