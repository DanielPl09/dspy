# DSPy RLM Parameter Guide

## What is RLM?

RLM (Recursive Language Model) is a DSPy module where the LLM **writes Python code** to programmatically explore large contexts. Instead of feeding everything to the model at once, RLM lets the LLM:

1. Write code to examine data
2. Chunk/split data intelligently using Python
3. Call sub-LLMs on specific chunks via `llm_query(prompt)` or `llm_query_batched(prompts)`
4. Iterate and build up the answer programmatically

**Key insight**: RLM doesn't auto-split your data. The LLM decides HOW to chunk it by writing code.

## Core Parameters

### 1. `max_iterations` (default: 20)
**Controls**: Maximum REPL interaction loops

The LLM writes code, sees output, writes more code, etc. Each loop = 1 iteration.

```python
rlm = dspy.RLM(
    "context, query -> answer",
    max_iterations=20  # Up to 20 code execution loops
)
```

**When to increase**:
- Complex multi-step reasoning tasks
- Need more exploration of the data
- LLM hits the limit before finishing

**When to decrease**:
- Simple extraction tasks
- Want faster execution
- Reduce cost

---

### 2. `max_llm_calls` (default: 50)
**Controls**: Maximum sub-LLM queries via `llm_query()` and `llm_query_batched()`

This limits how many chunks the RLM can send to sub-LLMs for semantic analysis.

```python
rlm = dspy.RLM(
    "discussion, request -> ideas: list[str]",
    max_llm_calls=50  # Can call llm_query up to 50 times total
)
```

**How the LLM uses this**:
```python
# Inside the RLM's generated code:

# Option 1: One at a time (uses 3 calls)
chunk1_result = llm_query("Analyze this chunk: ...")
chunk2_result = llm_query("Analyze this chunk: ...")
chunk3_result = llm_query("Analyze this chunk: ...")

# Option 2: Batched (uses 10 calls, but much faster!)
results = llm_query_batched([
    "Analyze chunk 1: ...",
    "Analyze chunk 2: ...",
    # ... 10 prompts total
])
```

**When to increase**:
- Very long documents (16+ months of Slack!)
- Need fine-grained chunk analysis
- LLM hits the limit

**When to decrease**:
- Shorter contexts
- Reduce cost (each call = API cost)
- Force more aggregation in code vs LLM calls

---

### 3. `max_output_chars` (default: 100,000)
**Controls**: Maximum characters to include from REPL output

When the LLM's code prints output, it's truncated to this length.

```python
rlm = dspy.RLM(
    "context, query -> answer",
    max_output_chars=100_000  # Truncate outputs over 100k chars
)
```

**When to increase**:
- Large intermediate results
- Need to see full outputs for debugging

**When to decrease**:
- Reduce token usage in the REPL history
- Faster iteration

---

### 4. `verbose` (default: False)
**Controls**: Whether to log detailed execution info

Shows you the actual Python code the LLM writes!

```python
rlm = dspy.RLM(
    "context, query -> answer",
    verbose=True  # See the code being generated!
)
```

**Output example**:
```
RLM iteration 1/20
Reasoning: I need to first explore the data structure...
Code:
print(len(discussion))
print(discussion[:200])
```

**Use this for**:
- Understanding how RLM works
- Debugging why results are wrong
- Learning what code patterns work best

---

### 5. `sub_lm` (default: None, uses `dspy.settings.lm`)
**Controls**: Which model to use for `llm_query()` calls

Lets you use a **cheaper/faster model** for sub-queries!

```python
# Main LM for code generation
main_lm = dspy.LM('openai/gpt-4o')
dspy.configure(lm=main_lm)

# Cheaper LM for chunk analysis
sub_lm = dspy.LM('openai/gpt-3.5-turbo')

rlm = dspy.RLM(
    "discussion, request -> ideas: list[str]",
    sub_lm=sub_lm  # Use gpt-3.5-turbo for llm_query calls
)
```

**Cost optimization strategy**:
- Main LM (code generation): Use smarter model (gpt-4o)
- Sub LM (chunk analysis): Use cheaper model (gpt-3.5-turbo)
- Save 10-20x on cost since most calls are `llm_query()`!

---

### 6. `tools` (default: None)
**Controls**: Custom functions the LLM can call from code

Add domain-specific tools the LLM can use!

```python
def search_database(query: str) -> str:
    """Search the internal database."""
    # Your implementation
    return results

rlm = dspy.RLM(
    "query -> answer",
    tools={"search_database": search_database}
)
```

**Built-in tools** (always available):
- `llm_query(prompt)` - Query sub-LLM
- `llm_query_batched(prompts)` - Query multiple prompts concurrently
- `print()` - Print to see results
- `SUBMIT(...)` - Submit final output
- Standard library: `re`, `json`, `collections`, `math`, etc.

---

### 7. `interpreter` (default: None, uses `PythonInterpreter`)
**Controls**: Code execution environment

Advanced: Provide custom interpreter (e.g., E2B, Modal).

```python
from dspy.primitives.code_interpreter import CodeInterpreter

# Custom interpreter implementation
my_interpreter = MyCustomInterpreter()

rlm = dspy.RLM(
    "context, query -> answer",
    interpreter=my_interpreter
)
```

---

## How RLM Splits Data

**RLM doesn't auto-split**. The LLM writes code to decide how to chunk!

### Example: Splitting by regex

```python
# The LLM might write:
import re

# Split into message blocks
messages = re.findall(r'\[.*?\].*?(?=\[|$)', discussion, re.DOTALL)
print(f"Found {len(messages)} messages")

# Analyze each message for ideas
prompts = [f"Extract ideas from: {msg}" for msg in messages]
results = llm_query_batched(prompts)  # Parallel analysis!

# Aggregate results
all_ideas = []
for result in results:
    all_ideas.extend(result.split('\n'))
```

### Example: Splitting by length

```python
# The LLM might write:
chunks = []
chunk_size = 5000  # chars per chunk

for i in range(0, len(discussion), chunk_size):
    chunks.append(discussion[i:i+chunk_size])

# Process chunks
summaries = llm_query_batched([f"Summarize: {c}" for c in chunks])
```

### Example: Smart splitting

```python
# The LLM might write:
import re

# Find natural boundaries (dates in Slack)
dates = re.findall(r'\[(\d{4}-\d{2}-\d{2})', discussion)
unique_dates = sorted(set(dates))

# Split by month
monthly_chunks = {}
for date in unique_dates:
    month = date[:7]  # "2024-03"
    if month not in monthly_chunks:
        monthly_chunks[month] = []
    # Extract messages for this date...
```

---

## Recommended Configurations

### For very long contexts (16 months of Slack):

```python
rlm = dspy.RLM(
    "discussion, request -> ideas: list[str]",
    max_iterations=30,       # More exploration
    max_llm_calls=100,       # More chunking budget
    max_output_chars=200000, # See more output
    verbose=True,            # Debug what it's doing
    sub_lm=cheap_lm          # Save cost on sub-queries
)
```

### For medium contexts (a few docs):

```python
rlm = dspy.RLM(
    "context, query -> answer",
    max_iterations=15,
    max_llm_calls=30,
    verbose=False
)
```

### For quick extraction:

```python
rlm = dspy.RLM(
    "text, query -> result: str",
    max_iterations=5,
    max_llm_calls=10,
    verbose=False
)
```

---

## Debugging Tips

1. **Always start with `verbose=True`** to see what code the LLM writes
2. **Check the trajectory** to see the execution history:
   ```python
   output = rlm(...)
   print(f"Iterations used: {len(output.trajectory)}")
   for step in output.trajectory:
       print(step['reasoning'])
       print(step['code'])
   ```
3. **Monitor token usage** by tracking `max_llm_calls`
4. **Use cheaper sub_lm** for cost optimization

---

## Summary

| Parameter | Controls | When to Increase | When to Decrease |
|-----------|----------|------------------|------------------|
| `max_iterations` | REPL loops | Complex tasks | Simple tasks, cost |
| `max_llm_calls` | Sub-LLM queries | Long docs, fine chunks | Cost, force aggregation |
| `max_output_chars` | Output size | Large intermediates | Token usage |
| `verbose` | Logging | Debugging | Production |
| `sub_lm` | Model for chunks | Quality | Cost (use cheaper) |
| `tools` | Custom functions | Domain tools | - |
| `interpreter` | Execution env | Custom sandboxes | - |

**Key insight**: The LLM decides how to split your data by writing Python code. You control the constraints (iterations, calls, output size).
