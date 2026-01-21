# DSPy RLM with Google Gemini Setup

This guide shows how to set up and use DSPy's RLM (Recursive Language Model) with Google Gemini API.

## What is RLM?

RLM (Recursive Language Model) is an advanced inference strategy in DSPy that treats long contexts as external environments. Instead of feeding everything to the model at once, RLM:

- Lets the LLM write Python code to examine and process data programmatically
- Uses a sandboxed REPL environment for safe code execution
- **Enables multi-hop reasoning** through iterative sub-LM calls
- **Delegates chunks to sub-LMs** for focused analysis and parallel processing
- Iteratively refines understanding by:
  1. Analyzing the current state
  2. Writing Python code to explore or solve parts of the problem
  3. Calling sub-LMs (`llm_query()`, `llm_query_batched()`) for semantic analysis
  4. Receiving output and continuing iteration
  5. Calling `SUBMIT()` when complete

This is particularly useful for:
- **Multi-hop reasoning**: Break complex queries into iterative sub-tasks
- **Processing very long documents**: Chunk and delegate to sub-LMs (e.g., 16 months of Slack messages)
- **Complex reasoning tasks**: Combine programmatic logic with LM semantic understanding
- **Iterative refinement**: Each sub-LM call builds on previous results
- **Parallel processing**: Use `llm_query_batched()` for independent analyses

## Files Included

1. **test_simple_gemini.py** - Simple test to verify API connection works
2. **test_rlm_gemini.py** - Full RLM test with 2-week dataset optimized for multi-hop reasoning
3. **RLM_MULTIHOP_ANALYSIS.md** - Detailed analysis of expected multi-hop behavior and chunking strategy
4. **GEMINI_RLM_SETUP.md** - This documentation

## Prerequisites

```bash
pip install dspy-ai
```

## Configuration

### API Key

The test scripts use the provided API key:
```python
api_key="AIzaSyAqfZRxJbxPfElPxVi79L9r0zBDCyEaZTc"
```

**Important:** For production use, store API keys securely using environment variables:
```python
import os
api_key = os.environ.get("GOOGLE_API_KEY")
```

### Model Selection

The scripts use `gemini/gemini-1.5-flash` which is:
- Fast and cost-effective
- Good for testing
- Sufficient for most RLM tasks

Alternative models:
- `gemini/gemini-1.5-pro` - More capable, higher cost
- `gemini/gemini-2.5-flash` - Latest flash model

## Multi-Hop Reasoning with RLM

The test script (`test_rlm_gemini.py`) is designed to demonstrate **multi-hop reasoning** where the RLM:

1. **Breaks down complex queries** into focused sub-tasks
2. **Delegates chunks to sub-LMs** for parallel analysis
3. **Iteratively refines** understanding based on sub-LM results
4. **Combines code + LM reasoning** for optimal efficiency

### Example Multi-Hop Flow

For the query: "What are the top 3 most promising ideas and how did they evolve?"

**Hop 1**: Extract all ideas
```python
ideas = llm_query("Extract all distinct ideas from this discussion")
```

**Hop 2**: Analyze each idea's timeline (parallel)
```python
timelines = llm_query_batched([
    f"When was '{idea}' first mentioned and how did it evolve?"
    for idea in ideas
])
```

**Hop 3**: Rank by promising-ness
```python
ranking = llm_query("Rank these ideas by frequency, depth, and connections")
```

**Hop 4**: Format and submit
```python
SUBMIT(ideas=format_top_3(ranking))
```

See **RLM_MULTIHOP_ANALYSIS.md** for detailed expected behavior.

## Usage

### Step 1: Test Basic Connection

First, verify your API connection works:

```bash
python test_simple_gemini.py
```

Expected output:
```
================================================================================
Testing basic dspy.Predict with Google Gemini
================================================================================

Question: What is 2 + 2?
Answer: 4
✓ Google Gemini API connection works!
```

### Step 2: Test RLM

Once basic connection works, run the RLM test:

```bash
python test_rlm_gemini.py
```

This will:
- Use a minimal mock Slack discussion (not the full 16 months)
- Limit iterations to 5 (vs default 20) to minimize API calls
- Limit LLM calls to 10 (vs default 50) to minimize API calls
- Show verbose output to see RLM reasoning process

Expected output format:
```
================================================================================
Testing dspy.RLM with Google Gemini
================================================================================

[RLM multi-hop reasoning traces showing:]
- Iteration 1: Extract all ideas using llm_query()
- Iteration 2: Analyze timelines using llm_query_batched()
- Iteration 3: Rank ideas using llm_query()
- Iteration 4: Format and SUBMIT()

================================================================================
RESULTS
================================================================================

Ideas found:
1. Recursive Language Models (RLM) (First mentioned: Jan 9, 2024) - Started with
   concept of programmatic context exploration, evolved through sandboxing discussions,
   successful testing, and recognized as general approach by Jan 19.

2. Multi-hop reasoning with iterative refinement (First mentioned: Jan 11, 2024) -
   Branching vs linear reasoning, successfully implemented with 5-8 iterations,
   recognized as key RLM capability.

3. Caching intermediate results (First mentioned: Jan 16, 2024) - Memoization for
   redundant sub-LM calls, connected to multi-hop efficiency and cost optimization.

First idea: Recursive Language Models (RLM) (First mentioned: Jan 9, 2024) - ...

================================================================================
EXECUTION TRAJECTORY (Multi-hop reasoning trace)
================================================================================
[Shows each iteration's reasoning, code, and sub-LM results]
```

## Cost Management

The scripts are designed to demonstrate multi-hop reasoning while keeping costs low:

### In test_rlm_gemini.py:

```python
# 2-week Slack dataset (realistic but manageable)
slack_dump = """
[~30 messages over 2 weeks covering RLM development]
"""

# Optimized for multi-hop reasoning
rlm = dspy.RLM(
    "discussion, request -> ideas: list[str]",
    max_iterations=10,     # Allow multi-hop reasoning (4-5 expected)
    max_llm_calls=15,      # Allow sub-LM calls for chunking (8-10 expected)
    verbose=True           # See multi-hop behavior
)
```

**Expected usage**:
- Iterations: 4-5 (breaking down the task into hops)
- Sub-LM calls: 8-10 (parallel analyses of chunks)
- Total cost: ~$0.002 per run (less than a quarter cent)

### For Production Use:

When ready to use your real dataset:

```python
# Load your real data
with open('slack_dump.txt', 'r') as f:
    slack_dump = f.read()

# Adjust limits based on your needs and budget
rlm = dspy.RLM(
    "discussion, request -> ideas: list[str]",
    max_iterations=20,     # Increase for complex tasks
    max_llm_calls=50,      # Increase for complex tasks
    verbose=False          # Disable verbose for production
)

output = rlm(
    discussion=slack_dump,
    request="What are the 5 coolest unfinished ideas that Omar & Isaac keep coming back to?"
)

for i, idea in enumerate(output.ideas, 1):
    print(f"{i}. {idea}")
```

## RLM Parameters Explained

```python
RLM(
    signature: str,                    # Input/output schema, e.g., "context, query -> answer"
    max_iterations: int = 20,          # Max REPL interaction rounds
    max_llm_calls: int = 50,           # Max sub-LLM calls (via llm_query)
    max_output_chars: int = 100_000,   # Max characters from REPL output
    verbose: bool = False,             # Show detailed execution logs
    tools: dict = None,                # Custom Python functions for the REPL
    sub_lm: dspy.LM = None,           # LM for llm_query (defaults to configured LM)
)
```

## Built-in RLM Tools

Inside the RLM REPL environment, the LLM has access to:

- `llm_query(prompt: str) -> str` - Query the sub-LLM with a prompt
- `llm_query_batched(prompts: list[str]) -> list[str]` - Batch queries
- `SUBMIT(field1=val1, field2=val2, ...)` - Return final results
- `print(...)` - Debug output
- Standard Python libraries: `re`, `json`, `collections`, `math`, etc.

## Advanced Features

### Custom Tools

Add your own tools to the RLM environment:

```python
def search_database(query: str) -> list[str]:
    """Search a custom database."""
    return database.search(query)

rlm = dspy.RLM(
    "query -> results: list[str]",
    tools={"search_database": search_database}
)
```

### Type-Safe Outputs

RLM supports typed output fields:

```python
rlm = dspy.RLM(
    "document, question -> summary: str, confidence: float, keywords: list[str]"
)

result = rlm(document="...", question="...")
print(result.summary)        # str
print(result.confidence)     # float
print(result.keywords)       # list[str]
```

### Async Support

```python
result = await rlm.aforward(discussion=slack_dump, request="...")
```

## Troubleshooting

### API Key Issues

If you get authentication errors:
1. Verify the API key is correct
2. Check that the key has Gemini API access enabled
3. Verify there are no usage limits or quotas exceeded

### Network Issues

If you see connection errors:
1. Check internet connectivity
2. Verify no proxy/firewall blocking Google APIs
3. Try unsetting proxy environment variables:
   ```bash
   unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
   python test_simple_gemini.py
   ```

### Model Not Found

If you get "model not found" errors:
1. Verify the model name format: `gemini/model-name`
2. Check that the model is available in your region
3. Try alternative models like `gemini/gemini-1.5-flash`

## Next Steps

1. ✅ Run `test_simple_gemini.py` to verify basic setup
2. ✅ Run `test_rlm_gemini.py` to test RLM with minimal data
3. 📝 Prepare your full dataset (e.g., 16 months of Slack messages)
4. 🎯 Adjust `max_iterations` and `max_llm_calls` based on complexity
5. 🚀 Run RLM on your real data
6. 📊 Analyze results and iterate on your prompt/signature

## Reference

- **RLM Paper:** "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
- **RLM Implementation:** `dspy/predict/rlm.py`
- **Tests:** `tests/predict/test_rlm.py`
- **DSPy Docs:** https://dspy-docs.vercel.app/

## Cost Estimation

For Gemini 1.5 Flash (prices as of 2024):
- Input: ~$0.075 per 1M tokens
- Output: ~$0.30 per 1M tokens

**Test script estimate (2-week dataset, multi-hop):**
- Main iterations: ~1,500 tokens/iteration × 5 iterations = 7,500 tokens
- Sub-LM calls: ~500 tokens/call × 10 calls = 5,000 tokens
- Output: ~200 tokens/iteration × 5 iterations = 1,000 tokens
- **Total: ~15,000 input tokens, ~2,500 output tokens**
- **Cost: ~$0.002** (less than a quarter cent)

**Production with 16-month Slack dump:**
- RLM will chunk the data and delegate to sub-LMs
- Expected: 20-30 iterations with 50-100 sub-LM calls
- Estimated cost: $0.05-0.10 per run with Gemini Flash
- Monitor first run with `verbose=True` to track actual costs
- The multi-hop approach is actually **more cost-effective** than sending
  the entire 16-month dump in a single prompt
