# Running the RLM Multi-Hop Test

## Quick Start

### 1. Set up your OpenAI API key

```bash
export OPENAI_API_KEY="sk-proj-your-key-here"
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your key
```

### 2. Install dependencies

```bash
pip install -e .
```

### 3. Run the test

```bash
python test_rlm_gemini.py
```

## What You'll See

The test demonstrates **multi-hop reasoning** where RLM:

1. **Hop 1**: Extracts all ideas from the 2-week discussion
2. **Hop 2**: Analyzes each idea's evolution (parallel sub-LM calls)
3. **Hop 3**: Ranks ideas by how "cool" and "unfinished" they are
4. **Hop 4**: Formats and returns the top 5

### Example Output

```
================================================================================
Testing dspy.RLM with OpenAI (GPT-4o-mini)
================================================================================

Iteration 1: Extracting ideas from discussion...
[RLM writes code to call llm_query()]

Iteration 2: Analyzing idea timelines in parallel...
[RLM writes code to call llm_query_batched()]

Iteration 3: Ranking ideas...
[RLM writes code to call llm_query()]

Iteration 4: Formatting results...
[RLM writes code to format and SUBMIT()]

================================================================================
RESULTS
================================================================================

Ideas found:
1. Recursive Language Models (RLM) - Core concept, mentioned throughout
2. Multi-hop reasoning with branching - Discussed Jan 11, expanded Jan 15
3. Prompt optimization through iteration - Jan 10, connected to RLM
4. Caching intermediate results - Jan 16, addresses cost/performance
5. Visual debugger for reasoning traces - Jan 17, unfinished but valuable

First idea: Recursive Language Models (RLM) - Core concept, mentioned throughout

================================================================================
EXECUTION TRAJECTORY (Multi-hop reasoning trace)
================================================================================
[Shows all iterations with reasoning, code, and sub-LM results]
```

## Cost Estimate

With GPT-4o-mini:
- **Iterations**: 4-5
- **Sub-LM calls**: 8-10
- **Total tokens**: ~15K input, ~3K output
- **Cost**: ~$0.002-0.003 per run (less than a third of a cent)

## Understanding the Multi-Hop Behavior

The key insight is that RLM **breaks down complex queries** into focused sub-tasks:

- Instead of: "Analyze this whole discussion and tell me the coolest ideas"
- RLM does:
  1. `llm_query("What ideas are discussed?")`
  2. `llm_query_batched(["When was idea X mentioned?", "When was idea Y mentioned?", ...])`
  3. `llm_query("Which of these are coolest and unfinished?")`
  4. Python code to format results

This is **more efficient** and **more accurate** than a single large prompt.

## Scaling to Larger Datasets

The same approach scales to much larger datasets:

```python
# For 16-month Slack dump
rlm = dspy.RLM(
    "discussion, request -> ideas: list[str]",
    max_iterations=30,     # More time periods to analyze
    max_llm_calls=100,     # More parallel analyses
    verbose=True
)
```

The multi-hop approach actually becomes **more valuable** as your data grows:
- Chunks large context into manageable pieces
- Parallel processing for efficiency
- Iterative refinement for accuracy
- Lower cost than single massive prompt

## Troubleshooting

### "Please set OPENAI_API_KEY environment variable"
Make sure you've exported your API key or created a .env file.

### Connection errors
Check that you're not behind a proxy that blocks OpenAI's API.

### "Unable to find the Deno cache dir"
This is just a warning - the test will still work. RLM uses Python interpreter as fallback.

## Files in This Setup

- **test_rlm_gemini.py** - Main test script (misnamed, but uses OpenAI)
- **RLM_MULTIHOP_ANALYSIS.md** - Detailed analysis of multi-hop behavior
- **TEST_RUN_SUMMARY.md** - Complete summary of expected results
- **GEMINI_RLM_SETUP.md** - General RLM setup guide
- **.env.example** - Template for environment variables
- **RUN_RLM_TEST.md** - This file

## Next Steps

1. Run the test to see multi-hop reasoning in action
2. Review the execution trajectory to understand RLM's strategy
3. Try modifying the query to see how RLM adapts
4. Scale up to your real dataset when ready

For more details on RLM internals, see `RLM_MULTIHOP_ANALYSIS.md`.
