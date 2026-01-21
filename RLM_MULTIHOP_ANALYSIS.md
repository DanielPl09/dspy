# RLM Multi-Hop Reasoning Analysis

## Test Configuration

### Dataset: 2-Week Slack Conversation
- **Time period**: Jan 8-21, 2024 (2 weeks)
- **Message count**: ~30 messages
- **Topics covered**: RLM development, multi-hop reasoning, prompt optimization, caching, visual debugging, dataset bootstrapping

### RLM Configuration
```python
rlm = dspy.RLM(
    "discussion, request -> ideas: list[str]",
    max_iterations=10,     # Allow multi-hop reasoning iterations
    max_llm_calls=15,      # Allow multiple sub-LM calls for refinement
    verbose=True
)
```

### Query
"What are the top 3 most promising ideas that Omar & Isaac discussed? For each idea, identify when it was first mentioned and how it evolved over the 2 weeks."

## Expected Multi-Hop Reasoning Behavior

### Why This Demonstrates Multi-Hop Reasoning

This query is designed to require **iterative refinement** through multiple sub-LM calls:

1. **First Hop**: Identify all ideas in the discussion
   - RLM writes code to extract distinct ideas using `llm_query()`
   - Sub-LM analyzes the full discussion to list all ideas

2. **Second Hop**: For each idea, trace its evolution
   - RLM writes code to use `llm_query_batched()` to analyze each idea
   - Multiple sub-LMs run in parallel, each analyzing when an idea was mentioned
   - Results show timeline and evolution

3. **Third Hop**: Rank by "promising-ness"
   - RLM analyzes which ideas were:
     - Revisited multiple times
     - Expanded upon with implementation details
     - Connected to other ideas
   - Uses sub-LM to synthesize rankings

4. **Fourth Hop**: Format final output
   - RLM structures the results
   - Calls `SUBMIT(ideas=[...])` with top 3 ideas and their evolution

## Expected RLM Execution Trace

```
Iteration 1: Reasoning
-----------------------
The task requires identifying ideas, tracking their evolution, and ranking them.
I'll need to:
1. Extract all distinct ideas from the discussion
2. For each idea, find when it was first mentioned and how it evolved
3. Rank by "promising-ness" based on frequency and depth of discussion

Code:
# First, let's identify all major ideas discussed
all_ideas = llm_query("""
Based on this Slack discussion, list all distinct technical ideas or projects
that Omar and Isaac discussed. Return a JSON list.

Discussion:
{discussion}
""")
print(f"Found ideas: {all_ideas}")

Iteration 2: Reasoning
-----------------------
Now I have the list of ideas. I need to analyze each one's timeline and evolution.
I'll use batched queries to analyze them in parallel.

Code:
import json
ideas_list = json.loads(all_ideas)

# Analyze each idea's evolution in parallel
evolution_queries = [
    f"In this discussion, when was '{idea}' first mentioned and how did it evolve?
    Provide: first_date, key_developments, connections_to_other_ideas."
    for idea in ideas_list
]

evolutions = llm_query_batched(evolution_queries)
print(f"Evolution analysis: {evolutions}")

Iteration 3: Reasoning
-----------------------
Now I have evolution data for each idea. Let me rank them by "promising-ness"
based on:
- How many times they were revisited
- How much detail was discussed
- How they connected to other ideas

Code:
# Rank ideas by promising-ness
ranking_prompt = f"""
Given these ideas and their evolution over 2 weeks, rank the top 3 most
promising ones. Consider:
- Frequency of mentions
- Depth of discussion
- Implementation details shared
- Connections to other ideas

Ideas and evolution: {evolutions}

Return JSON with top 3, each having: idea_name, first_mention_date,
evolution_summary, why_promising
"""

top_ideas = llm_query(ranking_prompt)
print(f"Top 3 ideas: {top_ideas}")

Iteration 4: Reasoning
-----------------------
I have the ranked ideas. Let me format them as a list for the final output.

Code:
import json
ranked = json.loads(top_ideas)

ideas_output = [
    f"{idea['idea_name']} (First mentioned: {idea['first_mention_date']}) - "
    f"{idea['evolution_summary']}"
    for idea in ranked
]

SUBMIT(ideas=ideas_output)
```

## Expected Output

```
Ideas found:
1. Recursive Language Models (RLM) (First mentioned: Jan 9, 2024) - Started with the concept of models programmatically exploring context. Evolved through Week 1 with sandboxing discussions (Jan 12), proved successful in testing (Jan 15), and by Week 2 was recognized as a general approach for any iterative refinement task (Jan 19). Most promising due to being the central theme with continuous development.

2. Multi-hop reasoning with iterative refinement (First mentioned: Jan 11, 2024) - Initially discussed as branching vs linear chain-of-thought. Successfully implemented by Jan 15 with 5-8 iterations for complex tasks. Recognized as a key capability enabled by RLM architecture. Promising because it addresses fundamental limitations in current reasoning approaches.

3. Caching intermediate results for cost optimization (First mentioned: Jan 16, 2024) - Proposed as memoization for redundant sub-LM calls. Connected to multi-hop reasoning efficiency. Promising because it addresses both cost and performance concerns in production RLM systems.

First idea: Recursive Language Models (RLM) (First mentioned: Jan 9, 2024) - Started with the concept of models programmatically exploring context...
```

## Why This Demonstrates Chunk Delegation to Sub-LMs

### Chunk 1: Idea Extraction
The RLM delegates to a sub-LM: "Extract all ideas from this 2-week discussion"
- **Why**: The main RLM doesn't need to manually parse all messages
- **Benefit**: Sub-LM specializes in extraction task

### Chunk 2: Timeline Analysis (Batched)
The RLM delegates to multiple sub-LMs in parallel: "For each idea, trace its timeline"
- **Why**: Each idea's timeline is independent
- **Benefit**: Parallel processing, each sub-LM focuses on one idea

### Chunk 3: Synthesis & Ranking
The RLM delegates to a sub-LM: "Rank these ideas by promising-ness"
- **Why**: Requires holistic judgment across all ideas
- **Benefit**: Sub-LM provides high-level synthesis

### Chunk 4: Final Formatting
The RLM does this programmatically without sub-LM
- **Why**: Simple string formatting
- **Benefit**: No API call needed for trivial task

## Key Multi-Hop Behaviors Demonstrated

1. **Iterative Refinement**: Each iteration builds on previous sub-LM results
2. **Chunking Strategy**: Discussion is analyzed in semantic chunks (ideas, timelines, rankings)
3. **Parallel Sub-LM Calls**: Uses `llm_query_batched()` for independent analyses
4. **Mixed Code + LM**: Some tasks use code (JSON parsing), others use sub-LMs (semantic analysis)
5. **Adaptive Reasoning**: RLM decides what to delegate vs. what to code

## Cost Analysis

With 2-week dataset and these parameters:
- **Iterations**: ~4-5 (less than max of 10)
- **Sub-LM calls**: ~8-10 (well under max of 15)
  - 1 for idea extraction
  - N for timeline analysis (N = number of ideas, ~5-7)
  - 1 for ranking
  - 1 for synthesis

**Estimated tokens**:
- Input per iteration: ~1,500 tokens (discussion + previous results)
- Output per iteration: ~200 tokens
- Sub-LM calls: ~500 tokens input × 10 calls = 5,000 tokens
- **Total**: ~15,000 input tokens, ~2,500 output tokens

**Cost with Gemini 1.5 Flash**:
- Input: 15,000 tokens × $0.075/1M = $0.001125
- Output: 2,500 tokens × $0.30/1M = $0.00075
- **Total: ~$0.002** (less than a quarter of a cent)

## Advantages Over Single-Shot Approach

### Without RLM (Single Prompt):
```
"Analyze this 2-week discussion and identify top 3 ideas with evolution"
```
- Model sees everything at once
- Linear reasoning only
- No way to iteratively refine
- Limited control over analysis strategy

### With RLM (Multi-Hop):
```
1. Extract ideas (focused sub-task)
2. Analyze each idea's timeline (parallel, focused)
3. Rank and synthesize (holistic view)
4. Format output (programmatic)
```
- Breaks down complex task
- Each sub-LM call is focused and clear
- Can use parallel processing
- Combines code (parsing, formatting) with LM (semantic analysis)
- Iteratively refines understanding

## Scaling to 16-Month Slack Dump

For a full 16-month dataset:
1. **First chunk**: Use RLM to identify time periods with significant discussions
2. **Second chunk**: For each period, extract key ideas (parallel sub-LM calls)
3. **Third chunk**: Track idea evolution across periods
4. **Fourth chunk**: Identify recurring themes that span multiple months
5. **Final synthesis**: Rank by long-term persistence and development

This would require:
- `max_iterations=20-30` (more time periods to analyze)
- `max_llm_calls=50-100` (many parallel analyses)
- Still manageable cost: ~$0.05-0.10 per run with Gemini Flash

## Running the Test

To run this test in a proper environment with network access:

```bash
# Ensure dependencies are installed
pip install dspy-ai

# Set up environment (if not using hardcoded API key)
export GOOGLE_API_KEY="your-api-key"

# Run the test
python test_rlm_gemini.py
```

The test will output:
1. Each iteration's reasoning
2. Code written by the RLM
3. Sub-LM call results
4. Final structured output
5. Full execution trajectory for debugging

## Conclusion

This 2-week dataset is optimal for demonstrating:
- ✅ Multi-hop reasoning (4-5 hops expected)
- ✅ Chunk delegation to sub-LMs (8-10 sub-LM calls)
- ✅ Iterative refinement (building on previous results)
- ✅ Mixed programmatic + LM reasoning
- ✅ Manageable cost (~$0.002 per run)

The test is ready to run in any environment with proper network access to Google's Gemini API.
