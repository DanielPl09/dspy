"""
Minimal test script for dspy.RLM with Google Gemini API
This script keeps API calls minimal by using:
- Small test dataset instead of 16-month slack dump
- Low max_iterations and max_llm_calls limits
"""

import dspy

# Configure dspy to use Google Gemini with the provided API key
# Note: Use "gemini/" prefix for Google Gemini models
google_lm = dspy.LM("gemini/gemini-1.5-flash", api_key="AIzaSyAqfZRxJbxPfElPxVi79L9r0zBDCyEaZTc")
dspy.configure(lm=google_lm)

# 2-week Slack discussion with enough complexity to require multi-hop reasoning
# The dataset is structured to encourage RLM to break it into chunks and use sub-LMs
slack_dump = """
Week 1 (Jan 8-14, 2024):

2024-01-08 Omar: We need to rethink how we handle long contexts. Current approaches just truncate or summarize, but we're losing structure.
2024-01-08 Isaac: Agreed. What if the model could programmatically explore the context instead of seeing it all at once?
2024-01-08 Omar: Like giving it tools to query specific parts? That's interesting...

2024-01-09 Isaac: Been sketching out an idea - recursive language models. The LM writes code to call sub-LMs.
2024-01-09 Omar: Mind blown 🤯 So it's like the model is writing its own retrieval program?
2024-01-09 Isaac: Exactly! And it can iterate - call one sub-LM, use that result to inform the next call.

2024-01-10 Omar: This could solve the prompt optimization problem we've been stuck on.
2024-01-10 Omar: If the model can iteratively refine prompts by testing them with sub-LMs...
2024-01-10 Isaac: Yes! Auto-optimization through iteration. The RLM explores the prompt space programmatically.

2024-01-11 Isaac: Also thinking about multi-hop reasoning. Current chain-of-thought is linear.
2024-01-11 Omar: But with RLM, it could branch? Try multiple reasoning paths and combine results?
2024-01-11 Isaac: Exactly. More like a tree of reasoning than a chain.

2024-01-12 Omar: Started prototyping the REPL environment. Sandboxing is tricky.
2024-01-12 Isaac: Security is critical. We can't let arbitrary code execution go wild.
2024-01-12 Omar: Right. Thinking Pyodide or Deno for the sandbox.

Week 2 (Jan 15-21, 2024):

2024-01-15 Isaac: The multi-hop reasoning is working! Tested on some complex questions.
2024-01-15 Omar: What's the iteration count looking like? Are we hitting limits?
2024-01-15 Isaac: Usually 5-8 iterations for complex tasks. Need to tune max_iterations.

2024-01-16 Omar: Idea: what if we cache intermediate results? Some sub-LM calls are redundant.
2024-01-16 Isaac: Smart. Like memoization for LM calls. Could save a lot of API costs.
2024-01-16 Omar: Exactly. And it would speed things up too.

2024-01-17 Isaac: Been thinking about the visual debugger idea again.
2024-01-17 Omar: For RLM specifically? That would be super valuable - seeing the whole reasoning tree.
2024-01-17 Isaac: Yeah. Show each iteration, what code it wrote, what sub-LMs returned, how it refined.

2024-01-18 Omar: Dataset bootstrapping could work really well with RLM.
2024-01-18 Isaac: How so?
2024-01-18 Omar: The RLM could programmatically generate examples, test them with sub-LMs, refine based on quality.
2024-01-18 Isaac: Ooh. So it's like automated data augmentation with quality control built in.

2024-01-19 Isaac: This RLM approach is more general than I thought. It's not just for long contexts.
2024-01-19 Omar: Right. Any task where you want the model to iteratively refine using multiple LM calls.
2024-01-19 Isaac: Prompt optimization, multi-hop QA, dataset generation, even meta-learning...

2024-01-20 Omar: We should write this up. This feels like a fundamental shift in how we use LMs.
2024-01-20 Isaac: Agreed. "Recursive Language Models" - treating LMs as programmable reasoning environments.
2024-01-21 Omar: Let's do it. I'll start the draft this week.
"""

# Create RLM with signature optimized for multi-hop reasoning
# Allow enough iterations to see sub-LM calls and iterative refinement
rlm = dspy.RLM(
    "discussion, request -> ideas: list[str]",
    max_iterations=10,     # Allow multi-hop reasoning iterations
    max_llm_calls=15,      # Allow multiple sub-LM calls for refinement
    verbose=True           # Show what's happening to observe multi-hop behavior
)

print("=" * 80)
print("Testing dspy.RLM with Google Gemini")
print("=" * 80)
print()

# Run the RLM with a request that encourages multi-hop reasoning
# The RLM should break this down: first identify all ideas, then analyze which are unfinished,
# then rank by "coolness" based on how often they're revisited and expanded upon
output = rlm(
    discussion=slack_dump,
    request="What are the top 3 most promising ideas that Omar & Isaac discussed? For each idea, identify when it was first mentioned and how it evolved over the 2 weeks."
)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print("Ideas found:")
for i, idea in enumerate(output.ideas, 1):
    print(f"{i}. {idea}")

print()
print("First idea:", output.ideas[0])

# Optional: Print the trajectory to see how RLM reasoned through the problem
print()
print("=" * 80)
print("EXECUTION TRAJECTORY (for debugging)")
print("=" * 80)
print(output.trajectory)
