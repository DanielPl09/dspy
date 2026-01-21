#!/usr/bin/env python3
"""
Test DSPy RLM (Recursive Language Model) with a mock Slack dump.

RLM lets the LLM write Python code to programmatically explore large contexts.
It doesn't auto-split - instead the LLM chunks data intelligently using code.

Key parameters:
- max_iterations: Max REPL iterations (default: 20)
- max_llm_calls: Max sub-LLM calls via llm_query/llm_query_batched (default: 50)
- max_output_chars: Max chars from REPL output (default: 100,000)
- verbose: Show detailed execution logs
- sub_lm: Use a different (e.g., cheaper) model for sub-queries
"""

import dspy
import os

# Mock Slack dump with 15 chunks - mix of technical discussions
MOCK_SLACK_DUMP = """
[2024-03-15 10:23] Omar: Been thinking about that adaptive compiler idea again.
What if we had a system that learns optimal compilation strategies per-codebase?

[2024-03-15 10:45] Isaac: Yeah! Like a meta-compiler that profiles your code patterns
and auto-tunes optimization passes. We could use RL to learn which passes help most.

[2024-03-16 14:12] Omar: The distributed tracing visualization tool idea is still
bugging me. Current tools can't handle the scale we need.

[2024-03-16 14:30] Isaac: We talked about building a timeline view that can collapse
redundant spans. Could use graph compression algorithms.

[2024-03-20 09:15] Omar: What if we built a natural language query interface for logs?
Instead of regex hell, just ask "show me all authentication failures in the last hour"

[2024-03-20 09:45] Isaac: Love it. Could combine semantic search with structured log parsing.
The challenge is making it fast enough for real-time use.

[2024-03-25 11:20] Omar: Remember that idea about version control for ML model checkpoints?
Git doesn't handle large binary files well.

[2024-03-25 11:40] Isaac: Yeah, we need something that does content-addressable storage
but with awareness of model architecture. Delta compression for weight updates.

[2024-04-02 16:30] Omar: Still want to build that code review bot that learns your team's
style preferences. Not just linting, but opinionated design feedback.

[2024-04-02 16:55] Isaac: Could train it on approved PRs vs requested changes.
The hard part is explaining WHY it suggests changes, not just what to change.

[2024-04-10 13:45] Omar: The self-healing infrastructure idea keeps coming up.
Systems that detect anomalies and auto-remediate without human intervention.

[2024-04-10 14:10] Isaac: We'd need really good anomaly detection + a safe rollback mechanism.
Maybe start with read-only "what would I do" mode before auto-fixing.

[2024-05-01 10:00] Omar: What about a development environment that predicts what you're
about to type based on your codebase context? Like autocomplete but smarter.

[2024-05-01 10:25] Isaac: Context-aware completion using the entire repo as training data.
Could also suggest refactorings when it detects patterns.

[2024-05-15 15:30] Omar: I still think about the API compatibility checker - a tool that
analyzes API changes and predicts what will break for consumers.

[2024-05-15 15:50] Isaac: Static analysis + runtime usage patterns from telemetry.
We could show impact radius before deploying breaking changes.

[2024-06-01 09:30] Omar: Random note - filed expense reports finally.

[2024-06-01 10:15] Isaac: Nice! I need to do mine too.

[2024-06-20 14:00] Omar: The idea about collaborative debugging where multiple devs
can share a debug session in real-time. Like pair programming but for debugging.

[2024-06-20 14:30] Isaac: Could use CRDTs for state synchronization. Main challenge is
keeping breakpoints and watches consistent across clients.

[2024-07-05 11:45] Omar: Been revisiting the config drift detector. Compare actual
infrastructure state vs declared config and highlight divergence.

[2024-07-05 12:10] Isaac: Infrastructure as code but with continuous reconciliation.
Could auto-generate PRs to update config when drift is detected.
"""

def test_rlm_basic():
    """Basic RLM test - let the LLM explore the data programmatically."""

    # Configure DSPy with OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return

    lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
    dspy.configure(lm=lm)

    print("=== Testing RLM with Mock Slack Dump ===\n")

    # Create RLM with explicit parameters
    rlm = dspy.RLM(
        "discussion, request -> ideas: list[str]",
        max_iterations=20,      # Max REPL loops
        max_llm_calls=50,       # Max calls to llm_query/llm_query_batched
        max_output_chars=100000, # Max output size
        verbose=True            # Show what's happening!
    )

    print("RLM Configuration:")
    print(f"  max_iterations: {rlm.max_iterations}")
    print(f"  max_llm_calls: {rlm.max_llm_calls}")
    print(f"  max_output_chars: {rlm.max_output_chars}")
    print(f"  verbose: {rlm.verbose}\n")

    # Run RLM - it will write Python code to chunk and analyze the data
    print("Running RLM (this will show the code it writes)...\n")
    print("="*80)

    output = rlm(
        discussion=MOCK_SLACK_DUMP,
        request="What are the 5 coolest unfinished ideas that Omar & Isaac keep coming back to?"
    )

    print("="*80)
    print("\n=== RESULTS ===\n")
    print("Ideas found:")
    for i, idea in enumerate(output.ideas, 1):
        print(f"{i}. {idea}")

    print("\n=== Trajectory Info ===")
    print(f"Total iterations: {len(output.trajectory)}")
    print(f"\nFinal reasoning: {output.final_reasoning}")


def test_rlm_with_sub_lm():
    """Test RLM with a different sub_lm for cheaper sub-queries."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return

    # Main LM for orchestration
    main_lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
    dspy.configure(lm=main_lm)

    # Cheaper LM for sub-queries (llm_query calls)
    sub_lm = dspy.LM('openai/gpt-3.5-turbo', api_key=api_key)

    print("\n\n=== Testing RLM with Separate sub_lm ===\n")
    print("Main LM (orchestration): gpt-4o-mini")
    print("Sub LM (llm_query calls): gpt-3.5-turbo (cheaper!)\n")

    rlm = dspy.RLM(
        "discussion, request -> summary: str",
        max_iterations=10,
        max_llm_calls=20,
        sub_lm=sub_lm,  # Use cheaper model for sub-queries!
        verbose=False   # Less verbose this time
    )

    output = rlm(
        discussion=MOCK_SLACK_DUMP,
        request="Summarize the key themes in this discussion"
    )

    print("Summary:")
    print(output.summary)


if __name__ == "__main__":
    # Run basic test with verbose output to see how RLM works
    test_rlm_basic()

    # Uncomment to test with separate sub_lm
    # test_rlm_with_sub_lm()
