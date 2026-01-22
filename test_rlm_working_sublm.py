#!/usr/bin/env python3
"""
Test RLM with WORKING sub-LLM calls - simplified to ensure success.
"""

import dspy
import os

os.environ['PATH'] = f"/root/.deno/bin:{os.environ.get('PATH', '')}"

def test_working_sublm():
    """Test with smaller documents to ensure llm_query works."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
    dspy.configure(lm=lm)

    print("="*80)
    print("RLM Test with Working Sub-LLM Calls")
    print("="*80)

    # Smaller, distinct documents
    context = """You have 3 separate documents. Analyze each one separately using llm_query().

=== DOCUMENT 1: Movie ===
The movie 'Inception' was directed by Christopher Nolan in 2010.

=== DOCUMENT 2: Director ===
Christopher Nolan was born on July 30, 1970 in London, England.

=== DOCUMENT 3: Other ===
Christopher Nolan is known for complex, non-linear storytelling.
"""

    question = "What year was the director of Inception born?"

    print(f"\nQuestion: {question}\n")
    print("Running RLM with verbose=True...")
    print("="*80)

    rlm = dspy.RLM(
        signature="context, question -> answer: str",
        max_iterations=15,
        max_llm_calls=10,
        verbose=True,
    )

    result = rlm(context=context, question=question)

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Answer: {result.answer}")
    print(f"Iterations: {len(result.trajectory)}")

    # Count llm_query usage
    total_llm_query = 0
    for i, step in enumerate(result.trajectory, 1):
        code = step.get('code', '')
        output = step.get('output', '')

        calls = code.count('llm_query(')
        if calls > 0:
            total_llm_query += calls
            print(f"\nIteration {i}: {calls} llm_query() calls")
            print(f"  Output: {output[:200] if output else '(no output)'}...")

    print(f"\n{'='*80}")
    print(f"Total llm_query() calls: {total_llm_query}")
    print(f"{'='*80}")

    return result

if __name__ == "__main__":
    test_working_sublm()
