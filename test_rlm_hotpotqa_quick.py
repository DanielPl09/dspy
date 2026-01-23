#!/usr/bin/env python3
"""Quick test with 3 HotPotQA questions to verify multi-hop reasoning works."""

import os
import dspy
from dspy.datasets import HotPotQA

# Check API key
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set")
    print("Set it with: export OPENAI_API_KEY='your-key'")
    exit(1)

print("="*70)
print("Quick HotPotQA Multi-Hop Test (3 questions, verbose)")
print("="*70)

# Setup
lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)
dspy.configure(lm=lm)
print("\n✓ DSPy configured with gpt-4o-mini")

# Load 3 questions
print("\n✓ Loading 3 HotPotQA questions...")
dataset = HotPotQA(train_size=0, dev_size=3, test_size=0)

# Create RLM with verbose mode to see the reasoning
print("\n✓ Creating RLM (verbose=True to see multi-hop reasoning)...")
rlm = dspy.RLM(
    "question -> answer: str",
    max_iterations=12,
    max_llm_calls=20,
    verbose=True  # See what's happening!
)

# Test each question
for i, example in enumerate(dataset.dev, 1):
    print(f"\n{'='*70}")
    print(f"Question {i}/3")
    print(f"{'='*70}")
    print(f"Q: {example.question}")
    print(f"Gold Answer: {example.answer}")
    print(f"\n{'-'*70}")
    print("RLM Processing (watch for multi-hop reasoning):")
    print(f"{'-'*70}\n")

    try:
        result = rlm(question=example.question)

        print(f"\n{'-'*70}")
        print(f"RLM Answer: {result.answer}")
        print(f"Gold Answer: {example.answer}")

        # Check if correct
        pred = result.answer.strip().lower()
        gold = example.answer.strip().lower()
        if pred == gold or gold in pred or pred in gold:
            print("✓ CORRECT!")
        else:
            print("✗ Incorrect")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("Test complete! Check the verbose output above to see:")
print("  - How RLM breaks down multi-hop questions")
print("  - Sub-LLM queries being made")
print("  - Iterative refinement in action")
print(f"{'='*70}\n")
