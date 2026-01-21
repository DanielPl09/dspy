#!/usr/bin/env python3
"""
DSPy Basic Test with Real OpenAI API
"""
import dspy
import os

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set OPENAI_API_KEY environment variable")
    print("Usage: export OPENAI_API_KEY='your-key-here' && python test_openai_basic.py")
    exit(1)

print("=" * 60)
print("Testing DSPy with Real OpenAI API")
print("=" * 60)

# Initialize LM
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm)

# Test 1: Simple Predict
print("\nTest 1: Simple Question")
print("-" * 60)
qa = dspy.Predict("question -> answer")
result = qa(question="What is 2 + 2?")
print(f"Question: What is 2 + 2?")
print(f"Answer: {result.answer}")

# Test 2: Chain of Thought
print("\nTest 2: Chain of Thought Reasoning")
print("-" * 60)
class QA(dspy.Signature):
    """Answer questions with reasoning."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

cot = dspy.ChainOfThought(QA)
result2 = cot(question="What is the capital of France?")
print(f"Question: What is the capital of France?")
if hasattr(result2, 'rationale'):
    print(f"Reasoning: {result2.rationale}")
print(f"Answer: {result2.answer}")

print("\n" + "=" * 60)
print("✓ SUCCESS! DSPy is working with OpenAI API!")
print("=" * 60)
