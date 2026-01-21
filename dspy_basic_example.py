#!/usr/bin/env python3
"""
DSPy with Real API Calls - OpenAI Example

This demonstrates how to use DSPy with real OpenAI API calls.
For testing without API calls, use getting_started_dry_run.py instead.

Setup:
    export OPENAI_API_KEY='sk-...'
    python dspy_basic_example.py
"""

import dspy
import os

def main():
    """Main function demonstrating DSPy with real API calls."""

    # Initialize the language model
    # Using OpenAI's GPT-4o-mini for cost-effective API calls
    lm = dspy.LM(model='openai/gpt-4o-mini')

    # Configure DSPy to use this language model globally
    dspy.configure(lm=lm)

    print("=" * 60)
    print("DSPy with Real OpenAI API Calls")
    print("=" * 60)

    # Example 1: Basic Question Answering with Predict
    print("\n1. Basic Question Answering (using Predict):")
    print("-" * 60)

    class BasicQA(dspy.Signature):
        """Answer questions with short factual answers."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    qa = dspy.Predict(BasicQA)
    response = qa(question="What is the capital of France?")
    print(f"Q: What is the capital of France?")
    print(f"A: {response.answer}")

    # Example 2: Chain of Thought Reasoning
    print("\n2. Chain of Thought Reasoning:")
    print("-" * 60)

    cot = dspy.ChainOfThought(BasicQA)
    response = cot(question="What is 15 + 27?")
    print(f"Q: What is 15 + 27?")
    if hasattr(response, 'reasoning'):
        print(f"Reasoning: {response.reasoning}")
    print(f"A: {response.answer}")

    # Example 3: Using string signature (shorthand)
    print("\n3. Using String Signature (shorthand):")
    print("-" * 60)

    summarizer = dspy.Predict("text -> summary")
    text = "DSPy is a framework for programming language models. It provides a way to build modular AI systems."
    result = summarizer(text=text)
    print(f"Text: {text}")
    print(f"Summary: {result.summary}")

    print("\n" + "=" * 60)
    print("✓ All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set!")
        print("\nTo use this script, set your OpenAI API key:")
        print("  export OPENAI_API_KEY='sk-proj-...'")
        print("\nOther supported providers:")
        print("  - Anthropic: lm = dspy.LM('anthropic/claude-3-5-sonnet-20241022')")
        print("  - Google: lm = dspy.LM('google/gemini-pro')")
        print("  - Azure OpenAI: lm = dspy.LM('azure/your-deployment')")
        print("\nFor testing WITHOUT API calls, use: python getting_started_dry_run.py")
        exit(1)

    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nIf you see a network/proxy error, this environment may block external API calls.")
        print("In that case, use the dry run example: python getting_started_dry_run.py")
        exit(1)
