#!/usr/bin/env python3
"""
DSPy Getting Started - Dry Run Example
This uses DummyLM to test DSPy without making real API calls.
"""

import dspy
from dspy.utils.dummies import DummyLM

def basic_example():
    """Most basic DSPy example with DummyLM"""
    print("=" * 60)
    print("Example 1: Basic Question Answering")
    print("=" * 60)

    # Create a DummyLM with predefined answers
    lm = DummyLM([
        {"answer": "Paris"},
        {"answer": "4"},
        {"answer": "Blue"}
    ])

    # Configure DSPy to use this dummy model
    dspy.configure(lm=lm)

    # Define a simple signature for Q&A
    class BasicQA(dspy.Signature):
        """Answer questions with short factual answers."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # Create a predictor
    qa = dspy.Predict(BasicQA)

    # Make predictions
    questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "What color is the sky?"
    ]

    for q in questions:
        response = qa(question=q)
        print(f"Q: {q}")
        print(f"A: {response.answer}\n")


def chain_of_thought_example():
    """Example using ChainOfThought with DummyLM"""
    print("=" * 60)
    print("Example 2: Chain of Thought Reasoning")
    print("=" * 60)

    # DummyLM with reasoning enabled
    lm = DummyLM([
        {"reasoning": "Let me think step by step. France is a country in Europe, and its capital is Paris.", "answer": "Paris"},
        {"reasoning": "Adding 2 + 2 means combining two groups of 2, which equals 4.", "answer": "4"}
    ])

    dspy.configure(lm=lm)

    # Define signature
    class QAWithReasoning(dspy.Signature):
        """Answer questions with reasoning."""
        question: str = dspy.InputField()
        reasoning: str = dspy.OutputField(desc="Think step by step")
        answer: str = dspy.OutputField(desc="Final answer")

    # Create a ChainOfThought predictor
    cot = dspy.ChainOfThought(QAWithReasoning)

    # Make predictions
    questions = [
        "What is the capital of France?",
        "What is 2 + 2?"
    ]

    for q in questions:
        response = cot(question=q)
        print(f"Q: {q}")
        print(f"Reasoning: {response.reasoning}")
        print(f"A: {response.answer}\n")


def dictionary_mode_example():
    """Example using dictionary mode for context-aware responses"""
    print("=" * 60)
    print("Example 3: Context-Aware Responses (Dictionary Mode)")
    print("=" * 60)

    # DummyLM in dictionary mode - matches questions to answers
    # The key should match part of the question text
    lm = DummyLM({
        "France": {"answer": "Paris"},
        "Japan": {"answer": "Tokyo"},
        "sky": {"answer": "Blue"},
        "ocean": {"answer": "Pacific Ocean"}
    })

    dspy.configure(lm=lm)

    class BasicQA(dspy.Signature):
        """Answer questions."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    qa = dspy.Predict(BasicQA)

    questions = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What color is the sky?",
        "What is the largest ocean?"
    ]

    for q in questions:
        response = qa(question=q)
        print(f"Q: {q}")
        print(f"A: {response.answer}\n")


def module_example():
    """Example creating a custom DSPy Module"""
    print("=" * 60)
    print("Example 4: Custom DSPy Module")
    print("=" * 60)

    lm = DummyLM([
        {"summary": "DSPy is a framework for programming language models."},
        {"summary": "Python is a versatile programming language."}
    ])

    dspy.configure(lm=lm)

    # Define a custom module
    class Summarizer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("text -> summary")

        def forward(self, text):
            return self.predictor(text=text)

    # Use the module
    summarizer = Summarizer()

    texts = [
        "DSPy is a framework for programming—rather than prompting—language models. It allows you to iterate fast on building modular AI systems.",
        "Python is a high-level, interpreted programming language known for its simplicity and readability."
    ]

    for text in texts:
        result = summarizer(text=text)
        print(f"Text: {text[:50]}...")
        print(f"Summary: {result.summary}\n")


def main():
    print("\n" + "=" * 60)
    print("DSPy Getting Started - Dry Run Examples")
    print("Using DummyLM (no real API calls needed!)")
    print("=" * 60 + "\n")

    # Run all examples
    basic_example()
    print("\n")

    chain_of_thought_example()
    print("\n")

    dictionary_mode_example()
    print("\n")

    module_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nTo use real LLMs, replace DummyLM with:")
    print("  lm = dspy.LM(model='openai/gpt-4o-mini')  # Requires OPENAI_API_KEY")
    print("  lm = dspy.LM(model='anthropic/claude-3-5-sonnet-20241022')  # Requires ANTHROPIC_API_KEY")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
