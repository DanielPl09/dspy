#!/usr/bin/env python3
"""
Basic DSPy Example - Simple API Call
This demonstrates the most vanilla DSPy setup with an API call.
"""

import dspy
import os

# Configure the language model
# You can use OpenAI, Anthropic, or other providers supported by DSPy
# For this example, we'll use OpenAI (make sure to set OPENAI_API_KEY)

def main():
    # Initialize the language model
    # Using OpenAI's GPT-4o-mini for a cost-effective example
    lm = dspy.LM(model='openai/gpt-4o-mini')

    # Configure DSPy to use this language model
    dspy.configure(lm=lm)

    # Create a simple prompt - DSPy uses signatures for structured prompts
    # Basic example: asking a simple question
    class BasicQA(dspy.Signature):
        """Answer questions with short factual answers."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # Create a predictor (this is what makes the API call)
    qa_predictor = dspy.ChainOfThought(BasicQA)

    # Make the API call with a simple question
    question = "What is the capital of France?"
    print(f"\nQuestion: {question}")

    # Execute the prediction (this makes the actual API call)
    response = qa_predictor(question=question)

    print(f"Answer: {response.answer}")
    print(f"\nFull response: {response}")

    # You can also use the simpler Predict module instead of ChainOfThought
    print("\n" + "="*50)
    print("Using simple Predict module:")
    print("="*50)

    simple_predictor = dspy.Predict(BasicQA)
    response2 = simple_predictor(question="What is 2 + 2?")
    print(f"Question: What is 2 + 2?")
    print(f"Answer: {response2.answer}")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set!")
        print("Please set it before running this script:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nAlternatively, you can use other providers like Anthropic:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        print("  and change the model to 'anthropic/claude-3-5-sonnet-20241022'")
    else:
        main()
