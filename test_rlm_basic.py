#!/usr/bin/env python3
"""Basic test for dspy.RLM() with OpenAI."""

import os
import dspy

def main():
    # Check if OpenAI key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return 1

    print("✓ OpenAI API key found in environment")

    # Configure DSPy with OpenAI
    print("\n1. Configuring DSPy with OpenAI (gpt-4o-mini)...")
    lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)
    dspy.configure(lm=lm)
    print("✓ DSPy configured")

    # Create an RLM instance with a simple signature
    print("\n2. Creating RLM instance...")
    rlm = dspy.RLM(
        "discussion, request -> ideas: list[str]",
        max_iterations=10,
        max_llm_calls=20,
        verbose=True
    )
    print("✓ RLM instance created")

    # Test with a simple query
    print("\n3. Running RLM with a simple query...")
    print("-" * 60)

    discussion = """
    Team Discussion (Jan 2024):

    Alice: We should build a dark mode for the app.
    Bob: What about adding offline support?
    Alice: Great idea! We could also add export to PDF.
    Bob: I've been thinking about real-time collaboration.
    Alice: And we definitely need better search functionality.
    """

    request = "What are the 3 main feature ideas mentioned in this discussion?"

    result = rlm(discussion=discussion, request=request)

    print("-" * 60)
    print("\n4. Results:")
    print(f"Type: {type(result.ideas)}")
    print(f"Number of ideas: {len(result.ideas) if isinstance(result.ideas, list) else 'N/A'}")
    print("\nIdeas extracted:")
    if isinstance(result.ideas, list):
        for i, idea in enumerate(result.ideas, 1):
            print(f"  {i}. {idea}")
    else:
        print(f"  {result.ideas}")

    print("\n✓ RLM test completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
