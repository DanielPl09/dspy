#!/usr/bin/env python3
"""Basic sanity test for DSPy with OpenAI API key from environment."""

import dspy
import os

def test_basic_openai():
    """Test basic OpenAI integration with DSPy."""

    # Verify API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False

    print("✓ OPENAI_API_KEY found in environment")

    # Configure DSPy with OpenAI
    try:
        lm = dspy.LM('openai/gpt-3.5-turbo', api_key=api_key)
        dspy.configure(lm=lm)
        print("✓ DSPy configured with OpenAI LM")
    except Exception as e:
        print(f"❌ Failed to configure DSPy: {e}")
        return False

    # Simple test query
    try:
        response = lm("Say 'Hello from DSPy!' and nothing else.")
        print(f"✓ API call successful")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

if __name__ == "__main__":
    print("=== DSPy OpenAI Basic Sanity Test ===\n")
    success = test_basic_openai()
    print("\n" + ("="*40))
    if success:
        print("✓ All checks passed!")
        exit(0)
    else:
        print("❌ Test failed!")
        exit(1)
