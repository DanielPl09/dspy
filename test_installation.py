#!/usr/bin/env python3
"""
Test that all required packages are installed correctly
"""

print("Checking DSPy installation...\n")

# 1. Check DSPy import
try:
    import dspy
    print("✓ dspy imported successfully")
    print(f"  Version: {dspy.__version__ if hasattr(dspy, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"✗ Failed to import dspy: {e}")
    exit(1)

# 2. Check LiteLLM (DSPy's LLM abstraction layer)
try:
    import litellm
    print("✓ litellm imported successfully")
    try:
        print(f"  Version: {litellm.version}")
    except:
        print("  Version: installed")
except ImportError as e:
    print(f"✗ Failed to import litellm: {e}")
    exit(1)

# 3. Check OpenAI SDK
try:
    import openai
    print("✓ openai imported successfully")
    print(f"  Version: {openai.__version__}")
except ImportError as e:
    print(f"✗ Failed to import openai: {e}")
    exit(1)

# 4. Test DSPy with DummyLM (no API needed)
print("\n" + "="*60)
print("Testing DSPy with DummyLM (no API key needed)...")
print("="*60)

from dspy.utils.dummies import DummyLM

lm = DummyLM([{"answer": "Paris"}])
dspy.configure(lm=lm)

qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print(f"Q: What is the capital of France?")
print(f"A: {result.answer}")
print("✓ DSPy is working correctly!")

# 5. Show how to use with real API
print("\n" + "="*60)
print("To use with real OpenAI API:")
print("="*60)
print("""
import os
os.environ['OPENAI_API_KEY'] = 'sk-proj-...'

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print(result.answer)
""")

print("\n✓ All packages installed correctly!")
print("  - No need to separately 'pip install openai'")
print("  - DSPy uses LiteLLM which handles all LLM providers")
print("  - Just set your API key and call dspy.LM()")
