"""
Simple test to verify Google Gemini API connection works with DSPy
Tests basic Predict before trying RLM
"""

import dspy

# Configure dspy to use Google Gemini
google_lm = dspy.LM("gemini/gemini-1.5-flash", api_key="AIzaSyAqfZRxJbxPfElPxVi79L9r0zBDCyEaZTc")
dspy.configure(lm=google_lm)

print("=" * 80)
print("Testing basic dspy.Predict with Google Gemini")
print("=" * 80)
print()

# Create a simple signature
class SimpleQA(dspy.Signature):
    """Answer questions briefly."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# Create predictor
predictor = dspy.Predict(SimpleQA)

# Test it
result = predictor(question="What is 2 + 2?")

print("Question: What is 2 + 2?")
print(f"Answer: {result.answer}")
print()
print("✓ Google Gemini API connection works!")
