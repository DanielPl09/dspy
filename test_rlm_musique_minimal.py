"""
Minimal RLM test on MuSiQue dataset - fail fast verification
"""
from datasets import load_dataset
import dspy
from dspy import RLM, configure
import os

# Configure dspy with OpenAI
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
configure(lm=lm)

# Load 10 MuSiQue validation examples
print("Loading MuSiQue dataset...")
data = load_dataset("bdsaglam/musique", split="validation")
questions = data.select(range(10))

# Initialize RLM with a signature for question answering
print("Initializing RLM for question answering...")
rlm = RLM("question -> answer", max_iterations=15, max_llm_calls=30)

# Run minimal test
print("\n" + "="*80)
print("RUNNING RLM ON 10 MUSIQUE QUESTIONS")
print("="*80 + "\n")

for i, q in enumerate(questions):
    print(f"\n[{i+1}/10]")
    print(f"Q: {q['question'][:80]}...")

    try:
        result = rlm(question=q['question'])
        print(f"A: {result.answer}")
        print(f"Gold: {q['answer']}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

    print("---")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
