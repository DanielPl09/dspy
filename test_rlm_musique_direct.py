"""
RLM test with direct paragraph access - simpler than tool-based approach

RLM gets structured paragraph data and must:
1. Explore titles/metadata programmatically
2. Use llm_query() on selected paragraph content for semantic analysis
3. Chain multi-hop reasoning
"""
from datasets import load_dataset
import dspy
from dspy import RLM, configure
import os

# Configure
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
configure(lm=lm)

# Load 1 example
data = load_dataset("bdsaglam/musique", split="validation")
q = data[1]  # Zubly Cemetery question

print("="*80)
print("RLM with direct paragraph access")
print("="*80)
print(f"\nQuestion: {q['question']}")
print(f"Gold: {q['answer']}")
print(f"Documents: {len(q['paragraphs'])}")

# Structure paragraphs as list of dicts (RLM can iterate/filter in code)
paragraphs = [
    {
        "idx": p['idx'],
        "title": p['title'],
        "text": p['paragraph_text'][:500]  # Limit text to force llm_query use
    }
    for p in q['paragraphs']
]

print("\nInitializing RLM...")
rlm = RLM(
    "paragraphs, question -> answer",
    max_iterations=15,
    max_llm_calls=25,
    verbose=True
)

print("\n" + "="*80)
print("RUNNING")
print("="*80 + "\n")

try:
    result = rlm(paragraphs=paragraphs, question=q['question'])

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"RLM: {result.answer}")
    print(f"Gold: {q['answer']}")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
