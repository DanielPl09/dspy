"""
Diagnostic RLM test - verbose output to see how RLM explores chunked documents
"""
from datasets import load_dataset
import dspy
from dspy import RLM, configure
import os

# Configure dspy
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
configure(lm=lm)

# Load just 1 example
data = load_dataset("bdsaglam/musique", split="validation")
q = data[1]  # The Zubly Cemetery question

print("="*80)
print("DIAGNOSTIC: RLM exploring chunked documents")
print("="*80)
print(f"\nQuestion: {q['question']}")
print(f"Gold answer: {q['answer']}")
print(f"\nQuestion decomposition:")
for i, sub_q in enumerate(q['question_decomposition'], 1):
    print(f"  {i}. {sub_q['question']} → {sub_q['answer']}")

print(f"\nAvailable documents: {len(q['paragraphs'])}")
print("\nSupporting paragraphs:")
for p in q['paragraphs']:
    if p['is_supporting']:
        print(f"  [{p['idx']}] {p['title']}")
        print(f"      {p['paragraph_text'][:100]}...")

# Setup metadata and retrieval tool
paragraph_metadata = [
    {"idx": p['idx'], "title": p['title']}
    for p in q['paragraphs']
]

paragraphs_store = {p['idx']: p['paragraph_text'] for p in q['paragraphs']}

def get_paragraph(idx: int) -> str:
    """Retrieve paragraph content by index."""
    print(f"  → TOOL CALLED: get_paragraph({idx})")  # Track calls
    if idx not in paragraphs_store:
        return f"Error: Paragraph {idx} not found"
    return paragraphs_store[idx]

# Initialize RLM with verbose output
print("\n" + "="*80)
print("RUNNING RLM (VERBOSE)")
print("="*80 + "\n")

rlm = RLM(
    "paragraph_metadata, question -> answer",
    max_iterations=10,  # Reduced for diagnostics
    max_llm_calls=20,
    tools={"get_paragraph": get_paragraph},
    verbose=True  # See what RLM is thinking
)

try:
    result = rlm(
        paragraph_metadata=paragraph_metadata,
        question=q['question']
    )

    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    print(f"RLM Answer: {result.answer}")
    print(f"Gold Answer: {q['answer']}")

    # Show full trajectory
    print("\n" + "="*80)
    print("FULL TRAJECTORY")
    print("="*80)
    if hasattr(result, 'trajectory'):
        for i, step in enumerate(result.trajectory, 1):
            print(f"\n--- Iteration {i} ---")
            print(f"Reasoning: {step.get('reasoning', 'N/A')[:200]}...")
            print(f"Code:\n{step.get('code', 'N/A')}")
            print(f"Output: {step.get('output', 'N/A')[:300]}...")

except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
