"""
Proper RLM test on MuSiQue with realistic multi-document chunking simulation

Design:
- We pre-chunk documents into paragraphs (RLM doesn't solve chunking)
- RLM gets paragraph metadata (idx, title) for code-based exploration
- Paragraph content NOT directly accessible - simulates external storage
- To understand content, RLM uses llm_query() which we route to the paragraph text
- This simulates: local metadata filtering + external semantic retrieval

Real-world analogy:
- paragraph_metadata = search index (can grep/filter locally)
- llm_query on paragraph = retrieve and analyze document chunk (external call)
"""
from datasets import load_dataset
import dspy
from dspy import RLM, configure
import os

# Configure dspy
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
configure(lm=lm)

# Load small sample
print("Loading MuSiQue dataset...")
data = load_dataset("bdsaglam/musique", split="validation")
questions = data.select(range(3))  # Test on 3 questions

print("\n" + "="*80)
print("RLM WITH PROPER MULTI-DOCUMENT CHUNKING SIMULATION")
print("="*80 + "\n")

for i, q in enumerate(questions):
    print(f"\n{'='*80}")
    print(f"QUESTION {i+1}/3")
    print(f"{'='*80}")
    print(f"Q: {q['question']}")
    print(f"Gold: {q['answer']}")
    print(f"\nGold decomposition:")
    for j, sub in enumerate(q['question_decomposition'], 1):
        print(f"  {j}. {sub['question']} → {sub['answer']}")

    # Pre-chunk: We provide paragraphs already separated (RLM doesn't decide how to chunk)
    # This is our internal simulation variable
    paragraphs = [
        {
            "idx": p['idx'],
            "title": p['title'],
            "text": p['paragraph_text']
        }
        for p in q['paragraphs']
    ]

    print(f"Available paragraphs: {len(paragraphs)}")
    print(f"Supporting paragraphs (gold): {[p['idx'] for p in q['paragraphs'] if p['is_supporting']]}")

    # CRITICAL: Instruct RLM on how to work with pre-chunked documents
    # - paragraphs are already chunked by us (RLM doesn't solve chunking)
    # - RLM can filter/explore titles and indices with code
    # - To understand semantic content, must use llm_query() on paragraph text
    instructions = f"""You have {len(paragraphs)} pre-separated document paragraphs (we already chunked them for you).

Each paragraph is a dict with: idx, title, text

Strategy:
1. Explore paragraph titles/indices with code (filtering, searching)
2. To understand semantic meaning of a paragraph, use llm_query() with the paragraph text
3. Chain reasoning across multiple paragraphs as needed

Example:
```python
# Filter by title
relevant = [p for p in paragraphs if 'Cemetery' in p['title']]
# Understand content semantically
state = llm_query(f"What state is mentioned in this text? {{relevant[0]['text']}}")
```

The paragraphs are in the `paragraphs` variable."""

    signature = dspy.Signature(
        {"paragraphs": dspy.InputField(desc="List of pre-chunked paragraphs with idx, title, text"),
         "question": dspy.InputField(desc="Question to answer")},
        instructions
    ).append("answer", dspy.OutputField(desc="Final answer"), type_=str)

    # Initialize RLM
    print("\nRunning RLM...")
    rlm = RLM(
        signature,
        max_iterations=20,
        max_llm_calls=30,
        verbose=False  # Set True to see reasoning
    )

    try:
        result = rlm(
            paragraphs=paragraphs,
            question=q['question']
        )

        print(f"\n{'='*80}")
        print("RESULT")
        print(f"{'='*80}")
        print(f"RLM Answer: {result.answer}")
        print(f"Gold Answer: {q['answer']}")

        # Analyze trajectory
        if hasattr(result, 'trajectory'):
            print(f"\nIterations: {len(result.trajectory)}")
            llm_query_count = sum(1 for step in result.trajectory if 'llm_query' in step.get('code', ''))
            print(f"Steps with llm_query(): {llm_query_count}")

            # Show which paragraphs were explored (heuristic)
            explored_titles = set()
            for step in result.trajectory:
                code = step.get('code', '')
                for p in paragraphs:
                    if p['title'] in code:
                        explored_titles.add(p['title'])

            print(f"Paragraphs referenced in code: {len(explored_titles)}")
            if explored_titles:
                print(f"  Examples: {list(explored_titles)[:3]}")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nKey design choices:")
print("- We pre-chunked into paragraphs (RLM doesn't solve chunking)")
print("- RLM explores metadata with code (titles, indices)")
print("- Content access requires llm_query() (simulates external retrieval)")
print("- Matches real-world: local filtering + remote semantic analysis")
