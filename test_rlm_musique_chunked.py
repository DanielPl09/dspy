"""
RLM test on MuSiQue with realistic multi-document chunking simulation

Simulates real-world scenario where:
- RLM gets metadata to grep/filter programmatically
- Each paragraph's content requires a separate tool call (simulating external retrieval)
- Forces RLM to selectively retrieve and reason over chunks
"""
from datasets import load_dataset
import dspy
from dspy import RLM, configure
import os

# Configure dspy with OpenAI
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
configure(lm=lm)

# Load 3 MuSiQue validation examples (reduced for testing)
print("Loading MuSiQue dataset...")
data = load_dataset("bdsaglam/musique", split="validation")
questions = data.select(range(3))

print("\n" + "="*80)
print("RUNNING RLM ON MUSIQUE WITH CHUNKED DOCUMENT SIMULATION")
print("="*80 + "\n")

for i, q in enumerate(questions):
    print(f"\n{'='*80}")
    print(f"QUESTION {i+1}/3")
    print(f"{'='*80}")
    print(f"Q: {q['question']}")
    print(f"Gold: {q['answer']}")
    print(f"Documents available: {len(q['paragraphs'])}")

    # Structure data like real-world scenario:
    # 1. Metadata that RLM can grep/filter in code
    paragraph_metadata = [
        {
            "idx": p['idx'],
            "title": p['title'],
            "is_supporting": p['is_supporting']  # Include for post-hoc analysis
        }
        for p in q['paragraphs']
    ]

    # 2. Tool that simulates retrieving paragraph content (like external API/RAG)
    #    Each call simulates a separate LLM context/retrieval
    paragraphs_store = {p['idx']: p['paragraph_text'] for p in q['paragraphs']}

    def get_paragraph(idx: int) -> str:
        """Retrieve paragraph content by index. Each call simulates fetching from external source."""
        if idx not in paragraphs_store:
            return f"Error: Paragraph {idx} not found"
        return paragraphs_store[idx]

    # Initialize RLM with tool for retrieving paragraphs
    print("\nInitializing RLM with paragraph retrieval tool...")
    rlm = RLM(
        "paragraph_metadata, question -> answer",
        max_iterations=20,
        max_llm_calls=40,
        tools={"get_paragraph": get_paragraph},
        verbose=False
    )

    try:
        print("Running RLM...")
        result = rlm(
            paragraph_metadata=paragraph_metadata,
            question=q['question']
        )

        print(f"\n{'='*80}")
        print("RESULT")
        print(f"{'='*80}")
        print(f"RLM Answer: {result.answer}")
        print(f"Gold Answer: {q['answer']}")

        # Analyze which paragraphs RLM explored
        if hasattr(result, 'trajectory'):
            print(f"\nTrajectory length: {len(result.trajectory)} iterations")

            # Count get_paragraph calls
            get_para_calls = []
            for step in result.trajectory:
                code = step.get('code', '')
                # Simple heuristic: count get_paragraph calls in code
                if 'get_paragraph(' in code:
                    get_para_calls.append(step)

            print(f"Steps with get_paragraph calls: {len(get_para_calls)}")

            # Show supporting paragraphs for comparison
            supporting_idxs = [p['idx'] for p in q['paragraphs'] if p['is_supporting']]
            print(f"Gold supporting paragraphs: {supporting_idxs}")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nThis simulation tests whether RLM can:")
print("- Explore paragraph metadata programmatically (titles, indices)")
print("- Selectively retrieve relevant paragraphs via get_paragraph()")
print("- Use llm_query() to extract semantic info from retrieved chunks")
print("- Chain multi-hop reasoning across documents")
