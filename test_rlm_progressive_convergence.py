"""
Test RLM's progressive convergence on multi-hop questions.

Goal: Prove that RLM progressively discovers facts across iterations,
and the answer emerges FAST enough for async streaming.

Key metrics:
- Which iteration discovered each required fact?
- How many iterations to reach 100% answer?
- Progressive fact accumulation over time

Vision: Async streaming shows "50% there... 75% there... 100% - answer!"
"""

import os
import json
from datasets import load_dataset
import dspy
import re


def chunk_document_strategically(title, content, chunk_size=150):
    """
    Split document into chunks to force sub-LLM decomposition.

    For multi-hop questions, key facts are in different chunks,
    requiring RLM to query multiple times.
    """
    words = content.split()
    chunks = {}

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunk_id = f"{title}_chunk_{i//chunk_size + 1}"
        chunks[chunk_id] = chunk_text

    return chunks


def prepare_chunked_context_strategic(example, chunk_size=150):
    """
    Strategically chunk documents to force multi-hop decomposition.

    Each chunk is small enough that RLM MUST make multiple sub-LM
    queries to find all required facts for multi-hop questions.
    """
    all_chunks = {}

    if example.get('context') and example['context'].get('sentences'):
        sentences = example['context']['sentences']
        titles = example['context']['title']

        for title, sents in zip(titles, sentences):
            content = ' '.join(sents)
            # Split into chunks
            title_chunks = chunk_document_strategically(title, content, chunk_size)
            all_chunks.update(title_chunks)

    return all_chunks


def analyze_progressive_convergence(trajectory, question, gold_answer):
    """
    Analyze how the answer emerged progressively across iterations.

    Returns statistics on:
    - Which iterations found key information
    - Progressive convergence toward answer
    - Fact discovery timeline
    """
    convergence_stats = {
        'iterations_to_answer': len(trajectory),
        'fact_discovery': [],
        'query_pattern': [],
        'progressive_info': []
    }

    # Track what information was gathered at each iteration
    for idx, step in enumerate(trajectory, 1):
        iteration_info = {
            'iteration': idx,
            'reasoning': step.get('reasoning', '')[:100],
            'queries_made': 0,
            'info_gathered': None
        }

        # Count queries
        code = step.get('code', '')
        iteration_info['queries_made'] = code.count('llm_query')

        # Check if meaningful output
        output = step.get('output', '')
        if output and not output.startswith('[Error]') and len(output) > 20:
            iteration_info['info_gathered'] = output[:150]

        # Check if answer was found
        if 'SUBMIT' in code or 'FINAL' in output:
            iteration_info['answer_submitted'] = True
            convergence_stats['iterations_to_answer'] = idx

        convergence_stats['fact_discovery'].append(iteration_info)

    return convergence_stats


def test_progressive_convergence(num_questions=3, chunk_size=150):
    """
    Test RLM with proper chunking and track progressive convergence.

    This shows how fast the answer emerges across iterations,
    proving async streaming would show rapid progress.
    """

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return None

    print("="*80)
    print("RLM PROGRESSIVE CONVERGENCE TEST")
    print("="*80)
    print(f"\nChunking Strategy:")
    print(f"  - Chunk size: ~{chunk_size} words")
    print(f"  - Forces multi-hop: Facts spread across chunks")
    print(f"  - RLM must query multiple chunks to find all facts")

    # Configure
    lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)
    dspy.configure(lm=lm)

    # Load dataset
    print(f"\n📚 Loading HotPotQA...")
    hf_dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
    multihop_examples = [ex for ex in hf_dataset if ex["level"] == "hard"][:num_questions]
    print(f"✓ Loaded {len(multihop_examples)} multi-hop questions")

    # Create RLM
    class ChunkedHotPotQA(dspy.Signature):
        """Answer multi-hop questions using chunked documents.

        Documents are split into small chunks. You MUST query multiple chunks
        via llm_query() to find all facts needed for multi-hop reasoning.
        Don't try to read everything - query strategically for specific facts.
        """
        document_chunks = dspy.InputField(desc="Dictionary of document chunks")
        question = dspy.InputField(desc="Multi-hop question requiring 2+ facts")
        answer = dspy.OutputField(desc="Final answer")

    rlm = dspy.RLM(
        signature=ChunkedHotPotQA,
        max_iterations=10,  # Should finish faster with good chunking
        max_llm_calls=20,   # Fewer calls needed per chunk
        verbose=True
    )

    print(f"\n✓ RLM configured (max_iter=10, max_calls=20)")
    print(f"\n{'='*80}")
    print(f"TESTING PROGRESSIVE CONVERGENCE")
    print(f"{'='*80}\n")

    results = []
    all_convergence_stats = []

    for idx, example in enumerate(multihop_examples, 1):
        question = example['question']
        gold_answer = example['answer']

        # Chunk documents strategically
        chunks = prepare_chunked_context_strategic(example, chunk_size)

        print(f"\n{'─'*80}")
        print(f"Question {idx}: {question}")
        print(f"Gold Answer: {gold_answer}")
        print(f"Document chunks: {len(chunks)}")
        print(f"{'─'*80}\n")

        try:
            # Run RLM
            result = rlm(document_chunks=chunks, question=question)
            predicted = result.answer if hasattr(result, 'answer') else ""

            # Analyze progressive convergence
            convergence = analyze_progressive_convergence(
                result.trajectory, question, gold_answer
            )

            # Show convergence timeline
            print(f"\n📊 CONVERGENCE ANALYSIS:")
            print(f"   Answer found at iteration: {convergence['iterations_to_answer']}/10")
            print(f"\n   Fact Discovery Timeline:")
            for fact in convergence['fact_discovery']:
                if fact.get('info_gathered'):
                    print(f"   ├─ Iteration {fact['iteration']}: {fact['queries_made']} queries")
                    print(f"   │  Info: {fact['info_gathered'][:80]}...")
                if fact.get('answer_submitted'):
                    print(f"   └─ ✅ Answer submitted!")

            # Check correctness
            is_correct = (
                gold_answer.lower() in predicted.lower() or
                predicted.lower() in gold_answer.lower()
            )

            print(f"\n   Predicted: {predicted[:200]}")
            print(f"   Correct: {'✅' if is_correct else '❌'}")

            results.append({
                'question': question,
                'gold': gold_answer,
                'predicted': predicted,
                'correct': is_correct,
                'chunks_available': len(chunks),
                'convergence': convergence
            })
            all_convergence_stats.append(convergence)

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    # Overall statistics
    print(f"\n{'='*80}")
    print("PROGRESSIVE CONVERGENCE STATISTICS")
    print(f"{'='*80}\n")

    if results:
        # Accuracy
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = (correct_count / len(results)) * 100
        print(f"📈 Accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")

        # Convergence speed
        iterations_list = [c['iterations_to_answer'] for c in all_convergence_stats]
        avg_iterations = sum(iterations_list) / len(iterations_list)

        print(f"\n⚡ Convergence Speed:")
        print(f"   Avg iterations to answer: {avg_iterations:.1f}/10")
        print(f"   Range: {min(iterations_list)} - {max(iterations_list)} iterations")
        print(f"   Early finish rate: {sum(1 for i in iterations_list if i < 10)/len(iterations_list)*100:.0f}%")

        # Fact discovery pattern
        print(f"\n🔍 Fact Discovery Pattern:")
        total_queries = 0
        iterations_with_info = 0
        for conv in all_convergence_stats:
            for fact in conv['fact_discovery']:
                total_queries += fact['queries_made']
                if fact.get('info_gathered'):
                    iterations_with_info += 1

        avg_queries_per_question = total_queries / len(results)
        print(f"   Avg queries per question: {avg_queries_per_question:.1f}")
        print(f"   Iterations with useful info: {iterations_with_info}/{sum(len(c['fact_discovery']) for c in all_convergence_stats)}")

        # Progressive convergence visualization
        print(f"\n📊 Progressive Answer Emergence:")
        print(f"   (Shows how fast answers emerge - key for async streaming)\n")
        for i, conv in enumerate(all_convergence_stats, 1):
            q_text = results[i-1]['question'][:50]
            iters = conv['iterations_to_answer']
            pct = (1 - iters/10) * 100  # How much "budget" was saved
            bar = '█' * min(int(pct/5), 20)
            print(f"   Q{i}: {q_text}...")
            print(f"       Converged at iter {iters}/10 | Efficiency: {bar} {pct:.0f}%")

        print(f"\n💡 Async Streaming Implications:")
        if avg_iterations <= 5:
            print(f"   ✅ FAST convergence (avg {avg_iterations:.1f} iterations)")
            print(f"      → Async streaming would show answer in ~{avg_iterations*2:.0f}s")
            print(f"      → Users see progressive facts being discovered")
        else:
            print(f"   ⚠️ Slower convergence (avg {avg_iterations:.1f} iterations)")
            print(f"      → May need better chunking or prompting")

        # Save results
        output_file = "rlm_progressive_convergence_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'num_questions': num_questions,
                    'chunk_size': chunk_size,
                    'max_iterations': 10,
                    'max_llm_calls': 20
                },
                'statistics': {
                    'accuracy': accuracy,
                    'avg_iterations_to_answer': avg_iterations,
                    'avg_queries': avg_queries_per_question,
                },
                'results': results
            }, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    import sys

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)

    print("\n🎬 Testing RLM Progressive Convergence...")
    print("Goal: Prove answer emerges FAST through iterative refinement\n")

    # Test with different chunk sizes
    test_progressive_convergence(num_questions=3, chunk_size=150)
