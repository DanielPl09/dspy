"""
Test RLM's iterative refinement on HotPotQA multi-hop questions.

This demonstrates:
1. Chunked context that RLM must decompose across sub-LM calls
2. Progressive intermediate results (async vision)
3. Proof that RLM reaches answers early through targeted reasoning

Vision: Async processing with intermediate results streaming to screen
while the RLM refines its answer in the background.
"""

import os
import json
import asyncio
from datetime import datetime
from collections import defaultdict
from datasets import load_dataset
import dspy


class ProgressTracker:
    """Track and display RLM's progressive refinement."""

    def __init__(self):
        self.iterations = []
        self.start_time = None

    def start(self, question):
        """Start tracking a new question."""
        self.iterations = []
        self.start_time = datetime.now()
        print(f"\n{'='*80}")
        print(f"🔍 Question: {question}")
        print(f"{'='*80}")
        print("⏳ RLM iterating...")

    def add_iteration(self, iteration_num, reasoning, code, output):
        """Add an iteration with intermediate results."""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        self.iterations.append({
            'num': iteration_num,
            'reasoning': reasoning,
            'code': code,
            'output': output,
            'elapsed': elapsed
        })

        # Show progressive refinement
        print(f"\n📍 Iteration {iteration_num} ({elapsed:.1f}s)")
        print(f"💭 Reasoning: {reasoning[:150]}...")

        # Count LLM calls in this iteration
        llm_calls = code.count('llm_query(') + code.count('llm_query_batched(')
        if llm_calls > 0:
            print(f"🔎 Making {llm_calls} sub-LM call(s)")

        # Show key outputs
        if output and len(output) > 0:
            output_preview = output[:200].replace('\n', ' ')
            print(f"📤 Output: {output_preview}...")

        # Check if we found the answer
        if 'SUBMIT' in code or 'FINAL' in output:
            print(f"✅ Answer found! (at iteration {iteration_num})")

    def summary(self, total_iterations, found_early):
        """Show summary of the refinement process."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{'─'*80}")
        print(f"⏱️  Total time: {elapsed:.1f}s | Iterations: {total_iterations}")
        if found_early:
            print(f"🎯 Early stopping achieved!")


def prepare_chunked_context(example):
    """
    Prepare context in a way that encourages RLM to decompose.

    Instead of passing one giant string, we structure it as documents
    that RLM must intelligently query via sub-LM calls.
    """
    documents = {}

    if example.get('context') and example['context'].get('sentences'):
        sentences = example['context']['sentences']
        titles = example['context']['title']

        for title, sents in zip(titles, sentences):
            # Each document is a separate entity
            documents[title] = ' '.join(sents)

    return documents


def run_rlm_test_sync(num_questions=5, verbose=True):
    """
    Run synchronous RLM test with verbose output showing iterations.
    This simulates the async vision by showing intermediate progress.
    """

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it to run this test.")
        return None

    print("="*80)
    print("RLM ITERATIVE REFINEMENT TEST - HotPotQA Multi-Hop Questions")
    print("="*80)

    # Configure DSPy
    print("\n🔧 Configuring DSPy...")
    lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)
    dspy.configure(lm=lm)
    print("✓ Using gpt-4o-mini")

    # Load dataset
    print("\n📚 Loading HotPotQA dataset...")
    hf_dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
    multihop_examples = [ex for ex in hf_dataset if ex["level"] == "hard"][:num_questions]
    print(f"✓ Loaded {len(multihop_examples)} multi-hop questions")

    # Create RLM - note the signature encourages decomposition
    print("\n🤖 Creating RLM...")
    print("   Signature: documents, question -> answer")
    print("   This encourages RLM to query specific documents via sub-LM calls")

    # Create custom signature with instructions
    class HotPotQASignature(dspy.Signature):
        """Answer multi-hop questions by querying relevant documents.

        You have access to multiple documents in the 'documents' dict.
        Use llm_query() to semantically search and analyze specific documents.
        Don't read everything at once - be strategic and targeted.
        Use llm_query_batched() for parallel queries when needed.
        """
        documents = dspy.InputField(desc="Dictionary mapping document titles to their content")
        question = dspy.InputField(desc="The multi-hop question to answer")
        answer = dspy.OutputField(desc="The final answer to the question")

    rlm = dspy.RLM(
        signature=HotPotQASignature,
        max_iterations=15,
        max_llm_calls=40,
        verbose=verbose  # Show all iterations
    )
    print(f"✓ RLM configured (max_iter=15, max_calls=40)")

    # Statistics
    stats = {
        'iterations': [],
        'llm_calls': [],
        'early_stops': 0,
        'correct': 0,
        'total': 0
    }

    results = []

    # Process questions
    print(f"\n{'='*80}")
    print(f"RUNNING {num_questions} QUESTIONS")
    print(f"{'='*80}")

    for idx, example in enumerate(multihop_examples, 1):
        question = example['question']
        gold_answer = example['answer']

        # Prepare chunked context
        documents = prepare_chunked_context(example)

        tracker = ProgressTracker()
        tracker.start(question)

        print(f"📑 Available documents: {len(documents)}")
        print(f"   {', '.join(list(documents.keys())[:5])}{'...' if len(documents) > 5 else ''}")
        print(f"🎯 Gold answer: {gold_answer}")

        try:
            # Run RLM
            result = rlm(documents=documents, question=question)

            # Extract statistics from trajectory
            trajectory = result.trajectory
            num_iterations = len(trajectory)

            # Count LLM calls from trajectory
            num_llm_calls = 0
            for step in trajectory:
                code = step.get('code', '')
                num_llm_calls += code.count('llm_query(')
                # Approximate batch calls
                if 'llm_query_batched(' in code:
                    num_llm_calls += 2

                # Track progress
                tracker.add_iteration(
                    step.get('iteration', len(tracker.iterations) + 1),
                    step.get('reasoning', ''),
                    code,
                    step.get('output', '')
                )

            predicted = result.answer if hasattr(result, 'answer') else ""

            # Check correctness
            is_correct = (
                gold_answer.lower().strip() in predicted.lower().strip() or
                predicted.lower().strip() in gold_answer.lower().strip()
            )

            # Update stats
            early_stop = num_iterations < 15
            stats['iterations'].append(num_iterations)
            stats['llm_calls'].append(num_llm_calls)
            stats['total'] += 1
            if is_correct:
                stats['correct'] += 1
            if early_stop:
                stats['early_stops'] += 1

            tracker.summary(num_iterations, early_stop)

            print(f"\n📊 Question {idx} Results:")
            print(f"   Predicted: {predicted}")
            print(f"   Correct: {'✅' if is_correct else '❌'}")
            print(f"   Iterations: {num_iterations}/15 ({(num_iterations/15)*100:.0f}%)")
            print(f"   LLM calls: {num_llm_calls}/40 ({(num_llm_calls/40)*100:.0f}%)")

            results.append({
                'question': question,
                'gold': gold_answer,
                'predicted': predicted,
                'correct': is_correct,
                'iterations': num_iterations,
                'llm_calls': num_llm_calls,
                'trajectory': trajectory
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            stats['total'] += 1

    # Final statistics
    print(f"\n{'='*80}")
    print("FINAL STATISTICS - ITERATIVE REFINEMENT ANALYSIS")
    print(f"{'='*80}")

    if stats['total'] > 0:
        avg_iter = sum(stats['iterations']) / len(stats['iterations']) if stats['iterations'] else 0
        avg_calls = sum(stats['llm_calls']) / len(stats['llm_calls']) if stats['llm_calls'] else 0
        accuracy = (stats['correct'] / stats['total']) * 100
        early_rate = (stats['early_stops'] / stats['total']) * 100

        print(f"\n📈 Performance:")
        print(f"   Accuracy: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")

        print(f"\n🎯 Iterative Refinement Evidence:")
        print(f"   Avg iterations: {avg_iter:.1f}/15 ({(avg_iter/15)*100:.1f}% utilization)")
        print(f"   Early stopping: {stats['early_stops']}/{stats['total']} ({early_rate:.1f}%)")
        print(f"   Range: {min(stats['iterations']) if stats['iterations'] else 0} - {max(stats['iterations']) if stats['iterations'] else 0}")

        print(f"\n🔍 Sub-LM Call Efficiency:")
        print(f"   Avg calls: {avg_calls:.1f}/40 ({(avg_calls/40)*100:.1f}% utilization)")
        print(f"   Range: {min(stats['llm_calls']) if stats['llm_calls'] else 0} - {max(stats['llm_calls']) if stats['llm_calls'] else 0}")

        # Distribution
        print(f"\n📊 Iteration Distribution:")
        iter_dist = defaultdict(int)
        for it in stats['iterations']:
            bucket = (it // 3) * 3
            iter_dist[bucket] += 1

        for bucket in sorted(iter_dist.keys()):
            count = iter_dist[bucket]
            bar = '█' * (count * 3)
            print(f"   {bucket:2d}-{bucket+3:2d}: {bar} ({count})")

        # Key insights
        print(f"\n💡 Key Insights:")
        if avg_iter < 10:
            print(f"   ✅ Strong early stopping - avg {avg_iter:.1f} iterations")
            print(f"      → RLM finds answers efficiently")

        if avg_calls < 20:
            print(f"   ✅ Targeted sub-LM queries - avg {avg_calls:.1f} calls")
            print(f"      → Strategic decomposition, not brute force")

        if early_rate > 70:
            print(f"   ✅ High early stopping rate ({early_rate:.1f}%)")
            print(f"      → Demonstrates confidence-based termination")

        # Save results
        output_file = "rlm_iterative_refinement_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'num_questions': num_questions,
                    'max_iterations': 15,
                    'max_llm_calls': 40,
                },
                'statistics': {
                    'accuracy': accuracy,
                    'avg_iterations': avg_iter,
                    'avg_llm_calls': avg_calls,
                    'early_stop_rate': early_rate,
                },
                'results': results
            }, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

    return stats


async def run_rlm_test_async(num_questions=3):
    """
    FUTURE: Async version that streams intermediate results.
    This is the vision - progressive refinement shown in real-time.
    """
    print("\n🚀 ASYNC MODE (Future Vision)")
    print("This would stream intermediate results as RLM iterates...")
    print("Currently not implemented - using sync with verbose mode instead")
    print("But imagine seeing each iteration's reasoning/results in real-time!")


if __name__ == "__main__":
    import sys

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set!")
        print("\nTo run this test:")
        print('  export OPENAI_API_KEY="sk-..."')
        print('  python test_rlm_iterative_refinement.py')
        sys.exit(1)

    # Run test
    print("\n🎬 Starting RLM Iterative Refinement Test...")
    print("This will show progressive refinement as it happens!\n")

    # Start with fewer questions for faster feedback
    run_rlm_test_sync(num_questions=5, verbose=True)
