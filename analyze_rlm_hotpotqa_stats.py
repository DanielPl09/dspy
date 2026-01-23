"""
Analyze RLM performance on HotPotQA multi-hop questions.

This script evaluates whether RLM demonstrates:
1. Iterative refinement over sub-LM calls
2. Early stopping (reaching answers before max iterations/calls)
3. Efficient multi-hop reasoning

Statistics collected:
- Number of iterations used per question
- Number of sub-LM calls made per question
- Success rate (correct answers)
- Distribution of iterations and calls
"""

import os
import json
from collections import defaultdict
from datasets import load_dataset
import dspy


def analyze_rlm_trajectory(trajectory):
    """Extract statistics from RLM trajectory."""
    num_iterations = len(trajectory)

    # Count llm_query calls by looking for "llm_query" in the code
    num_llm_calls = 0
    for step in trajectory:
        code = step.get('code', '')
        # Count single llm_query calls
        num_llm_calls += code.count('llm_query(')
        # Count batched calls - each batch counts as multiple
        if 'llm_query_batched(' in code:
            # This is approximate - would need to parse to get exact count
            # For now, estimate based on context
            num_llm_calls += 2  # Conservative estimate

    return {
        'num_iterations': num_iterations,
        'num_llm_calls': num_llm_calls,
    }


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if not answer:
        return ""
    return str(answer).lower().strip()


def check_answer_correctness(predicted, gold):
    """Check if predicted answer matches gold answer."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    # Exact match
    if pred_norm == gold_norm:
        return True

    # Check if gold is contained in prediction or vice versa
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return True

    return False


def run_rlm_analysis(num_questions=10, max_iterations=15, max_llm_calls=40):
    """Run RLM on HotPotQA questions and collect statistics."""

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it to run this analysis.")
        return None

    print("=" * 80)
    print("RLM Performance Analysis on HotPotQA Multi-Hop Questions")
    print("=" * 80)

    # Configure DSPy
    print("\n1. Configuring DSPy with OpenAI...")
    lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)
    dspy.configure(lm=lm)
    print("✓ Configured with gpt-4o-mini")

    # Load HotPotQA dataset
    print("\n2. Loading HotPotQA dataset...")
    hf_dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
    multihop_examples = [ex for ex in hf_dataset if ex["level"] == "hard"]
    print(f"✓ Loaded {len(multihop_examples)} multi-hop questions")

    # Create RLM instance
    print(f"\n3. Creating RLM instance...")
    print(f"   - max_iterations: {max_iterations}")
    print(f"   - max_llm_calls: {max_llm_calls}")

    rlm = dspy.RLM(
        "question, context -> answer: str",
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        verbose=False
    )
    print("✓ RLM instance created")

    # Run analysis
    print(f"\n4. Running RLM on {num_questions} questions...")
    print("-" * 80)

    results = []
    stats = {
        'iterations': [],
        'llm_calls': [],
        'correct': 0,
        'total': 0,
        'early_stop': 0,  # Stopped before max iterations
        'efficient_calls': 0,  # Used < 75% of max_llm_calls
    }

    for idx, example in enumerate(multihop_examples[:num_questions], 1):
        question = example['question']
        gold_answer = example['answer']

        # Build context from supporting facts
        context_parts = []
        if example.get('context') and example['context'].get('sentences'):
            # Get all context sentences
            sentences = example['context']['sentences']
            titles = example['context']['title']
            for title, sents in zip(titles, sentences):
                for sent in sents:
                    context_parts.append(f"{title}: {sent}")

        context = "\n".join(context_parts[:20])  # Limit to first 20 sentences

        print(f"\nQuestion {idx}/{num_questions}")
        print(f"Q: {question}")
        print(f"Gold Answer: {gold_answer}")

        try:
            # Run RLM
            result = rlm(question=question, context=context)
            predicted_answer = result.answer if hasattr(result, 'answer') else ""

            # Analyze trajectory
            trajectory_stats = analyze_rlm_trajectory(result.trajectory)

            # Check correctness
            is_correct = check_answer_correctness(predicted_answer, gold_answer)

            print(f"Predicted: {predicted_answer}")
            print(f"Correct: {'✓' if is_correct else '✗'}")
            print(f"Iterations used: {trajectory_stats['num_iterations']}/{max_iterations}")
            print(f"LLM calls (est.): {trajectory_stats['num_llm_calls']}/{max_llm_calls}")

            # Update statistics
            stats['iterations'].append(trajectory_stats['num_iterations'])
            stats['llm_calls'].append(trajectory_stats['num_llm_calls'])
            stats['total'] += 1
            if is_correct:
                stats['correct'] += 1
            if trajectory_stats['num_iterations'] < max_iterations:
                stats['early_stop'] += 1
            if trajectory_stats['num_llm_calls'] < max_llm_calls * 0.75:
                stats['efficient_calls'] += 1

            results.append({
                'question': question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'correct': is_correct,
                'iterations': trajectory_stats['num_iterations'],
                'llm_calls': trajectory_stats['num_llm_calls'],
                'trajectory': result.trajectory,
            })

        except Exception as e:
            print(f"Error: {e}")
            stats['total'] += 1

    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)

    if stats['total'] == 0:
        print("No questions were processed.")
        return None

    # Calculate statistics
    avg_iterations = sum(stats['iterations']) / len(stats['iterations']) if stats['iterations'] else 0
    avg_llm_calls = sum(stats['llm_calls']) / len(stats['llm_calls']) if stats['llm_calls'] else 0
    accuracy = (stats['correct'] / stats['total']) * 100
    early_stop_rate = (stats['early_stop'] / stats['total']) * 100
    efficient_rate = (stats['efficient_calls'] / stats['total']) * 100

    print(f"\nOverall Performance:")
    print(f"  Total questions: {stats['total']}")
    print(f"  Correct answers: {stats['correct']} ({accuracy:.1f}%)")

    print(f"\nIterative Refinement Evidence:")
    print(f"  Average iterations used: {avg_iterations:.1f}/{max_iterations}")
    print(f"  Early stopping rate: {early_stop_rate:.1f}% (stopped before max iterations)")
    print(f"  Min iterations: {min(stats['iterations']) if stats['iterations'] else 0}")
    print(f"  Max iterations: {max(stats['iterations']) if stats['iterations'] else 0}")

    print(f"\nSub-LM Call Efficiency:")
    print(f"  Average LLM calls: {avg_llm_calls:.1f}/{max_llm_calls}")
    print(f"  Efficient usage rate: {efficient_rate:.1f}% (used <75% of max calls)")
    print(f"  Min calls: {min(stats['llm_calls']) if stats['llm_calls'] else 0}")
    print(f"  Max calls: {max(stats['llm_calls']) if stats['llm_calls'] else 0}")

    # Distribution analysis
    print(f"\nIteration Distribution:")
    iteration_dist = defaultdict(int)
    for it in stats['iterations']:
        iteration_dist[it] += 1
    for it in sorted(iteration_dist.keys()):
        count = iteration_dist[it]
        bar = '█' * count
        print(f"  {it:2d} iterations: {bar} ({count})")

    print(f"\nKey Insights:")
    if early_stop_rate > 50:
        print(f"  ✓ RLM shows strong early stopping ({early_stop_rate:.1f}% of cases)")
        print(f"    → Demonstrates efficient answer discovery")
    else:
        print(f"  ⚠ Low early stopping rate ({early_stop_rate:.1f}%)")

    if avg_llm_calls < max_llm_calls * 0.5:
        print(f"  ✓ RLM uses sub-LM calls efficiently (avg {avg_llm_calls:.1f}/{max_llm_calls})")
        print(f"    → Shows targeted semantic queries vs. brute force")
    else:
        print(f"  ⚠ High sub-LM call usage (avg {avg_llm_calls:.1f}/{max_llm_calls})")

    if accuracy > 70:
        print(f"  ✓ Strong accuracy on multi-hop questions ({accuracy:.1f}%)")
    elif accuracy > 40:
        print(f"  ~ Moderate accuracy ({accuracy:.1f}%) - may need tuning")
    else:
        print(f"  ⚠ Low accuracy ({accuracy:.1f}%) - needs improvement")

    print("\n" + "=" * 80)

    # Save detailed results
    output_file = "rlm_hotpotqa_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'num_questions': num_questions,
                'max_iterations': max_iterations,
                'max_llm_calls': max_llm_calls,
            },
            'statistics': {
                'accuracy': accuracy,
                'avg_iterations': avg_iterations,
                'avg_llm_calls': avg_llm_calls,
                'early_stop_rate': early_stop_rate,
                'efficient_rate': efficient_rate,
            },
            'results': results,
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    return stats


if __name__ == "__main__":
    # Run analysis on 10 questions
    # Adjust num_questions, max_iterations, max_llm_calls as needed
    run_rlm_analysis(
        num_questions=10,
        max_iterations=15,
        max_llm_calls=40
    )
