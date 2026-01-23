#!/usr/bin/env python3
"""Test dspy.RLM with HotPotQA multi-hop reasoning questions.

This test evaluates RLM's ability to use iterative refinement and sub-LLM calls
to answer complex multi-hop questions from the HotPotQA dataset.
"""

import os
import json
import time
from typing import List, Dict, Any
import dspy
from dspy.datasets import HotPotQA


def setup_dspy(model: str = "openai/gpt-4o-mini", temperature: float = 0.0):
    """Initialize DSPy with the specified model."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    lm = dspy.LM(model, temperature=temperature)
    dspy.configure(lm=lm)
    print(f"✓ DSPy configured with {model}")
    return lm


def load_hotpotqa_samples(num_samples: int = 30) -> List[Dict[str, Any]]:
    """Load a subset of HotPotQA dev questions.

    Args:
        num_samples: Number of questions to load (default: 30)

    Returns:
        List of question examples with format: {"question": str, "answer": str}
    """
    print(f"\nLoading {num_samples} HotPotQA examples...")

    # Load dataset - using dev set for testing
    dataset = HotPotQA(
        train_size=0,
        dev_size=num_samples,
        test_size=0,
        only_hard_examples=True
    )

    samples = []
    for example in dataset.dev[:num_samples]:
        samples.append({
            "question": example.question,
            "answer": example.answer,
        })

    print(f"✓ Loaded {len(samples)} HotPotQA examples")
    return samples


def create_rlm_qa_module(max_iterations: int = 15, max_llm_calls: int = 30, verbose: bool = False):
    """Create an RLM module for question answering.

    The signature takes a question and returns an answer using iterative refinement.
    RLM will be able to:
    - Break down multi-hop questions into sub-questions
    - Use sub-LLM calls to gather information
    - Synthesize information across multiple reasoning steps

    Args:
        max_iterations: Maximum reasoning iterations
        max_llm_calls: Maximum sub-LLM calls allowed
        verbose: Whether to show detailed logs

    Returns:
        RLM module configured for QA
    """
    rlm = dspy.RLM(
        "question -> answer: str",
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        verbose=verbose
    )
    return rlm


def evaluate_answer(predicted: str, gold: str) -> bool:
    """Simple exact match evaluation (case-insensitive, stripped).

    Note: This is a simple baseline. HotPotQA typically uses F1 and EM metrics
    with more sophisticated normalization, but this suffices for basic testing.
    """
    pred_normalized = predicted.strip().lower()
    gold_normalized = gold.strip().lower()

    # Exact match
    if pred_normalized == gold_normalized:
        return True

    # Check if gold answer is contained in prediction
    # (since RLM might return more verbose answers)
    if gold_normalized in pred_normalized:
        return True

    # Check if prediction is contained in gold
    # (in case prediction is abbreviated)
    if pred_normalized in gold_normalized:
        return True

    return False


def run_hotpotqa_test(
    samples: List[Dict[str, Any]],
    rlm_module,
    verbose: bool = False,
    save_results: bool = True
):
    """Run RLM on HotPotQA samples and evaluate.

    Args:
        samples: List of HotPotQA examples
        rlm_module: Configured RLM module
        verbose: Print detailed output
        save_results: Save results to JSON file

    Returns:
        Dictionary with evaluation metrics and detailed results
    """
    print(f"\n{'='*70}")
    print(f"Running RLM on {len(samples)} HotPotQA questions...")
    print(f"{'='*70}\n")

    results = []
    correct = 0
    total = 0
    total_time = 0

    for i, example in enumerate(samples, 1):
        question = example["question"]
        gold_answer = example["answer"]

        print(f"\n[{i}/{len(samples)}] Question: {question[:100]}{'...' if len(question) > 100 else ''}")

        try:
            start_time = time.time()
            prediction = rlm_module(question=question)
            elapsed = time.time() - start_time
            total_time += elapsed

            predicted_answer = prediction.answer
            is_correct = evaluate_answer(predicted_answer, gold_answer)

            if is_correct:
                correct += 1
                status = "✓ CORRECT"
            else:
                status = "✗ INCORRECT"

            total += 1

            result_entry = {
                "question_num": i,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "correct": is_correct,
                "time_seconds": elapsed
            }
            results.append(result_entry)

            print(f"  Gold:      {gold_answer}")
            print(f"  Predicted: {predicted_answer}")
            print(f"  {status} (took {elapsed:.2f}s)")
            print(f"  Accuracy so far: {correct}/{total} = {100*correct/total:.1f}%")

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            result_entry = {
                "question_num": i,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": None,
                "correct": False,
                "error": str(e),
                "time_seconds": 0
            }
            results.append(result_entry)
            total += 1

    # Calculate final metrics
    accuracy = 100 * correct / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0

    metrics = {
        "total_questions": total,
        "correct": correct,
        "accuracy_percent": accuracy,
        "total_time_seconds": total_time,
        "avg_time_per_question": avg_time,
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total questions:        {total}")
    print(f"Correct answers:        {correct}")
    print(f"Accuracy:               {accuracy:.2f}%")
    print(f"Total time:             {total_time:.2f}s")
    print(f"Avg time per question:  {avg_time:.2f}s")
    print(f"{'='*70}\n")

    # Save results to file
    if save_results:
        output_file = "test_rlm_hotpotqa_results.json"
        output_data = {
            "metrics": metrics,
            "results": results
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Results saved to {output_file}\n")

    return {
        "metrics": metrics,
        "results": results
    }


def main():
    """Main test execution."""
    print("="*70)
    print("Testing dspy.RLM with HotPotQA Multi-Hop Questions")
    print("="*70)

    # Configuration - tuned for budget awareness
    NUM_QUESTIONS = 25  # 20-30 range as requested
    MAX_ITERATIONS = 12  # Reasonable for multi-hop
    MAX_LLM_CALLS = 25   # Budget-conscious limit
    VERBOSE = False      # Set to True for debugging
    MODEL = "openai/gpt-4o-mini"  # Budget-friendly model

    print(f"\nConfiguration:")
    print(f"  Model:           {MODEL}")
    print(f"  Questions:       {NUM_QUESTIONS}")
    print(f"  Max iterations:  {MAX_ITERATIONS}")
    print(f"  Max LLM calls:   {MAX_LLM_CALLS}")
    print(f"  Verbose:         {VERBOSE}")

    # Setup
    setup_dspy(model=MODEL)

    # Load HotPotQA samples
    samples = load_hotpotqa_samples(num_samples=NUM_QUESTIONS)

    # Show a few examples
    print("\nSample questions:")
    for i, sample in enumerate(samples[:3], 1):
        print(f"\n  {i}. Q: {sample['question']}")
        print(f"     A: {sample['answer']}")

    # Create RLM module
    print(f"\nCreating RLM module...")
    rlm = create_rlm_qa_module(
        max_iterations=MAX_ITERATIONS,
        max_llm_calls=MAX_LLM_CALLS,
        verbose=VERBOSE
    )
    print("✓ RLM module created")

    # Run evaluation
    results = run_hotpotqa_test(
        samples=samples,
        rlm_module=rlm,
        verbose=VERBOSE,
        save_results=True
    )

    # Success criteria
    accuracy = results["metrics"]["accuracy_percent"]
    if accuracy > 20:  # Multi-hop is hard, 20%+ is reasonable for initial test
        print(f"✓ Test PASSED - Accuracy {accuracy:.2f}% > 20%")
        return 0
    else:
        print(f"✗ Test needs improvement - Accuracy {accuracy:.2f}% <= 20%")
        print("  (Note: Multi-hop QA is challenging; this may need tuning)")
        return 1


if __name__ == "__main__":
    exit(main())
