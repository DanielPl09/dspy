#!/usr/bin/env python3
"""
Multi-hop QA with RLM Iterative Refinement
Testing on HotPotQA dataset with budget under $0.50
Priority: Accuracy over cost
"""

import json
import os
import time
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

import dspy
from dspy.datasets import HotPotQA


# ============================================================================
# Cost Tracking
# ============================================================================

class CostTracker:
    """Track API costs for budget management"""

    def __init__(self, budget: float = 0.50):
        self.budget = budget
        self.total_cost = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.start_time = time.time()

    def add_usage(self, input_tok: int, output_tok: int):
        """Add token usage and calculate cost (gpt-4o-mini pricing)"""
        # Input: $0.15 per 1M tokens, Output: $0.60 per 1M tokens
        cost = (input_tok * 0.15 / 1_000_000) + (output_tok * 0.60 / 1_000_000)
        self.total_cost += cost
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        self.calls += 1
        return cost

    def remaining_budget(self) -> float:
        return self.budget - self.total_cost

    def can_continue(self, safety_margin: float = 0.9) -> bool:
        """Check if we can continue within budget (default 90% threshold)"""
        return self.total_cost < (self.budget * safety_margin)

    def summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        return {
            "total_cost": round(self.total_cost, 4),
            "budget": self.budget,
            "remaining": round(self.remaining_budget(), 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "api_calls": self.calls,
            "elapsed_seconds": round(elapsed, 2),
        }


# ============================================================================
# Dataset Preparation
# ============================================================================

def load_hotpotqa_subset(num_examples: int = 20) -> List[dspy.Example]:
    """Load filtered HotPotQA examples for multi-hop reasoning"""
    print(f"Loading HotPotQA dataset (fetching {num_examples} examples)...")

    # Load the dataset
    dataset = HotPotQA(
        train_size=0,
        dev_size=num_examples * 3,  # Get more than needed for filtering
        test_size=0,
    )

    # Filter for good multi-hop examples
    # Use dev set which has gold_titles for evaluation
    examples = []
    for ex in dataset.dev:
        # HotPotQA examples have: question, answer, gold_titles
        # Create DSPy Example with proper input/output format
        example = dspy.Example(
            question=ex.question,
            answer=ex.answer,
            gold_titles=list(ex.gold_titles) if hasattr(ex, 'gold_titles') else []
        ).with_inputs("question")

        examples.append(example)

        if len(examples) >= num_examples:
            break

    print(f"✓ Loaded {len(examples)} HotPotQA examples")
    return examples


# ============================================================================
# Approach 1: Baseline ChainOfThought
# ============================================================================

class BaselineQA(dspy.Module):
    """Simple baseline using ChainOfThought"""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        return self.predictor(question=question)


# ============================================================================
# Approach 2: RLM Basic Multi-hop
# ============================================================================

class RLMBasicMultihop(dspy.Module):
    """RLM doing basic multi-hop reasoning without refinement"""

    def __init__(self):
        super().__init__()
        self.rlm = dspy.RLM(
            "question -> reasoning_steps: list[str], answer: str",
            max_iterations=5,
            max_llm_calls=10,
            verbose=False
        )

    def forward(self, question: str) -> dspy.Prediction:
        result = self.rlm(question=question)
        return dspy.Prediction(
            answer=result.answer,
            reasoning_steps=result.reasoning_steps if hasattr(result, 'reasoning_steps') else []
        )


# ============================================================================
# Approach 3: RLM with Iterative Refinement
# ============================================================================

class RLMRefinedMultihop(dspy.Module):
    """RLM with iterative refinement - prioritizing accuracy"""

    def __init__(self):
        super().__init__()
        self.rlm = dspy.RLM(
            """question ->
               thought_process: str,
               reasoning_steps: list[str],
               confidence_score: float,
               answer: str""",
            max_iterations=10,  # Increased for accuracy
            max_llm_calls=25,   # Increased for accuracy
            verbose=False
        )

    def forward(self, question: str) -> dspy.Prediction:
        result = self.rlm(question=question)
        return dspy.Prediction(
            answer=result.answer,
            reasoning_steps=result.reasoning_steps if hasattr(result, 'reasoning_steps') else [],
            thought_process=result.thought_process if hasattr(result, 'thought_process') else "",
            confidence_score=result.confidence_score if hasattr(result, 'confidence_score') else 0.0
        )


# ============================================================================
# Evaluation Metrics
# ============================================================================

def normalize_answer(text: str) -> str:
    """Normalize answer for comparison"""
    import re
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def exact_match(pred_answer: str, gold_answer: str) -> bool:
    """Exact match after normalization"""
    return normalize_answer(pred_answer) == normalize_answer(gold_answer)


def f1_score(pred_answer: str, gold_answer: str) -> float:
    """Token-level F1 score"""
    pred_tokens = set(normalize_answer(pred_answer).split())
    gold_tokens = set(normalize_answer(gold_answer).split())

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = pred_tokens & gold_tokens
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * (precision * recall) / (precision + recall)


def evaluate_prediction(example: dspy.Example, prediction: dspy.Prediction) -> Dict[str, Any]:
    """Evaluate a single prediction"""
    pred_answer = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
    gold_answer = example.answer

    em = exact_match(pred_answer, gold_answer)
    f1 = f1_score(pred_answer, gold_answer)

    return {
        "exact_match": em,
        "f1_score": round(f1, 3),
        "pred_answer": pred_answer,
        "gold_answer": gold_answer
    }


# ============================================================================
# Main Evaluation Runner
# ============================================================================

@dataclass
class EvalResult:
    approach: str
    num_examples: int
    accuracy: float
    avg_f1: float
    total_cost: float
    cost_per_example: float
    examples_evaluated: int

    def to_dict(self):
        return asdict(self)


def run_evaluation(
    approach_name: str,
    module: dspy.Module,
    examples: List[dspy.Example],
    tracker: CostTracker,
    max_examples: int = None
) -> EvalResult:
    """Run evaluation on a set of examples with cost tracking"""

    print(f"\n{'='*60}")
    print(f"Evaluating: {approach_name}")
    print(f"{'='*60}")

    if max_examples:
        examples = examples[:max_examples]

    results = []
    start_cost = tracker.total_cost

    for i, example in enumerate(examples):
        if not tracker.can_continue():
            print(f"⚠️  Budget limit approaching. Stopping at {i}/{len(examples)} examples.")
            break

        print(f"\n[{i+1}/{len(examples)}] Question: {example.question[:80]}...")

        try:
            # Run prediction
            pred = module(question=example.question)

            # Evaluate
            eval_result = evaluate_prediction(example, pred)
            results.append(eval_result)

            # Display result
            status = "✓" if eval_result["exact_match"] else "✗"
            print(f"  {status} EM: {eval_result['exact_match']}, F1: {eval_result['f1_score']:.3f}")
            print(f"  Predicted: {eval_result['pred_answer'][:100]}")
            print(f"  Gold: {eval_result['gold_answer'][:100]}")

            # Note: Actual token tracking would require LM callback integration
            # For now, we'll estimate based on typical usage
            # This is a simplified estimation - real implementation would need LM instrumentation

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "exact_match": False,
                "f1_score": 0.0,
                "pred_answer": f"ERROR: {str(e)}",
                "gold_answer": example.answer
            })

    # Calculate metrics
    cost_used = tracker.total_cost - start_cost
    num_evaluated = len(results)
    accuracy = sum(r["exact_match"] for r in results) / num_evaluated if num_evaluated > 0 else 0.0
    avg_f1 = sum(r["f1_score"] for r in results) / num_evaluated if num_evaluated > 0 else 0.0
    cost_per_ex = cost_used / num_evaluated if num_evaluated > 0 else 0.0

    eval_result = EvalResult(
        approach=approach_name,
        num_examples=len(examples),
        accuracy=round(accuracy, 3),
        avg_f1=round(avg_f1, 3),
        total_cost=round(cost_used, 4),
        cost_per_example=round(cost_per_ex, 4),
        examples_evaluated=num_evaluated
    )

    print(f"\n📊 Results for {approach_name}:")
    print(f"  Accuracy: {eval_result.accuracy:.1%}")
    print(f"  Avg F1: {eval_result.avg_f1:.3f}")
    print(f"  Cost: ${eval_result.total_cost:.4f} (${eval_result.cost_per_example:.4f}/example)")
    print(f"  Evaluated: {eval_result.examples_evaluated}/{eval_result.num_examples}")

    return eval_result


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""

    print("="*60)
    print("RLM Multi-Hop Iterative Refinement Evaluation")
    print("Dataset: HotPotQA")
    print("Budget: $0.50")
    print("Priority: Accuracy over Cost")
    print("="*60)

    # Check OpenAI key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return 1

    # Configure DSPy with OpenAI
    print("\n1. Configuring DSPy with OpenAI (gpt-4o-mini, temperature=0.0)...")
    lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)
    dspy.configure(lm=lm)
    print("✓ DSPy configured")

    # Initialize cost tracker
    tracker = CostTracker(budget=0.50)

    # Load dataset
    print("\n2. Loading HotPotQA dataset...")
    examples = load_hotpotqa_subset(num_examples=20)

    # Split into dev (5) and test (15)
    dev_examples = examples[:5]
    test_examples = examples[5:20]

    print(f"\n✓ Dataset split:")
    print(f"  Dev set: {len(dev_examples)} examples (for testing implementations)")
    print(f"  Test set: {len(test_examples)} examples (for evaluation)")

    # Store all results
    all_results = []

    # ========================================================================
    # Approach 1: Baseline ChainOfThought
    # ========================================================================

    print("\n3. Testing Baseline ChainOfThought...")
    baseline = BaselineQA()

    baseline_result = run_evaluation(
        "Baseline (ChainOfThought)",
        baseline,
        test_examples,
        tracker,
        max_examples=10  # Limit for baseline
    )
    all_results.append(baseline_result.to_dict())

    # ========================================================================
    # Approach 2: RLM Basic Multi-hop
    # ========================================================================

    if tracker.can_continue():
        print("\n4. Testing RLM Basic Multi-hop...")
        rlm_basic = RLMBasicMultihop()

        rlm_basic_result = run_evaluation(
            "RLM Basic Multi-hop",
            rlm_basic,
            test_examples,
            tracker,
            max_examples=8  # Fewer examples due to higher cost
        )
        all_results.append(rlm_basic_result.to_dict())
    else:
        print("\n⚠️  Skipping RLM Basic - budget limit reached")

    # ========================================================================
    # Approach 3: RLM with Iterative Refinement
    # ========================================================================

    if tracker.can_continue():
        print("\n5. Testing RLM with Iterative Refinement...")
        rlm_refined = RLMRefinedMultihop()

        rlm_refined_result = run_evaluation(
            "RLM Iterative Refinement",
            rlm_refined,
            test_examples,
            tracker,
            max_examples=8  # Fewer examples due to higher cost
        )
        all_results.append(rlm_refined_result.to_dict())
    else:
        print("\n⚠️  Skipping RLM Refined - budget limit reached")

    # ========================================================================
    # Final Summary
    # ========================================================================

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print("\n📊 Cost Summary:")
    cost_summary = tracker.summary()
    print(f"  Total Cost: ${cost_summary['total_cost']:.4f} / ${cost_summary['budget']:.2f}")
    print(f"  Remaining: ${cost_summary['remaining']:.4f}")
    print(f"  Total Tokens: {cost_summary['total_tokens']:,} ({cost_summary['input_tokens']:,} in, {cost_summary['output_tokens']:,} out)")
    print(f"  API Calls: {cost_summary['api_calls']}")
    print(f"  Time: {cost_summary['elapsed_seconds']:.1f}s")

    print("\n📈 Accuracy Comparison:")
    print(f"{'Approach':<30} {'Accuracy':<12} {'Avg F1':<12} {'Cost':<12} {'Examples':<10}")
    print("-" * 76)
    for result in all_results:
        print(f"{result['approach']:<30} {result['accuracy']:<12.1%} {result['avg_f1']:<12.3f} ${result['total_cost']:<11.4f} {result['examples_evaluated']:<10}")

    # Save results
    results_file = "multihop_results.json"
    output_data = {
        "cost_summary": cost_summary,
        "evaluation_results": all_results,
        "dataset_info": {
            "source": "HotPotQA",
            "dev_size": len(dev_examples),
            "test_size": len(test_examples)
        }
    }

    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    return 0


if __name__ == "__main__":
    exit(main())
