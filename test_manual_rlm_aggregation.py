#!/usr/bin/env python3
"""
Manual RLM-style implementation showing ACTUAL sub-LLM aggregation.

This demonstrates:
1. Splitting context into relevant chunks
2. Processing each chunk with separate sub-LLM calls
3. Iterative refinement of intermediate results
4. Final aggregation with statistics

Budget: < $0.50
"""

import dspy
import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass


# ============================================================================
# Multi-hop Dataset with Explicitly Separated Documents
# ============================================================================

@dataclass
class MultiHopQuestion:
    """A multi-hop question requiring cross-document reasoning."""
    question: str
    answer: str
    documents: Dict[str, str]  # Each document is separate
    reasoning_path: List[str]  # Expected chain of reasoning


DATASET = [
    MultiHopQuestion(
        question="What year was the director of 'The Social Network' born?",
        answer="1962",
        documents={
            "The Social Network": "The Social Network is a 2010 American biographical drama film directed by David Fincher and written by Aaron Sorkin.",
            "David Fincher": "David Andrew Leo Fincher (born August 28, 1962) is an American film director known for psychological thrillers.",
            "Aaron Sorkin": "Aaron Benjamin Sorkin (born June 9, 1961) is an American playwright and screenwriter.",
            "Facebook": "Facebook is an online social media platform founded in 2004 by Mark Zuckerberg.",
        },
        reasoning_path=["Find director", "Find director's birth year", "Aggregate"]
    ),
    MultiHopQuestion(
        question="Which university did the founder of Tesla attend for undergraduate studies?",
        answer="University of Pennsylvania",
        documents={
            "Tesla, Inc.": "Tesla, Inc. is an American electric vehicle company. Elon Musk joined as chairman in 2004 and became CEO in 2008.",
            "Elon Musk": "Elon Reeve Musk (born June 28, 1971) attended the University of Pennsylvania, where he earned degrees in economics and physics.",
            "University of Pennsylvania": "The University of Pennsylvania is a private Ivy League research university in Philadelphia, founded in 1740.",
            "SpaceX": "SpaceX is an American spacecraft manufacturer founded in 2002 by Elon Musk.",
        },
        reasoning_path=["Find founder", "Find founder's university", "Aggregate"]
    ),
    MultiHopQuestion(
        question="In which year did the author of '1984' die?",
        answer="1950",
        documents={
            "1984 (novel)": "Nineteen Eighty-Four is a dystopian novel published in 1949 by English author George Orwell.",
            "George Orwell": "Eric Arthur Blair (1903-1950), known by his pen name George Orwell, was an English novelist. He died of tuberculosis on January 21, 1950.",
            "Animal Farm": "Animal Farm is a satirical novella by George Orwell, published in 1945.",
            "Dystopian fiction": "Dystopian fiction explores dark, nightmare worlds. Famous examples include 1984 and Brave New World.",
        },
        reasoning_path=["Find author", "Find death year", "Aggregate"]
    ),
]


# ============================================================================
# Sub-LLM Call Tracking
# ============================================================================

@dataclass
class SubLLMResult:
    """Result from a single sub-LLM call."""
    step: int
    chunk_name: str
    prompt: str
    response: str
    timestamp: float


@dataclass
class IterativeRefinement:
    """Tracks iterative refinement process."""
    question: str
    sub_llm_calls: List[SubLLMResult]
    intermediate_results: List[str]
    final_answer: str
    reasoning_trace: List[str]
    total_time: float
    correct: bool


# ============================================================================
# Manual RLM: Orchestrator with Sub-LLM Calls
# ============================================================================

class ManualRLM:
    """Manual implementation of RLM's sub-LLM aggregation pattern."""

    def __init__(self, lm: dspy.LM):
        self.lm = lm
        self.sub_llm_calls = []
        self.intermediate_results = []
        self.reasoning_trace = []

    def sub_llm_query(self, chunk_name: str, prompt: str, step: int) -> str:
        """Simulate llm_query() - call sub-LLM on a specific chunk."""
        print(f"\n  [Step {step}] Sub-LLM Query to '{chunk_name}'")
        print(f"  Prompt: {prompt[:100]}...")

        # Make actual LLM call
        start = time.time()
        response = self.lm(prompt)
        elapsed = time.time() - start

        # Extract response
        if isinstance(response, list) and response:
            response_text = response[0] if isinstance(response[0], str) else str(response[0])
        else:
            response_text = str(response)

        print(f"  Response: {response_text[:200]}...")
        print(f"  Time: {elapsed:.2f}s")

        # Track the call
        self.sub_llm_calls.append(SubLLMResult(
            step=step,
            chunk_name=chunk_name,
            prompt=prompt,
            response=response_text,
            timestamp=time.time()
        ))

        return response_text

    def aggregate(self, intermediate_results: List[str], question: str, step: int) -> str:
        """Aggregate intermediate results from sub-LLMs."""
        print(f"\n  [Step {step}] Aggregating {len(intermediate_results)} intermediate results...")

        # Create aggregation prompt
        results_text = "\n".join([f"- {r}" for r in intermediate_results])
        prompt = f"""Given these intermediate findings:
{results_text}

Answer the question: {question}

Provide a concise, direct answer."""

        response = self.sub_llm_query("Aggregator", prompt, step)
        return response

    def solve_multihop(self, question: str, documents: Dict[str, str]) -> IterativeRefinement:
        """Solve multi-hop question using iterative refinement with sub-LLMs."""

        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        print(f"Available documents: {list(documents.keys())}")

        start_time = time.time()
        self.sub_llm_calls = []
        self.intermediate_results = []
        self.reasoning_trace = []

        # STEP 1: Process each document separately with sub-LLM
        print(f"\n--- PHASE 1: Process Each Document Separately ---")
        step = 1
        for doc_name, doc_content in documents.items():
            prompt = f"""Analyze this document and extract information relevant to: "{question}"

Document: {doc_name}
Content: {doc_content}

Extract any relevant facts. If not relevant, say "Not relevant"."""

            result = self.sub_llm_query(doc_name, prompt, step)
            if "not relevant" not in result.lower():
                self.intermediate_results.append(f"From '{doc_name}': {result}")
                self.reasoning_trace.append(f"Extracted from {doc_name}")
            step += 1

        # STEP 2: First aggregation - combine relevant findings
        print(f"\n--- PHASE 2: First Aggregation ---")
        if self.intermediate_results:
            first_aggregate = self.aggregate(self.intermediate_results, question, step)
            self.reasoning_trace.append("First aggregation")
            step += 1

            # STEP 3: Refinement - verify and refine the answer
            print(f"\n--- PHASE 3: Refinement ---")
            refinement_prompt = f"""Review this answer and refine it if needed: "{first_aggregate}"

Question: {question}

Provide the most concise, accurate answer possible."""

            final_answer = self.sub_llm_query("Refiner", refinement_prompt, step)
            self.reasoning_trace.append("Final refinement")
        else:
            final_answer = "Unable to find relevant information"

        total_time = time.time() - start_time

        print(f"\n{'='*80}")
        print(f"Final Answer: {final_answer}")
        print(f"Sub-LLM Calls: {len(self.sub_llm_calls)}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"{'='*80}")

        return IterativeRefinement(
            question=question,
            sub_llm_calls=self.sub_llm_calls,
            intermediate_results=self.intermediate_results,
            final_answer=final_answer,
            reasoning_trace=self.reasoning_trace,
            total_time=total_time,
            correct=False  # Will be set later
        )


# ============================================================================
# Run Test with Statistics
# ============================================================================

def run_manual_rlm_test(dataset: List[MultiHopQuestion]) -> List[IterativeRefinement]:
    """Run manual RLM test showing sub-LLM aggregation."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    # Use gpt-4o-mini for cost efficiency
    lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)

    print("="*80)
    print("Manual RLM: Iterative Refinement with Sub-LLM Aggregation")
    print("="*80)
    print(f"\nDataset: {len(dataset)} multi-hop questions")
    print("Model: gpt-4o-mini")
    print("\nThis will show:")
    print("  1. Separate sub-LLM calls for each document")
    print("  2. Intermediate results from each sub-LLM")
    print("  3. Iterative aggregation and refinement")
    print("  4. Complete statistics\n")

    manual_rlm = ManualRLM(lm)
    results = []

    for i, example in enumerate(dataset):
        print(f"\n\n{'#'*80}")
        print(f"EXAMPLE {i+1}/{len(dataset)}")
        print(f"{'#'*80}")

        refinement = manual_rlm.solve_multihop(
            question=example.question,
            documents=example.documents
        )

        # Check correctness
        refinement.correct = (
            example.answer.lower() in refinement.final_answer.lower() or
            refinement.final_answer.lower() in example.answer.lower()
        )

        results.append(refinement)

        # Print intermediate analysis
        print(f"\n--- Intermediate Results ---")
        for j, intermediate in enumerate(refinement.intermediate_results, 1):
            print(f"{j}. {intermediate}")

        print(f"\n--- Reasoning Trace ---")
        for j, trace in enumerate(refinement.reasoning_trace, 1):
            print(f"{j}. {trace}")

        status = "✓" if refinement.correct else "✗"
        print(f"\n{status} Expected: {example.answer}")
        print(f"{status} Got: {refinement.final_answer}")

    return results


def print_statistics(results: List[IterativeRefinement]):
    """Print comprehensive statistics."""

    print(f"\n\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}\n")

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = (correct / total * 100) if total > 0 else 0

    total_sub_calls = sum(len(r.sub_llm_calls) for r in results)
    avg_sub_calls = total_sub_calls / total if total > 0 else 0
    avg_time = sum(r.total_time for r in results) / total if total > 0 else 0

    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Total Sub-LLM Calls: {total_sub_calls}")
    print(f"Avg Sub-LLM Calls per Question: {avg_sub_calls:.1f}")
    print(f"Avg Time per Question: {avg_time:.2f}s")

    # Estimate cost
    # Approximate: 1500 tokens per sub-call (500 in, 1000 out)
    estimated_input_tokens = total_sub_calls * 500
    estimated_output_tokens = total_sub_calls * 1000
    cost = (estimated_input_tokens / 1_000_000 * 0.15 +
            estimated_output_tokens / 1_000_000 * 0.60)

    print(f"\nEstimated Cost: ${cost:.3f}")
    print(f"  Input tokens: ~{estimated_input_tokens:,}")
    print(f"  Output tokens: ~{estimated_output_tokens:,}")
    print(f"  Cost per question: ${cost/total:.3f}")

    # Per-question breakdown
    print(f"\n{'='*80}")
    print("PER-QUESTION SUB-LLM BREAKDOWN")
    print(f"{'='*80}\n")

    for i, result in enumerate(results, 1):
        status = "✓" if result.correct else "✗"
        print(f"{status} Q{i}: {len(result.sub_llm_calls)} sub-LLM calls | {result.total_time:.1f}s")
        print(f"  Question: {result.question[:60]}...")

        # Show sub-call breakdown
        for call in result.sub_llm_calls:
            print(f"    Step {call.step}: {call.chunk_name}")

        print()

    # Aggregation pattern analysis
    print(f"{'='*80}")
    print("AGGREGATION PATTERN ANALYSIS")
    print(f"{'='*80}\n")

    for i, result in enumerate(results, 1):
        print(f"Question {i}:")
        print(f"  Phase 1 (Document Processing): {len(result.intermediate_results)} documents yielded results")
        print(f"  Phase 2 (Aggregation): Combined into preliminary answer")
        print(f"  Phase 3 (Refinement): Final answer polished")
        print(f"  Total sub-LLM calls: {len(result.sub_llm_calls)}")
        print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Run test
    results = run_manual_rlm_test(DATASET)

    # Print statistics
    print_statistics(results)

    print(f"\n{'='*80}")
    print("KEY INSIGHT: This demonstrates the EXACT pattern RLM uses:")
    print("  1. Split data into chunks (documents)")
    print("  2. Process each chunk with separate sub-LLM calls")
    print("  3. Collect intermediate results")
    print("  4. Iteratively aggregate and refine")
    print("  5. Produce final answer with full traceability")
    print(f"{'='*80}")
