#!/usr/bin/env python3
"""
Multi-hop QA test with RLM to demonstrate iterative refinement.

This test shows:
1. How RLM splits across documents to find answers
2. Statistics on iterations, LLM calls, and cost
3. Multi-hop reasoning performance (similar to HotPotQA)

Budget: < $0.50 (using gpt-4o-mini)
"""

import dspy
import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass


# ============================================================================
# Multi-hop Dataset (HotPotQA-style)
# ============================================================================

@dataclass
class MultiHopExample:
    """A multi-hop question with supporting documents and answer."""
    question: str
    answer: str
    supporting_facts: List[str]  # Titles of docs needed
    documents: Dict[str, str]    # title -> content

    def get_context(self) -> str:
        """Concatenate all documents into a single context."""
        parts = []
        for title, content in self.documents.items():
            parts.append(f"=== {title} ===\n{content}\n")
        return "\n".join(parts)


MULTIHOP_DATASET = [
    MultiHopExample(
        question="What year was the director of 'The Social Network' born?",
        answer="1962",
        supporting_facts=["The Social Network", "David Fincher"],
        documents={
            "The Social Network": "The Social Network is a 2010 American biographical drama film directed by David Fincher and written by Aaron Sorkin. The film depicts the founding of Facebook.",
            "David Fincher": "David Andrew Leo Fincher (born August 28, 1962) is an American film director. He is known for his psychological thrillers including Se7en, Fight Club, and Gone Girl.",
            "Aaron Sorkin": "Aaron Benjamin Sorkin (born June 9, 1961) is an American playwright, screenwriter, and film director. He is known for his fast-paced dialogue.",
            "Facebook": "Facebook is an American online social media and social networking service owned by Meta Platforms. It was founded in 2004 by Mark Zuckerberg.",
        }
    ),
    MultiHopExample(
        question="Which university did the founder of Tesla attend for his undergraduate degree?",
        answer="University of Pennsylvania",
        supporting_facts=["Tesla, Inc.", "Elon Musk"],
        documents={
            "Tesla, Inc.": "Tesla, Inc. is an American electric vehicle and clean energy company founded in 2003. Elon Musk joined as chairman in 2004 and became CEO in 2008.",
            "Elon Musk": "Elon Reeve Musk (born June 28, 1971) is a business magnate and investor. He attended the University of Pennsylvania, where he earned bachelor's degrees in economics and physics.",
            "University of Pennsylvania": "The University of Pennsylvania is a private Ivy League research university in Philadelphia. It was founded in 1740 by Benjamin Franklin.",
            "SpaceX": "Space Exploration Technologies Corp. (SpaceX) is an American spacecraft manufacturer founded in 2002 by Elon Musk.",
        }
    ),
    MultiHopExample(
        question="What is the capital of the country where the Eiffel Tower is located?",
        answer="Paris",
        supporting_facts=["Eiffel Tower", "France"],
        documents={
            "Eiffel Tower": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was completed in 1889 and named after engineer Gustave Eiffel.",
            "France": "France, officially the French Republic, is a country in Western Europe. Its capital and largest city is Paris.",
            "Gustave Eiffel": "Alexandre Gustave Eiffel (1832-1923) was a French civil engineer. He is best known for the Eiffel Tower and the Statue of Liberty's internal structure.",
            "Paris": "Paris is the capital and most populous city of France. It is known for its art, culture, and landmarks like the Louvre and Notre-Dame.",
        }
    ),
    MultiHopExample(
        question="In which year did the author of '1984' die?",
        answer="1950",
        supporting_facts=["1984 (novel)", "George Orwell"],
        documents={
            "1984 (novel)": "Nineteen Eighty-Four is a dystopian novel published in 1949 by English author George Orwell. It depicts a totalitarian society under constant surveillance.",
            "George Orwell": "Eric Arthur Blair (1903-1950), known by his pen name George Orwell, was an English novelist and essayist. He died of tuberculosis on January 21, 1950.",
            "Animal Farm": "Animal Farm is a satirical allegorical novella by George Orwell, published in 1945. It uses animals on a farm to represent the Russian Revolution.",
            "Dystopian fiction": "Dystopian fiction is a genre that explores social and political structures in a dark, nightmare world. Famous examples include 1984 and Brave New World.",
        }
    ),
    MultiHopExample(
        question="What is the population of the birthplace of the painter of the Mona Lisa?",
        answer="approximately 15,000",
        supporting_facts=["Mona Lisa", "Leonardo da Vinci", "Vinci, Italy"],
        documents={
            "Mona Lisa": "The Mona Lisa is a portrait painting by the Italian artist Leonardo da Vinci, created between 1503-1519. It is displayed at the Louvre Museum in Paris.",
            "Leonardo da Vinci": "Leonardo di ser Piero da Vinci (1452-1519) was an Italian polymath of the Renaissance. He was born in Vinci, a town in Tuscany, Italy.",
            "Vinci, Italy": "Vinci is a comune in the Metropolitan City of Florence, Tuscany, Italy. As of 2021, it has a population of approximately 15,000 residents.",
            "Louvre Museum": "The Louvre is the world's most-visited museum, located in Paris, France. It houses thousands of works of art, including the Mona Lisa.",
        }
    ),
    MultiHopExample(
        question="What company did the co-founder of Apple start after leaving in 1985?",
        answer="NeXT",
        supporting_facts=["Apple Inc.", "Steve Jobs", "NeXT"],
        documents={
            "Apple Inc.": "Apple Inc. is an American technology company founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne. Jobs left the company in 1985 after a power struggle.",
            "Steve Jobs": "Steven Paul Jobs (1955-2011) was an American business magnate and co-founder of Apple Inc. After leaving Apple in 1985, he founded NeXT Computer.",
            "NeXT": "NeXT Computer (later NeXT Software) was a computer company founded by Steve Jobs in 1985 after he left Apple. Apple acquired NeXT in 1997.",
            "Pixar": "Pixar Animation Studios is an American computer animation film studio. Steve Jobs became its majority shareholder in 1986 after purchasing it from Lucasfilm.",
        }
    ),
]


# ============================================================================
# Statistics Tracking
# ============================================================================

@dataclass
class RLMStats:
    """Statistics for a single RLM execution."""
    question: str
    answer: str
    predicted_answer: str
    correct: bool
    iterations_used: int
    llm_calls_estimate: int  # Approximate based on trajectory
    time_seconds: float
    trajectory_length: int

    def __str__(self):
        status = "✓" if self.correct else "✗"
        return (
            f"{status} Q: {self.question[:60]}...\n"
            f"  Predicted: {self.predicted_answer}\n"
            f"  Expected: {self.answer}\n"
            f"  Iterations: {self.iterations_used} | Time: {self.time_seconds:.2f}s"
        )


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost based on model and token counts."""
    # Prices per 1M tokens (approximate as of Jan 2024)
    prices = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 2.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }

    # Find matching model
    for model_key, price in prices.items():
        if model_key in model_name.lower():
            input_cost = (input_tokens / 1_000_000) * price["input"]
            output_cost = (output_tokens / 1_000_000) * price["output"]
            return input_cost + output_cost

    # Default to gpt-4o-mini pricing
    return (input_tokens / 1_000_000) * 0.15 + (output_tokens / 1_000_000) * 0.60


# ============================================================================
# RLM Multi-hop Test
# ============================================================================

def test_multihop_rlm(
    dataset: List[MultiHopExample],
    max_iterations: int = 15,
    max_llm_calls: int = 30,
    verbose: bool = False,
) -> List[RLMStats]:
    """Run RLM on multi-hop dataset and collect statistics."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    # Use gpt-4o-mini for cost efficiency
    lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
    dspy.configure(lm=lm)

    # Create RLM with explicit parameters
    rlm = dspy.RLM(
        signature="context, question -> answer: str",
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        verbose=verbose,
    )

    results = []

    for i, example in enumerate(dataset):
        print(f"\n{'='*80}")
        print(f"Example {i+1}/{len(dataset)}: {example.question}")
        print(f"{'='*80}")

        context = example.get_context()

        # Run RLM and time it
        start_time = time.time()
        try:
            output = rlm(context=context, question=example.question)
            elapsed = time.time() - start_time

            predicted = output.answer.strip()
            expected = example.answer.strip()

            # Flexible matching (contains or close match)
            correct = (
                expected.lower() in predicted.lower() or
                predicted.lower() in expected.lower()
            )

            # Estimate LLM calls from trajectory
            # Each iteration typically has 1 main call, plus any llm_query calls in the code
            # For now, use trajectory length as proxy
            llm_calls_estimate = len(output.trajectory) + 5  # Conservative estimate

            stats = RLMStats(
                question=example.question,
                answer=expected,
                predicted_answer=predicted,
                correct=correct,
                iterations_used=len(output.trajectory),
                llm_calls_estimate=llm_calls_estimate,
                time_seconds=elapsed,
                trajectory_length=len(output.trajectory),
            )

            print(f"\n{stats}")
            results.append(stats)

        except Exception as e:
            print(f"❌ Error: {e}")
            elapsed = time.time() - start_time
            stats = RLMStats(
                question=example.question,
                answer=example.answer,
                predicted_answer=f"ERROR: {str(e)[:50]}",
                correct=False,
                iterations_used=0,
                llm_calls_estimate=0,
                time_seconds=elapsed,
                trajectory_length=0,
            )
            results.append(stats)

    return results


def print_summary(results: List[RLMStats], model_name: str = "gpt-4o-mini"):
    """Print summary statistics."""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = (correct / total * 100) if total > 0 else 0

    avg_iterations = sum(r.iterations_used for r in results) / total if total > 0 else 0
    avg_time = sum(r.time_seconds for r in results) / total if total > 0 else 0
    total_llm_calls = sum(r.llm_calls_estimate for r in results)

    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Avg Iterations: {avg_iterations:.1f}")
    print(f"Avg Time: {avg_time:.2f}s")
    print(f"Total Estimated LLM Calls: {total_llm_calls}")

    # Estimate cost (rough approximation)
    # Assume ~2000 input tokens and ~500 output tokens per call
    estimated_input_tokens = total_llm_calls * 2000
    estimated_output_tokens = total_llm_calls * 500
    estimated_cost = estimate_cost(model_name, estimated_input_tokens, estimated_output_tokens)

    print(f"\nEstimated Cost: ${estimated_cost:.3f} (using {model_name})")
    print(f"  Input tokens: ~{estimated_input_tokens:,}")
    print(f"  Output tokens: ~{estimated_output_tokens:,}")

    # Per-question breakdown
    print(f"\n{'='*80}")
    print("PER-QUESTION BREAKDOWN")
    print(f"{'='*80}\n")

    for i, stat in enumerate(results, 1):
        print(f"{i}. {stat}")
        print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Multi-hop QA Test with RLM (Iterative Refinement)")
    print("="*80)
    print(f"\nDataset: {len(MULTIHOP_DATASET)} multi-hop questions")
    print("Model: gpt-4o-mini (cost-efficient)")
    print("Budget: < $0.50\n")

    # Run test with limited iterations to control cost
    results = test_multihop_rlm(
        dataset=MULTIHOP_DATASET,
        max_iterations=15,   # Limit iterations
        max_llm_calls=20,    # Limit sub-LLM calls
        verbose=False,       # Set to True to see code execution
    )

    # Print summary
    print_summary(results, model_name="gpt-4o-mini")

    print("\n" + "="*80)
    print("TIP: Set verbose=True to see the Python code RLM writes!")
    print("="*80)
