#!/usr/bin/env python3
"""
Multi-hop RLM test showing ACTUAL sub-LLM calls and iterative aggregation.

This version tracks and displays:
1. Each llm_query/llm_query_batched call
2. Intermediate results from sub-LMs
3. How answers are aggregated across chunks
4. The progression of iterative refinement

Budget: < $0.50
"""

import dspy
import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass, field


# ============================================================================
# Enhanced Multi-hop Dataset - Explicitly Chunked
# ============================================================================

@dataclass
class MultiHopExample:
    """A multi-hop question with explicitly separated document chunks."""
    question: str
    answer: str
    supporting_facts: List[str]
    documents: Dict[str, str]

    def get_chunked_context(self) -> str:
        """Return context formatted to encourage chunking."""
        parts = []
        for title, content in self.documents.items():
            parts.append(f"=== Document: {title} ===\n{content}\n")

        header = f"You have {len(self.documents)} separate documents to analyze.\n"
        header += "IMPORTANT: Process each document separately using llm_query() or llm_query_batched().\n\n"
        return header + "\n".join(parts)


# Same dataset as before
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
]


# ============================================================================
# Sub-LLM Call Tracking
# ============================================================================

@dataclass
class SubLMCall:
    """Record of a single sub-LLM call."""
    call_id: int
    prompt: str
    response: str
    timestamp: float


@dataclass
class RLMExecutionTrace:
    """Detailed trace of RLM execution with sub-LLM calls."""
    question: str
    answer: str
    predicted_answer: str
    correct: bool
    total_iterations: int
    sub_llm_calls: List[SubLMCall] = field(default_factory=list)
    time_seconds: float = 0.0

    def add_sub_call(self, prompt: str, response: str):
        """Add a sub-LLM call to the trace."""
        self.sub_llm_calls.append(SubLMCall(
            call_id=len(self.sub_llm_calls) + 1,
            prompt=prompt[:200] + "..." if len(prompt) > 200 else prompt,
            response=response[:300] + "..." if len(response) > 300 else response,
            timestamp=time.time()
        ))

    def print_trace(self):
        """Print the execution trace."""
        status = "✓" if self.correct else "✗"
        print(f"\n{'='*80}")
        print(f"{status} Question: {self.question}")
        print(f"{'='*80}")
        print(f"Expected: {self.answer}")
        print(f"Predicted: {self.predicted_answer}")
        print(f"Iterations: {self.total_iterations}")
        print(f"Sub-LLM Calls: {len(self.sub_llm_calls)}")
        print(f"Time: {self.time_seconds:.2f}s")

        if self.sub_llm_calls:
            print(f"\n--- Sub-LLM Call Trace ---")
            for call in self.sub_llm_calls:
                print(f"\nCall #{call.call_id}:")
                print(f"  Prompt: {call.prompt}")
                print(f"  Response: {call.response}")


# ============================================================================
# Custom RLM with Call Tracking
# ============================================================================

class TrackedRLM(dspy.RLM):
    """RLM that tracks sub-LLM calls for analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_trace = None

    def _make_llm_tools(self, max_workers: int = 8):
        """Override to track sub-LLM calls."""
        # Get original tools
        tools = super()._make_llm_tools(max_workers)

        # Wrap llm_query to track calls
        original_llm_query = tools["llm_query"]
        original_llm_query_batched = tools["llm_query_batched"]

        def tracked_llm_query(prompt: str) -> str:
            result = original_llm_query(prompt)
            if self.current_trace:
                self.current_trace.add_sub_call(prompt, result)
            return result

        def tracked_llm_query_batched(prompts: List[str]) -> List[str]:
            results = original_llm_query_batched(prompts)
            if self.current_trace:
                for prompt, result in zip(prompts, results):
                    self.current_trace.add_sub_call(prompt, result)
            return results

        tools["llm_query"] = tracked_llm_query
        tools["llm_query_batched"] = tracked_llm_query_batched

        return tools

    def forward(self, **input_args):
        """Forward with tracking."""
        # Create trace
        question = input_args.get("question", "")
        self.current_trace = RLMExecutionTrace(
            question=question,
            answer="",  # Will be set later
            predicted_answer="",
            correct=False,
            total_iterations=0
        )

        start_time = time.time()
        result = super().forward(**input_args)
        self.current_trace.time_seconds = time.time() - start_time
        self.current_trace.total_iterations = len(result.trajectory)
        self.current_trace.predicted_answer = result.answer

        return result


# ============================================================================
# Run Test with Tracking
# ============================================================================

def run_tracked_multihop_test(
    dataset: List[MultiHopExample],
    max_iterations: int = 15,
    max_llm_calls: int = 25,
    verbose: bool = True,
) -> List[RLMExecutionTrace]:
    """Run multi-hop test with sub-LLM call tracking."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    # Use gpt-4o-mini
    lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
    dspy.configure(lm=lm)

    # Create tracked RLM
    rlm = TrackedRLM(
        signature="context, question -> answer: str",
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        verbose=verbose,
    )

    traces = []

    for i, example in enumerate(dataset):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}/{len(dataset)}")
        print(f"{'='*80}")

        context = example.get_chunked_context()

        try:
            output = rlm(context=context, question=example.question)

            # Complete the trace
            trace = rlm.current_trace
            trace.answer = example.answer
            trace.correct = (
                example.answer.lower() in output.answer.lower() or
                output.answer.lower() in example.answer.lower()
            )

            # Print trace
            trace.print_trace()
            traces.append(trace)

        except Exception as e:
            print(f"❌ Error: {e}")
            trace = RLMExecutionTrace(
                question=example.question,
                answer=example.answer,
                predicted_answer=f"ERROR: {str(e)[:50]}",
                correct=False,
                total_iterations=0,
            )
            traces.append(trace)

    return traces


def print_aggregation_summary(traces: List[RLMExecutionTrace]):
    """Print summary showing aggregation patterns."""
    print(f"\n{'='*80}")
    print("AGGREGATION ANALYSIS")
    print(f"{'='*80}\n")

    total_sub_calls = sum(len(t.sub_llm_calls) for t in traces)
    avg_sub_calls = total_sub_calls / len(traces) if traces else 0

    print(f"Total Questions: {len(traces)}")
    print(f"Total Sub-LLM Calls: {total_sub_calls}")
    print(f"Avg Sub-LLM Calls per Question: {avg_sub_calls:.1f}")
    print(f"Accuracy: {sum(1 for t in traces if t.correct)}/{len(traces)}")

    print(f"\n--- Sub-LLM Call Distribution ---")
    for i, trace in enumerate(traces, 1):
        print(f"Q{i}: {len(trace.sub_llm_calls)} sub-calls | "
              f"{trace.total_iterations} iterations | "
              f"{'✓' if trace.correct else '✗'}")

    # Cost estimate
    # Rough: main calls = sum(iterations), sub calls = total_sub_calls
    total_main_calls = sum(t.total_iterations for t in traces)
    total_calls = total_main_calls + total_sub_calls
    estimated_cost = (total_calls * 2000 / 1_000_000 * 0.15 +  # input
                      total_calls * 500 / 1_000_000 * 0.60)   # output

    print(f"\n--- Cost Estimate ---")
    print(f"Main LLM Calls (iterations): {total_main_calls}")
    print(f"Sub LLM Calls (llm_query): {total_sub_calls}")
    print(f"Total LLM Calls: {total_calls}")
    print(f"Estimated Cost: ${estimated_cost:.3f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Multi-hop RLM with Sub-LLM Call Tracking")
    print("="*80)
    print("\nThis test shows:")
    print("1. Actual llm_query() calls to sub-LLMs")
    print("2. Intermediate results from each sub-call")
    print("3. How answers are aggregated across chunks")
    print("4. Iterative refinement statistics")
    print()

    # Run test with verbose to see code + tracking for sub-calls
    traces = run_tracked_multihop_test(
        dataset=MULTIHOP_DATASET[:3],  # Run 3 questions to control cost
        max_iterations=15,
        max_llm_calls=25,
        verbose=True,  # See the code being written!
    )

    # Print aggregation summary
    print_aggregation_summary(traces)

    print(f"\n{'='*80}")
    print("KEY INSIGHT: RLM writes code that calls llm_query() to process")
    print("each document separately, then aggregates the intermediate results!")
    print(f"{'='*80}")
