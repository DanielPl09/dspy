"""
Visualize RLM's progressive refinement across sub-LM calls.

Shows:
- Timeline of iterations
- Sub-LM queries made at each step
- Information discovered incrementally
- Progressive convergence toward final answer
"""

import json
from collections import defaultdict


def visualize_iteration_flow(trajectory, question, answer):
    """
    Create ASCII visualization of RLM's refinement process.
    Shows progressive information gathering across iterations.
    """
    print("\n" + "═" * 100)
    print(f"REFINEMENT PROCESS VISUALIZATION")
    print("═" * 100)
    print(f"\n📋 Question: {question[:80]}...")
    print(f"🎯 Final Answer: {answer}")
    print(f"📊 Total Iterations: {len(trajectory)}")
    print("\n" + "─" * 100)

    # Track cumulative information
    cumulative_info = []
    total_queries = 0

    for idx, step in enumerate(trajectory, 1):
        reasoning = step.get('reasoning', '')[:120]
        code = step.get('code', '')
        output = step.get('output', '')

        # Count queries in this iteration
        queries_this_iter = code.count('llm_query')
        total_queries += queries_this_iter

        # Determine iteration type
        has_query = 'llm_query' in code
        has_error = '[Error]' in output
        has_answer = 'SUBMIT' in code or 'FINAL' in output

        # Visual markers
        if has_answer:
            marker = "✅"
            status = "ANSWER"
        elif has_query and not has_error:
            marker = "🔎"
            status = "QUERY"
        elif has_error:
            marker = "❌"
            status = "ERROR"
        else:
            marker = "⚙️"
            status = "REFINE"

        # Progress bar
        progress = int((idx / len(trajectory)) * 20)
        progress_bar = "▓" * progress + "░" * (20 - progress)

        print(f"\n┌─ Iteration {idx}/{len(trajectory)} {marker} [{status}] " + "─" * 60)
        print(f"│ Progress: [{progress_bar}] {idx}/{len(trajectory)}")
        print(f"│")
        print(f"│ 💭 Reasoning: {reasoning}...")

        if queries_this_iter > 0:
            print(f"│ 🔎 Sub-LM Queries: {queries_this_iter} (Total so far: {total_queries})")

            # Extract query content
            if 'llm_query_batched' in code:
                print(f"│    └─ Batch query (parallel)")
            elif 'llm_query(' in code:
                # Try to extract query text
                import re
                query_match = re.search(r'llm_query\(["\'](.+?)["\']\)', code)
                if query_match:
                    query_text = query_match.group(1)[:60]
                    print(f"│    └─ Query: \"{query_text}...\"")

        # Show output if meaningful
        if output and not has_error and len(output) > 20:
            output_preview = output[:80].replace('\n', ' ')
            print(f"│ 📤 Discovery: {output_preview}...")
            cumulative_info.append(f"Iter {idx}: {output_preview[:50]}")
        elif has_error:
            print(f"│ ⚠️  Error encountered (retrying with different strategy)")

        if has_answer:
            print(f"│ ✅ Final answer submitted!")

        print(f"└" + "─" * 80)

    # Summary
    print("\n" + "═" * 100)
    print("REFINEMENT SUMMARY")
    print("═" * 100)
    print(f"\n📊 Statistics:")
    print(f"   Total iterations: {len(trajectory)}")
    print(f"   Total sub-LM queries: {total_queries}")
    print(f"   Queries per iteration: {total_queries/len(trajectory):.1f}")
    print(f"   Information pieces gathered: {len(cumulative_info)}")

    if cumulative_info:
        print(f"\n📈 Cumulative Information Discovery:")
        for i, info in enumerate(cumulative_info[:5], 1):
            print(f"   {i}. {info}")
        if len(cumulative_info) > 5:
            print(f"   ... and {len(cumulative_info) - 5} more")


def create_convergence_timeline(trajectory):
    """
    Create a timeline showing when queries were made and information discovered.
    """
    print("\n" + "═" * 100)
    print("CONVERGENCE TIMELINE")
    print("═" * 100 + "\n")

    timeline = []
    cumulative_queries = 0

    for idx, step in enumerate(trajectory, 1):
        code = step.get('code', '')
        output = step.get('output', '')

        queries = code.count('llm_query')
        cumulative_queries += queries

        has_info = output and not output.startswith('[Error]') and len(output) > 20
        has_answer = 'SUBMIT' in code or 'FINAL' in output

        timeline.append({
            'iteration': idx,
            'queries': queries,
            'cumulative_queries': cumulative_queries,
            'has_info': has_info,
            'has_answer': has_answer
        })

    # ASCII timeline
    print("Time →")
    print("│")

    for entry in timeline:
        iter_num = entry['iteration']
        queries = entry['queries']
        cum_queries = entry['cumulative_queries']

        # Build visual
        indent = "│   " * (iter_num - 1)

        if entry['has_answer']:
            symbol = "✓"
            label = f"ANSWER (Total queries: {cum_queries})"
        elif entry['has_info']:
            symbol = "●"
            label = f"Info discovered (Queries: {queries}, Total: {cum_queries})"
        elif queries > 0:
            symbol = "○"
            label = f"Querying ({queries} calls, Total: {cum_queries})"
        else:
            symbol = "·"
            label = f"Refining strategy"

        print(f"├──[{iter_num}] {symbol} {label}")

    print("│")
    print("▼")


def visualize_query_pattern(trajectory):
    """
    Show the pattern of sub-LM queries across iterations.
    """
    print("\n" + "═" * 100)
    print("SUB-LM QUERY PATTERN")
    print("═" * 100 + "\n")

    # Collect query data
    query_data = []
    for idx, step in enumerate(trajectory, 1):
        code = step.get('code', '')
        queries = code.count('llm_query')
        query_data.append((idx, queries))

    # ASCII bar chart
    max_queries = max(q for _, q in query_data) if query_data else 1

    print("Iteration │ Queries")
    print("──────────┼" + "─" * 50)

    for iter_num, queries in query_data:
        if queries > 0:
            bar_length = int((queries / max(max_queries, 1)) * 40)
            bar = "█" * bar_length
            print(f"    {iter_num:2d}    │ {bar} {queries}")
        else:
            print(f"    {iter_num:2d}    │ ·")

    print("\nPattern Analysis:")

    # Find query clusters
    query_iters = [i for i, q in query_data if q > 0]
    if query_iters:
        print(f"   Queries made in iterations: {query_iters}")
        print(f"   Total query calls: {sum(q for _, q in query_data)}")
        print(f"   Avg queries per active iteration: {sum(q for _, q in query_data if q > 0) / len(query_iters):.1f}")


def visualize_all_questions(results_file):
    """
    Visualize refinement process for all questions in results.
    """
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return

    results = data.get('results', [])

    print("\n" + "█" * 100)
    print("RLM PROGRESSIVE REFINEMENT VISUALIZATION")
    print("█" * 100)
    print(f"\nAnalyzing {len(results)} questions from: {results_file}")
    print(f"Configuration: {data.get('config', {})}")

    for idx, result in enumerate(results[:3], 1):  # Show first 3 in detail
        question = result.get('question', 'Unknown')
        answer = result.get('predicted', 'Unknown')
        trajectory = result.get('trajectory', [])

        print("\n\n" + "█" * 100)
        print(f"QUESTION {idx}/{len(results)}")
        print("█" * 100)

        # Main visualization
        visualize_iteration_flow(trajectory, question, answer)

        # Timeline
        create_convergence_timeline(trajectory)

        # Query pattern
        visualize_query_pattern(trajectory)

    # Overall statistics across all questions
    print("\n\n" + "█" * 100)
    print("OVERALL STATISTICS ACROSS ALL QUESTIONS")
    print("█" * 100 + "\n")

    total_iterations = sum(len(r.get('trajectory', [])) for r in results)
    total_queries = 0
    productive_iterations = 0

    for result in results:
        for step in result.get('trajectory', []):
            code = step.get('code', '')
            queries = code.count('llm_query')
            total_queries += queries
            if queries > 0:
                productive_iterations += 1

    print(f"📊 Summary Statistics:")
    print(f"   Total questions: {len(results)}")
    print(f"   Total iterations: {total_iterations}")
    print(f"   Total sub-LM queries: {total_queries}")
    print(f"   Productive iterations (with queries): {productive_iterations}")
    print(f"\n📈 Averages:")
    print(f"   Iterations per question: {total_iterations/len(results):.1f}")
    print(f"   Queries per question: {total_queries/len(results):.1f}")
    print(f"   Productive iteration rate: {productive_iterations/total_iterations*100:.1f}%")

    # Efficiency metrics
    print(f"\n⚡ Efficiency:")
    config = data.get('config', {})
    max_calls = config.get('max_llm_calls', 40)
    avg_queries = total_queries / len(results)
    print(f"   Budget usage: {avg_queries}/{max_calls} ({avg_queries/max_calls*100:.1f}%)")
    print(f"   Demonstrates: {'✅ Targeted queries' if avg_queries < max_calls * 0.3 else '⚠️ Room for optimization'}")


if __name__ == "__main__":
    import sys

    # Visualize both test results
    print("\n🎨 RLM Refinement Process Visualizer\n")

    # First, the progressive convergence test (with proper chunking)
    if len(sys.argv) > 1:
        visualize_all_questions(sys.argv[1])
    else:
        # Default: visualize the progressive convergence results
        print("Visualizing Progressive Convergence Results (with proper chunking)...")
        visualize_all_questions('rlm_progressive_convergence_results.json')

        print("\n\n" + "="*100)
        print("\nTo visualize the initial test (no chunking), run:")
        print("  python visualize_rlm_refinement.py rlm_iterative_refinement_results.json")
