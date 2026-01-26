"""
Analyze progressive convergence patterns for source-category-based chunking.

Focus:
1. Which categories are queried first vs. later?
2. How does information accumulate across categories?
3. Category-switching patterns (e.g., CLIENT_DB → POLICY → KNOWLEDGE)
4. Async streaming implications for enterprise UIs
"""

import json
from collections import defaultdict


def analyze_category_query_timeline(results_file='rlm_source_category_results.json'):
    """
    Analyze WHEN each category is queried during RLM execution.

    For enterprise async UIs:
    - Show users which source category is being checked
    - Display progressive information from different sources
    - Visualize cross-referencing across categories
    """

    print("="*100)
    print("CATEGORY-BASED PROGRESSIVE CONVERGENCE ANALYSIS")
    print("Enterprise Async Streaming: Show Which Sources Being Queried")
    print("="*100)

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n❌ Results file not found: {results_file}")
        print("   Run: python test_rlm_source_categories.py first")
        return

    results = data.get('results', [])
    config = data.get('config', {})
    categories = data.get('categories', [])

    print(f"\nDataset: HotPotQA multi-hop (enterprise simulation)")
    print(f"Questions analyzed: {len(results)}")
    print(f"Max iterations: {config.get('max_iterations', 'N/A')}")
    print(f"Source categories: {', '.join(categories)}")

    print(f"\n{'='*100}")
    print("PER-QUESTION CATEGORY TIMELINE ANALYSIS")
    print(f"{'='*100}")

    all_category_timelines = []

    for idx, result in enumerate(results, 1):
        question = result.get('question', 'Unknown')
        trajectory = result.get('trajectory', [])
        categories_available = result.get('categories_available', {})
        categories_queried = result.get('categories_queried', [])

        print(f"\n\nQuestion {idx}: {question[:80]}...")
        print(f"{'─'*100}")

        print(f"\n📂 Available Categories:")
        for cat, chunk_count in categories_available.items():
            status = "✅ QUERIED" if cat in categories_queried else "⚪ NOT USED"
            print(f"   {cat}: {chunk_count} chunks - {status}")

        # Track category usage per iteration
        category_timeline = []

        for iter_idx, step in enumerate(trajectory, 1):
            code = step.get('code', '')
            reasoning = step.get('reasoning', '')[:100]

            # Detect which categories queried this iteration
            categories_this_iter = []
            for cat in categories:
                cat_lower = cat.lower()
                if cat_lower in code.lower():
                    categories_this_iter.append(cat)

            queries_this_iter = code.count('llm_query')

            if categories_this_iter or queries_this_iter > 0:
                category_timeline.append({
                    'iteration': iter_idx,
                    'categories': categories_this_iter,
                    'queries': queries_this_iter,
                    'reasoning': reasoning
                })

        if category_timeline:
            print(f"\n🔍 Category Query Timeline:")
            print(f"   Iter │ Categories Queried │ Queries │ Reasoning")
            print(f"   ─────┼────────────────────┼─────────┼──────────────────────")

            for entry in category_timeline[:10]:  # Show first 10
                iter_num = entry['iteration']
                cats = ', '.join(entry['categories']) if entry['categories'] else 'General'
                queries = entry['queries']
                reasoning = entry['reasoning']

                print(f"   {iter_num:4d} │ {cats:18s} │ {queries:7d} │ {reasoning}...")

            if len(category_timeline) > 10:
                print(f"   ... ({len(category_timeline) - 10} more iterations)")

            # Identify category switching pattern
            category_sequence = []
            for entry in category_timeline:
                if entry['categories']:
                    category_sequence.extend(entry['categories'])

            if category_sequence:
                print(f"\n📊 Category Switching Pattern:")
                print(f"   {' → '.join(category_sequence[:8])}")
                if len(category_sequence) > 8:
                    print(f"   ... ({len(category_sequence) - 8} more)")

                # Identify first category queried
                first_category = category_sequence[0] if category_sequence else None
                print(f"\n   First category queried: {first_category}")

                # Category diversity
                unique_categories = len(set(category_sequence))
                print(f"   Unique categories used: {unique_categories}")

            all_category_timelines.append({
                'question_idx': idx,
                'timeline': category_timeline,
                'category_sequence': category_sequence,
                'categories_available': categories_available,
                'categories_queried': categories_queried
            })

        else:
            print(f"\n   ⚠️ No category-specific queries detected")

    # Aggregate analysis
    print(f"\n\n{'='*100}")
    print("AGGREGATE CATEGORY ANALYSIS")
    print(f"{'='*100}\n")

    # Count category usage
    category_usage_count = defaultdict(int)
    category_first_query = defaultdict(int)

    for timeline_data in all_category_timelines:
        for cat in timeline_data['categories_queried']:
            category_usage_count[cat] += 1

        # Track first category queried
        if timeline_data['category_sequence']:
            first_cat = timeline_data['category_sequence'][0]
            category_first_query[first_cat] += 1

    print(f"📊 Category Usage Frequency:")
    print(f"   Category       │ Used in N questions │ As First Query")
    print(f"   ───────────────┼─────────────────────┼───────────────")
    for cat in sorted(category_usage_count.keys()):
        usage = category_usage_count[cat]
        first = category_first_query.get(cat, 0)
        usage_pct = (usage / len(results)) * 100
        first_pct = (first / len(results)) * 100 if first > 0 else 0
        print(f"   {cat:14s} │ {usage}/{len(results)} ({usage_pct:4.0f}%)         │ {first}/{len(results)} ({first_pct:4.0f}%)")

    # Category switching patterns
    print(f"\n🔄 Category Switching Insights:")

    # Most common first category
    if category_first_query:
        most_common_first = max(category_first_query.items(), key=lambda x: x[1])
        print(f"   Most common starting category: {most_common_first[0]} ({most_common_first[1]}/{len(results)} questions)")

    # Average categories per question
    avg_categories = sum(
        len(set(t['category_sequence']))
        for t in all_category_timelines
    ) / len(all_category_timelines) if all_category_timelines else 0

    print(f"   Avg categories per question: {avg_categories:.1f}")

    # Category pairs (which categories are queried together?)
    category_pairs = defaultdict(int)
    for timeline_data in all_category_timelines:
        cats_used = set(timeline_data['categories_queried'])
        for cat1 in cats_used:
            for cat2 in cats_used:
                if cat1 != cat2:
                    pair = tuple(sorted([cat1, cat2]))
                    category_pairs[pair] += 1

    if category_pairs:
        print(f"\n🔗 Category Co-occurrence (cross-referencing):")
        sorted_pairs = sorted(category_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        for pair, count in sorted_pairs:
            print(f"   {pair[0]} + {pair[1]}: {count}/{len(results)} questions")

    # Async streaming implications
    print(f"\n{'='*100}")
    print("ASYNC STREAMING UI IMPLICATIONS")
    print(f"{'='*100}\n")

    print("🎨 Progressive UI Updates:")
    print()
    print("For enterprise users, show real-time category querying:")
    print()
    print("   Example Timeline (Async Streaming):")
    print("   ───────────────────────────────────────────────────────────")
    print("   [2s]  🔍 Checking CLIENT_DB...")
    print("         💾 Found: Customer record for 'John Doe'")
    print("         Progress: ▓▓▓▓▓░░░░░░░░░░░░░░░░ 33%")
    print()
    print("   [4s]  🔍 Checking KNOWLEDGE base...")
    print("         📚 Found: Product specifications")
    print("         Progress: ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░ 67%")
    print()
    print("   [6s]  🔍 Cross-referencing POLICY documents...")
    print("         📋 Found: Warranty terms")
    print("         Progress: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 100%")
    print()
    print("   [8s]  ✅ Answer: Customer eligible for replacement under warranty")
    print("   ───────────────────────────────────────────────────────────")
    print()

    print("✅ Benefits of Category-Aware Streaming:")
    print("   1. Users see WHICH source is being checked")
    print("   2. Transparent cross-referencing visible")
    print("   3. Progress tied to source categories (not just iterations)")
    print("   4. Builds trust: 'It's checking the right sources'")
    print()

    print("📊 Expected Performance (from statistics):")
    stats = data.get('statistics', {})
    avg_queries = stats.get('avg_queries', 0)
    avg_cats = stats.get('avg_categories_per_question', 0)
    avg_iters = stats.get('avg_iterations', 0)

    print(f"   Avg queries: {avg_queries:.1f}")
    print(f"   Avg categories queried: {avg_cats:.1f}")
    print(f"   Avg iterations: {avg_iters:.1f}")
    print(f"   Expected time: ~{avg_iters * 2:.0f} seconds")
    print()

    print("🎯 Progressive Convergence Pattern:")
    if avg_cats >= 2:
        print(f"   • After 1st category query: ~{100/avg_cats:.0f}% complete")
        print(f"   • After 2nd category query: ~{200/avg_cats:.0f}% complete")
        if avg_cats >= 3:
            print(f"   • After 3rd category query: ~{300/avg_cats:.0f}% complete")
    print()

    print("🚀 Recommended UI Updates:")
    print("   • Show category badges: [CLIENT_DB] [POLICY] [KNOWLEDGE]")
    print("   • Highlight active category being queried")
    print("   • Display facts discovered per category")
    print("   • Cross-reference arrows: CLIENT_DB → POLICY")
    print()


def visualize_category_timeline(results_file='rlm_source_category_results.json'):
    """
    Create ASCII visualization of category query patterns.
    """
    print(f"\n{'='*100}")
    print("CATEGORY QUERY PATTERN VISUALIZATION")
    print(f"{'='*100}\n")

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return

    results = data.get('results', [])

    for idx, result in enumerate(results[:3], 1):  # Show first 3
        question = result.get('question', 'Unknown')[:60]
        trajectory = result.get('trajectory', [])

        print(f"Question {idx}: {question}...")
        print(f"{'─'*100}")

        # Create timeline
        print("\nCategory Usage Timeline:")
        print("Iteration │ Categories")
        print("──────────┼" + "─"*80)

        categories = ['POLICY', 'MANUAL', 'CLIENT_DB', 'CONTRACT', 'KNOWLEDGE']

        for iter_idx, step in enumerate(trajectory[:15], 1):
            code = step.get('code', '')

            # Detect categories
            cats_found = []
            for cat in categories:
                if cat.lower() in code.lower():
                    cats_found.append(cat)

            if cats_found:
                cat_str = '  '.join(f"[{c}]" for c in cats_found)
                print(f"    {iter_idx:2d}    │ {cat_str}")
            else:
                if 'llm_query' in code:
                    print(f"    {iter_idx:2d}    │ [General query]")
                else:
                    print(f"    {iter_idx:2d}    │ ·")

        print()


if __name__ == "__main__":
    print("\n🏢 Enterprise Category-Based Convergence Analysis\n")

    # Main analysis
    analyze_category_query_timeline()

    # Visualization
    visualize_category_timeline()

    print(f"\n{'='*100}")
    print("SUMMARY: Source-Category Chunking for Enterprise")
    print(f"{'='*100}\n")

    print("✅ Key Findings:")
    print("   • Documents organized by SOURCE TYPE (not arbitrary chunks)")
    print("   • RLM queries specific categories for targeted information")
    print("   • Cross-referencing visible: CLIENT_DB + POLICY + KNOWLEDGE")
    print("   • Enterprise-ready: Mirrors real document organization")
    print()

    print("🎯 Async Streaming Advantages:")
    print("   • Show users WHICH source category being checked")
    print("   • Progressive information from different sources")
    print("   • Transparent reasoning: 'Checking client records...'")
    print("   • Builds trust through visible cross-referencing")
    print()

    print("📋 Next Steps:")
    print("   1. Run source-category test: python test_rlm_source_categories.py")
    print("   2. Implement enterprise UI with category badges")
    print("   3. Test on real enterprise document sets")
    print("   4. Add category-specific progress indicators")
