"""
RLM with Source-Category-Based Chunking for Enterprise Use Cases

Uses REAL enterprise dataset with actual multi-source documents:
- CLIENT_DB: Customer/employee records
- CONTRACT: Legal agreements, warranties
- POLICY: Company policies, guidelines
- KNOWLEDGE: FAQs, reference material
- MANUAL: Procedures, process guides

Tests progressive refinement across actual enterprise source categories.
"""

import dspy
from typing import Dict, List
import json


class EnterpriseMultiHopSignature(dspy.Signature):
    """
    Answer multi-source questions by querying across source categories.

    Available categories: POLICY, MANUAL, CLIENT_DB, CONTRACT, KNOWLEDGE
    Use llm_query() to retrieve specific chunks by category prefix.
    """

    source_categories = dspy.InputField(
        desc="Summary of available source categories and chunk counts"
    )
    documents = dspy.InputField(
        desc="All document chunks with category-prefixed IDs (e.g., 'client_db_JohnDoe_chunk_1')"
    )
    question = dspy.InputField(
        desc="Multi-hop question requiring cross-category information"
    )
    answer = dspy.OutputField(
        desc="Final answer synthesized from multiple source categories"
    )


def test_source_category_chunking():
    """
    Test RLM with source-category-based chunking on REAL enterprise dataset.

    Real enterprise questions requiring:
    - Cross-referencing multiple source categories
    - CLIENT_DB + CONTRACT + POLICY coordination
    - Multi-source fact discovery
    """

    print("="*100)
    print("RLM WITH SOURCE-CATEGORY-BASED CHUNKING")
    print("Enterprise Use Case: Cross-Reference Across Document Categories")
    print("="*100)

    # Load REAL enterprise dataset
    print("\n📥 Loading enterprise dataset...")
    with open('enterprise_dataset.json', 'r') as f:
        dataset = json.load(f)

    questions = dataset['questions']
    print(f"✅ Loaded {len(questions)} enterprise questions\n")

    # Setup LLM
    lm = dspy.LM(model='openai/gpt-4o-mini', temperature=0.0)
    dspy.configure(lm=lm, experimental=True)

    # Configure RLM
    config = {
        'max_iterations': 8,
        'max_llm_calls': 20,
        'num_questions': len(questions)
    }

    print(f"⚙️  Configuration:")
    print(f"   Dataset: Real enterprise multi-source questions")
    print(f"   Chunking: By SOURCE CATEGORY (POLICY, CLIENT_DB, CONTRACT, etc.)")
    print(f"   Max iterations: {config['max_iterations']}")
    print(f"   Max LLM calls: {config['max_llm_calls']}")
    print(f"   Test questions: {config['num_questions']}")

    # Run test
    rlm = dspy.ReAct(EnterpriseMultiHopSignature, max_iters=config['max_iterations'])

    results = []

    for idx, qa_item in enumerate(questions, 1):
        question = qa_item['question']
        gold_answer = qa_item['answer']
        sources = qa_item['sources']

        print(f"\n{'='*100}")
        print(f"QUESTION {idx}/{len(questions)}")
        print(f"{'='*100}")
        print(f"\n❓ Question: {question}")
        print(f"🎯 Gold Answer: {gold_answer}")

        # Prepare documents from source categories (already categorized!)
        all_docs = {}
        category_summary_parts = []

        print(f"\n📂 Source Categories:")
        for category, docs in sources.items():
            doc_count = len(docs)
            print(f"   {category}: {doc_count} documents")
            category_summary_parts.append(f"{category}: {doc_count} documents")

            # Add documents with category prefix
            for doc_id, content in docs.items():
                prefixed_id = f"{category.lower()}_{doc_id}"
                all_docs[prefixed_id] = content

        category_summary = "AVAILABLE SOURCE CATEGORIES:\n" + "\n".join(
            f"- {part}" for part in category_summary_parts
        )

        print(f"\n   Total documents: {len(all_docs)}")
        print(f"\n📋 Category Summary:")
        print(category_summary)

        # Run RLM
        print(f"\n🔄 Running RLM with source-category documents...")

        try:
            pred = rlm(
                source_categories=category_summary,
                documents=all_docs,
                question=question
            )

            predicted_answer = pred.answer if hasattr(pred, 'answer') else str(pred)
            trajectory = pred.trajectory if hasattr(pred, 'trajectory') else []

            # Analyze category usage
            categories_queried = set()
            categories_available = list(sources.keys())

            for step in trajectory:
                code = step.get('code', '')
                for category in categories_available:
                    if category.lower() in code.lower():
                        categories_queried.add(category)

            print(f"\n✅ RLM Completed")
            print(f"   Predicted: {predicted_answer}")
            print(f"   Gold: {gold_answer}")
            print(f"   Categories available: {', '.join(categories_available)}")
            print(f"   Categories queried: {', '.join(sorted(categories_queried)) if categories_queried else 'None detected'}")
            print(f"   Iterations used: {len(trajectory)}/{config['max_iterations']}")

            # Count queries
            total_queries = sum(
                step.get('code', '').count('llm_query')
                for step in trajectory
            )
            print(f"   Total queries: {total_queries}")

            results.append({
                'question_id': qa_item['id'],
                'question': question,
                'gold_answer': gold_answer,
                'predicted': predicted_answer,
                'trajectory': trajectory,
                'categories_available': categories_available,
                'categories_queried': list(categories_queried),
                'total_queries': total_queries,
                'iterations': len(trajectory)
            })

        except Exception as e:
            print(f"\n❌ Error: {e}")
            results.append({
                'question': question,
                'gold_answer': gold_answer,
                'predicted': f"ERROR: {e}",
                'trajectory': [],
                'error': str(e)
            })

    # Save results
    output = {
        'config': config,
        'chunking_strategy': 'source_category',
        'categories': ['POLICY', 'MANUAL', 'CLIENT_DB', 'CONTRACT', 'KNOWLEDGE'],
        'results': results,
        'statistics': {
            'total_questions': len(results),
            'avg_queries': sum(r.get('total_queries', 0) for r in results) / len(results),
            'avg_iterations': sum(r.get('iterations', 0) for r in results) / len(results),
            'avg_categories_per_question': sum(
                len(r.get('categories_queried', []))
                for r in results
            ) / len(results)
        }
    }

    output_file = 'rlm_source_category_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*100}")
    print("FINAL STATISTICS")
    print(f"{'='*100}\n")

    stats = output['statistics']
    print(f"📊 Overall Performance:")
    print(f"   Questions tested: {stats['total_questions']}")
    print(f"   Avg queries per question: {stats['avg_queries']:.1f}")
    print(f"   Avg iterations: {stats['avg_iterations']:.1f}")
    print(f"   Avg categories queried: {stats['avg_categories_per_question']:.1f}")

    print(f"\n💡 Category Usage Analysis:")
    all_categories_used = set()
    for r in results:
        all_categories_used.update(r.get('categories_queried', []))

    print(f"   Categories used: {', '.join(sorted(all_categories_used))}")

    # Per-category usage count
    category_usage = {}
    for r in results:
        for cat in r.get('categories_queried', []):
            category_usage[cat] = category_usage.get(cat, 0) + 1

    if category_usage:
        print(f"\n   Usage frequency:")
        for cat in sorted(category_usage.keys()):
            count = category_usage[cat]
            pct = (count / len(results)) * 100
            print(f"      {cat}: {count}/{len(results)} questions ({pct:.0f}%)")

    print(f"\n💾 Results saved to: {output_file}")

    print(f"\n{'='*100}")
    print("KEY INSIGHTS: Source-Category Chunking")
    print(f"{'='*100}\n")

    print("✅ Enterprise Benefits:")
    print("   1. Documents organized by SOURCE TYPE (policy, client, manual, etc.)")
    print("   2. RLM can target specific categories in queries")
    print("   3. Enables cross-referencing (e.g., policy + client history)")
    print("   4. Mirrors real enterprise document organization")
    print()
    print("🎯 Use Cases:")
    print("   • Legal: Cross-reference contracts + client records + policies")
    print("   • Support: Link manuals + client tickets + knowledge base")
    print("   • Compliance: Check policies + contracts + regulatory docs")
    print()
    print("📈 Progressive Refinement:")
    print("   • Query CLIENT_DB for entity info → 33% complete")
    print("   • Query KNOWLEDGE for reference → 67% complete")
    print("   • Cross-reference categories → 100% complete")
    print()
    print("🚀 Async Streaming Potential:")
    print("   • Show which category being queried in real-time")
    print("   • Users see cross-category reasoning")
    print("   • Transparent: 'Checking client records...' → 'Referencing policy...'")


if __name__ == "__main__":
    print("\n🏢 Enterprise Source-Category Chunking Test\n")
    test_source_category_chunking()

    print("\n\n" + "="*100)
    print("NEXT: Analyze category-based progressive convergence")
    print("="*100)
    print("\nRun: python analyze_category_convergence.py")
