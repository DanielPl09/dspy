"""
RLM with Source-Category-Based Chunking for Enterprise Use Cases

Enterprise Scenario:
- Documents organized by SOURCE CATEGORY (policies, manuals, client DB, etc.)
- RLM queries specific categories to find cross-referenced information
- Simulates real-world: legal case requiring policy + client history + contract terms

This tests progressive refinement across heterogeneous source categories.
"""

import dspy
from datasets import load_dataset
from typing import Dict, List
import json


class SourceCategoryChunker:
    """
    Chunks documents by source category for enterprise scenarios.

    Categories:
    - POLICY: Company policies, legal documents
    - MANUAL: Technical manuals, product docs
    - CLIENT_DB: Customer records, support tickets
    - CONTRACT: Agreements, terms of service
    - KNOWLEDGE: FAQ, wiki articles
    """

    def __init__(self, chunk_size: int = 150):
        self.chunk_size = chunk_size

    def categorize_source(self, title: str, content: str) -> str:
        """
        Determine source category from document metadata.

        For HotPotQA simulation:
        - Person bios → CLIENT_DB (customer/entity records)
        - Film/TV/Book info → KNOWLEDGE (reference material)
        - Location/Organization → POLICY (entity policies/info)
        - Legal/Government → CONTRACT (agreements/terms)
        """
        title_lower = title.lower()
        content_lower = content.lower()

        # Simulate category assignment
        if any(word in title_lower for word in ['film', 'album', 'song', 'book', 'series']):
            return 'KNOWLEDGE'
        elif any(word in content_lower[:200] for word in ['born', 'nationality', 'actor', 'director', 'writer']):
            return 'CLIENT_DB'  # Person records
        elif any(word in title_lower for word in ['company', 'organization', 'government']):
            return 'POLICY'
        elif any(word in content_lower[:200] for word in ['agreement', 'contract', 'terms']):
            return 'CONTRACT'
        else:
            return 'MANUAL'  # Default reference

    def chunk_by_source_category(
        self,
        titles: List[str],
        contents: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Create category-aware chunks.

        Returns:
            {
                'POLICY': {
                    'policy_CompanyX_chunk_1': 'content...',
                    'policy_CompanyX_chunk_2': 'content...'
                },
                'CLIENT_DB': {
                    'client_JohnDoe_chunk_1': 'content...',
                    'client_JohnDoe_chunk_2': 'content...'
                },
                ...
            }
        """
        categorized_chunks = {
            'POLICY': {},
            'MANUAL': {},
            'CLIENT_DB': {},
            'CONTRACT': {},
            'KNOWLEDGE': {}
        }

        for title, content in zip(titles, contents):
            # Determine category
            category = self.categorize_source(title, content)

            # Chunk the document
            words = content.split()

            for i in range(0, len(words), self.chunk_size):
                chunk_words = words[i:i + self.chunk_size]
                chunk_num = i // self.chunk_size + 1

                # Create category-prefixed chunk ID
                chunk_id = f"{category.lower()}_{title}_chunk_{chunk_num}"
                chunk_content = ' '.join(chunk_words)

                categorized_chunks[category][chunk_id] = chunk_content

        return categorized_chunks

    def flatten_for_rlm(self, categorized_chunks: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """Flatten category structure for RLM input."""
        flat_chunks = {}
        for category, chunks in categorized_chunks.items():
            flat_chunks.update(chunks)
        return flat_chunks

    def get_category_summary(self, categorized_chunks: Dict[str, Dict[str, str]]) -> str:
        """Create summary of available sources for RLM context."""
        summary = "AVAILABLE SOURCE CATEGORIES:\n"
        for category, chunks in categorized_chunks.items():
            if chunks:
                summary += f"\n{category}:\n"
                summary += f"  - {len(chunks)} chunks available\n"
                # Show sample chunk IDs
                sample_ids = list(chunks.keys())[:3]
                for chunk_id in sample_ids:
                    summary += f"    • {chunk_id}\n"
                if len(chunks) > 3:
                    summary += f"    • ... and {len(chunks) - 3} more\n"
        return summary


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
    Test RLM with source-category-based chunking on HotPotQA.

    Simulates enterprise scenario:
    - Documents from different categories
    - RLM must identify relevant categories
    - Query across categories to find answer
    """

    print("="*100)
    print("RLM WITH SOURCE-CATEGORY-BASED CHUNKING")
    print("Enterprise Use Case: Cross-Reference Across Document Categories")
    print("="*100)

    # Load dataset
    print("\n📥 Loading HotPotQA dataset...")
    dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
    multihop = [ex for ex in dataset if ex["level"] == "hard"][:5]

    print(f"✅ Loaded {len(multihop)} multi-hop questions\n")

    # Setup LLM
    lm = dspy.LM(model='openai/gpt-4o-mini', temperature=0.0)
    dspy.configure(lm=lm, experimental=True)

    # Create chunker
    chunker = SourceCategoryChunker(chunk_size=150)

    # Configure RLM
    config = {
        'max_iterations': 12,
        'max_llm_calls': 30,
        'num_questions': len(multihop)
    }

    print(f"⚙️  Configuration:")
    print(f"   Chunking strategy: SOURCE CATEGORY (POLICY, CLIENT_DB, KNOWLEDGE, etc.)")
    print(f"   Chunk size: ~150 words per chunk")
    print(f"   Max iterations: {config['max_iterations']}")
    print(f"   Max LLM calls: {config['max_llm_calls']}")
    print(f"   Test questions: {config['num_questions']}")

    # Run test
    rlm = dspy.ReAct(EnterpriseMultiHopSignature, max_iters=config['max_iterations'])

    results = []

    for idx, example in enumerate(multihop, 1):
        question = example['question']
        gold_answer = example['answer']
        titles = example['supporting_facts']['title']
        sentences = example['supporting_facts']['sent']

        # Reconstruct full documents
        title_to_content = {}
        for title, sent in zip(titles, sentences):
            if title not in title_to_content:
                title_to_content[title] = []
            title_to_content[title].append(sent)

        doc_titles = list(title_to_content.keys())
        doc_contents = [' '.join(title_to_content[t]) for t in doc_titles]

        print(f"\n{'='*100}")
        print(f"QUESTION {idx}/{len(multihop)}")
        print(f"{'='*100}")
        print(f"\n❓ Question: {question}")
        print(f"🎯 Gold Answer: {gold_answer}")

        # Create category-based chunks
        categorized_chunks = chunker.chunk_by_source_category(doc_titles, doc_contents)
        flat_chunks = chunker.flatten_for_rlm(categorized_chunks)
        category_summary = chunker.get_category_summary(categorized_chunks)

        print(f"\n📂 Source Categories:")
        for category, chunks in categorized_chunks.items():
            if chunks:
                print(f"   {category}: {len(chunks)} chunks")

        print(f"\n   Total chunks: {len(flat_chunks)}")
        print(f"\n📋 Category Summary:")
        print(category_summary)

        # Run RLM
        print(f"\n🔄 Running RLM with category-aware chunking...")

        try:
            pred = rlm(
                source_categories=category_summary,
                documents=flat_chunks,
                question=question
            )

            predicted_answer = pred.answer if hasattr(pred, 'answer') else str(pred)
            trajectory = pred.trajectory if hasattr(pred, 'trajectory') else []

            # Analyze category usage
            categories_queried = set()
            for step in trajectory:
                code = step.get('code', '')
                for category in ['policy', 'client_db', 'knowledge', 'manual', 'contract']:
                    if category in code.lower():
                        categories_queried.add(category.upper())

            print(f"\n✅ RLM Completed")
            print(f"   Predicted: {predicted_answer}")
            print(f"   Gold: {gold_answer}")
            print(f"   Categories queried: {', '.join(sorted(categories_queried)) if categories_queried else 'None detected'}")
            print(f"   Iterations used: {len(trajectory)}/{config['max_iterations']}")

            # Count queries
            total_queries = sum(
                step.get('code', '').count('llm_query')
                for step in trajectory
            )
            print(f"   Total queries: {total_queries}")

            results.append({
                'question': question,
                'gold_answer': gold_answer,
                'predicted': predicted_answer,
                'trajectory': trajectory,
                'categories_available': {
                    cat: len(chunks)
                    for cat, chunks in categorized_chunks.items()
                    if chunks
                },
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
