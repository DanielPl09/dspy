"""
Test file to peek at 20 multi-hop questions from HotPotQA dataset.

HotPotQA is a question answering dataset that requires reasoning over multiple
supporting documents (multi-hop reasoning). Questions are categorized as:
- bridge: questions that require bridging between entities
- comparison: questions that require comparing properties of entities
"""

from datasets import load_dataset


def peek_multihop_questions(num_questions=20):
    """Download and display multi-hop questions from HotPotQA dataset."""

    print("=" * 80)
    print("Loading HotPotQA Dataset...")
    print("=" * 80)

    # Load the dataset directly using Hugging Face datasets
    # The 'fullwiki' configuration contains the multi-hop questions
    hf_dataset = load_dataset("hotpot_qa", "fullwiki", split="train")

    print(f"\nDataset loaded successfully!")
    print(f"Total examples in dataset: {len(hf_dataset)}")

    # Filter for hard (multi-hop) questions only
    multihop_examples = [ex for ex in hf_dataset if ex["level"] == "hard"]
    print(f"Total multi-hop (hard) questions: {len(multihop_examples)}")

    print(f"\n{'=' * 80}")
    print(f"Displaying {num_questions} Multi-Hop Questions from HotPotQA")
    print(f"{'=' * 80}\n")

    # Display 20 multi-hop questions with details
    for idx, example in enumerate(multihop_examples[:num_questions], 1):
        print(f"\n{'─' * 80}")
        print(f"Question #{idx}")
        print(f"{'─' * 80}")
        print(f"Question: {example['question']}")
        print(f"Answer: {example['answer']}")
        print(f"Question Type: {example['type']}")
        print(f"Difficulty Level: {example['level']}")

        # Display supporting facts (titles of documents needed for answer)
        if example['supporting_facts'] and example['supporting_facts']['title']:
            supporting_titles = set(example['supporting_facts']['title'])
            print(f"Supporting Documents ({len(supporting_titles)}): {', '.join(sorted(supporting_titles))}")

    print(f"\n{'=' * 80}")
    print("Sample complete!")
    print(f"{'=' * 80}\n")

    # Show some statistics
    question_types = {}
    for example in multihop_examples[:num_questions]:
        qtype = example['type']
        question_types[qtype] = question_types.get(qtype, 0) + 1

    print("\nQuestion Type Distribution (in sample):")
    for qtype, count in sorted(question_types.items()):
        print(f"  {qtype}: {count}")


if __name__ == "__main__":
    peek_multihop_questions(num_questions=20)
