"""
Analyze MuSiQue dataset structure for recursive refinement suitability
"""
from datasets import load_dataset
import statistics

print("Loading MuSiQue validation dataset...")
data = load_dataset("bdsaglam/musique", split="validation")

print("\n" + "="*80)
print("DATASET SCHEMA")
print("="*80)
print(f"\nDataset features:\n{data.features}\n")

print("\n" + "="*80)
print("DETAILED VIEW: FIRST 5 EXAMPLES")
print("="*80)

for i in range(min(5, len(data))):
    example = data[i]
    print(f"\n{'='*80}")
    print(f"EXAMPLE {i+1}")
    print(f"{'='*80}")

    # Print all available fields
    print(f"\nAvailable fields: {list(example.keys())}")

    # Print each field
    for key, value in example.items():
        print(f"\n--- {key.upper()} ---")
        if isinstance(value, str):
            if len(value) > 500:
                print(f"{value[:500]}... (truncated, total length: {len(value)})")
            else:
                print(value)
        elif isinstance(value, list):
            print(f"List with {len(value)} items:")
            if value and isinstance(value[0], dict):
                # Print first item structure
                print(f"  First item keys: {list(value[0].keys())}")
                print(f"  First item sample:")
                for k, v in value[0].items():
                    if isinstance(v, str) and len(v) > 100:
                        print(f"    {k}: {v[:100]}...")
                    else:
                        print(f"    {k}: {v}")
                if len(value) > 1:
                    print(f"  ... and {len(value)-1} more items")
            else:
                # Print all items if simple list
                for item in value[:5]:
                    if isinstance(item, str) and len(item) > 100:
                        print(f"  - {item[:100]}...")
                    else:
                        print(f"  - {item}")
                if len(value) > 5:
                    print(f"  ... and {len(value)-5} more items")
        else:
            print(value)

print("\n\n" + "="*80)
print("STATISTICS ACROSS 50 EXAMPLES")
print("="*80)

sample_size = min(50, len(data))
sample = data.select(range(sample_size))

# Collect statistics
paragraph_counts = []
question_lengths = []
answer_lengths = []
field_presence = {}

for example in sample:
    # Track which fields are present
    for key in example.keys():
        field_presence[key] = field_presence.get(key, 0) + 1

    # Question and answer lengths
    if 'question' in example:
        question_lengths.append(len(example['question']))
    if 'answer' in example:
        answer_lengths.append(len(str(example['answer'])))

    # Count paragraphs/documents
    if 'paragraphs' in example and isinstance(example['paragraphs'], list):
        paragraph_counts.append(len(example['paragraphs']))
    elif 'context' in example and isinstance(example['context'], list):
        paragraph_counts.append(len(example['context']))

print(f"\nSample size: {sample_size} examples")

print("\n--- FIELD CONSISTENCY ---")
for field, count in sorted(field_presence.items()):
    percentage = (count / sample_size) * 100
    print(f"{field}: {count}/{sample_size} ({percentage:.1f}%)")

if paragraph_counts:
    print("\n--- PARAGRAPH/DOCUMENT STATISTICS ---")
    print(f"Average paragraphs per question: {statistics.mean(paragraph_counts):.2f}")
    print(f"Min paragraphs: {min(paragraph_counts)}")
    print(f"Max paragraphs: {max(paragraph_counts)}")
    print(f"Median paragraphs: {statistics.median(paragraph_counts)}")

if question_lengths:
    print("\n--- QUESTION LENGTH STATISTICS ---")
    print(f"Average question length: {statistics.mean(question_lengths):.2f} chars")
    print(f"Min question length: {min(question_lengths)} chars")
    print(f"Max question length: {max(question_lengths)} chars")

if answer_lengths:
    print("\n--- ANSWER LENGTH STATISTICS ---")
    print(f"Average answer length: {statistics.mean(answer_lengths):.2f} chars")
    print(f"Min answer length: {min(answer_lengths)} chars")
    print(f"Max answer length: {max(answer_lengths)} chars")

print("\n\n" + "="*80)
print("MULTI-DOCUMENT STRUCTURE ASSESSMENT")
print("="*80)

# Analyze if suitable for recursive refinement
has_multi_doc = any('paragraphs' in example or 'context' in example for example in sample)
avg_paragraphs = statistics.mean(paragraph_counts) if paragraph_counts else 0

print(f"\nHas multi-document structure: {has_multi_doc}")
print(f"Average documents/paragraphs: {avg_paragraphs:.2f}")

if avg_paragraphs >= 5:
    print("\n✓ SUITABLE for recursive refinement testing")
    print(f"  - Questions have {avg_paragraphs:.1f} documents on average")
    print("  - Can test iterative retrieval and reasoning")
elif avg_paragraphs >= 2:
    print("\n⚠ POSSIBLY SUITABLE for recursive refinement testing")
    print(f"  - Questions have {avg_paragraphs:.1f} documents on average")
    print("  - Limited multi-hop structure")
else:
    print("\n✗ NOT SUITABLE for recursive refinement testing")
    print(f"  - Questions have {avg_paragraphs:.1f} documents on average")
    print("  - Insufficient multi-document structure")

print("\n" + "="*80)
