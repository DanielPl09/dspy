# RLM Multi-Hop Iterative Refinement Plan

## Objective
Test RLM's iterative refinement capabilities on multi-hop question answering with a budget under $0.50

## Approved Configuration

### Dataset
- **HotPotQA** real-world benchmark (filtered subset)
- 5 examples for development
- 15-20 examples for evaluation
- Focus on 2-3 hop questions

### Priorities
- **Accuracy over cost** - maximize correctness
- Summary statistics output
- Stay under $0.50 total budget

## Token Budget Analysis

### OpenAI gpt-4o-mini Pricing (Jan 2026)
- Input: $0.15 per 1M tokens (~$0.00015 per 1K tokens)
- Output: $0.60 per 1M tokens (~$0.00060 per 1K tokens)

### Budget Allocation ($0.50 total)
- **Development/Testing**: $0.20 (40%)
  - Initial RLM tests: 3-5 examples
  - Debugging and iteration
- **Baseline Evaluation**: $0.10 (20%)
  - Simple multi-hop without refinement
  - 10-15 examples
- **Iterative Refinement**: $0.20 (40%)
  - RLM with refinement loop
  - 8-12 examples

### Estimated Token Usage per Example
**Without RLM (baseline):**
- Input: ~2K tokens (question + context)
- Output: ~500 tokens (answer)
- Cost per example: ~$0.0006

**With RLM (iterative refinement):**
- Main LM calls: 3-5 iterations × 3K tokens input × 800 tokens output
- Sub-LM calls (llm_query): 5-10 calls × 1K tokens each
- Estimated: ~25-40K tokens total per example
- Cost per example: ~$0.015-$0.025

**Budget allows: ~15-30 RLM examples within $0.50**

## Multi-Hop Scenario: HotPotQA

**Task**: Answer questions requiring 2-3 hops of reasoning
**Dataset**: HotPotQA subset (5-10 examples)
**Example Question**:
"What is the capital of the country where the director of Inception was born?"
- Hop 1: Find director of Inception (Christopher Nolan)
- Hop 2: Find Nolan's birthplace (UK)
- Hop 3: Find capital of UK (London)

**Why this approach:**
- Real-world benchmark dataset
- Clear evaluation metrics
- Demonstrates actual multi-hop reasoning
- Industry-standard evaluation

## RLM Implementation Strategy

### Phase 1: Baseline (Simple Predict)
```python
# Simple baseline without RLM
baseline = dspy.ChainOfThought("question, context -> answer")
```

### Phase 2: RLM Without Refinement
```python
# RLM doing basic multi-hop reasoning
rlm = dspy.RLM(
    "question, context -> reasoning_steps: list[str], answer: str",
    max_iterations=5,
    max_llm_calls=10
)
```

### Phase 3: RLM With Iterative Refinement
```python
# RLM with self-refinement loop - PRIORITIZE ACCURACY
rlm_refine = dspy.RLM(
    """question, context, previous_attempts: list[str] ->
       analysis: str,
       refined_reasoning: list[str],
       confidence: float,
       final_answer: str""",
    max_iterations=10,  # Increased for accuracy
    max_llm_calls=25    # Increased for accuracy
)
```

## Evaluation Metrics

1. **Accuracy**: Exact match on final answer (PRIMARY METRIC)
2. **F1 Score**: Token-level overlap with gold answer
3. **Hop Coverage**: % of reasoning steps that match gold reasoning path
4. **Cost per example**: Track spending
5. **Iteration Analysis**: How many iterations needed for correct answer

## Implementation Steps

### Step 1: Setup & Data Preparation (5 examples)
- Load HotPotQA dataset
- Filter 2-3 hop questions
- Prepare evaluation function
- Set up cost tracking

### Step 2: Baseline Evaluation (~10 examples, ~$0.05)
- Run simple ChainOfThought
- Measure baseline accuracy
- Document failure modes

### Step 3: RLM Without Refinement (~8 examples, ~$0.12)
- Implement basic RLM
- Compare to baseline
- Analyze reasoning traces

### Step 4: RLM With Iterative Refinement (~8 examples, ~$0.15)
- Add refinement loop
- Compare improvements
- Measure when refinement helps

### Step 5: Analysis & Report (~$0.08 buffer)
- Compare all approaches
- Identify best practices
- Generate summary statistics

## Expected Files to Create

1. `test_rlm_multihop.py` - Main test script with cost tracking
2. `multihop_data.json` - HotPotQA filtered dataset (5-20 examples)
3. `multihop_results.json` - Evaluation results with metrics
4. `RESULTS_multihop.md` - Summary statistics and findings

## Success Criteria

✓ Stay under $0.50 budget
✓ Test at least 20 total examples across all approaches
✓ Maximize accuracy with iterative refinement
✓ Document when/why refinement helps
✓ Provide reusable code for multi-hop RLM patterns
✓ Generate clear summary statistics

## Risk Mitigation

1. **Token Overflow**: Monitor token usage, adjust max_llm_calls if needed
2. **Cost Overrun**: Track costs after each example, stop at $0.45 (safety margin)
3. **API Failures**: Implement retry logic with exponential backoff
4. **Poor Results**: If accuracy is low, increase max_iterations (accuracy priority)

## Cost Tracking Implementation

```python
import time
from typing import Dict

class CostTracker:
    def __init__(self, budget: float = 0.50):
        self.budget = budget
        self.total_cost = 0.0
        self.input_tokens = 0
        self.output_tokens = 0

    def add_tokens(self, input_tok: int, output_tok: int):
        # gpt-4o-mini pricing
        cost = (input_tok * 0.00015 / 1000) + (output_tok * 0.00060 / 1000)
        self.total_cost += cost
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        return cost

    def remaining_budget(self) -> float:
        return self.budget - self.total_cost

    def can_continue(self) -> bool:
        return self.total_cost < (self.budget * 0.9)  # 90% safety threshold
```

## Timeline

1. ✓ Plan approved
2. → Create HotPotQA dataset loader and filter
3. → Implement baseline ChainOfThought
4. → Implement RLM basic variant
5. → Implement RLM refinement variant
6. → Run evaluations with cost tracking
7. → Generate summary statistics report
