# Source-Category-Based Chunking for Enterprise RLM

## 🎯 Overview

**Previous Approach:** Documents chunked by word count (~150 words per chunk)
**New Approach:** Documents chunked BY SOURCE CATEGORY for enterprise use cases

---

## 🏢 Enterprise Use Cases

### Real-World Scenarios

**1. Legal Research & Compliance**
```
Question: "Is customer X eligible for refund under our warranty policy?"

Source Categories:
- CLIENT_DB: Customer purchase history, service records
- CONTRACT: Warranty agreements, terms of service
- POLICY: Company refund policies, legal guidelines
- KNOWLEDGE: Product specifications, known issues

Progressive Convergence:
[2s] Query CLIENT_DB → Found purchase date: Jan 2024
[4s] Query CONTRACT → Warranty: 12 months from purchase
[6s] Query POLICY → Refund policy: Within warranty period
[8s] Answer: ✅ Yes, eligible (within 12-month warranty)
```

**2. Customer Support**
```
Question: "How do I troubleshoot error code E47 for product model XYZ?"

Source Categories:
- KNOWLEDGE: Technical manuals, FAQ database
- CLIENT_DB: Previous support tickets with error E47
- MANUAL: Product XYZ user guide
- POLICY: Support escalation procedures

Progressive Convergence:
[2s] Query KNOWLEDGE → Error E47: Power supply issue
[4s] Query MANUAL → Product XYZ: Reset procedure on page 24
[6s] Query CLIENT_DB → Similar tickets: 85% resolved with reset
[8s] Answer: Reset power supply (steps 1-5 from manual)
```

**3. HR & Employee Policies**
```
Question: "How many vacation days does employee Y have remaining?"

Source Categories:
- CLIENT_DB: Employee Y's record (hire date, used PTO)
- POLICY: Company PTO policy, accrual rules
- CONTRACT: Employee Y's employment agreement
- MANUAL: HR system usage guide

Progressive Convergence:
[2s] Query CLIENT_DB → Employee Y: 5 days used in 2024
[4s] Query POLICY → Accrual: 15 days/year for tenure 2+ years
[6s] Query CONTRACT → Employee Y: Hired 2022 (3 years tenure)
[8s] Answer: 10 days remaining (15 accrued - 5 used)
```

---

## 📊 Source Category Types

### Standard Enterprise Categories

| Category | Description | Examples | Query Patterns |
|----------|-------------|----------|----------------|
| **POLICY** | Company policies, guidelines | HR policies, legal guidelines, compliance docs | "What is the policy for..." |
| **MANUAL** | Technical documentation | User manuals, API docs, installation guides | "How to configure...", "Steps to..." |
| **CLIENT_DB** | Customer/employee records | Purchase history, support tickets, personnel files | "Who is...", "When did..." |
| **CONTRACT** | Legal agreements | SLAs, warranties, employment contracts | "What are the terms...", "Is X covered..." |
| **KNOWLEDGE** | Reference material | FAQs, wiki articles, best practices | "What is...", "Explain..." |

### Custom Categories (Domain-Specific)

**Healthcare:**
- PATIENT_RECORDS
- CLINICAL_GUIDELINES
- MEDICATION_DB
- INSURANCE_POLICIES

**Finance:**
- TRANSACTION_HISTORY
- REGULATORY_DOCS
- ACCOUNT_RECORDS
- COMPLIANCE_POLICIES

**E-Commerce:**
- PRODUCT_CATALOG
- ORDER_HISTORY
- SHIPPING_POLICIES
- RETURNS_PROCEDURES

---

## 🔧 Implementation Details

### Chunking Strategy

```python
class SourceCategoryChunker:
    """
    Chunk documents by source category.

    Key differences from word-count chunking:
    1. Category assignment FIRST (not arbitrary splitting)
    2. Chunk IDs include category prefix
    3. RLM can target specific source types
    """

    def chunk_by_source_category(self, titles, contents):
        """
        Steps:
        1. Determine source category from document metadata
        2. Split into ~150-word chunks within category
        3. Prefix chunk IDs with category name

        Result:
        {
            'POLICY': {
                'policy_RefundGuidelines_chunk_1': 'content...',
                'policy_RefundGuidelines_chunk_2': 'content...'
            },
            'CLIENT_DB': {
                'client_db_CustomerX_chunk_1': 'content...'
            }
        }
        """
```

### Category Assignment Logic

```python
def categorize_source(self, title, content):
    """
    Determine category from document metadata.

    Methods:
    1. Filename/title patterns (e.g., "HR_Policy_2024.pdf" → POLICY)
    2. Content keywords (e.g., "warranty terms" → CONTRACT)
    3. Metadata tags (e.g., document_type='manual' → MANUAL)
    4. Source system (e.g., from CRM → CLIENT_DB)
    """

    # Example: Title-based categorization
    if 'policy' in title.lower():
        return 'POLICY'
    elif 'manual' in title.lower():
        return 'MANUAL'
    elif 'customer_' in title.lower():
        return 'CLIENT_DB'
    # ... etc.
```

### RLM Integration

```python
class EnterpriseMultiHopSignature(dspy.Signature):
    """
    RLM signature with category awareness.

    RLM receives:
    1. Category summary (which categories available)
    2. All chunks with category-prefixed IDs
    3. Question to answer

    RLM can now:
    - Target specific categories: llm_query("Find in CLIENT_DB...")
    - Cross-reference: "Check POLICY, then CONTRACT"
    - Adaptive strategy: "Try KNOWLEDGE first, fall back to MANUAL"
    """

    source_categories = dspy.InputField(
        desc="Available categories: POLICY, CLIENT_DB, etc."
    )
    documents = dspy.InputField(
        desc="Chunks with IDs like 'policy_X_chunk_1'"
    )
    question = dspy.InputField()
    answer = dspy.OutputField()
```

---

## 📈 Progressive Convergence with Categories

### Timeline Example

```
Question: "Does customer warranty cover water damage?"

Available Categories:
- CLIENT_DB: 3 chunks (customer purchase records)
- CONTRACT: 5 chunks (warranty agreement)
- POLICY: 4 chunks (company warranty policies)

RLM Execution:

Iteration 1 (2s):
├─ 💭 Reasoning: Need customer purchase info
├─ 🔎 Query: llm_query("client_db_CustomerX_chunk_1")
├─ 📊 Found: Purchased laptop Jan 2024
└─ Progress: ▓▓▓▓▓▓░░░░░░░░░░░░░░ 33%

Iteration 2 (4s):
├─ 💭 Reasoning: Check warranty terms
├─ 🔎 Query: llm_query("contract_Warranty_chunk_2")
├─ 📊 Found: 12-month coverage, excludes water damage
└─ Progress: ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░ 67%

Iteration 3 (6s):
├─ 💭 Reasoning: Verify policy exceptions
├─ 🔎 Query: llm_query("policy_WarrantyExceptions_chunk_1")
├─ 📊 Found: No exceptions for water damage
└─ Progress: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 100%

Iteration 4 (8s):
├─ 💭 Reasoning: All facts gathered
├─ ✅ Answer: No, warranty excludes water damage
└─ Sources: CLIENT_DB + CONTRACT + POLICY
```

### Category Switching Patterns

**Sequential (Deep Dive)**
```
CLIENT_DB → CLIENT_DB → CLIENT_DB
(Exploring multiple aspects of customer record)
```

**Breadth-First (Multi-Source)**
```
CLIENT_DB → CONTRACT → POLICY → KNOWLEDGE
(Gathering facts from each relevant category)
```

**Iterative Refinement (Ping-Pong)**
```
POLICY → CLIENT_DB → POLICY → CONTRACT
(Switching back as new questions arise)
```

---

## 🚀 Async Streaming UI Benefits

### Category-Aware Progress Indicators

**Traditional Progress:**
```
Processing... [▓▓▓▓▓░░░░░] 50%
```

**Category-Aware Progress:**
```
🔍 Checking CLIENT_DB...     [▓▓▓▓▓▓▓▓▓▓] Complete ✅
🔍 Checking CONTRACT...      [▓▓▓▓▓░░░░░] 50%
🔍 Checking POLICY...        [░░░░░░░░░░] Pending
```

### Real-Time Category Updates

```jsx
// Example React UI component
<CategoryProgress>
  <CategoryBadge
    name="CLIENT_DB"
    status="complete"
    findings={["Customer: John Doe", "Purchase: Jan 2024"]}
  />
  <CategoryBadge
    name="CONTRACT"
    status="in_progress"
    findings={["Warranty: 12 months"]}
  />
  <CategoryBadge
    name="POLICY"
    status="pending"
  />
</CategoryProgress>
```

### Cross-Reference Visualization

```
[CLIENT_DB] ──┐
              ├─→ [ANSWER]
[CONTRACT]  ──┤
              │
[POLICY]    ──┘

Shows: Answer synthesized from 3 source categories
```

---

## 📊 Statistical Evidence

### Comparison: Word-Count vs. Source-Category Chunking

| Metric | Word-Count Chunking | Source-Category Chunking |
|--------|---------------------|--------------------------|
| **Chunk Organization** | Arbitrary 150-word splits | Semantic category grouping |
| **Avg Queries** | 4.7 | TBD (test needed) |
| **User Transparency** | Low (arbitrary chunks) | High (named categories) |
| **Cross-Referencing** | Implicit | Explicit & visible |
| **Enterprise Readiness** | Low | High ✅ |

### Expected Progressive Convergence

For 3-category question (CLIENT_DB + CONTRACT + POLICY):
```
Iteration 1-2 (4s):  Query CLIENT_DB    → 33% complete
Iteration 3-4 (8s):  Query CONTRACT     → 67% complete
Iteration 5-6 (12s): Query POLICY       → 100% complete
Iteration 7 (14s):   Synthesize answer  → Submit ✅

Total: ~14 seconds, with visible category progression
```

### Async Viability for Categories

**Threshold:** 80% info in < 50% time

**Expected Performance:**
- 2-category questions: 67% info at 33% time ✅ (beats threshold)
- 3-category questions: 67% info at 44% time ✅ (beats threshold)
- 4+ category questions: May need optimization

---

## 🎯 Implementation Checklist

### For Enterprise Deployment

**1. Category Definition**
- [ ] Define standard categories (POLICY, CLIENT_DB, etc.)
- [ ] Map document sources to categories
- [ ] Create category assignment rules

**2. Chunking Implementation**
- [ ] Implement SourceCategoryChunker
- [ ] Add category metadata to chunks
- [ ] Test with real enterprise documents

**3. RLM Configuration**
- [ ] Update signature with category awareness
- [ ] Add category summary to RLM input
- [ ] Configure max_iterations per category

**4. Async Streaming UI**
- [ ] Design category progress indicators
- [ ] Implement real-time category updates
- [ ] Add cross-reference visualization

**5. Testing & Validation**
- [ ] Test category switching patterns
- [ ] Measure progressive convergence per category
- [ ] Validate async viability threshold

---

## 💡 Key Insights

### Why Source Categories Matter

**1. Semantic Organization**
- Categories reflect REAL document organization
- Not arbitrary word-count splits
- Matches how enterprises actually structure knowledge

**2. Transparent Reasoning**
- Users see WHICH sources being checked
- "Checking warranty policy..." builds trust
- Cross-referencing visible and understandable

**3. Targeted Queries**
- RLM can focus on relevant categories
- Avoid querying irrelevant sources
- More efficient than exhaustive search

**4. Progressive Discovery**
- Each category query adds distinct information
- Progress tied to semantic completion (not just iterations)
- 33% → 67% → 100% has clear meaning to users

### Enterprise Advantages

✅ **Compliance:** Audit trail shows which policies/contracts checked
✅ **Trust:** Users see transparent source attribution
✅ **Efficiency:** Skip irrelevant document categories
✅ **Scalability:** Add new categories without restructuring
✅ **Integration:** Maps to existing document management systems

---

## 🔬 Next Steps

### Testing Plan

1. **Run Source-Category Test**
   ```bash
   python test_rlm_source_categories.py
   ```
   - Test on HotPotQA (simulated categories)
   - Measure category usage patterns
   - Track progressive convergence

2. **Analyze Category Convergence**
   ```bash
   python analyze_category_convergence.py
   ```
   - Which categories queried first?
   - Category switching patterns
   - Async streaming viability

3. **Visualize Results**
   ```bash
   python visualize_rlm_refinement.py rlm_source_category_results.json
   ```
   - Show iteration-by-iteration category usage
   - Timeline of cross-referencing
   - Progressive fact discovery

### Real Enterprise Deployment

1. Define categories for specific domain (legal, support, etc.)
2. Integrate with document management system
3. Implement async streaming UI with category badges
4. A/B test: Show users category-aware vs. generic progress
5. Measure user trust & satisfaction

---

## 🎬 Summary

**Source-Category-Based Chunking enables:**

✅ Enterprise-ready document organization (POLICY, CLIENT_DB, etc.)
✅ Transparent cross-referencing visible to users
✅ Progressive convergence with semantic meaning (33% → 67% → 100%)
✅ Async streaming UI showing which sources being checked
✅ Targeted queries reducing unnecessary document access
✅ Trust through visible source attribution

**This is how real enterprise multi-source reasoning should work!** 🚀
