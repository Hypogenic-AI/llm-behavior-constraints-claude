# Research Plan: Constraining LLM Behavior Through Abstention

## Research Question

**Can LLMs be motivated to add constraints to their behavior such that, instead of generating hallucinated responses, they choose to abstain from answering ("I don't know"), and by choosing this option, they achieve better overall performance?**

## Background and Motivation

Large language models are fundamentally generative systems, which makes selective action (abstention) challenging. When faced with questions beyond their knowledge, LLMs often generate plausible-sounding but incorrect responses (hallucinations). This is problematic for real-world applications where reliability matters.

The literature review identifies several promising approaches:
1. **R-Tuning**: Training-based refusal-aware instruction tuning
2. **SelfCheckGPT**: Inference-time self-consistency checking
3. **Conformal Abstention**: Statistical guarantees via conformal prediction
4. **Semantic Entropy**: Principled uncertainty quantification

**Gap we're addressing**: While these methods exist, we need empirical evidence on how well *prompt-based strategies* (accessible without fine-tuning) can induce abstention behavior in current SOTA models.

## Hypothesis Decomposition

**H1 (Primary)**: Prompt-based strategies can significantly increase appropriate abstention rates on answerable/unanswerable question classification.

**H1a**: Self-consistency checking (sampling multiple responses and checking agreement) improves abstention accuracy.

**H1b**: Explicit uncertainty instruction prompts ("If you're unsure, say 'I don't know'") improve abstention behavior.

**H1c**: Chain-of-thought reasoning with uncertainty awareness improves calibration.

**H2 (Secondary)**: Different models exhibit different abstention behaviors when given the same prompts.

**H3 (Tertiary)**: There is a measurable trade-off between abstention rate and answer accuracy.

## Proposed Methodology

### Approach: Prompt-Based Abstention Experiments on Real LLMs

We will conduct experiments using real LLM API calls (GPT-4o, Claude Sonnet 4.5 via OpenRouter) to test whether various prompting strategies can induce appropriate abstention behavior.

**Why this approach?**
- Accessible without fine-tuning (practical for deployment)
- Tests current SOTA models' inherent capabilities
- Can be combined with self-consistency methods
- Provides insights for when to apply more expensive methods

### Experimental Steps

#### Step 1: Dataset Preparation
- Use **TruthfulQA** subset (questions designed to elicit false answers)
- Use **SelfAware** dataset (answerable vs. unanswerable questions)
- Sample 100-150 questions from each for computational efficiency

#### Step 2: Prompt Strategy Design
Design 4 prompting conditions:
1. **Baseline**: Standard prompt without abstention instruction
2. **Explicit Abstention**: "If you are uncertain or don't know the answer, respond with 'I don't know'"
3. **Chain-of-Thought + Uncertainty**: "Think step by step about what you know. State your confidence level."
4. **Self-Consistency**: Sample N=5 responses, measure agreement

#### Step 3: Model Selection
- **GPT-4o-mini** (cost-effective, strong reasoning)
- **Claude Sonnet 4.5** via OpenRouter (current SOTA)
Compare both to understand model-specific behaviors.

#### Step 4: Metrics Collection
For each prompt-model combination:
- Abstention rate (% of "I don't know" responses)
- Answer accuracy (on questions answered)
- Calibration (alignment of stated confidence with actual accuracy)
- F1 score for answerability classification

#### Step 5: Self-Consistency Implementation
For the self-consistency condition:
- Sample 5 responses per question (temperature=0.7)
- Measure semantic agreement (using cosine similarity of embeddings)
- Flag inconsistent responses as potential hallucinations
- Measure improvement over single-sample baseline

### Baselines

1. **No abstention instruction** (standard prompting)
2. **Random abstention** (abstain on 20% of questions randomly) - statistical baseline
3. **Always answer** (0% abstention) - shows accuracy ceiling without abstention

### Evaluation Metrics

| Metric | Description | Why Important |
|--------|-------------|---------------|
| **Abstention Rate** | % of questions where model says "I don't know" | Measures abstention behavior |
| **Answer Accuracy** | Accuracy on answered questions | Measures quality of non-abstained responses |
| **F1 Score** | Balance of precision/recall for correct abstention | Holistic abstention quality |
| **AUROC** | Area under ROC for abstention vs. correctness | Discrimination ability |
| **Expected Calibration Error (ECE)** | Calibration of stated confidence | Reliability of uncertainty estimates |

### Statistical Analysis Plan

- **Paired t-tests** between prompt conditions (same questions)
- **Chi-squared tests** for abstention rate differences
- **Bootstrap confidence intervals** (1000 resamples) for all metrics
- **Effect sizes** (Cohen's d) for practical significance
- Significance level: α = 0.05

## Expected Outcomes

**If H1 is supported:**
- Explicit abstention prompts increase abstention rate by 10-30%
- Abstention correlates with questions model gets wrong (appropriate abstention)
- Self-consistency flagging shows >0.6 AUROC for hallucination detection

**If H1 is refuted:**
- Abstention rate remains low (<5%) regardless of prompts
- Or abstention is not correlated with actual uncertainty
- Models may over-abstain on easy questions or under-abstain on hard ones

**Mixed outcome (likely):**
- Some improvement but models still hallucinate
- Trade-off curves between coverage and accuracy
- Prompt sensitivity varies by model

## Timeline and Milestones

1. **Environment Setup** (~10 min): Set up uv environment, install dependencies
2. **Data Preparation** (~15 min): Load and preprocess TruthfulQA and SelfAware samples
3. **Baseline Implementation** (~30 min): Implement API calls, baseline prompts
4. **Prompt Variants** (~30 min): Implement 4 prompting conditions
5. **Self-Consistency** (~20 min): Implement multi-sample consistency checking
6. **Run Experiments** (~45 min): Execute all experiments (API calls)
7. **Analysis** (~30 min): Statistical analysis and visualization
8. **Documentation** (~20 min): Create REPORT.md and README.md

## Potential Challenges

| Challenge | Mitigation |
|-----------|------------|
| API rate limits | Use exponential backoff, batch requests |
| High API costs | Use 100-150 samples per dataset, GPT-4o-mini |
| Model variability | Fixed seeds where possible, multiple runs |
| Parsing "I don't know" | Robust regex + semantic classification |
| Semantic similarity | Use simple embedding cosine similarity |

## Success Criteria

**Strong success**:
- Abstention prompts show statistically significant improvement in F1 (p < 0.05)
- Effect size d > 0.5 for at least one prompt strategy
- Self-consistency achieves AUROC > 0.65 for hallucination detection

**Moderate success**:
- Trends in expected direction but smaller effect sizes
- Some prompt strategies work, others don't
- Insights about model-specific behaviors

**Partial success**:
- Clear documentation of what doesn't work
- Identification of limitations of prompt-based approaches
- Recommendations for when fine-tuning is needed

## Resource Requirements

- **API Costs**: ~$20-50 estimated (GPT-4o-mini + Claude via OpenRouter)
- **Compute**: CPU sufficient (API-based)
- **Storage**: <100MB for results
- **Time**: ~3-4 hours total

## Files to Create

```
src/
├── experiment.py          # Main experiment runner
├── prompts.py             # Prompt templates
├── metrics.py             # Evaluation metrics
├── data_loader.py         # Dataset loading
└── self_consistency.py    # Multi-sample consistency

results/
├── raw/                   # Raw API responses
├── metrics.json           # Computed metrics
└── plots/                 # Visualizations

REPORT.md                  # Final research report
README.md                  # Project overview
```
