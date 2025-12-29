# LLM Behavior Constraints: Teaching Models to Say "I Don't Know"

This research investigates whether Large Language Models can be motivated to abstain from answering uncertain questions instead of generating hallucinated responses.

## Key Findings

- **Prompt-based abstention works**: Simple prompt modifications increase abstention rates from 2% to 62%, with large effect sizes (Cohen's h > 0.98, all p < 0.0001)
- **Explicit abstention instruction is most effective**: Achieves F1 scores of 0.61-0.67 for appropriate abstention
- **Answer accuracy improves when models abstain**: From 36% (baseline) to 62% (with abstention prompts)
- **Self-consistency over-abstains**: 89% abstention rate is higher than optimal for balanced datasets

## Quick Start

```bash
# Create environment
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"

# Run experiments
python src/experiment.py

# Analyze results
python src/analyze_results.py
```

## Results Summary

| Prompting Strategy | Abstention Rate | F1 Score | Answer Accuracy |
|--------------------|-----------------|----------|-----------------|
| Baseline | 2.2% | 0.00 | 36.4% |
| **Explicit Abstention** | 62.2% | **0.64** | **61.8%** |
| CoT + Uncertainty | 35.6% | 0.53 | 50.9% |
| Self-Consistency | 88.9% | 0.49 | 55.0% |

## File Structure

```
├── REPORT.md                  # Full research report with analysis
├── README.md                  # This file
├── planning.md                # Research plan
├── literature_review.md       # Literature synthesis
├── resources.md               # Resource catalog
├── src/
│   ├── experiment.py          # Main experiment runner
│   ├── prompts.py             # Prompt templates
│   ├── api_client.py          # API clients (OpenAI, OpenRouter)
│   ├── self_consistency.py    # Multi-sample consistency checking
│   ├── metrics.py             # Evaluation metrics
│   └── analyze_results.py     # Statistical analysis & visualization
├── results/
│   ├── metrics.json           # Computed metrics
│   ├── statistical_analysis.json  # Statistical test results
│   ├── raw/                   # Raw API responses
│   └── plots/                 # Visualizations
├── papers/                    # Reference papers (PDFs)
└── datasets/                  # Dataset documentation
```

## Methodology

We tested 4 prompting strategies on 2 models (GPT-4o-mini, Claude Sonnet 4):

1. **Baseline**: Standard prompting
2. **Explicit Abstention**: "If uncertain, say 'I don't know'"
3. **Chain-of-Thought + Uncertainty**: Multi-step reasoning with confidence
4. **Self-Consistency**: Sample 3 responses, flag inconsistency

Datasets: TruthfulQA (30 tricky questions) + SQuAD 2.0 (60 questions, 50% unanswerable)

## Key Insight

> Models almost never abstain on their own (1-3%), but simple prompts can dramatically increase abstention (60-90%). The explicit instruction "If uncertain, say I don't know" is the most effective strategy, balancing high recall (catching uncertain questions) with acceptable precision (avoiding over-abstention).

## References

See [REPORT.md](REPORT.md) for full bibliography. Key papers:
- R-Tuning (Zhang et al., NAACL 2024)
- Know Your Limits Survey (Feng et al., 2024)
- SelfCheckGPT (Manakul et al., EMNLP 2023)

---

*Research conducted: December 2024*
