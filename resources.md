# Resource Catalog: Constraining LLM Behavior Through Abstention

## Research Question

**Can LLMs be trained to constrain their behavior by abstaining from answering when uncertain, rather than generating hallucinated responses?**

This catalog consolidates all resources gathered for investigating training signals and verifier models that enable LLMs to recognize when they are incapable of answering a question.

---

## Quick Reference

| Category | Count | Location |
|----------|-------|----------|
| Research Papers | 10 | `papers/` |
| Datasets | 8+ | See [Datasets](#datasets) |
| Code Repositories | 15+ | See [Code](#code-repositories) |
| Literature Review | 1 | `literature_review.md` |

---

## Key Papers

### Training-Based Approaches

| Paper | Year | Key Contribution | arXiv |
|-------|------|------------------|-------|
| **R-Tuning** | 2024 | Refusal-Aware Instruction Tuning; NAACL Outstanding Paper | 2311.09677 |
| **UA-CLM** | 2024 | Uncertainty-aware loss function for fine-tuning | 2412.02904 |

### Inference-Time Approaches

| Paper | Year | Key Contribution | arXiv |
|-------|------|------------------|-------|
| **Conformal Abstention** | 2024 | Statistical guarantees via conformal prediction | 2405.01563 |
| **SelfCheckGPT** | 2023 | Zero-resource hallucination detection | 2303.08896 |
| **Semantic Entropy** | 2024 | Nature paper on uncertainty via semantic clustering | 2406.15927 |

### Surveys and Foundations

| Paper | Year | Key Contribution | arXiv |
|-------|------|------------------|-------|
| **Know Your Limits** | 2024 | Comprehensive abstention survey | 2407.18418 |
| **Hallucination Survey** | 2023 | Taxonomy of LLM hallucinations | 2311.05232 |
| **Calibrated LMs Must Hallucinate** | 2023 | Fundamental calibration-factuality tension | 2311.14648 |
| **TruthfulQA** | 2022 | Benchmark for truthfulness | 2109.07958 |
| **Do LLMs Know What They Don't Know** | 2023 | SelfAware dataset paper | 2305.18153 |

---

## Datasets

### Primary Evaluation Datasets

| Dataset | Purpose | Size | Link |
|---------|---------|------|------|
| **AbstentionBench** | Holistic abstention evaluation | 35K+ queries, 20 datasets | [GitHub](https://github.com/facebookresearch/AbstentionBench) |
| **SelfAware** | Answerable vs unanswerable classification | 3,369 questions | [GitHub](https://github.com/yinzhangyue/SelfAware) |
| **TruthfulQA** | Truthfulness against misconceptions | 817 questions | [GitHub](https://github.com/sylinrl/TruthfulQA) |
| **HaluEval** | Hallucination detection | 35K samples | [GitHub](https://github.com/RUCAIBox/HaluEval) |

### Training Datasets

| Dataset | Purpose | Size | Link |
|---------|---------|------|------|
| **SQuAD 2.0** | Reading comprehension with abstention | 150K questions | [HuggingFace](https://huggingface.co/datasets/rajpurkar/squad_v2) |
| **FalseQA** | False premise detection | Paired Q&A | [GitHub](https://github.com/thunlp/FalseQA) |
| **Natural Questions** | Real queries with null answers | 307K examples | [Google AI](https://ai.google.com/research/NaturalQuestions) |

**Full dataset catalog**: `datasets/datasets_catalog.md`

---

## Code Repositories

### Core Implementations

| Repository | Purpose | Stars | Link |
|------------|---------|-------|------|
| **R-Tuning** | Training LLMs to say "I don't know" | - | [GitHub](https://github.com/shizhediao/R-Tuning) |
| **SelfCheckGPT** | Consistency-based hallucination detection | - | [GitHub](https://github.com/potsawee/selfcheckgpt) |
| **Semantic Uncertainty** | Semantic entropy implementation | - | [GitHub](https://github.com/jlko/semantic_uncertainty) |
| **Semantic Entropy Probes** | Efficient probe-based uncertainty | - | [GitHub](https://github.com/OATML/semantic-entropy-probes) |

### Conformal Prediction

| Repository | Purpose | Link |
|------------|---------|------|
| **VLM-Uncertainty** | Learnable conformal abstention | [GitHub](https://github.com/sinatayebati/vlm-uncertainty) |
| **LLM-Uncertainty-Bench** | Conformal prediction benchmarking | [GitHub](https://github.com/smartyfh/LLM-Uncertainty-Bench) |

### RLHF Frameworks

| Repository | Purpose | Link |
|------------|---------|------|
| **OpenRLHF** | Scalable RLHF training | [GitHub](https://github.com/OpenRLHF/OpenRLHF) |
| **Safe-RLHF** | Constrained value alignment | [GitHub](https://github.com/PKU-Alignment/safe-rlhf) |
| **RLHF-V** | Fine-grained correctional feedback | [GitHub](https://github.com/RLHF-V/RLHF-V) |

### Curated Lists

| Repository | Focus | Link |
|------------|-------|------|
| **Awesome-LLM-Uncertainty** | Uncertainty, reliability, robustness | [GitHub](https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness) |
| **Awesome-RLHF** | RLHF resources | [GitHub](https://github.com/opendilab/awesome-RLHF) |

**Full code catalog**: `code/code_repositories.md`

---

## Key Findings Summary

### The Research Hypothesis is Supported

The literature strongly supports that LLMs can be trained to abstain appropriately:

1. **Abstention is learnable**: R-Tuning demonstrates refusal as a generalizable meta-skill
2. **Uncertainty can be calibrated**: UA-CLM improves calibration without accuracy loss
3. **Statistical guarantees are achievable**: Conformal prediction bounds hallucination rates
4. **Self-consistency works**: SelfCheckGPT detects hallucinations without external knowledge

### Recommended Approach

1. **Training**: Apply R-Tuning methodology (certain/uncertain data splits)
2. **Loss Function**: Incorporate UA-CLM uncertainty-aware objectives
3. **Inference**: Add conformal abstention for statistical guarantees
4. **Verification**: Use SelfCheckGPT or semantic entropy for hallucination detection

### Open Challenges

- Reasoning fine-tuning degrades abstention by 24% (AbstentionBench finding)
- Long-form generation abstention remains difficult
- Black-box settings limit many approaches
- Compositional uncertainty in multi-step reasoning

---

## File Structure

```
llm-behavior-constraints-claude/
├── README.md                      # Project overview
├── literature_review.md           # Comprehensive synthesis
├── resources.md                   # This catalog
├── papers/                        # Downloaded PDFs
│   ├── 2407.18418_Know_Your_Limits_Abstention_Survey.pdf
│   ├── 2311.09677_R-Tuning_Say_IDK.pdf
│   ├── 2303.08896_SelfCheckGPT_Original.pdf
│   ├── 2405.01563_Conformal_Abstention.pdf
│   ├── 2406.15927_Semantic_Entropy_Probes.pdf
│   ├── 2311.14648_Calibrated_LMs_Must_Hallucinate.pdf
│   ├── 2109.07958_TruthfulQA.pdf
│   ├── 2311.05232_Hallucination_Survey.pdf
│   ├── 2305.18153_Do_LLMs_Know_What_They_Dont_Know.pdf
│   └── 2412.02904_Uncertainty_Aware_Fine_Tuning.pdf
├── datasets/
│   └── datasets_catalog.md        # Detailed dataset information
├── code/
│   └── code_repositories.md       # Detailed repository information
└── .resource_finder_complete      # Completion marker
```

---

## Quick Start Commands

```bash
# Install key packages
pip install selfcheckgpt
pip install lmflow  # For R-Tuning

# Clone essential repositories
git clone https://github.com/shizhediao/R-Tuning.git
git clone https://github.com/potsawee/selfcheckgpt.git
git clone https://github.com/facebookresearch/AbstentionBench.git

# Load datasets via HuggingFace
python -c "from datasets import load_dataset; ds = load_dataset('rajpurkar/squad_v2')"
```

---

## References

See `literature_review.md` for complete bibliography with 10 primary sources and additional references.

---

*Generated: 2025-12-29*
*Research Focus: LLM Abstention and Uncertainty-Aware Training*
