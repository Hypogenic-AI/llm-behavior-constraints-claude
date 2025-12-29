# Code Repositories for LLM Abstention and Uncertainty Research

This catalog documents open-source implementations and evaluation frameworks for LLM abstention, uncertainty quantification, and hallucination detection.

---

## Core Abstention Training Methods

### 1. R-Tuning (NAACL 2024 Outstanding Paper)

**Purpose**: Training LLMs to say "I don't know" through Refusal-Aware Instruction Tuning

**Description**: Identifies the disparity between pre-trained parametric knowledge and instruction tuning data, then constructs refusal-aware training data based on the knowledge intersection.

**Key Features**:
- LMFlow-based training pipeline
- Multi-dataset evaluation (ParaRel, MMLU, WiCE, HotpotQA, FEVER)
- Improved calibration as a byproduct of uncertainty learning

**GitHub**: https://github.com/shizhediao/R-Tuning

**Training Command**:
```bash
cd ~/LMFlow
./scripts/run_finetune.sh \
    --model_name_or_path openlm-research/open_llama_3b \
    --dataset_path ../training/training_data \
    --output_model_path output_models/finetuned_llama_3b
```

**Paper**: https://arxiv.org/abs/2311.09677

---

## Hallucination Detection

### 2. SelfCheckGPT

**Purpose**: Zero-resource black-box hallucination detection through self-consistency

**Description**: Detects hallucinations by sampling multiple responses and checking for consistency. If LLM has knowledge, responses are consistent; hallucinated facts lead to contradictions.

**Variants**:
- SelfCheckBERTScore
- SelfCheckMQAG (Multiple-choice QA Generation)
- SelfCheckNgram
- SelfCheckNLI
- SelfCheckPrompt (best performance with GPT-3.5/ChatGPT)

**GitHub**: https://github.com/potsawee/selfcheckgpt

**Installation**: `pip install selfcheckgpt`

**Usage**:
```python
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
selfcheck = SelfCheckBERTScore()
scores = selfcheck.predict(sentences, sampled_passages)
```

**Paper**: https://arxiv.org/abs/2303.08896 (EMNLP 2023)

---

## Semantic Entropy and Uncertainty Quantification

### 3. Semantic Uncertainty (Nature 2024)

**Purpose**: Detecting hallucinations using semantic entropy

**Description**: Clusters semantically equivalent generations and computes entropy over semantic clusters rather than token sequences.

**GitHub**: https://github.com/jlko/semantic_uncertainty (Updated)
**GitHub (deprecated)**: https://github.com/lorenzkuhn/semantic_uncertainty (Original ICLR 2023)

**Dependencies**: Weights & Biases, Hugging Face

**Paper**: "Detecting Hallucinations in Large Language Models Using Semantic Entropy" (Nature 2024)

---

### 4. Semantic Entropy Probes

**Purpose**: Efficient uncertainty estimation via linear probes on hidden states

**Description**: Trains linear probes on model hidden states to predict semantic uncertainty and correctness, avoiding expensive sampling.

**GitHub**: https://github.com/OATML/semantic-entropy-probes

**Key Innovation**: Uses two token positions (TBG, SLT) for probe training.

**Paper**: https://arxiv.org/abs/2406.15927

---

### 5. Kernel Language Entropy (NeurIPS 2024)

**Purpose**: Fine-grained uncertainty quantification using semantic similarities

**Description**: Defines positive semidefinite kernels to encode semantic similarities and quantifies uncertainty using von Neumann entropy. Considers pairwise dependencies rather than hard clustering.

**GitHub**: https://github.com/AlexanderVNikitin/kernel-language-entropy

**Paper**: "Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities"

---

## Conformal Prediction for Abstention

### 6. VLM-Uncertainty (Conformal Abstention)

**Purpose**: Learnable conformal abstention integrating RL with conformal prediction

**Description**: Optimizes abstention thresholds dynamically using reinforcement learning. Balances prediction set size reduction with reliable coverage.

**Key Results**:
- 21.17% boost in AUARC (uncertainty-guided selective generation)
- 70-85% reduction in calibration error
- 90% coverage target maintained

**Outputs**: Single prediction, set of plausible predictions, or abstention

**GitHub**: https://github.com/sinatayebati/vlm-uncertainty

**Paper**: https://arxiv.org/abs/2502.06884

---

### 7. LLM-Uncertainty-Bench

**Purpose**: Benchmarking LLMs via uncertainty quantification with conformal prediction

**Description**: Compares LLM uncertainty measured as average prediction set size. Larger sets indicate higher uncertainty.

**GitHub**: https://github.com/smartyfh/LLM-Uncertainty-Bench

**Features**:
- UAcc metric (uncertainty-aware accuracy)
- Conformal prediction implementation
- Multi-model comparison

---

## RLHF and Alignment

### 8. RLHF-V

**Purpose**: Trustworthy MLLMs via fine-grained correctional human feedback

**Description**: Reduces hallucination by collecting fine-grained correctional feedback. Annotators correct hallucinated segments directly.

**Key Result**: 34.8% hallucination reduction in 1 hour on 8 A100 GPUs

**GitHub**: https://github.com/RLHF-V/RLHF-V

**Paper**: CVPR 2024

---

### 9. Safe-RLHF (PKU-Alignment)

**Purpose**: Constrained value alignment via safe reinforcement learning

**Description**: Modular RLHF framework for alignment research, especially constrained alignment.

**GitHub**: https://github.com/PKU-Alignment/safe-rlhf

**Paper**: "Safe RLHF: Safe Reinforcement Learning from Human Feedback"

---

### 10. OpenRLHF

**Purpose**: Scalable RLHF training framework

**Description**: Built on Ray, vLLM, ZeRO-3, and HuggingFace. Scales to 70B+ parameter models.

**GitHub**: https://github.com/OpenRLHF/OpenRLHF

**Features**: PPO, GRPO, REINFORCE++, dynamic sampling

---

## Confidence and Calibration

### 11. LLM-Uncertainty (ICLR 2024)

**Purpose**: Evaluating confidence elicitation in LLMs

**Description**: Systematic framework for prompting strategies, sampling methods, and aggregation techniques for confidence calibration.

**GitHub**: https://github.com/MiaoXiong2320/llm-uncertainty

**Paper**: "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs"

---

### 12. Reliable-LLM

**Purpose**: LLM hallucination mitigation via uncertainty and knowledge

**Description**: Framework for improving factuality perception and eliciting factual expressions.

**GitHub**: https://github.com/AmourWaltz/Reliable-LLM

**Focus**: Addressing over-confidence from maximum likelihood training

---

## Curated Lists and Surveys

### 13. Awesome-LLM-Uncertainty-Reliability-Robustness

**Description**: Comprehensive curated list covering:
- Uncertainty estimation methods
- Calibration techniques
- RLHF approaches
- Robustness methods

**GitHub**: https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness

**Notable Papers Linked**:
- "Calibrating Large Language Models Using Their Generations Only" (ACL 2024)
- "Batch Calibration: Rethinking Calibration for In-Context Learning" (ICLR 2024)
- "Just Ask for Calibration" (arXiv 2023)

---

### 14. Awesome-RLHF (OpenDILab)

**Description**: Curated list of RLHF resources

**GitHub**: https://github.com/opendilab/awesome-RLHF

---

## Benchmarks

### 15. AbstentionBench (Facebook Research)

**Purpose**: Holistic benchmark for LLM abstention evaluation

**GitHub**: https://github.com/facebookresearch/AbstentionBench

**Datasets**: 20 datasets, 35K+ queries across 6 abstention scenarios

---

## Quick Start Commands

```bash
# Clone core repositories
git clone https://github.com/shizhediao/R-Tuning.git
git clone https://github.com/potsawee/selfcheckgpt.git
git clone https://github.com/jlko/semantic_uncertainty.git
git clone https://github.com/OATML/semantic-entropy-probes.git
git clone https://github.com/sinatayebati/vlm-uncertainty.git
git clone https://github.com/smartyfh/LLM-Uncertainty-Bench.git
git clone https://github.com/facebookresearch/AbstentionBench.git

# Install key packages
pip install selfcheckgpt
pip install lmflow  # For R-Tuning
```

---

## Recommended Implementation Order

1. **Baseline Evaluation**: Start with AbstentionBench to establish baseline abstention rates
2. **Hallucination Detection**: Implement SelfCheckGPT for detecting when model is hallucinating
3. **Uncertainty Quantification**: Add semantic entropy for principled uncertainty estimates
4. **Training for Abstention**: Apply R-Tuning methodology to train model to say "I don't know"
5. **Conformal Abstention**: Add conformal prediction for statistical guarantees on coverage
