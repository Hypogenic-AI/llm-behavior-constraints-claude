# Literature Review: Constraining LLM Behavior Through Abstention and Uncertainty-Aware Training

## Executive Summary

This literature review synthesizes research on training Large Language Models (LLMs) to constrain their behavior by abstaining from answering when uncertain, rather than generating hallucinated responses. The research hypothesis under investigation is: **It is possible to motivate LLMs to add constraints to their behavior such that, instead of generating hallucinated responses, they can choose to abstain from answering, say "I don't know," or "I need more data," and by choosing this option, they are better off.**

The reviewed literature strongly supports this hypothesis, with multiple complementary approaches demonstrating that LLMs can be trained or calibrated to recognize their knowledge boundaries and abstain appropriately.

---

## 1. Conceptual Framework for LLM Abstention

### 1.1 The Know Your Limits Survey (Feng et al., 2024)

The most comprehensive framework for understanding LLM abstention comes from "Know Your Limits: A Survey of Abstention in Large Language Models" (arXiv:2407.18418). This survey organizes abstention research around three fundamental perspectives:

**Query Perspective (Input-Level)**
- Questions may be inherently unanswerable due to:
  - Lack of sufficient context
  - False premises in the question
  - Ambiguous or underspecified queries
  - Questions about future events or unknowable facts

**Model Knowledge Perspective (Capability-Level)**
- The model's parametric knowledge may be insufficient:
  - Knowledge gaps in pre-training data
  - Information beyond the training cutoff date
  - Domain-specific knowledge not well-represented
  - Confidence-accuracy misalignment

**Human Values Perspective (Safety-Level)**
- Abstention required for alignment with human values:
  - Harmful content requests
  - Privacy-violating queries
  - Biased or discriminatory outputs
  - Legally problematic content

### 1.2 The Hallucination Problem

Hallucinations are defined as "plausible-sounding but incorrect or unfaithful model generations" (Ji et al., 2023). The "Hallucination Survey" (arXiv:2311.05232) categorizes hallucinations as:

- **Intrinsic hallucinations**: Contradicting the source material
- **Extrinsic hallucinations**: Information that cannot be verified from the source
- **Confabulations**: A subset characterized by generating arbitrary and incorrect responses confidently (Farquhar et al., 2024)

---

## 2. Training-Based Approaches to Abstention

### 2.1 R-Tuning: Refusal-Aware Instruction Tuning (NAACL 2024 Outstanding Paper)

Zhang et al. (2024) present **R-Tuning** (arXiv:2311.09677), a seminal approach for training LLMs to say "I don't know."

**Core Insight**: Standard instruction tuning forces models to generate answers regardless of whether they possess the requisite knowledge. This creates a problematic training signal where the model is rewarded for generating plausible-sounding but potentially incorrect outputs.

**Methodology**:
1. **Knowledge Gap Identification**: Split training data into:
   - **Certain set (D₁)**: Questions where model predictions match ground-truth labels
   - **Uncertain set (D₀)**: Questions where model predictions differ from labels

2. **Refusal-Aware Data Construction**:
   - For certain questions: Append "I am sure" after the answer
   - For uncertain questions: Append "I am unsure" or construct refusal responses

3. **Training Objective**: Fine-tune the model on the combined dataset to learn both answering and refusal behaviors

**Key Results**:
- Improved ability to answer known questions while refraining from unknown ones
- **Meta-skill generalization**: The refusal ability transfers to out-of-domain tasks
- **Better calibration**: Learning uncertainty leads to improved Expected Calibration Error (ECE)
- Evaluation across ParaRel, MMLU, WiCE, HotpotQA, FEVER, SelfAware, HaluEval

**Evaluation Metrics**:
- **Accuracy**: Correctness on questions the model answers
- **Abstention Rate**: Proportion of questions refused
- **Average Precision (AP)**: Area under precision-recall curve
- **F1 Score**: Balancing precision and recall of abstention decisions

### 2.2 Uncertainty-Aware Causal Language Modeling (UA-CLM)

Krishnan et al. (2024) propose **UA-CLM** (arXiv:2412.02904), an uncertainty-aware fine-tuning approach grounded in decision theory.

**Key Innovation**: A novel loss function that explicitly incorporates uncertainty into the training objective:

```
L_UA-CLM = -1/|C̃| Σ_{i∈C̃} P_θ(w_i|w_{0:i-1}) log(tanh(H_i))   [incorrect tokens]
           -1/|C| Σ_{i∈C} (1-P_θ(w_i|w_{0:i-1})) log(1-tanh(H_i)) [correct tokens]
```

Where H_i is the token-level entropy.

**Desiderata**:
- Correctly generated tokens → low uncertainty, high probability
- Incorrectly generated tokens → high uncertainty, low probability

**Results** (Llama-2-7B/13B, Gemma-2B on CoQA, TriviaQA):
- Up to **17.1% improvement** in hallucination detection AUROC
- Up to **23.6% improvement** on VQA tasks
- Significant reduction in Expected Calibration Error (ECE)
- **No accuracy degradation** compared to standard CLM fine-tuning

**Applications**:
1. Hallucination detection
2. Uncertainty-guided selective generation
3. Out-of-domain prompt detection
4. Improved calibration

---

## 3. Inference-Time Approaches

### 3.1 Conformal Abstention (Abbasi-Yadkori et al., 2024)

"Mitigating LLM Hallucinations via Conformal Abstention" (arXiv:2405.01563) provides a **principled statistical framework** for abstention with formal guarantees.

**Core Framework**:
- Given classifier f: X → Y and abstention function a_λ(X)
- Loss function: ℓ(X,Y;λ) = (1 - a_λ(X))(1 - m(X; f(X), Y))
- Goal: Minimize abstention rate T(λ) subject to hallucination risk R(λ) ≤ α

**Conformal Risk Control (CRC)**:
```
λ̂_n = inf{λ : (n/(n+1))L_n(λ) + 1/(n+1) ≤ α}
```

**Theoretical Guarantee**: E[R(λ̂_n)] = E[ℓ(X,Y;λ̂_n)] ≤ α

**Score Functions**:
1. **Match Count**: Number of sampled responses similar to reference response
2. **Expected Match Count**: Probabilistic version using log-probabilities
3. **Log-probability baseline**: Raw confidence scores

**Key Finding**: Self-consistency-based scores (match counts) significantly outperform log-probability scores, especially for long-form responses.

**Results on Temporal Sequences and TriviaQA**:
- Reliably bounds hallucination rate to specified α levels
- Match count scores achieve lower abstention rates than log-probability baselines
- Method works with small calibration sets (as few as 10-30 examples)

### 3.2 SelfCheckGPT: Zero-Resource Hallucination Detection

Manakul et al. (2023) propose **SelfCheckGPT** (arXiv:2303.08896), a black-box approach to hallucination detection.

**Core Insight**: If an LLM has knowledge of a concept, sampled responses are likely to be consistent. For hallucinated facts, stochastically sampled responses will diverge and contradict.

**Variants**:
1. **SelfCheckBERTScore**: Semantic similarity via BERT embeddings
2. **SelfCheckMQAG**: Multiple-choice question generation and answering
3. **SelfCheckNgram**: N-gram overlap statistics
4. **SelfCheckNLI**: Natural language inference for consistency
5. **SelfCheckPrompt**: LLM self-evaluation (best performance with GPT-3.5/ChatGPT)

**Sentence-Level Scoring**: Each sentence is compared against sampled passages to produce factuality scores.

### 3.3 Semantic Entropy (Nature 2024)

Kuhn et al. (2023) and Farquhar et al. (2024) develop **Semantic Entropy** as a principled uncertainty measure.

**Problem with Token-Level Entropy**: Multiple semantically equivalent answers have high token-level entropy but low actual uncertainty.

**Solution**: Compute entropy over semantic equivalence classes rather than token sequences:
1. Generate multiple responses
2. Cluster semantically equivalent responses
3. Compute entropy over cluster distribution

**Semantic Entropy Probes** (arXiv:2406.15927): Train linear probes on hidden states to predict semantic uncertainty efficiently, avoiding expensive sampling.

---

## 4. Calibration and Confidence Estimation

### 4.1 The Calibration Challenge

"Calibrated Language Models Must Hallucinate" (arXiv:2311.14648) establishes a fundamental tension:

**Theorem**: Perfectly calibrated language models must sometimes produce hallucinations for rare facts, as calibration requires matching the true probability distribution.

**Implications**:
- Calibration alone is insufficient for factuality
- Need complementary mechanisms (abstention, retrieval augmentation)
- Trade-offs between calibration and accuracy

### 4.2 Verbalized Confidence

Xiong et al. (2024) study confidence elicitation in "Can LLMs Express Their Uncertainty?"

**Finding**: LLMs are often poorly calibrated, exhibiting overconfidence in predictions. RLHF-tuned models show particular miscalibration issues.

**Approaches**:
1. Direct confidence elicitation via prompting
2. Consistency-based confidence estimation
3. Calibrated training objectives

---

## 5. Benchmark Datasets and Evaluation

### 5.1 AbstentionBench (Facebook Research, 2024)

The most comprehensive benchmark for abstention evaluation:
- **20 datasets** across **6 abstention scenarios**
- **35,000+ unanswerable queries**
- Includes modified versions of GSM8K, MMLU, GPQA

**Key Finding**: "Surprisingly, reasoning fine-tuning degrades abstention (by 24% on average), even for math and science domains."

### 5.2 Key Evaluation Datasets

| Dataset | Purpose | Size |
|---------|---------|------|
| SelfAware | Answerable vs. unanswerable questions | 1,032 unanswerable + 2,337 answerable |
| TruthfulQA | Truthfulness against misconceptions | 817 questions, 38 categories |
| SQuAD 2.0 | Reading comprehension with abstention | 100K answerable + 50K unanswerable |
| HaluEval | Hallucination evaluation | 35K samples |
| FalseQA | False premise detection | Binary labeled Q&A pairs |

### 5.3 Evaluation Metrics

**Abstention Quality**:
- **AUROC**: Area under ROC curve for hallucination detection
- **AUARC**: Area under accuracy-rejection curve
- **AP Score**: Average precision for abstention decisions

**Calibration**:
- **ECE**: Expected Calibration Error
- **Brier Score**: Proper scoring rule for probabilistic predictions

**Text Quality**:
- **ROUGE-L**: Longest common subsequence similarity
- **BERTScore**: Semantic similarity via embeddings
- **Exact Match**: Strict correctness criterion

---

## 6. Synthesis: A Multi-Pronged Approach

The literature suggests that effective abstention requires combining multiple approaches:

### 6.1 Training-Time Interventions

1. **R-Tuning**: Explicitly teach refusal through modified training data
2. **UA-CLM**: Incorporate uncertainty into the loss function
3. **Calibration Tuning**: Optimize for well-calibrated confidence

### 6.2 Inference-Time Mechanisms

1. **Conformal Prediction**: Statistical guarantees on error rates
2. **Self-Consistency**: Multiple sampling with agreement checking
3. **Semantic Entropy**: Principled uncertainty quantification

### 6.3 Hybrid Systems

The most effective systems likely combine:
- Trained abstention capabilities (R-Tuning style)
- Calibrated uncertainty estimates (UA-CLM)
- Inference-time verification (SelfCheckGPT, Conformal)
- Retrieval augmentation for factual grounding

---

## 7. Open Challenges and Future Directions

### 7.1 Unsolved Problems

1. **Scaling Abstention**: AbstentionBench shows reasoning models perform worse
2. **Black-Box Settings**: Many methods require logit access
3. **Long-Form Generation**: Abstention for extended outputs remains challenging
4. **Compositional Uncertainty**: Handling multi-step reasoning

### 7.2 Research Opportunities

1. **End-to-End Training**: Unified objectives for accuracy and abstention
2. **Retrieval-Augmented Abstention**: Combining RAG with uncertainty
3. **Interactive Clarification**: Models that ask for more information
4. **Domain Adaptation**: Transfer of abstention capabilities

---

## 8. Conclusions

The literature strongly supports the research hypothesis. Key findings:

1. **Abstention is learnable**: R-Tuning demonstrates that LLMs can be trained to recognize and refuse uncertain questions as a generalizable meta-skill.

2. **Uncertainty can be calibrated**: UA-CLM shows that incorporating uncertainty into training objectives improves calibration without sacrificing accuracy.

3. **Statistical guarantees are achievable**: Conformal prediction provides rigorous bounds on hallucination rates with finite calibration sets.

4. **Self-consistency works**: Methods like SelfCheckGPT and semantic entropy can detect hallucinations without external knowledge bases.

5. **Trade-offs exist**: Perfect calibration and complete factuality are fundamentally in tension; abstention provides a practical resolution.

The research direction of training LLMs to constrain their behavior through principled abstention represents a promising path toward more trustworthy AI systems that acknowledge their limitations rather than confabulating responses.

---

## References

1. Feng, S. et al. (2024). Know Your Limits: A Survey of Abstention in Large Language Models. TACL. arXiv:2407.18418

2. Zhang, H. et al. (2024). R-Tuning: Instructing Large Language Models to Say 'I Don't Know'. NAACL Outstanding Paper. arXiv:2311.09677

3. Krishnan, R. et al. (2024). Enhancing Trust in Large Language Models with Uncertainty-Aware Fine-Tuning. arXiv:2412.02904

4. Abbasi-Yadkori, Y. et al. (2024). Mitigating LLM Hallucinations via Conformal Abstention. arXiv:2405.01563

5. Manakul, P. et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection. EMNLP. arXiv:2303.08896

6. Kuhn, L. et al. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG. ICLR.

7. Farquhar, S. et al. (2024). Detecting Hallucinations in Large Language Models Using Semantic Entropy. Nature.

8. Lin, S. et al. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. arXiv:2109.07958

9. Yin, Z. et al. (2023). Do Large Language Models Know What They Don't Know? ACL. (SelfAware dataset)

10. Li, J. et al. (2023). HaluEval: A Large-Scale Hallucination Evaluation Benchmark. EMNLP. arXiv:2305.11747
