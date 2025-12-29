# Datasets for LLM Abstention and Hallucination Research

This catalog documents datasets relevant to training and evaluating LLMs on abstention behavior, uncertainty quantification, and hallucination detection.

---

## 1. AbstentionBench (Facebook Research, 2024)

**Purpose**: Holistic evaluation of LLM abstention capabilities

**Description**: Large-scale benchmark spanning 20 datasets with over 35,000 unanswerable queries across 6 abstention scenarios:
- Unknown answers (unsolved problems, future events)
- Underspecified queries (altered GSM8K, MMLU, GPQA)
- False premise questions
- Subjective/ambiguous questions
- Outdated information queries

**Key Finding**: Reasoning fine-tuning degrades abstention by 24% on average.

**Links**:
- GitHub: https://github.com/facebookresearch/AbstentionBench
- Hugging Face: https://huggingface.co/datasets/facebook/AbstentionBench
- Paper: https://arxiv.org/abs/2506.09038

**License**: Check repository

---

## 2. SelfAware Dataset

**Purpose**: Evaluating LLM self-knowledge on answerable vs. unanswerable questions

**Description**: Contains 1,032 unanswerable questions and 2,337 answerable questions. Evaluates whether LLMs know what they don't know using F1 score metrics.

**Data Format**: JSON with fields for question ID, question text, answer (array or null), answerability boolean, and source.

**Links**:
- GitHub: https://github.com/yinzhangyue/SelfAware
- Paper: "Do Large Language Models Know What They Don't Know?" (ACL 2023)

**License**: CC-BY-SA-4.0

---

## 3. TruthfulQA

**Purpose**: Measuring LLM truthfulness and resistance to common misconceptions

**Description**: 817 questions across 38 categories (health, law, finance, politics). Questions designed to elicit false answers based on common human misconceptions.

**Key Statistics**:
- Best model (at time of publication): 58% truthful
- Human performance: 94%

**Links**:
- GitHub: https://github.com/sylinrl/TruthfulQA
- Hugging Face: https://huggingface.co/datasets/HiTZ/truthfulqa-multi
- Paper: https://arxiv.org/abs/2109.07958

**Evaluation**: Includes GPT-judge and GPT-info fine-tuned evaluators

---

## 4. HaluEval

**Purpose**: Large-scale hallucination evaluation for LLMs

**Description**: 35,000 samples including:
- 5,000 general user queries with ChatGPT responses
- 30,000 task-specific examples across:
  - Question Answering (10K, based on HotpotQA)
  - Knowledge-grounded Dialogue
  - Text Summarization

**Data Format**: JSON files with knowledge, question, right_answer, and hallucinated_answer fields.

**Links**:
- GitHub: https://github.com/RUCAIBox/HaluEval
- Paper: https://arxiv.org/abs/2305.11747 (EMNLP 2023)

**License**: MIT

---

## 5. SQuAD 2.0

**Purpose**: Reading comprehension with unanswerable questions

**Description**: Combines 100,000 answerable questions from SQuAD 1.1 with 50,000+ adversarially-written unanswerable questions. Systems must determine when no answer is supported by the passage.

**Key Innovation**: Unanswerable questions look similar to answerable ones, requiring models to abstain appropriately.

**Links**:
- Official: https://rajpurkar.github.io/SQuAD-explorer/
- Hugging Face: https://huggingface.co/datasets/rajpurkar/squad_v2
- Paper: https://arxiv.org/abs/1806.03822

**License**: CC BY-SA 4.0

---

## 6. FalseQA

**Purpose**: Evaluating LLM handling of false premise questions

**Description**: Dataset of questions with false premises paired with true premise questions. Models should detect and rebut false premises rather than hallucinate answers.

**Data Format**: CSV with question, answer (normal or rebuttal), and label (1=false premise, 0=true premise).

**Links**:
- GitHub: https://github.com/thunlp/FalseQA
- Paper: "Won't Get Fooled Again: Answering Questions with False Premises" (ACL 2023)

---

## 7. Natural Questions (NQ)

**Purpose**: Real user question answering with null answers

**Description**: 307,372 training examples of real Google search queries with Wikipedia-sourced answers. Includes null annotations for unanswerable questions.

**Annotation Types**:
- Long answer (paragraph-level)
- Short answer (entity-level)
- Null (no answer found)

**Links**:
- GitHub: https://github.com/google-research-datasets/natural-questions
- Download: https://ai.google.com/research/NaturalQuestions/download
- Hugging Face: https://huggingface.co/datasets/google-research-datasets/nq_open

**License**: CC BY-SA 3.0

---

## 8. Additional Datasets from Literature

### From R-Tuning Paper:
- **ParaRel**: Parametric knowledge probing
- **MMLU**: Massive Multitask Language Understanding
- **WiCE**: Word-in-Context Entailment
- **HotpotQA**: Multi-hop question answering
- **FEVER**: Fact Extraction and Verification
- **NEC**: Non-existent concepts

### From Know Your Limits Survey:
- **AmbigQA**: Ambiguous questions
- **PopQA**: Popularity-stratified QA
- **LFQA**: Long-form QA
- **Qwen-ReFL**: Reflection-based training data
- **PromptSource**: Prompt templates

---

## Dataset Selection Recommendations

### For Training Abstention Behavior:
1. **SelfAware** - Direct answerable/unanswerable classification
2. **SQuAD 2.0** - Large-scale with adversarial unanswerable questions
3. **FalseQA** - False premise detection and rebuttal

### For Evaluation:
1. **AbstentionBench** - Comprehensive multi-scenario evaluation
2. **TruthfulQA** - Truthfulness against misconceptions
3. **HaluEval** - Hallucination detection across tasks

### For Uncertainty Calibration:
1. **Natural Questions** - Real-world distribution with null answers
2. **TruthfulQA** - Calibrated confidence evaluation

---

## Download Scripts

```bash
# Clone key dataset repositories
git clone https://github.com/facebookresearch/AbstentionBench.git
git clone https://github.com/yinzhangyue/SelfAware.git
git clone https://github.com/sylinrl/TruthfulQA.git
git clone https://github.com/RUCAIBox/HaluEval.git
git clone https://github.com/thunlp/FalseQA.git

# Hugging Face datasets (via Python)
# from datasets import load_dataset
# abstention_bench = load_dataset("facebook/AbstentionBench")
# squad_v2 = load_dataset("rajpurkar/squad_v2")
# truthfulqa = load_dataset("HiTZ/truthfulqa-multi")
```
