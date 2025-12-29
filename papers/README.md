# Downloaded Papers

This directory contains research papers related to constraining LLM behavior, uncertainty quantification, and abstention.

## Paper List

### 1. TruthfulQA: Measuring How Models Mimic Human Falsehoods
- **File**: `2109.07958_TruthfulQA.pdf`
- **Authors**: Lin, Hilton, Evans (2021)
- **arXiv**: https://arxiv.org/abs/2109.07958
- **Why relevant**: Foundational benchmark for measuring LLM truthfulness; evaluates if models can avoid generating false answers

### 2. Training a Helpful and Harmless Assistant with RLHF
- **File**: `2207.05221_Training_HHH.pdf`
- **Authors**: Bai et al. (Anthropic, 2022)
- **arXiv**: https://arxiv.org/abs/2207.05221
- **Why relevant**: Foundational paper on RLHF for AI alignment including honesty; defines the "helpful, harmless, honest" framework

### 3. ReAct: Synergizing Reasoning and Acting in Language Models
- **File**: `2210.07662_ReAct.pdf`
- **Authors**: Yao et al. (2022)
- **arXiv**: https://arxiv.org/abs/2210.07662
- **Why relevant**: Shows how reasoning traces can improve LLM reliability and self-awareness

### 4. LLaMA: Open and Efficient Foundation Language Models
- **File**: `2302.13971_LLaMA.pdf`
- **Authors**: Touvron et al. (Meta, 2023)
- **arXiv**: https://arxiv.org/abs/2302.13971
- **Why relevant**: Key baseline model used in many uncertainty/abstention experiments

### 5. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection
- **File**: `2303.08896_SelfCheckGPT_Original.pdf`
- **Authors**: Manakul, Liusie, Gales (2023)
- **arXiv**: https://arxiv.org/abs/2303.08896
- **Why relevant**: Foundational method for self-consistency based hallucination detection

### 6. Do Large Language Models Know What They Don't Know?
- **File**: `2305.18153_Do_LLMs_Know_What_They_Dont_Know.pdf`
- **Authors**: Yin et al. (2023)
- **arXiv**: https://arxiv.org/abs/2305.18153
- **Why relevant**: Directly investigates LLM self-knowledge and ability to identify knowledge gaps

### 7. A Survey on Hallucination in Large Language Models
- **File**: `2311.05232_Hallucination_Survey.pdf`
- **Authors**: Huang et al. (2023)
- **arXiv**: https://arxiv.org/abs/2311.05232
- **Why relevant**: Comprehensive survey covering hallucination causes, detection, and mitigation

### 8. Confidence and Calibration in LLMs: A Survey
- **File**: `2311.08298_Confidence_Calibration_Survey.pdf`
- **Authors**: Unknown (2023)
- **arXiv**: https://arxiv.org/abs/2311.08298
- **Why relevant**: Survey on confidence estimation and calibration methods

### 9. R-Tuning: Instructing LLMs to Say 'I Don't Know'
- **File**: `2311.09677_R-Tuning_Say_IDK.pdf`
- **Authors**: Diao et al. (2023)
- **arXiv**: https://arxiv.org/abs/2311.09677
- **Why relevant**: **KEY PAPER** - NAACL 2024 Outstanding Paper on training LLMs to refuse unknown questions

### 10. Calibrated Language Models Must Hallucinate
- **File**: `2311.14648_Calibrated_LMs_Must_Hallucinate.pdf`
- **Authors**: Kalai & Vempala (2023)
- **arXiv**: https://arxiv.org/abs/2311.14648
- **Why relevant**: Theoretical analysis of calibration-hallucination tradeoff

### 11. SelfCheckGPT (Extended Version)
- **File**: `2402.03744_SelfCheckGPT.pdf`
- **Authors**: Manakul et al. (2024)
- **arXiv**: https://arxiv.org/abs/2402.03744
- **Why relevant**: Extended version of SelfCheckGPT with additional experiments

### 12. Supervised UQ Approach for LLMs
- **File**: `2404.15993_UQ_Supervised_Approach.pdf`
- **Authors**: Unknown (2024)
- **arXiv**: https://arxiv.org/abs/2404.15993
- **Why relevant**: Supervised approach to uncertainty quantification

### 13. More RLHF, More Trust? Impact of Preference Alignment on Trustworthiness
- **File**: `2404.18870_RLHF_Trust.pdf`
- **Authors**: Unknown (2024)
- **arXiv**: https://arxiv.org/abs/2404.18870
- **Why relevant**: Evaluates RLHF impact on truthfulness and other trust dimensions

### 14. Mitigating LLM Hallucinations via Conformal Abstention
- **File**: `2405.01563_Conformal_Abstention.pdf`
- **Authors**: Abbasi-Yadkori et al. (2024)
- **arXiv**: https://arxiv.org/abs/2405.01563
- **Why relevant**: **KEY PAPER** - Uses conformal prediction for principled abstention with guarantees

### 15. Semantic Entropy Probes: Cheap Hallucination Detection
- **File**: `2406.15927_Semantic_Entropy_Probes.pdf`
- **Authors**: Kossen et al. (2024)
- **arXiv**: https://arxiv.org/abs/2406.15927
- **Why relevant**: Efficient uncertainty quantification without sampling

### 16. Know Your Limits: A Survey of Abstention in LLMs
- **File**: `2407.18418_Know_Your_Limits_Abstention_Survey.pdf`
- **Authors**: Unknown (2024)
- **arXiv**: https://arxiv.org/abs/2407.18418
- **Why relevant**: **KEY SURVEY** - Comprehensive survey on abstention methods, benchmarks, and evaluation

### 17. Looking Inward: LMs Can Learn About Themselves by Introspection
- **File**: `2410.13787_LM_Introspection.pdf`
- **Authors**: Binder et al. (2024)
- **arXiv**: https://arxiv.org/abs/2410.13787
- **Why relevant**: Investigates whether LLMs can introspect and know their own behavior

## Key Papers for This Research

The most directly relevant papers for the research hypothesis are:
1. **R-Tuning** (2311.09677) - Training signal for refusal
2. **Conformal Abstention** (2405.01563) - Principled abstention with guarantees
3. **Know Your Limits Survey** (2407.18418) - Comprehensive abstention overview
4. **SelfCheckGPT** (2303.08896) - Self-consistency for hallucination detection
5. **Do LLMs Know What They Don't Know** (2305.18153) - Self-knowledge investigation
