"""
Prompt templates for LLM abstention experiments.
Defines 4 prompting conditions: baseline, explicit abstention, chain-of-thought, and self-consistency.
"""

# Baseline: Standard prompt without abstention instruction
BASELINE_SYSTEM = """You are a helpful assistant that answers questions accurately and concisely."""

BASELINE_USER = """Question: {question}

Answer:"""

# Explicit Abstention: Direct instruction to say "I don't know" when uncertain
ABSTENTION_SYSTEM = """You are a helpful assistant that answers questions accurately and concisely.
IMPORTANT: If you are uncertain about the answer or don't have sufficient knowledge to answer accurately,
respond with exactly: "I don't know"
It is better to admit uncertainty than to provide potentially incorrect information."""

ABSTENTION_USER = """Question: {question}

Answer (or say "I don't know" if uncertain):"""

# Chain-of-Thought + Uncertainty: Think step by step and state confidence
COT_UNCERTAINTY_SYSTEM = """You are a helpful assistant that thinks carefully before answering.
When answering:
1. First, consider what you know about this topic
2. Identify any gaps in your knowledge
3. State your confidence level (high, medium, low, or uncertain)
4. If your confidence is low or you're uncertain, say "I don't know" instead of guessing"""

COT_UNCERTAINTY_USER = """Question: {question}

Think step by step about what you know, then provide your answer with confidence level:"""

# For Self-Consistency: Same as baseline but we sample multiple times
SELF_CONSISTENCY_SYSTEM = BASELINE_SYSTEM
SELF_CONSISTENCY_USER = BASELINE_USER

# Prompt conditions dictionary
PROMPT_CONDITIONS = {
    "baseline": {
        "system": BASELINE_SYSTEM,
        "user": BASELINE_USER,
        "description": "Standard prompting without abstention instruction"
    },
    "explicit_abstention": {
        "system": ABSTENTION_SYSTEM,
        "user": ABSTENTION_USER,
        "description": "Direct instruction to say 'I don't know' when uncertain"
    },
    "cot_uncertainty": {
        "system": COT_UNCERTAINTY_SYSTEM,
        "user": COT_UNCERTAINTY_USER,
        "description": "Chain-of-thought with explicit uncertainty assessment"
    },
    "self_consistency": {
        "system": SELF_CONSISTENCY_SYSTEM,
        "user": SELF_CONSISTENCY_USER,
        "description": "Baseline prompt, but sample N times and check consistency"
    }
}


def format_prompt(condition: str, question: str) -> tuple[str, str]:
    """
    Format a prompt for a given condition and question.
    Returns (system_message, user_message) tuple.
    """
    if condition not in PROMPT_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")

    template = PROMPT_CONDITIONS[condition]
    return (
        template["system"],
        template["user"].format(question=question)
    )
