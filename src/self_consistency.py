"""
Self-consistency implementation for hallucination detection.
Samples multiple responses and measures agreement to detect potential hallucinations.
"""

import re
from dataclasses import dataclass
from typing import Optional
from collections import Counter

from api_client import LLMResponse, detect_abstention


@dataclass
class SelfConsistencyResult:
    """Result of self-consistency checking on multiple responses."""
    question_id: str
    num_samples: int
    responses: list[str]
    agreement_score: float  # 0-1, higher = more consistent
    majority_answer: str
    is_consistent: bool  # True if agreement above threshold
    should_abstain: bool  # Based on low consistency
    abstention_count: int


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    Removes punctuation, lowercases, and strips whitespace.
    """
    # Remove common prefixes
    text = re.sub(r'^(the answer is|answer:|i think|i believe|based on my knowledge,?)\s*', '', text.lower())
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


def compute_agreement(responses: list[str]) -> tuple[float, str]:
    """
    Compute agreement score among responses using simple text matching.
    Returns (agreement_score, majority_answer).

    Agreement is based on normalized answer overlap.
    """
    if not responses:
        return 0.0, ""

    if len(responses) == 1:
        return 1.0, responses[0]

    # Normalize all responses
    normalized = [normalize_answer(r) for r in responses]

    # Find most common response
    counter = Counter(normalized)
    majority, majority_count = counter.most_common(1)[0]

    # Agreement = proportion of responses matching majority
    agreement_score = majority_count / len(responses)

    # Find original (non-normalized) version of majority answer
    majority_original = responses[normalized.index(majority)]

    return agreement_score, majority_original


def check_self_consistency(
    responses: list[LLMResponse],
    consistency_threshold: float = 0.6
) -> SelfConsistencyResult:
    """
    Check self-consistency of multiple sampled responses.
    If responses are inconsistent, the model may be hallucinating.

    Args:
        responses: List of LLMResponse objects for the same question
        consistency_threshold: Minimum agreement score to consider consistent

    Returns:
        SelfConsistencyResult with analysis
    """
    if not responses:
        raise ValueError("No responses provided")

    question_id = responses[0].question_id
    response_texts = [r.response_text for r in responses]

    # Count abstentions
    abstention_count = sum(1 for r in responses if r.is_abstention)

    # If majority abstains, we should abstain
    if abstention_count > len(responses) / 2:
        return SelfConsistencyResult(
            question_id=question_id,
            num_samples=len(responses),
            responses=response_texts,
            agreement_score=abstention_count / len(responses),
            majority_answer="I don't know",
            is_consistent=True,  # Consistent in abstention
            should_abstain=True,
            abstention_count=abstention_count
        )

    # Filter out abstentions for agreement calculation
    non_abstention_texts = [r.response_text for r in responses if not r.is_abstention]

    if not non_abstention_texts:
        return SelfConsistencyResult(
            question_id=question_id,
            num_samples=len(responses),
            responses=response_texts,
            agreement_score=1.0,
            majority_answer="I don't know",
            is_consistent=True,
            should_abstain=True,
            abstention_count=abstention_count
        )

    agreement_score, majority_answer = compute_agreement(non_abstention_texts)

    return SelfConsistencyResult(
        question_id=question_id,
        num_samples=len(responses),
        responses=response_texts,
        agreement_score=agreement_score,
        majority_answer=majority_answer,
        is_consistent=agreement_score >= consistency_threshold,
        should_abstain=agreement_score < consistency_threshold,
        abstention_count=abstention_count
    )


def sample_multiple_responses(
    client,
    system_prompt: str,
    user_prompt: str,
    question_id: str,
    prompt_condition: str,
    n_samples: int = 5,
    temperature: float = 0.7
) -> list[LLMResponse]:
    """
    Sample multiple responses for self-consistency checking.

    Args:
        client: API client (OpenAI or OpenRouter)
        system_prompt: System message
        user_prompt: User message with question
        question_id: ID for tracking
        prompt_condition: Name of prompt condition
        n_samples: Number of samples to generate
        temperature: Sampling temperature (higher = more diverse)

    Returns:
        List of LLMResponse objects
    """
    responses = []
    for i in range(n_samples):
        response = client.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            question_id=f"{question_id}_sample{i}",
            prompt_condition=prompt_condition,
            temperature=temperature,
            max_tokens=256
        )
        responses.append(response)

    return responses


if __name__ == "__main__":
    # Test self-consistency checking
    from api_client import LLMResponse

    # Simulate consistent responses
    consistent_responses = [
        LLMResponse("test", "test", "q1", "The answer is 42.", False),
        LLMResponse("test", "test", "q1", "42", False),
        LLMResponse("test", "test", "q1", "The answer is 42!", False),
    ]

    result = check_self_consistency(consistent_responses)
    print(f"Consistent test: agreement={result.agreement_score:.2f}, should_abstain={result.should_abstain}")

    # Simulate inconsistent responses
    inconsistent_responses = [
        LLMResponse("test", "test", "q2", "The capital is Paris.", False),
        LLMResponse("test", "test", "q2", "I believe it's London.", False),
        LLMResponse("test", "test", "q2", "The capital is Berlin.", False),
    ]

    result = check_self_consistency(inconsistent_responses)
    print(f"Inconsistent test: agreement={result.agreement_score:.2f}, should_abstain={result.should_abstain}")
