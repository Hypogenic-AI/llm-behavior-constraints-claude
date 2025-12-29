"""
Data loading utilities for LLM abstention experiments.
Loads and preprocesses TruthfulQA and SelfAware datasets.
"""

import json
import random
from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class Question:
    """Represents a single question with metadata."""
    id: str
    question: str
    correct_answer: str
    is_answerable: bool
    source: str  # 'truthfulqa' or 'selfaware'
    category: str = ""


def load_truthfulqa(sample_size: int = 100, seed: int = 42) -> list[Question]:
    """
    Load TruthfulQA dataset.
    Questions are designed to elicit false answers based on common misconceptions.
    """
    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", trust_remote_code=True)

    questions = []
    for idx, item in enumerate(dataset["validation"]):
        # Use the best_answer as the correct answer
        q = Question(
            id=f"truthfulqa_{idx}",
            question=item["question"],
            correct_answer=item["best_answer"],
            is_answerable=True,  # All TruthfulQA questions are answerable (but tricky)
            source="truthfulqa",
            category=item.get("category", "")
        )
        questions.append(q)

    # Sample
    random.seed(seed)
    sampled = random.sample(questions, min(sample_size, len(questions)))
    print(f"Loaded {len(sampled)} TruthfulQA questions")
    return sampled


def load_selfaware(sample_size: int = 100, seed: int = 42) -> list[Question]:
    """
    Load SelfAware dataset.
    Contains answerable and unanswerable questions.
    """
    print("Loading SelfAware-style questions from SQuAD 2.0...")
    # SelfAware dataset is not on HuggingFace, using SQuAD 2.0 which has similar structure
    # SQuAD 2.0 contains unanswerable questions
    dataset = load_dataset("rajpurkar/squad_v2", trust_remote_code=True)

    answerable_qs = []
    unanswerable_qs = []

    for idx, item in enumerate(dataset["validation"]):
        has_answer = len(item["answers"]["text"]) > 0
        q = Question(
            id=f"squad2_{idx}",
            question=item["question"],
            correct_answer=item["answers"]["text"][0] if has_answer else "",
            is_answerable=has_answer,
            source="squad2",
            category=item["title"]
        )
        if has_answer:
            answerable_qs.append(q)
        else:
            unanswerable_qs.append(q)

    # Sample balanced set (half answerable, half unanswerable)
    random.seed(seed)
    n_each = sample_size // 2
    sampled_answerable = random.sample(answerable_qs, min(n_each, len(answerable_qs)))
    sampled_unanswerable = random.sample(unanswerable_qs, min(n_each, len(unanswerable_qs)))

    combined = sampled_answerable + sampled_unanswerable
    random.shuffle(combined)

    print(f"Loaded {len(sampled_answerable)} answerable + {len(sampled_unanswerable)} unanswerable = {len(combined)} SQuAD 2.0 questions")
    return combined


def load_mixed_dataset(truthful_size: int = 50, squad_size: int = 100, seed: int = 42) -> list[Question]:
    """
    Load a mixed dataset for comprehensive evaluation.
    Includes TruthfulQA (for testing resistance to misconceptions)
    and SQuAD 2.0 (for testing answerable vs. unanswerable classification).
    """
    truthful = load_truthfulqa(sample_size=truthful_size, seed=seed)
    squad = load_selfaware(sample_size=squad_size, seed=seed)

    combined = truthful + squad
    random.seed(seed)
    random.shuffle(combined)

    print(f"\nTotal: {len(combined)} questions")
    print(f"  - TruthfulQA: {len(truthful)}")
    print(f"  - SQuAD 2.0: {len(squad)}")

    return combined


def save_questions(questions: list[Question], filepath: str):
    """Save questions to JSON file."""
    data = [
        {
            "id": q.id,
            "question": q.question,
            "correct_answer": q.correct_answer,
            "is_answerable": q.is_answerable,
            "source": q.source,
            "category": q.category
        }
        for q in questions
    ]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Test loading
    questions = load_mixed_dataset(truthful_size=10, squad_size=20, seed=42)
    print("\nSample questions:")
    for q in questions[:3]:
        print(f"  [{q.source}] {q.question[:60]}... (answerable={q.is_answerable})")
