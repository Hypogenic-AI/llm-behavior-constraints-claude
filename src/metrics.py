"""
Evaluation metrics for LLM abstention experiments.
Computes accuracy, abstention rates, F1, AUROC, and calibration metrics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


@dataclass
class ExperimentMetrics:
    """Container for all computed metrics."""
    model: str
    prompt_condition: str
    n_questions: int

    # Basic metrics
    abstention_rate: float
    answer_accuracy: float  # Accuracy on answered questions
    overall_accuracy: float  # Including abstentions as incorrect

    # Abstention quality
    precision: float  # Of abstentions, what % were on questions model would get wrong
    recall: float  # Of questions model gets wrong, what % did it abstain on
    f1_score: float

    # If available
    auroc: Optional[float] = None
    ece: Optional[float] = None  # Expected Calibration Error


def compute_basic_metrics(
    predictions: list[dict],
    model: str,
    condition: str
) -> ExperimentMetrics:
    """
    Compute basic metrics from experiment predictions.

    Each prediction dict should have:
    - question_id: str
    - is_abstention: bool
    - response_text: str
    - is_correct: bool (if answered)
    - is_answerable: bool (ground truth)
    """
    n = len(predictions)
    if n == 0:
        raise ValueError("No predictions provided")

    abstentions = [p for p in predictions if p["is_abstention"]]
    answers = [p for p in predictions if not p["is_abstention"]]

    abstention_rate = len(abstentions) / n

    # Accuracy on answered questions
    if answers:
        correct_answers = sum(1 for p in answers if p.get("is_correct", False))
        answer_accuracy = correct_answers / len(answers)
    else:
        answer_accuracy = 0.0

    # Overall accuracy (treating abstentions as neither correct nor incorrect for answered questions)
    # For unanswerable questions, abstention IS correct
    # For answerable questions, need actual answer to be correct
    correct_count = 0
    for p in predictions:
        if p["is_abstention"]:
            # Abstention is correct if question was unanswerable
            if not p.get("is_answerable", True):
                correct_count += 1
        else:
            # Answer is correct if question was answerable AND answer is correct
            if p.get("is_correct", False):
                correct_count += 1
    overall_accuracy = correct_count / n

    # Compute precision/recall for abstention as a binary classification task
    # True positive: Abstained AND question is unanswerable
    # False positive: Abstained AND question is answerable
    # False negative: Didn't abstain AND question is unanswerable
    # True negative: Didn't abstain AND question is answerable

    tp = sum(1 for p in predictions if p["is_abstention"] and not p.get("is_answerable", True))
    fp = sum(1 for p in predictions if p["is_abstention"] and p.get("is_answerable", True))
    fn = sum(1 for p in predictions if not p["is_abstention"] and not p.get("is_answerable", True))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ExperimentMetrics(
        model=model,
        prompt_condition=condition,
        n_questions=n,
        abstention_rate=abstention_rate,
        answer_accuracy=answer_accuracy,
        overall_accuracy=overall_accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1
    )


def compute_auroc(
    predictions: list[dict],
    confidence_key: str = "confidence_score"
) -> Optional[float]:
    """
    Compute AUROC for abstention decisions.
    Uses confidence scores to predict correctness.

    Returns None if cannot compute (e.g., only one class).
    """
    y_true = []  # 1 if incorrect/unanswerable (should abstain), 0 otherwise
    y_scores = []  # Lower confidence = higher "abstention score"

    for p in predictions:
        if confidence_key not in p:
            continue

        # Ground truth: should abstain?
        should_abstain = not p.get("is_answerable", True) or not p.get("is_correct", True)
        y_true.append(1 if should_abstain else 0)

        # Score: inverse of confidence (low confidence -> should abstain)
        y_scores.append(1 - p[confidence_key])

    if len(set(y_true)) < 2:
        return None  # Need both classes

    try:
        return roc_auc_score(y_true, y_scores)
    except Exception:
        return None


def compute_ece(
    predictions: list[dict],
    n_bins: int = 10,
    confidence_key: str = "confidence_score"
) -> Optional[float]:
    """
    Compute Expected Calibration Error.
    Measures how well stated confidence aligns with actual accuracy.
    """
    confidences = []
    correct = []

    for p in predictions:
        if confidence_key not in p or p["is_abstention"]:
            continue

        confidences.append(p[confidence_key])
        correct.append(1 if p.get("is_correct", False) else 0)

    if len(confidences) < n_bins:
        return None

    confidences = np.array(confidences)
    correct = np.array(correct)

    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accuracy = correct[mask].mean()
            bin_confidence = confidences[mask].mean()
            bin_weight = mask.sum() / len(confidences)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def compare_conditions_ttest(
    metrics1: list[dict],
    metrics2: list[dict],
    metric_key: str
) -> dict:
    """
    Perform paired t-test between two conditions.
    Assumes questions are paired (same questions in both conditions).
    """
    values1 = [m.get(metric_key, 0) for m in metrics1]
    values2 = [m.get(metric_key, 0) for m in metrics2]

    if len(values1) != len(values2):
        # Fall back to independent t-test
        t_stat, p_value = stats.ttest_ind(values1, values2)
    else:
        t_stat, p_value = stats.ttest_rel(values1, values2)

    effect_size = (np.mean(values1) - np.mean(values2)) / np.std(values1 + values2)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "effect_size_d": effect_size,
        "mean1": np.mean(values1),
        "mean2": np.mean(values2),
        "significant": p_value < 0.05
    }


def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    Returns (mean, lower_bound, upper_bound).
    """
    np.random.seed(seed)
    values = np.array(values)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(boot_means, alpha / 2 * 100)
    upper = np.percentile(boot_means, (1 - alpha / 2) * 100)

    return float(np.mean(values)), float(lower), float(upper)


if __name__ == "__main__":
    # Test metrics computation
    test_predictions = [
        {"question_id": "1", "is_abstention": False, "is_correct": True, "is_answerable": True},
        {"question_id": "2", "is_abstention": False, "is_correct": False, "is_answerable": True},
        {"question_id": "3", "is_abstention": True, "is_correct": False, "is_answerable": False},
        {"question_id": "4", "is_abstention": True, "is_correct": False, "is_answerable": True},
        {"question_id": "5", "is_abstention": False, "is_correct": True, "is_answerable": True},
    ]

    metrics = compute_basic_metrics(test_predictions, "test_model", "test_condition")
    print(f"Abstention rate: {metrics.abstention_rate:.2%}")
    print(f"Answer accuracy: {metrics.answer_accuracy:.2%}")
    print(f"Overall accuracy: {metrics.overall_accuracy:.2%}")
    print(f"Precision: {metrics.precision:.2%}")
    print(f"Recall: {metrics.recall:.2%}")
    print(f"F1: {metrics.f1_score:.2%}")
