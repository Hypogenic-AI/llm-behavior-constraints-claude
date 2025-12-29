"""
Main experiment runner for LLM abstention research.
Tests multiple prompting strategies across different models.
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_mixed_dataset, Question
from prompts import format_prompt, PROMPT_CONDITIONS
from api_client import get_client, LLMResponse, detect_abstention
from self_consistency import sample_multiple_responses, check_self_consistency
from metrics import compute_basic_metrics, bootstrap_confidence_interval


# Configuration - optimized for meaningful results with reasonable runtime
CONFIG = {
    "seed": 42,
    "truthfulqa_size": 30,  # Tricky questions to test hallucination resistance
    "squad_size": 60,  # 30 answerable + 30 unanswerable - tests abstention ability
    "models": ["gpt-4o-mini", "anthropic/claude-sonnet-4"],
    "conditions": ["baseline", "explicit_abstention", "cot_uncertainty"],
    "self_consistency_samples": 3,  # Reduced for speed, still effective
    "self_consistency_temperature": 0.7,
    "results_dir": Path(__file__).parent.parent / "results",
    "max_retries": 3,
    "retry_delay": 2.0
}


def evaluate_answer(response_text: str, question: Question) -> bool:
    """
    Check if response is correct.
    For TruthfulQA: check if response aligns with correct answer.
    For SQuAD 2.0: check if response contains the correct answer text.
    """
    if question.source == "squad2":
        if not question.is_answerable:
            # For unanswerable questions, abstention is correct
            return False  # Not "correct" in traditional sense
        # Check if correct answer appears in response
        return question.correct_answer.lower() in response_text.lower()
    elif question.source == "truthfulqa":
        # Simple check: does response contain key elements of correct answer
        correct_words = set(question.correct_answer.lower().split())
        response_words = set(response_text.lower().split())
        overlap = len(correct_words & response_words) / len(correct_words) if correct_words else 0
        return overlap > 0.5
    return False


def run_single_experiment(
    client,
    question: Question,
    condition: str,
    retries: int = 3
) -> dict:
    """Run a single question through one condition."""
    system_prompt, user_prompt = format_prompt(condition, question.question)

    for attempt in range(retries):
        try:
            response = client.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                question_id=question.id,
                prompt_condition=condition,
                temperature=0.0,
                max_tokens=256
            )

            if "ERROR" not in response.response_text:
                return {
                    "question_id": question.id,
                    "question": question.question,
                    "correct_answer": question.correct_answer,
                    "is_answerable": question.is_answerable,
                    "source": question.source,
                    "model": response.model,
                    "condition": condition,
                    "response_text": response.response_text,
                    "is_abstention": response.is_abstention,
                    "is_correct": evaluate_answer(response.response_text, question) if not response.is_abstention else False,
                    "confidence": response.confidence,
                    "latency_ms": response.latency_ms,
                    "tokens_used": response.tokens_used
                }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(CONFIG["retry_delay"])

    # All retries failed
    return {
        "question_id": question.id,
        "question": question.question,
        "correct_answer": question.correct_answer,
        "is_answerable": question.is_answerable,
        "source": question.source,
        "model": client.model if hasattr(client, 'model') else "unknown",
        "condition": condition,
        "response_text": "ERROR: All retries failed",
        "is_abstention": False,
        "is_correct": False,
        "confidence": None,
        "latency_ms": 0,
        "tokens_used": 0
    }


def run_self_consistency_experiment(
    client,
    question: Question,
    n_samples: int = 5,
    temperature: float = 0.7
) -> dict:
    """Run self-consistency experiment for a single question."""
    system_prompt, user_prompt = format_prompt("self_consistency", question.question)

    responses = sample_multiple_responses(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        question_id=question.id,
        prompt_condition="self_consistency",
        n_samples=n_samples,
        temperature=temperature
    )

    # Check consistency
    consistency_result = check_self_consistency(responses)

    # Determine final answer based on consistency
    final_response = consistency_result.majority_answer
    is_abstention = consistency_result.should_abstain

    return {
        "question_id": question.id,
        "question": question.question,
        "correct_answer": question.correct_answer,
        "is_answerable": question.is_answerable,
        "source": question.source,
        "model": responses[0].model if responses else "unknown",
        "condition": "self_consistency",
        "response_text": final_response,
        "is_abstention": is_abstention,
        "is_correct": evaluate_answer(final_response, question) if not is_abstention else False,
        "confidence": None,
        "latency_ms": sum(r.latency_ms for r in responses),
        "tokens_used": sum(r.tokens_used for r in responses),
        "n_samples": n_samples,
        "agreement_score": consistency_result.agreement_score,
        "all_responses": consistency_result.responses
    }


def run_experiments(questions: list[Question], config: dict) -> dict:
    """Run all experiments across models and conditions."""
    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "experiments": []
    }

    for model_name in config["models"]:
        print(f"\n{'='*60}")
        print(f"Running experiments with model: {model_name}")
        print(f"{'='*60}")

        client = get_client(model_name)

        # Run standard conditions
        for condition in config["conditions"]:
            print(f"\n--- Condition: {condition} ---")
            condition_results = []

            for q in tqdm(questions, desc=f"{model_name}/{condition}"):
                result = run_single_experiment(client, q, condition)
                condition_results.append(result)
                # Small delay to avoid rate limits
                time.sleep(0.1)

            results["experiments"].extend(condition_results)

            # Print quick stats
            abstention_count = sum(1 for r in condition_results if r["is_abstention"])
            print(f"Abstention rate: {abstention_count}/{len(condition_results)} = {abstention_count/len(condition_results):.1%}")

        # Run self-consistency condition
        print(f"\n--- Condition: self_consistency ---")
        sc_results = []

        for q in tqdm(questions, desc=f"{model_name}/self_consistency"):
            result = run_self_consistency_experiment(
                client, q,
                n_samples=config["self_consistency_samples"],
                temperature=config["self_consistency_temperature"]
            )
            sc_results.append(result)
            time.sleep(0.2)  # Longer delay for multiple samples

        results["experiments"].extend(sc_results)

        abstention_count = sum(1 for r in sc_results if r["is_abstention"])
        print(f"Abstention rate: {abstention_count}/{len(sc_results)} = {abstention_count/len(sc_results):.1%}")

    return results


def analyze_results(results: dict) -> dict:
    """Compute metrics for all experiments."""
    analysis = {
        "summary": {},
        "by_model": {},
        "by_condition": {},
        "by_source": {}
    }

    experiments = results["experiments"]

    # Group by model and condition
    for model in results["config"]["models"]:
        model_short = model.split("/")[-1]  # Get short name
        analysis["by_model"][model_short] = {}

        for condition in results["config"]["conditions"] + ["self_consistency"]:
            subset = [e for e in experiments if e["model"] == model and e["condition"] == condition]

            if not subset:
                continue

            metrics = compute_basic_metrics(subset, model, condition)

            analysis["by_model"][model_short][condition] = {
                "n": metrics.n_questions,
                "abstention_rate": metrics.abstention_rate,
                "answer_accuracy": metrics.answer_accuracy,
                "overall_accuracy": metrics.overall_accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score
            }

    # Compute aggregate by condition
    for condition in results["config"]["conditions"] + ["self_consistency"]:
        subset = [e for e in experiments if e["condition"] == condition]
        if subset:
            metrics = compute_basic_metrics(subset, "all", condition)
            analysis["by_condition"][condition] = {
                "n": metrics.n_questions,
                "abstention_rate": metrics.abstention_rate,
                "answer_accuracy": metrics.answer_accuracy,
                "overall_accuracy": metrics.overall_accuracy,
                "f1_score": metrics.f1_score
            }

    # Compute by source dataset
    for source in ["truthfulqa", "squad2"]:
        subset = [e for e in experiments if e["source"] == source]
        if subset:
            abstentions = sum(1 for e in subset if e["is_abstention"])
            correct = sum(1 for e in subset if e.get("is_correct", False))
            analysis["by_source"][source] = {
                "n": len(subset),
                "abstention_rate": abstentions / len(subset),
                "correct_rate": correct / len(subset)
            }

    return analysis


def main():
    """Main entry point."""
    random.seed(CONFIG["seed"])

    # Create results directory
    results_dir = CONFIG["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw").mkdir(exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)

    print("Loading datasets...")
    questions = load_mixed_dataset(
        truthful_size=CONFIG["truthfulqa_size"],
        squad_size=CONFIG["squad_size"],
        seed=CONFIG["seed"]
    )

    print(f"\nStarting experiments with {len(questions)} questions...")
    print(f"Models: {CONFIG['models']}")
    print(f"Conditions: {CONFIG['conditions']} + self_consistency")

    # Run experiments
    results = run_experiments(questions, CONFIG)

    # Save raw results
    raw_path = results_dir / "raw" / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {raw_path}")

    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_results(results)

    # Save analysis
    analysis_path = results_dir / "metrics.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_path}")

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for model, conditions in analysis["by_model"].items():
        print(f"\n{model}:")
        for condition, metrics in conditions.items():
            print(f"  {condition}:")
            print(f"    Abstention rate: {metrics['abstention_rate']:.1%}")
            print(f"    Answer accuracy: {metrics['answer_accuracy']:.1%}")
            print(f"    F1 score: {metrics['f1_score']:.2f}")

    return results, analysis


if __name__ == "__main__":
    results, analysis = main()
