"""
Statistical analysis and visualization of LLM abstention experiment results.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Optional

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_results():
    """Load raw results and metrics."""
    # Find most recent raw results
    raw_files = sorted((RESULTS_DIR / "raw").glob("results_*.json"))
    if not raw_files:
        raise FileNotFoundError("No raw results found")

    with open(raw_files[-1]) as f:
        raw_results = json.load(f)

    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    return raw_results, metrics


def create_comparison_dataframe(raw_results: dict) -> pd.DataFrame:
    """Convert raw results to a DataFrame for analysis."""
    experiments = raw_results["experiments"]

    df = pd.DataFrame(experiments)
    # Clean up model names
    df["model_short"] = df["model"].apply(lambda x: x.split("/")[-1])
    return df


def statistical_tests(df: pd.DataFrame) -> dict:
    """Perform statistical tests comparing conditions."""
    results = {"comparisons": {}}

    # Get unique conditions and models
    conditions = df["condition"].unique()
    models = df["model_short"].unique()

    # Chi-square test for abstention rates across conditions
    for model in models:
        model_df = df[df["model_short"] == model]

        # Create contingency table: abstention (yes/no) vs condition
        contingency = pd.crosstab(model_df["condition"], model_df["is_abstention"])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

        results["comparisons"][f"{model}_chi2"] = {
            "test": "chi-squared",
            "statistic": float(chi2),
            "p_value": float(p_val),
            "degrees_of_freedom": int(dof),
            "interpretation": "significant" if p_val < 0.05 else "not significant"
        }

    # Pairwise comparisons: baseline vs each intervention
    for model in models:
        model_df = df[df["model_short"] == model]
        baseline_abs = model_df[model_df["condition"] == "baseline"]["is_abstention"].values
        baseline_rate = baseline_abs.mean()

        for condition in ["explicit_abstention", "cot_uncertainty", "self_consistency"]:
            cond_abs = model_df[model_df["condition"] == condition]["is_abstention"].values
            cond_rate = cond_abs.mean()

            # Two-proportion z-test
            n1, n2 = len(baseline_abs), len(cond_abs)
            p_pooled = (baseline_abs.sum() + cond_abs.sum()) / (n1 + n2)

            if p_pooled == 0 or p_pooled == 1:
                z_stat, z_pval = 0, 1
            else:
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                z_stat = (cond_rate - baseline_rate) / se if se > 0 else 0
                z_pval = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            # Effect size (Cohen's h for proportions)
            h1 = 2 * np.arcsin(np.sqrt(baseline_rate))
            h2 = 2 * np.arcsin(np.sqrt(cond_rate))
            cohens_h = abs(h2 - h1)

            results["comparisons"][f"{model}_{condition}_vs_baseline"] = {
                "test": "two-proportion z-test",
                "baseline_rate": float(baseline_rate),
                "intervention_rate": float(cond_rate),
                "difference": float(cond_rate - baseline_rate),
                "z_statistic": float(z_stat),
                "p_value": float(z_pval),
                "cohens_h": float(cohens_h),
                "effect_size_interpretation": (
                    "small" if cohens_h < 0.5 else
                    "medium" if cohens_h < 0.8 else "large"
                ),
                "significant": z_pval < 0.05
            }

    return results


def bootstrap_confidence_intervals(df: pd.DataFrame, metric: str = "is_abstention",
                                    n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """Compute bootstrap confidence intervals for each condition."""
    results = {}
    conditions = df["condition"].unique()

    for condition in conditions:
        cond_values = df[df["condition"] == condition][metric].values

        boot_means = []
        np.random.seed(42)
        for _ in range(n_bootstrap):
            sample = np.random.choice(cond_values, size=len(cond_values), replace=True)
            boot_means.append(np.mean(sample))

        alpha = 1 - ci
        lower = np.percentile(boot_means, alpha / 2 * 100)
        upper = np.percentile(boot_means, (1 - alpha / 2) * 100)

        results[condition] = {
            "mean": float(np.mean(cond_values)),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "ci_level": ci
        }

    return results


def plot_abstention_rates(df: pd.DataFrame, save_path: Path):
    """Create bar plot of abstention rates by condition and model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute abstention rates
    summary = df.groupby(["model_short", "condition"]).agg({
        "is_abstention": ["mean", "std", "count"]
    }).reset_index()
    summary.columns = ["model", "condition", "mean", "std", "n"]
    summary["se"] = summary["std"] / np.sqrt(summary["n"])

    # Define condition order
    condition_order = ["baseline", "explicit_abstention", "cot_uncertainty", "self_consistency"]
    summary["condition"] = pd.Categorical(summary["condition"], categories=condition_order, ordered=True)
    summary = summary.sort_values(["condition", "model"])

    # Plot grouped bar chart
    x = np.arange(len(condition_order))
    width = 0.35
    models = summary["model"].unique()

    for i, model in enumerate(models):
        model_data = summary[summary["model"] == model]
        offset = width * (i - 0.5)
        bars = ax.bar(x + offset, model_data["mean"], width,
                     yerr=model_data["se"] * 1.96,
                     label=model, capsize=3)

    ax.set_xlabel("Prompting Condition", fontsize=12)
    ax.set_ylabel("Abstention Rate", fontsize=12)
    ax.set_title("Abstention Rates by Prompting Strategy and Model", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Explicit\nAbstention", "CoT +\nUncertainty", "Self-\nConsistency"])
    ax.legend(title="Model")
    ax.set_ylim(0, 1)

    # Add significance stars
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.3, label='Random abstention')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_accuracy_abstention_tradeoff(df: pd.DataFrame, save_path: Path):
    """Plot accuracy vs abstention rate tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 8))

    markers = {"gpt-4o-mini": "o", "claude-sonnet-4": "s"}
    colors = {"baseline": "#1f77b4", "explicit_abstention": "#ff7f0e",
              "cot_uncertainty": "#2ca02c", "self_consistency": "#d62728"}

    for model in df["model_short"].unique():
        for condition in df["condition"].unique():
            subset = df[(df["model_short"] == model) & (df["condition"] == condition)]

            abstention_rate = subset["is_abstention"].mean()
            # Accuracy on answered questions only
            answered = subset[~subset["is_abstention"]]
            if len(answered) > 0:
                answer_accuracy = answered["is_correct"].mean()
            else:
                answer_accuracy = 0

            ax.scatter(abstention_rate, answer_accuracy,
                      marker=markers.get(model, "o"),
                      c=colors.get(condition, "gray"),
                      s=150, edgecolors='black', linewidth=1)
            ax.annotate(f"{model[:6]}\n{condition[:8]}", (abstention_rate, answer_accuracy),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel("Abstention Rate", fontsize=12)
    ax.set_ylabel("Accuracy (on answered questions)", fontsize=12)
    ax.set_title("Accuracy vs Abstention Trade-off", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_f1_comparison(metrics: dict, save_path: Path):
    """Plot F1 scores for abstention classification by condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = []
    f1_scores = []
    models = []

    for model, cond_metrics in metrics["by_model"].items():
        for condition, m in cond_metrics.items():
            conditions.append(condition)
            f1_scores.append(m["f1_score"])
            models.append(model)

    df_f1 = pd.DataFrame({
        "condition": conditions,
        "f1_score": f1_scores,
        "model": models
    })

    condition_order = ["baseline", "explicit_abstention", "cot_uncertainty", "self_consistency"]
    df_f1["condition"] = pd.Categorical(df_f1["condition"], categories=condition_order, ordered=True)

    x = np.arange(len(condition_order))
    width = 0.35
    unique_models = df_f1["model"].unique()

    for i, model in enumerate(unique_models):
        model_data = df_f1[df_f1["model"] == model].sort_values("condition")
        offset = width * (i - 0.5)
        ax.bar(x + offset, model_data["f1_score"], width, label=model)

    ax.set_xlabel("Prompting Condition", fontsize=12)
    ax.set_ylabel("F1 Score (Abstention Classification)", fontsize=12)
    ax.set_title("F1 Score for Appropriate Abstention by Condition", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Explicit\nAbstention", "CoT +\nUncertainty", "Self-\nConsistency"])
    ax.legend(title="Model")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_precision_recall(metrics: dict, save_path: Path):
    """Plot precision-recall for abstention classification."""
    fig, ax = plt.subplots(figsize=(10, 8))

    markers = {"gpt-4o-mini": "o", "claude-sonnet-4": "s"}
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    condition_labels = ["baseline", "explicit_abstention", "cot_uncertainty", "self_consistency"]

    for model, cond_metrics in metrics["by_model"].items():
        for i, condition in enumerate(condition_labels):
            if condition in cond_metrics:
                m = cond_metrics[condition]
                ax.scatter(m["recall"], m["precision"],
                          marker=markers.get(model, "o"),
                          c=colors[i], s=200, edgecolors='black', linewidth=1.5,
                          label=f"{model} - {condition}" if model == "gpt-4o-mini" else "")

    ax.set_xlabel("Recall (of unanswerable questions)", fontsize=12)
    ax.set_ylabel("Precision (of abstentions)", fontsize=12)
    ax.set_title("Precision vs Recall for Abstention Decisions", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Add diagonal for F1 reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='P=R line')

    # Legend for conditions
    condition_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                   markersize=10, label=condition_labels[i])
        for i in range(len(condition_labels))
    ]
    ax.legend(handles=condition_patches, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_source_comparison(df: pd.DataFrame, save_path: Path):
    """Compare abstention behavior on TruthfulQA vs SQuAD2."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, source in enumerate(["truthfulqa", "squad2"]):
        ax = axes[idx]
        source_df = df[df["source"] == source]

        summary = source_df.groupby(["model_short", "condition"]).agg({
            "is_abstention": "mean"
        }).reset_index()

        condition_order = ["baseline", "explicit_abstention", "cot_uncertainty", "self_consistency"]
        summary["condition"] = pd.Categorical(summary["condition"], categories=condition_order, ordered=True)

        x = np.arange(len(condition_order))
        width = 0.35

        for i, model in enumerate(summary["model_short"].unique()):
            model_data = summary[summary["model_short"] == model].sort_values("condition")
            offset = width * (i - 0.5)
            ax.bar(x + offset, model_data["is_abstention"], width, label=model)

        ax.set_xlabel("Condition")
        ax.set_ylabel("Abstention Rate")
        title = "TruthfulQA (all answerable, tests misconceptions)" if source == "truthfulqa" else "SQuAD 2.0 (50% unanswerable)"
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["Base", "Explicit", "CoT", "Self-Cons"])
        ax.legend()
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Run complete analysis."""
    print("Loading results...")
    raw_results, metrics = load_results()

    print("Creating DataFrame...")
    df = create_comparison_dataframe(raw_results)

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    # Statistical tests
    stat_results = statistical_tests(df)

    # Print key findings
    print("\n--- Abstention Rate Comparisons ---")
    for key, result in stat_results["comparisons"].items():
        if "_vs_baseline" in key:
            print(f"\n{key}:")
            print(f"  Baseline rate: {result['baseline_rate']:.1%}")
            print(f"  Intervention rate: {result['intervention_rate']:.1%}")
            print(f"  Difference: {result['difference']:+.1%}")
            print(f"  p-value: {result['p_value']:.4f}")
            print(f"  Effect size (Cohen's h): {result['cohens_h']:.2f} ({result['effect_size_interpretation']})")
            print(f"  Significant: {'YES' if result['significant'] else 'NO'}")

    # Bootstrap CIs
    print("\n--- Bootstrap Confidence Intervals (95%) ---")
    ci_results = bootstrap_confidence_intervals(df)
    for condition, ci in ci_results.items():
        print(f"  {condition}: {ci['mean']:.1%} [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    # Save statistical results
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(i) for i in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        return obj

    with open(RESULTS_DIR / "statistical_analysis.json", 'w') as f:
        json.dump(convert_to_json_serializable({
            "statistical_tests": stat_results,
            "bootstrap_ci": ci_results
        }), f, indent=2)
    print(f"\nStatistical analysis saved to: {RESULTS_DIR / 'statistical_analysis.json'}")

    # Generate plots
    print("\n--- Generating Visualizations ---")
    plot_abstention_rates(df, RESULTS_DIR / "plots" / "abstention_rates.png")
    plot_accuracy_abstention_tradeoff(df, RESULTS_DIR / "plots" / "accuracy_abstention_tradeoff.png")
    plot_f1_comparison(metrics, RESULTS_DIR / "plots" / "f1_comparison.png")
    plot_precision_recall(metrics, RESULTS_DIR / "plots" / "precision_recall.png")
    plot_source_comparison(df, RESULTS_DIR / "plots" / "source_comparison.png")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    # Summary statistics
    print("\n--- Summary ---")
    print(f"Total experiments: {len(df)}")
    print(f"Models tested: {df['model_short'].unique().tolist()}")
    print(f"Conditions tested: {df['condition'].unique().tolist()}")

    # Best performing condition
    by_condition = df.groupby("condition").agg({
        "is_abstention": "mean",
        "is_correct": "mean"
    }).reset_index()

    best_abstention = by_condition.loc[by_condition["is_abstention"].idxmax()]
    print(f"\nHighest abstention rate: {best_abstention['condition']} ({best_abstention['is_abstention']:.1%})")

    # F1 analysis
    print("\n--- F1 Scores (Abstention as Classification) ---")
    for model, conditions in metrics["by_model"].items():
        print(f"\n{model}:")
        for cond, m in conditions.items():
            print(f"  {cond}: F1={m['f1_score']:.3f}, P={m['precision']:.2f}, R={m['recall']:.2f}")


if __name__ == "__main__":
    main()
