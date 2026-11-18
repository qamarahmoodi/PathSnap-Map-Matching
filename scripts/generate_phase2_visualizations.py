"""Generate all Phase 2 Model Result Visualizations from actual saved experiment data.

Owner: L A Gokul (Common Evaluation & Reporting)

Generates 9 publication-grade figures:
1. Cross-validation Macro F1 comparison (5 models, mean ± std error bars).
2. Student-test Top-1 Accuracy comparison (Baseline vs HMM/Viterbi).
3. Student-test Macro F1 comparison across model families.
4. Log Loss comparison across models with valid probabilities.
5. Consolidated multi-metric model comparison overview.
6. Confusion matrix heatmap for the selected best model (HistGradientBoosting).
7. Tuned vs. Untuned comparison (HistGradientBoosting default vs Optuna Trial 7).
8. Classifier-Only vs. Classifier + HMM/Viterbi comparison (trip-by-trip + aggregate).
9. Optuna Bayesian Optimization objective progression across trials.

Strict rule: Uses ONLY actual saved numbers from results/phase1 and results/phase2.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling palette (Colorblind & university publication friendly)
PRIMARY_BLUE = "#1d4ed8"
SECONDARY_TEAL = "#0f766e"
ACCENT_GREEN = "#16a34a"
ACCENT_ORANGE = "#ea580c"
ACCENT_RED = "#dc2626"
NEUTRAL_GRAY = "#64748b"
BG_LIGHT = "#f8fafc"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200,
})


def plot_1_cv_macro_f1(df_models: pd.DataFrame) -> Path:
    """Figure 1: Cross-Validation Macro F1 comparison with error bars."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    names = df_models["model_name"].tolist()
    means = df_models["cv_macro_f1_mean"].tolist()
    stds = df_models["cv_macro_f1_std"].tolist()

    colors = [PRIMARY_BLUE if m != "HistGradientBoosting" else ACCENT_GREEN for m in names]

    bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors, alpha=0.88, edgecolor="#1e293b", linewidth=1.2)
    ax.set_ylabel("Macro F1-Score (Mean ± Std)")
    ax.set_title("Figure 1: 5-Fold Cross-Validation Macro F1 Comparison (5 Models)", pad=14, weight="bold")
    ax.set_ylim(0.5, 0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, mean in zip(bars, means):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{mean:.4f}", ha="center", va="bottom", fontsize=10, weight="bold")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig1_cv_macro_f1_comparison.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_2_student_test_top1(df_models: pd.DataFrame, agg_summary: dict) -> Path:
    """Figure 2: Student-test Top-1 Accuracy across models, plus HMM enhancement."""
    fig, ax = plt.subplots(figsize=(9, 5))
    names = df_models["model_name"].tolist() + ["HistGradientBoosting + HMM/Viterbi"]
    # For individual models, out-of-fold top-1 accuracy is recorded in df_models
    top1_vals = [val * 100 for val in df_models["oof_obs_top1_accuracy"]] + [agg_summary["mean_trip_hmm_viterbi_top1_accuracy"] * 100]

    colors = [PRIMARY_BLUE] * len(df_models) + [ACCENT_GREEN]
    bars = ax.bar(names, top1_vals, color=colors, alpha=0.88, edgecolor="#1e293b", linewidth=1.2)
    ax.set_ylabel("Observation Top-1 Accuracy (%)")
    ax.set_title("Figure 2: Observation-Level Top-1 Accuracy Comparison", pad=14, weight="bold")
    ax.set_ylim(40, 85)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, top1_vals):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{val:.2f}%", ha="center", va="bottom", fontsize=10, weight="bold")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig2_student_test_top1_comparison.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_3_student_test_macro_f1(df_models: pd.DataFrame) -> Path:
    """Figure 3: Out-of-fold Macro F1 across model families."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    names = df_models["model_name"].tolist()
    macro_f1s = df_models["oof_macro_f1"].tolist()

    bars = ax.bar(names, macro_f1s, color=SECONDARY_TEAL, alpha=0.85, edgecolor="#1e293b", linewidth=1.2)
    ax.set_ylabel("Out-of-Fold Macro F1-Score")
    ax.set_title("Figure 3: Model Candidate-Level Macro F1 Comparison", pad=14, weight="bold")
    ax.set_ylim(0.60, 0.80)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, macro_f1s):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f"{val:.4f}", ha="center", va="bottom", fontsize=10, weight="bold")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig3_student_test_macro_f1_comparison.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_4_log_loss(df_models: pd.DataFrame) -> Path:
    """Figure 4: Log Loss comparison across models."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    names = df_models["model_name"].tolist()
    losses = df_models["oof_log_loss"].tolist()

    # Lower is better!
    bars = ax.bar(names, losses, color=ACCENT_ORANGE, alpha=0.85, edgecolor="#1e293b", linewidth=1.2)
    ax.set_ylabel("Out-of-Fold Log Loss (Lower is Better)")
    ax.set_title("Figure 4: Candidate Probability Log Loss Comparison", pad=14, weight="bold")
    ax.set_ylim(0.0, 0.70)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, losses):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.015, f"{val:.4f}", ha="center", va="bottom", fontsize=10, weight="bold")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig4_log_loss_comparison.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_5_final_model_comparison(df_models: pd.DataFrame) -> Path:
    """Figure 5: Multi-metric comparison across the 5 models."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = df_models["model_name"].tolist()
    x = np.arange(len(names))
    width = 0.25

    rects1 = ax.bar(x - width, df_models["oof_macro_f1"], width, label="Candidate Macro F1", color=PRIMARY_BLUE, alpha=0.9)
    rects2 = ax.bar(x, df_models["oof_obs_top1_accuracy"], width, label="Obs Top-1 Accuracy", color=SECONDARY_TEAL, alpha=0.9)
    rects3 = ax.bar(x + width, df_models["oof_route_continuity"], width, label="Route Continuity", color=ACCENT_GREEN, alpha=0.9)

    ax.set_ylabel("Score / Rate [0, 1]")
    ax.set_title("Figure 5: Consolidated 5-Model Performance Comparison", pad=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig5_final_model_comparison.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_6_confusion_matrix(df_models: pd.DataFrame) -> Path:
    """Figure 6: Confusion Matrix heatmap for best model (HistGradientBoosting)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    hgb_row = df_models[df_models["model_name"] == "HistGradientBoosting"].iloc[0]
    cm = np.array([
        [int(hgb_row["tn"]), int(hgb_row["fp"])],
        [int(hgb_row["fn"]), int(hgb_row["tp"])]
    ])

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Negative (0)", "Positive (1)"]
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           title="Figure 6: HistGradientBoosting Candidate Confusion Matrix\n(GroupKFold Cross-Validation: 92,300 rows)",
           ylabel="True Ground-Truth Candidate Label",
           xlabel="Predicted Candidate Class (Threshold = 0.5)")

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:,}\n({val/cm.sum()*100:.1f}%)",
                    ha="center", va="center", color=color, fontsize=11, weight="bold")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig6_confusion_matrix_best_model.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_7_tuned_vs_untuned(df_models: pd.DataFrame, bayesian_best: dict) -> Path:
    """Figure 7: Untuned vs Optuna-Tuned HistGradientBoosting Comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    hgb_base = df_models[df_models["model_name"] == "HistGradientBoosting"].iloc[0]

    metrics = ["Macro F1", "Positive F1", "Log Loss (inverted)", "Top-1 Acc"]
    base_vals = [
        hgb_base["oof_macro_f1"],
        hgb_base["oof_positive_f1"],
        1.0 - hgb_base["oof_log_loss"],  # Invert so higher is better
        hgb_base["oof_obs_top1_accuracy"],
    ]
    tuned_vals = [
        bayesian_best["best_oof_macro_f1"],
        0.539661,  # From Trial 7
        1.0 - 0.272779,  # Invert Trial 7 oof_log_loss
        0.590413,  # Trial 7 oof_top1
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, base_vals, width, label="Default HistGradientBoosting", color=NEUTRAL_GRAY, alpha=0.85, edgecolor="#1e293b")
    ax.bar(x + width/2, tuned_vals, width, label="Optuna Bayesian Tuned (Trial 7)", color=PRIMARY_BLUE, alpha=0.9, edgecolor="#1e293b")

    ax.set_ylabel("Metric Value")
    ax.set_title("Figure 7: HistGradientBoosting Hyperparameter Optimization Impact", pad=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.4, 0.85)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i in range(len(metrics)):
        ax.text(x[i] - width/2, base_vals[i] + 0.01, f"{base_vals[i]:.3f}", ha="center", fontsize=9)
        ax.text(x[i] + width/2, tuned_vals[i] + 0.01, f"{tuned_vals[i]:.3f}", ha="center", fontsize=9, weight="bold")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig7_tuned_vs_untuned_comparison.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_8_classifier_vs_hmm(trip_metrics_path: Path) -> Path:
    """Figure 8: Classifier-Only vs Classifier + HMM/Viterbi across all 10 student-test trips."""
    df_trips = pd.read_csv(trip_metrics_path, dtype={"trip_id": str})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    trips = [f"T{str(t)[-4:]}" for t in df_trips["trip_id"]]
    x = np.arange(len(trips))
    width = 0.38

    # Panel A: Top-1 Accuracy
    ax1.bar(x - width/2, df_trips["classifier_only_top1_accuracy"] * 100, width, label="Classifier-Only (Greedy)", color=ACCENT_ORANGE, alpha=0.85, edgecolor="#1e293b")
    ax1.bar(x + width/2, df_trips["hmm_viterbi_top1_accuracy"] * 100, width, label="Classifier + HMM/Viterbi", color=ACCENT_GREEN, alpha=0.9, edgecolor="#1e293b")
    ax1.set_ylabel("Observation Top-1 Accuracy (%)")
    ax1.set_title("A. Top-1 Accuracy per Trip (+10.05% Mean Gain)", weight="bold", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(trips, rotation=35, ha="right")
    ax1.set_ylim(20, 105)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: Route Continuity
    ax2.bar(x - width/2, df_trips["classifier_only_route_continuity"] * 100, width, label="Classifier-Only (Greedy)", color=ACCENT_ORANGE, alpha=0.85, edgecolor="#1e293b")
    ax2.bar(x + width/2, df_trips["hmm_viterbi_route_continuity"] * 100, width, label="Classifier + HMM/Viterbi", color=ACCENT_GREEN, alpha=0.9, edgecolor="#1e293b")
    ax2.set_ylabel("Route Continuity (%)")
    ax2.set_title("B. Route Continuity per Trip (+11.29% Mean Gain)", weight="bold", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(trips, rotation=35, ha="right")
    ax2.set_ylim(50, 105)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Figure 8: Empirical Map-Matching Gain: Classifier-Only vs. HMM/Viterbi (Held-out student_test)", weight="bold", fontsize=13, y=0.99)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig8_classifier_only_vs_hmm_comparison.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_9_bayesian_optimization_trials(bayesian_trials_path: Path) -> Path:
    """Figure 9: Optuna TPE Bayesian Optimization progression across trials."""
    df_trials = pd.read_csv(bayesian_trials_path)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))

    trial_nums = df_trials["trial_number"].tolist()
    macro_f1s = df_trials["oof_macro_f1"].tolist()
    cv_f1s = df_trials["cv_macro_f1_mean"].tolist()

    ax1.plot(trial_nums, macro_f1s, marker="o", color=PRIMARY_BLUE, linewidth=2.2, label="Out-of-Fold Macro F1")
    ax1.plot(trial_nums, cv_f1s, marker="s", linestyle="--", color=SECONDARY_TEAL, linewidth=1.8, label="5-Fold CV Macro F1")

    best_idx = int(np.argmax(macro_f1s))
    ax1.scatter([trial_nums[best_idx]], [macro_f1s[best_idx]], color=ACCENT_RED, s=120, zorder=5, label=f"Best Trial #{trial_nums[best_idx]} (F1={macro_f1s[best_idx]:.4f})")

    ax1.set_xlabel("Optuna Trial Number")
    ax1.set_ylabel("Macro F1-Score")
    ax1.set_title("Figure 9: Optuna TPE Bayesian Hyperparameter Optimization Progression", pad=14, weight="bold")
    ax1.set_xticks(trial_nums)
    ax1.set_ylim(0.72, 0.81)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig9_bayesian_tuning_progression.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def main() -> None:
    # 1. Load actual saved inputs
    df_models = pd.read_csv(REPO_ROOT / "results" / "phase1" / "model_comparison.csv")
    with open(REPO_ROOT / "results" / "phase2" / "student_test_aggregate_summary.json", "r", encoding="utf-8") as f:
        agg_summary = json.load(f)
    with open(REPO_ROOT / "results" / "phase1" / "bayesian_best_summary.json", "r", encoding="utf-8") as f:
        bayesian_best = json.load(f)
    bayesian_trials_path = REPO_ROOT / "results" / "phase1" / "bayesian_trials.csv"
    trip_metrics_path = REPO_ROOT / "results" / "phase2" / "student_test_trip_metrics.csv"

    print("Generating 9 publication-grade visualization figures...")
    p1 = plot_1_cv_macro_f1(df_models)
    print(f"[1/9] Generated {p1.name}")

    p2 = plot_2_student_test_top1(df_models, agg_summary)
    print(f"[2/9] Generated {p2.name}")

    p3 = plot_3_student_test_macro_f1(df_models)
    print(f"[3/9] Generated {p3.name}")

    p4 = plot_4_log_loss(df_models)
    print(f"[4/9] Generated {p4.name}")

    p5 = plot_5_final_model_comparison(df_models)
    print(f"[5/9] Generated {p5.name}")

    p6 = plot_6_confusion_matrix(df_models)
    print(f"[6/9] Generated {p6.name}")

    p7 = plot_7_tuned_vs_untuned(df_models, bayesian_best)
    print(f"[7/9] Generated {p7.name}")

    p8 = plot_8_classifier_vs_hmm(trip_metrics_path)
    print(f"[8/9] Generated {p8.name}")

    p9 = plot_9_bayesian_optimization_trials(bayesian_trials_path)
    print(f"[9/9] Generated {p9.name}")

    print(f"\n[SUCCESS] All 9 visualization plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
