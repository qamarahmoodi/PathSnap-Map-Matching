"""Execute full Phase 2 ML Experimentation (80 train trips CV + 10 student_test trips), Optuna TPE Bayesian Tuning, HMM/Viterbi Decoding, and Artifact Export."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import folium
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import optuna
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.model_pipeline import (
    RANDOM_SEED,
    compute_observation_and_route_metrics,
    evaluate_five_models_group_cv,
    get_five_models,
    verify_candidate_contract,
)
from models.scorer import score_candidates
from optimization.bayesian_tuning import run_bayesian_tuning_hgb
from pathsnap.data.loader import list_trips, load_trip, resolve_data_root
from pathsnap.graph.road_graph import TripRoadGraph
from pathsnap.schemas.candidate_schema import (
    CANDIDATE_FEATURE_COLUMNS,
    CANDIDATE_TARGET_COLUMN,
)
from pathsnap.spatial.road_network import build_trip_road_network
from sequence.viterbi_decoder import ViterbiConfig, decode_trip_viterbi


def build_graphs_for_partition(
    partition: str,
    trip_ids: List[str],
    data_root: Path,
) -> Dict[str, TripRoadGraph]:
    """Build TripRoadGraph dictionary for a list of trip_ids in a partition."""
    graphs: Dict[str, TripRoadGraph] = {}
    for tid in trip_ids:
        raw = load_trip(str(tid), partition=partition, data_root=data_root, include_route=False)
        net = build_trip_road_network(raw.trip_id, raw.nodes, raw.arcs)
        graphs[str(tid)] = net.graph
    return graphs


def evaluate_model_on_partition(
    estimator: Any,
    eval_cands: pd.DataFrame,
    eval_obs: pd.DataFrame,
    graphs_by_trip: Dict[str, TripRoadGraph],
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluate a fitted estimator on a target partition (`train` or `student_test`)."""
    X_all = eval_cands[CANDIDATE_FEATURE_COLUMNS].astype("float32")
    t0 = time.perf_counter()
    probs_all = estimator.predict_proba(X_all)[:, 1]
    inf_time = time.perf_counter() - t0

    lab_mask = eval_cands[CANDIDATE_TARGET_COLUMN].notna().to_numpy()
    y_lab = eval_cands.loc[lab_mask, CANDIDATE_TARGET_COLUMN].astype("int8").to_numpy()
    probs_lab = np.clip(probs_all[lab_mask], 1e-6, 1.0 - 1e-6)
    preds_lab = (probs_lab >= 0.5).astype(np.int8)

    acc = float(accuracy_score(y_lab, preds_lab))
    prec = float(precision_score(y_lab, preds_lab, zero_division=0))
    rec = float(recall_score(y_lab, preds_lab, zero_division=0))
    f1_pos = float(f1_score(y_lab, preds_lab, pos_label=1, zero_division=0))
    f1_mac = float(f1_score(y_lab, preds_lab, average="macro", zero_division=0))
    ll = float(log_loss(y_lab, probs_lab, labels=[0, 1]))
    tn, fp, fn, tp = confusion_matrix(y_lab, preds_lab, labels=[0, 1]).ravel()

    obs_metrics = compute_observation_and_route_metrics(
        eval_cands, eval_obs, probs_all, graphs_by_trip
    )

    metrics = {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "positive_f1": round(f1_pos, 6),
        "macro_f1": round(f1_mac, 6),
        "log_loss": round(ll, 6),
        "obs_top1_accuracy": round(obs_metrics["obs_top1_accuracy_aligned"], 6),
        "obs_top1_eligible": round(obs_metrics["obs_top1_accuracy_eligible"], 6),
        "route_continuity": round(obs_metrics["route_continuity"], 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "inference_time_s": round(inf_time, 4),
    }
    return metrics, probs_all


def generate_phase2_plots(
    consolidated_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    tuned_comp_df: pd.DataFrame,
    hmm_comp_df: pd.DataFrame,
    hgb_staged_history: Dict[str, List[float]],
    out_dir: Path,
) -> None:
    """Generate mandatory comparison plots, confusion matrices, and iterative loss curves from real results."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 5-Model CV vs Student-Test Macro F1 and Top-1 Accuracy Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    names = consolidated_df["model_name"].tolist()
    y = np.arange(len(names))
    h = 0.35

    axes[0].barh(y - h / 2, consolidated_df["cv_macro_f1_mean"], height=h, color="#2563eb", label="80-Trip CV Macro F1")
    axes[0].barh(y + h / 2, consolidated_df["student_test_macro_f1"], height=h, color="#16a34a", label="10-Trip Student-Test Macro F1")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].set_xlim(0.5, 0.95)
    axes[0].set_xlabel("Macro F1-Score")
    axes[0].set_title("Macro F1: 80-Trip CV vs 10-Trip Student-Test")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    axes[1].barh(y - h / 2, consolidated_df["cv_obs_top1_accuracy_mean"], height=h, color="#3b82f6", label="80-Trip CV Top-1 Acc")
    axes[1].barh(y + h / 2, consolidated_df["student_test_top1_accuracy"], height=h, color="#10b981", label="10-Trip Student-Test Top-1 Acc")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names)
    axes[1].set_xlim(0.5, 0.95)
    axes[1].set_xlabel("Observation-Level Top-1 Accuracy")
    axes[1].set_title("Top-1 Accuracy: 80-Trip CV vs Student-Test")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    axes[2].barh(y - h / 2, consolidated_df["cv_log_loss_mean"], height=h, color="#f59e0b", label="80-Trip CV Log Loss")
    axes[2].barh(y + h / 2, consolidated_df["student_test_log_loss"], height=h, color="#dc2626", label="10-Trip Student-Test Log Loss")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(names)
    axes[2].set_xlabel("Binary Log Loss (Lower is Better)")
    axes[2].set_title("Probability Calibration: Log Loss")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_dir / "five_model_comparison_full.png", dpi=160)
    plt.close()

    # 2. Confusion Matrices for all 5 Models on Student-Test (1x5 grid)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for idx, row in consolidated_df.iterrows():
        cm = np.array(
            [[int(row["student_test_tn"]), int(row["student_test_fp"])],
             [int(row["student_test_fn"]), int(row["student_test_tp"])]]
        )
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{row['model_name']}\n(Test F1={row['student_test_macro_f1']:.3f})", fontsize=9.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=8)
        for r_i in range(2):
            for c_i in range(2):
                ax.text(c_i, r_i, f"{cm[r_i, c_i]:,}", ha="center", va="center", color="black", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "student_test_confusion_matrices.png", dpi=160)
    plt.close()

    # 3. Iterative Staged Log-Loss Curve ONLY for HistGradientBoosting (real staged_predict_proba history)
    if hgb_staged_history["iters"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(hgb_staged_history["iters"], hgb_staged_history["train_log_loss"], label="80-Trip Training Log Loss", color="#2563eb", linewidth=2)
        ax.plot(hgb_staged_history["iters"], hgb_staged_history["test_log_loss"], label="10-Trip Student-Test Log Loss", color="#dc2626", linewidth=2, linestyle="--")
        ax.set_xlabel("Boosting Iteration (Trees)")
        ax.set_ylabel("Binary Log Loss")
        ax.set_title("HistGradientBoosting Iteration-Wise Loss Curve (Tuned Final Model)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(out_dir / "hgb_iterative_loss_curve.png", dpi=160)
        plt.close()

    # 4. Bayesian Optimisation Progression Plot (80-Trip CV)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    t_nums = trials_df["trial_number"].to_numpy()
    f1_vals = trials_df["oof_macro_f1"].to_numpy()
    best_vals = np.maximum.accumulate(f1_vals)
    ax.plot(t_nums, f1_vals, "o--", color="#6b7280", label="Trial 80-Trip OOF Macro F1")
    ax.plot(t_nums, best_vals, "s-", color="#2563eb", linewidth=2, label="Best-So-Far OOF Macro F1")
    ax.set_xlabel("Optuna TPE Trial Number")
    ax.set_ylabel("80-Trip Out-of-Fold Macro F1")
    ax.set_title("Optuna TPE Bayesian Hyperparameter Optimisation (80-Trip GroupKFold)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "bayesian_tuning_80trip_progression.png", dpi=160)
    plt.close()

    # 5. Classifier-Only (Greedy) vs Classifier + HMM/Viterbi on 10 Student-Test Trips
    sub_hmm = hmm_comp_df[hmm_comp_df["test_type"] == "student_test_trip"].copy()
    if not sub_hmm.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        tids = sub_hmm["trip_or_case_id"].tolist()
        x = np.arange(len(tids))
        w = 0.36

        axes[0].bar(x - w / 2, sub_hmm["greedy_top1_accuracy"], width=w, color="#f59e0b", label="Classifier-Only (Greedy)")
        axes[0].bar(x + w / 2, sub_hmm["viterbi_top1_accuracy"], width=w, color="#16a34a", label="Classifier + HMM/Viterbi")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(tids, rotation=45, fontsize=8.5)
        axes[0].set_ylim(0.4, 1.05)
        axes[0].set_ylabel("Observation Top-1 Accuracy")
        axes[0].set_title("Student-Test Observation Top-1 Accuracy: Greedy vs HMM/Viterbi")
        axes[0].legend(fontsize=8.5)
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)

        axes[1].bar(x - w / 2, sub_hmm["greedy_route_continuity"], width=w, color="#f59e0b", label="Classifier-Only (Greedy)")
        axes[1].bar(x + w / 2, sub_hmm["viterbi_route_continuity"], width=w, color="#2563eb", label="Classifier + HMM/Viterbi")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(tids, rotation=45, fontsize=8.5)
        axes[1].set_ylim(0.5, 1.05)
        axes[1].set_ylabel("Route Continuity")
        axes[1].set_title("Student-Test Route Continuity: Greedy vs HMM/Viterbi")
        axes[1].legend(fontsize=8.5)
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(out_dir / "student_test_greedy_vs_viterbi.png", dpi=160)
        plt.close()


def main() -> None:
    t0_all = time.perf_counter()
    data_root = resolve_data_root()
    full_dir = REPO_ROOT / "outputs" / "full_candidates"
    phase2_dir = REPO_ROOT / "results" / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = phase2_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_cands = pd.read_parquet(full_dir / "train_candidates.parquet")
    train_obs = pd.read_parquet(full_dir / "train_observations.parquet")
    test_cands = pd.read_parquet(full_dir / "student_test_candidates.parquet")
    test_obs = pd.read_parquet(full_dir / "student_test_observations.parquet")

    train_contract = verify_candidate_contract(train_cands, train_obs)
    test_contract = verify_candidate_contract(test_cands, test_obs)
    print("80-Trip Train Contract Verified:", json.dumps(train_contract, indent=2))
    print("10-Trip Student-Test Contract Verified:", json.dumps(test_contract, indent=2))

    # Build road graphs for all 80 train and 10 student_test trips
    t_g = time.perf_counter()
    train_trip_ids = sorted(train_cands["trip_id"].unique())
    test_trip_ids = sorted(test_cands["trip_id"].unique())
    train_graphs = build_graphs_for_partition("train", train_trip_ids, data_root)
    test_graphs = build_graphs_for_partition("student_test", test_trip_ids, data_root)
    print(f"Built 90 TripRoadGraphs in {time.perf_counter() - t_g:.2f}s")

    # 1. Run 5-Model 5-Fold GroupKFold Cross-Validation on all 80 Training Trips
    print("Step 1: Running 5-Model 5-Fold GroupKFold CV across all 80 training trips...")
    cv_summary_df, cv_folds_df, oof_probs_by_model = evaluate_five_models_group_cv(
        train_cands, train_obs, train_graphs, n_splits=5, random_state=RANDOM_SEED
    )
    cv_folds_df.to_csv(phase2_dir / "cv_80trip_fold_results.csv", index=False)

    # Save Out-of-Fold (OOF) predictions for all 5 models in the exact agreed schema for Gokul
    base_id_cols = ["trip_id", "partition", "observation_id", "candidate_arc_id", "candidate_rank", "label"]
    for m_key, oof_p in oof_probs_by_model.items():
        if m_key == "nearest_segment_baseline":
            continue
        oof_df = train_cands[base_id_cols].copy()
        oof_df.insert(0, "model_key", m_key)
        oof_df["predicted_probability"] = oof_p.astype(np.float32)
        oof_df.to_parquet(preds_dir / f"train_oof_predictions_{m_key}.parquet", index=False)

    # 2. Fit each of the 5 models on the full 80-trip training set and evaluate on both Train (for overfitting check) and 10 Student-Test trips
    print("Step 2: Fitting all 5 models on full 80 training trips and evaluating on 10 student_test trips...")
    models_dict = get_five_models(random_state=RANDOM_SEED)
    train_lab_mask = train_cands[CANDIDATE_TARGET_COLUMN].notna()
    X_train_full = train_cands.loc[train_lab_mask, CANDIDATE_FEATURE_COLUMNS].astype("float32")
    y_train_full = train_cands.loc[train_lab_mask, CANDIDATE_TARGET_COLUMN].astype("int8").to_numpy()

    consolidated_rows: List[Dict[str, Any]] = []
    fitted_untuned_models: Dict[str, Any] = {}
    test_probs_by_model: Dict[str, np.ndarray] = {}

    # Key hyperparameter descriptions for the report table
    key_configs = {
        "logistic_regression": "L2 penalty, C=1.0, lbfgs, StandardScaler + SimpleImputer(median, add_indicator=True)",
        "distance_knn": "n_neighbors=25, weights='distance', metric='manhattan', StandardScaler + SimpleImputer",
        "decision_tree": "criterion='log_loss', max_depth=8, min_samples_leaf=30, SimpleImputer",
        "random_forest": "n_estimators=120, max_depth=12, min_samples_leaf=15, max_features='sqrt', SimpleImputer",
        "hist_gradient_boosting": "lr=0.08, max_iter=150, max_leaf_nodes=31, max_depth=8, min_samples_leaf=25, l2=1.0, native NaN routing",
    }

    for model_key, (model_name, family, base_est) in models_dict.items():
        est = clone(base_est)
        t_fit0 = time.perf_counter()
        est.fit(X_train_full, y_train_full)
        full_fit_time = time.perf_counter() - t_fit0
        fitted_untuned_models[model_key] = est

        # Evaluate on full 80-trip train set (to quantify train vs CV/test generalisation gap)
        tr_metrics, _ = evaluate_model_on_partition(est, train_cands, train_obs, train_graphs)
        # Evaluate on 10 held-out student_test trips
        te_metrics, te_probs = evaluate_model_on_partition(est, test_cands, test_obs, test_graphs)
        test_probs_by_model[model_key] = te_probs

        # Save reusable student_test scored predictions for Gokul & Qamar
        scored_te = score_candidates(test_cands, model_or_bundle=est, retain_all_columns=False)
        scored_te.insert(0, "model_key", model_key)
        scored_te.to_parquet(preds_dir / f"student_test_predictions_{model_key}.parquet", index=False)

        cv_row = cv_summary_df[cv_summary_df["model_key"] == model_key].iloc[0]
        consolidated_rows.append(
            {
                "model_key": model_key,
                "model_name": model_name,
                "algorithm_family": family,
                "key_configuration": key_configs[model_key],
                "random_seed": RANDOM_SEED,
                # Training partition (in-sample) metrics for overfitting diagnosis
                "train_macro_f1": tr_metrics["macro_f1"],
                "train_obs_top1_accuracy": tr_metrics["obs_top1_accuracy"],
                "train_log_loss": tr_metrics["log_loss"],
                # 80-Trip 5-Fold GroupKFold Cross-Validation metrics
                "cv_macro_f1_mean": cv_row["cv_macro_f1_mean"],
                "cv_macro_f1_std": cv_row["cv_macro_f1_std"],
                "oof_macro_f1": cv_row["oof_macro_f1"],
                "oof_positive_f1": cv_row["oof_positive_f1"],
                "cv_obs_top1_accuracy_mean": cv_row["cv_obs_top1_accuracy_mean"],
                "oof_obs_top1_accuracy": cv_row["oof_obs_top1_accuracy"],
                "oof_route_continuity": cv_row["oof_route_continuity"],
                "cv_log_loss_mean": cv_row["cv_log_loss_mean"],
                "oof_log_loss": cv_row["oof_log_loss"],
                # 10-Trip Held-Out Student-Testing metrics (Deliverables §3.1 columns)
                "student_test_macro_f1": te_metrics["macro_f1"],
                "student_test_positive_f1": te_metrics["positive_f1"],
                "student_test_accuracy": te_metrics["accuracy"],
                "student_test_precision": te_metrics["precision"],
                "student_test_recall": te_metrics["recall"],
                "student_test_top1_accuracy": te_metrics["obs_top1_accuracy"],
                "student_test_top1_eligible": te_metrics["obs_top1_eligible"],
                "student_test_route_continuity": te_metrics["route_continuity"],
                "student_test_log_loss": te_metrics["log_loss"],
                "student_test_tn": te_metrics["tn"],
                "student_test_fp": te_metrics["fp"],
                "student_test_fn": te_metrics["fn"],
                "student_test_tp": te_metrics["tp"],
                # Timings
                "full_80trip_train_time_s": round(full_fit_time, 4),
                "total_cv_train_time_s": cv_row["total_cv_train_time_s"],
                "student_test_inference_time_s": te_metrics["inference_time_s"],
            }
        )
        print(
            f"  [{model_name}] CV Macro F1={cv_row['cv_macro_f1_mean']:.4f} | "
            f"Student-Test Macro F1={te_metrics['macro_f1']:.4f} | "
            f"Student-Test Top-1={te_metrics['obs_top1_accuracy']:.4f} | "
            f"Student-Test LogLoss={te_metrics['log_loss']:.4f}"
        )

    consolidated_df = pd.DataFrame(consolidated_rows)
    consolidated_df.to_csv(phase2_dir / "model_comparison_80train_10test.csv", index=False)

    # 3. Bayesian Hyperparameter Optimisation (Optuna TPE) on the 80 Training Trips (25 Trials)
    print("Step 3: Running Optuna TPE Bayesian Hyperparameter Optimisation (25 trials, 80-trip GroupKFold)...")
    trials_df, best_bayes_summary, tuned_hgb_model, best_oof_probs = run_bayesian_tuning_hgb(
        train_cands, train_obs, train_graphs, n_trials=25, n_splits=5, random_state=RANDOM_SEED
    )
    trials_df.to_csv(phase2_dir / "bayesian_trials_80trip.csv", index=False)

    # Save OOF predictions for Tuned HGB
    oof_tuned_df = train_cands[base_id_cols].copy()
    oof_tuned_df.insert(0, "model_key", "hist_gradient_boosting_tuned")
    oof_tuned_df["predicted_probability"] = best_oof_probs.astype(np.float32)
    oof_tuned_df.to_parquet(preds_dir / "train_oof_predictions_tuned_hgb.parquet", index=False)

    # Evaluate Untuned HGB vs Bayesian-Tuned HGB on both 80-Trip CV and 10-Trip Student-Test
    untuned_hgb = fitted_untuned_models["hist_gradient_boosting"]
    untuned_te_metrics, _ = evaluate_model_on_partition(untuned_hgb, test_cands, test_obs, test_graphs)
    tuned_tr_metrics, _ = evaluate_model_on_partition(tuned_hgb_model, train_cands, train_obs, train_graphs)
    tuned_te_metrics, tuned_te_probs = evaluate_model_on_partition(tuned_hgb_model, test_cands, test_obs, test_graphs)

    # Save Tuned HGB student_test predictions for Gokul & Qamar
    scored_tuned_te = score_candidates(test_cands, model_or_bundle=tuned_hgb_model, retain_all_columns=False)
    scored_tuned_te.insert(0, "model_key", "hist_gradient_boosting_tuned")
    scored_tuned_te.to_parquet(preds_dir / "student_test_predictions_tuned_hgb.parquet", index=False)
    # Save a small human-readable CSV sample of scored predictions
    scored_tuned_te.head(300).to_csv(phase2_dir / "scored_candidates_sample_300.csv", index=False)

    untuned_cv_row = consolidated_df[consolidated_df["model_key"] == "hist_gradient_boosting"].iloc[0]
    best_trial_row = trials_df.loc[trials_df["oof_macro_f1"].idxmax()]

    tuned_vs_untuned_df = pd.DataFrame(
        [
            {
                "version": "Untuned HistGradientBoosting (Baseline)",
                "hyperparameters": key_configs["hist_gradient_boosting"],
                "cv_macro_f1_mean": untuned_cv_row["cv_macro_f1_mean"],
                "oof_macro_f1": untuned_cv_row["oof_macro_f1"],
                "oof_log_loss": untuned_cv_row["oof_log_loss"],
                "oof_obs_top1_accuracy": untuned_cv_row["oof_obs_top1_accuracy"],
                "student_test_macro_f1": untuned_te_metrics["macro_f1"],
                "student_test_positive_f1": untuned_te_metrics["positive_f1"],
                "student_test_top1_accuracy": untuned_te_metrics["obs_top1_accuracy"],
                "student_test_top1_eligible": untuned_te_metrics["obs_top1_eligible"],
                "student_test_route_continuity": untuned_te_metrics["route_continuity"],
                "student_test_log_loss": untuned_te_metrics["log_loss"],
                "student_test_inference_time_s": untuned_te_metrics["inference_time_s"],
            },
            {
                "version": f"Bayesian-Tuned HistGradientBoosting (Optuna TPE Trial #{int(best_trial_row['trial_number'])})",
                "hyperparameters": json.dumps(best_bayes_summary["best_params"]),
                "cv_macro_f1_mean": best_trial_row["cv_macro_f1_mean"],
                "oof_macro_f1": best_trial_row["oof_macro_f1"],
                "oof_log_loss": best_trial_row["oof_log_loss"],
                "oof_obs_top1_accuracy": best_trial_row["oof_obs_top1_accuracy"],
                "student_test_macro_f1": tuned_te_metrics["macro_f1"],
                "student_test_positive_f1": tuned_te_metrics["positive_f1"],
                "student_test_top1_accuracy": tuned_te_metrics["obs_top1_accuracy"],
                "student_test_top1_eligible": tuned_te_metrics["obs_top1_eligible"],
                "student_test_route_continuity": tuned_te_metrics["route_continuity"],
                "student_test_log_loss": tuned_te_metrics["log_loss"],
                "student_test_inference_time_s": tuned_te_metrics["inference_time_s"],
            },
        ]
    )
    tuned_vs_untuned_df.to_csv(phase2_dir / "tuned_vs_untuned_comparison.csv", index=False)

    best_bayes_summary["untuned_student_test"] = untuned_te_metrics
    best_bayes_summary["tuned_student_test"] = tuned_te_metrics
    with open(phase2_dir / "bayesian_best_summary_80trip.json", "w", encoding="utf-8") as f:
        json.dump(best_bayes_summary, f, indent=2)

    # Compute real iteration-wise loss history for Tuned HGB using staged_predict_proba
    iters_hist: List[int] = []
    tr_ll_hist: List[float] = []
    te_ll_hist: List[float] = []
    X_te_lab = test_cands.loc[test_cands[CANDIDATE_TARGET_COLUMN].notna(), CANDIDATE_FEATURE_COLUMNS].astype("float32")
    y_te_lab = test_cands.loc[test_cands[CANDIDATE_TARGET_COLUMN].notna(), CANDIDATE_TARGET_COLUMN].astype("int8").to_numpy()
    # Subsample 100k train rows for fast staged loss calculation
    rng = np.random.default_rng(RANDOM_SEED)
    sub_idx = rng.choice(len(X_train_full), size=min(100000, len(X_train_full)), replace=False)
    X_tr_sub = X_train_full.iloc[sub_idx]
    y_tr_sub = y_train_full[sub_idx]

    for step_idx, (p_tr_stage, p_te_stage) in enumerate(
        zip(tuned_hgb_model.staged_predict_proba(X_tr_sub), tuned_hgb_model.staged_predict_proba(X_te_lab)),
        start=1,
    ):
        if step_idx == 1 or step_idx % 5 == 0 or step_idx == tuned_hgb_model.n_iter_:
            iters_hist.append(step_idx)
            tr_ll_hist.append(float(log_loss(y_tr_sub, np.clip(p_tr_stage[:, 1], 1e-6, 1 - 1e-6), labels=[0, 1])))
            te_ll_hist.append(float(log_loss(y_te_lab, np.clip(p_te_stage[:, 1], 1e-6, 1 - 1e-6), labels=[0, 1])))

    hgb_staged_history = {"iters": iters_hist, "train_log_loss": tr_ll_hist, "test_log_loss": te_ll_hist}
    pd.DataFrame(hgb_staged_history).to_csv(phase2_dir / "hgb_iteration_loss_history.csv", index=False)

    # 4. Export Final Tuned Model Artifact & Metadata JSON for Qamar's Backend
    final_metadata = {
        "model_name": "Bayesian-Tuned HistGradientBoostingClassifier",
        "model_key": "hist_gradient_boosting_tuned",
        "algorithm_family": "Gradient Boosting",
        "feature_columns": list(CANDIDATE_FEATURE_COLUMNS),
        "id_columns_retained": ["trip_id", "observation_id", "candidate_arc_id", "arc_id", "candidate_rank"],
        "classes": [0, 1],
        "positive_class": 1,
        "positive_class_meaning": "Candidate road segment matches the vehicle's traversed reference route segment",
        "random_seed": RANDOM_SEED,
        "training_trips_count": 80,
        "training_labeled_rows": int(len(X_train_full)),
        "best_hyperparameters": best_bayes_summary["best_params"],
        "cv_macro_f1_mean": float(best_trial_row["cv_macro_f1_mean"]),
        "oof_macro_f1": float(best_trial_row["oof_macro_f1"]),
        "student_test_metrics": tuned_te_metrics,
        "dependencies": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "optuna": optuna.__version__,
            "joblib": joblib.__version__,
        },
    }
    joblib.dump(
        {"estimator": tuned_hgb_model, "metadata": final_metadata},
        artifacts_dir / "final_model.joblib",
    )
    joblib.dump(
        {"estimator": tuned_hgb_model, "metadata": final_metadata},
        artifacts_dir / "final_tuned_model.joblib",
    )
    from models.inference import compute_file_sha256
    final_metadata["artifact_sha256"] = compute_file_sha256(artifacts_dir / "final_model.joblib")
    with open(artifacts_dir / "final_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=2)

    # 5. Run HMM/Viterbi Route Reconstruction across all 10 Student-Test Trips
    print("Step 4: Running HMM/Viterbi route reconstruction across all 10 student_test trips...")
    test_cands_scored = test_cands.copy()
    test_cands_scored["pred_prob"] = tuned_te_probs

    hmm_rows: List[Dict[str, Any]] = []
    viterbi_cfg = ViterbiConfig()
    tot_obs = 0
    tot_aligned = 0
    g_corr = 0
    v_corr = 0
    tot_steps = 0
    g_conn = 0
    v_conn = 0
    tot_changed = 0

    for tid in test_trip_ids:
        sub_c = test_cands_scored[test_cands_scored["trip_id"] == tid]
        sub_o = test_obs[test_obs["trip_id"] == tid]
        res_v = decode_trip_viterbi(sub_c, sub_o, test_graphs[tid], prob_col="pred_prob", config=viterbi_cfg)
        diag = res_v.diagnostics
        al_cnt = int((sub_o["alignment_status"] == "aligned").sum())

        hmm_rows.append(
            {
                "test_type": "student_test_trip",
                "trip_or_case_id": str(tid),
                "model_source": "Bayesian-Tuned HistGradientBoosting",
                "num_observations": int(len(sub_o)),
                "aligned_observations": al_cnt,
                "greedy_top1_accuracy": diag.get("greedy_top1_accuracy", float("nan")),
                "viterbi_top1_accuracy": diag.get("viterbi_top1_accuracy", float("nan")),
                "top1_accuracy_delta": round(
                    diag.get("viterbi_top1_accuracy", 0.0) - diag.get("greedy_top1_accuracy", 0.0), 6
                ),
                "greedy_route_continuity": diag["greedy_route_continuity"],
                "viterbi_route_continuity": diag["viterbi_route_continuity"],
                "continuity_delta": round(
                    diag["viterbi_route_continuity"] - diag["greedy_route_continuity"], 6
                ),
                "changed_observations_count": int(diag["changed_observations_count"]),
            }
        )
        tot_obs += len(sub_o)
        tot_aligned += al_cnt
        g_corr += int(round(diag.get("greedy_top1_accuracy", 0.0) * al_cnt))
        v_corr += int(round(diag.get("viterbi_top1_accuracy", 0.0) * al_cnt))
        st_cnt = max(len(sub_o) - 1, 0)
        tot_steps += st_cnt
        g_conn += int(round(diag["greedy_route_continuity"] * st_cnt))
        v_conn += int(round(diag["viterbi_route_continuity"] * st_cnt))
        tot_changed += int(diag["changed_observations_count"])

    g_acc_all = g_corr / max(tot_aligned, 1)
    v_acc_all = v_corr / max(tot_aligned, 1)
    g_con_all = g_conn / max(tot_steps, 1)
    v_con_all = v_conn / max(tot_steps, 1)
    hmm_rows.append(
        {
            "test_type": "student_test_aggregate",
            "trip_or_case_id": "ALL_10_STUDENT_TEST_TRIPS",
            "model_source": "Bayesian-Tuned HistGradientBoosting",
            "num_observations": tot_obs,
            "aligned_observations": tot_aligned,
            "greedy_top1_accuracy": round(g_acc_all, 6),
            "viterbi_top1_accuracy": round(v_acc_all, 6),
            "top1_accuracy_delta": round(v_acc_all - g_acc_all, 6),
            "greedy_route_continuity": round(g_con_all, 6),
            "viterbi_route_continuity": round(v_con_all, 6),
            "continuity_delta": round(v_con_all - g_con_all, 6),
            "changed_observations_count": tot_changed,
        }
    )
    hmm_comp_df = pd.DataFrame(hmm_rows)
    hmm_comp_df.to_csv(phase2_dir / "student_test_hmm_viterbi_comparison.csv", index=False)

    # 6. Generate all Phase 2 diagnostic charts
    generate_phase2_plots(
        consolidated_df, trials_df, tuned_vs_untuned_df, hmm_comp_df, hgb_staged_history, phase2_dir
    )
    print(f"Complete Phase 2 ML + Bayesian + HMM pipeline finished in {time.perf_counter() - t0_all:.2f}s")


if __name__ == "__main__":
    main()
