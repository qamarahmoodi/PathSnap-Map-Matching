"""Bayesian hyperparameter optimisation using Optuna TPE with Trip-Grouped Cross-Validation."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from models.model_pipeline import RANDOM_SEED, compute_observation_and_route_metrics
from pathsnap.graph.road_graph import TripRoadGraph
from pathsnap.schemas.candidate_schema import CANDIDATE_FEATURE_COLUMNS, CANDIDATE_TARGET_COLUMN

# Silence verbose Optuna stdout logs during clean evaluation runs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_tuning_hgb(
    candidates_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    graphs_by_trip: Dict[str, TripRoadGraph],
    n_trials: int = 8,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any], HistGradientBoostingClassifier, np.ndarray]:
    """Run genuine Optuna TPE (Tree-structured Parzen Estimator) Bayesian optimisation for HistGradientBoosting.

    Optimises Out-of-Fold Macro F1-score (with secondary tracking of Log Loss and Observation Top-1 Accuracy)
    using trip-grouped cross-validation (`GroupKFold`).
    """
    unique_trips = candidates_df["trip_id"].unique()
    trip_df = pd.DataFrame({"trip_id": unique_trips})
    gkf = GroupKFold(n_splits=n_splits)
    fold_trip_sets: List[Tuple[set, set]] = []
    for tr_idx, val_idx in gkf.split(trip_df, groups=trip_df["trip_id"]):
        fold_trip_sets.append(
            (set(trip_df.iloc[tr_idx]["trip_id"]), set(trip_df.iloc[val_idx]["trip_id"]))
        )

    trial_records: List[Dict[str, Any]] = []
    best_oof_probs: np.ndarray = np.full(len(candidates_df), np.nan, dtype=np.float64)
    best_score_seen = -1.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_oof_probs, best_score_seen
        t_start = time.perf_counter()

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.18, log=True),
            "max_iter": trial.suggest_int("max_iter", 80, 240, step=20),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-2, 10.0, log=True),
            "random_state": random_state,
        }

        oof_probs = np.full(len(candidates_df), np.nan, dtype=np.float64)
        fold_f1s: List[float] = []
        fold_lls: List[float] = []

        for tr_trips, val_trips in fold_trip_sets:
            tr_mask = candidates_df["trip_id"].isin(tr_trips) & candidates_df[CANDIDATE_TARGET_COLUMN].notna()
            val_all_mask = candidates_df["trip_id"].isin(val_trips)

            X_tr = candidates_df.loc[tr_mask, CANDIDATE_FEATURE_COLUMNS].astype("float32")
            y_tr = candidates_df.loc[tr_mask, CANDIDATE_TARGET_COLUMN].astype("int8").to_numpy()
            X_val_all = candidates_df.loc[val_all_mask, CANDIDATE_FEATURE_COLUMNS].astype("float32")

            clf = HistGradientBoostingClassifier(**params)
            clf.fit(X_tr, y_tr)
            probs_val = clf.predict_proba(X_val_all)[:, 1]
            oof_probs[val_all_mask.to_numpy()] = probs_val

            lab_sub = candidates_df.loc[val_all_mask, CANDIDATE_TARGET_COLUMN].notna().to_numpy()
            y_val_lab = candidates_df.loc[
                val_all_mask & candidates_df[CANDIDATE_TARGET_COLUMN].notna(),
                CANDIDATE_TARGET_COLUMN,
            ].astype("int8").to_numpy()
            probs_lab = probs_val[lab_sub]
            preds_lab = (probs_lab >= 0.5).astype(np.int8)

            fold_f1s.append(float(f1_score(y_val_lab, preds_lab, average="macro", zero_division=0)))
            fold_lls.append(float(log_loss(y_val_lab, np.clip(probs_lab, 1e-6, 1 - 1e-6), labels=[0, 1])))

        lab_mask = candidates_df[CANDIDATE_TARGET_COLUMN].notna().to_numpy()
        y_all_lab = candidates_df.loc[lab_mask, CANDIDATE_TARGET_COLUMN].astype("int8").to_numpy()
        oof_lab_probs = oof_probs[lab_mask]
        oof_lab_preds = (oof_lab_probs >= 0.5).astype(np.int8)

        oof_macro_f1 = float(f1_score(y_all_lab, oof_lab_preds, average="macro", zero_division=0))
        oof_pos_f1 = float(f1_score(y_all_lab, oof_lab_preds, pos_label=1, zero_division=0))
        oof_ll = float(log_loss(y_all_lab, np.clip(oof_lab_probs, 1e-6, 1 - 1e-6), labels=[0, 1]))
        obs_metrics = compute_observation_and_route_metrics(
            candidates_df, observations_df, oof_probs, graphs_by_trip
        )
        elapsed_s = time.perf_counter() - t_start

        # Composite score primarily driven by OOF Macro F1 + Observation Top-1 Accuracy
        mean_cv_f1 = float(np.mean(fold_f1s))

        if oof_macro_f1 > best_score_seen:
            best_score_seen = oof_macro_f1
            best_oof_probs = oof_probs.copy()

        trial_records.append(
            {
                "trial_number": trial.number + 1,
                "sampler": "Optuna TPESampler (Bayesian)",
                "evaluation_scope": f"Preliminary {len(unique_trips)}-Trip GroupKFold ({n_splits} folds)",
                "learning_rate": round(params["learning_rate"], 6),
                "max_iter": params["max_iter"],
                "max_leaf_nodes": params["max_leaf_nodes"],
                "max_depth": params["max_depth"],
                "min_samples_leaf": params["min_samples_leaf"],
                "l2_regularization": round(params["l2_regularization"], 6),
                "cv_macro_f1_mean": round(mean_cv_f1, 6),
                "oof_macro_f1": round(oof_macro_f1, 6),
                "oof_positive_f1": round(oof_pos_f1, 6),
                "cv_log_loss_mean": round(float(np.mean(fold_lls)), 6),
                "oof_log_loss": round(oof_ll, 6),
                "oof_obs_top1_accuracy": round(obs_metrics["obs_top1_accuracy_aligned"], 6),
                "oof_route_continuity": round(obs_metrics["route_continuity"], 6),
                "trial_time_s": round(elapsed_s, 4),
            }
        )
        return oof_macro_f1

    sampler = optuna.samplers.TPESampler(seed=random_state, n_startup_trials=3)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Enqueue the untuned baseline configuration as Trial 1 for direct apple-to-apples comparison
    study.enqueue_trial(
        {
            "learning_rate": 0.08,
            "max_iter": 140,
            "max_leaf_nodes": 31,
            "max_depth": 8,
            "min_samples_leaf": 25,
            "l2_regularization": 1.0,
        }
    )

    t0_total = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    total_opt_time = time.perf_counter() - t0_total

    trials_df = pd.DataFrame(trial_records)
    best_params = dict(study.best_params)
    best_params["random_state"] = random_state

    # Refit best model on all available labeled training rows
    lab_mask = candidates_df[CANDIDATE_TARGET_COLUMN].notna()
    X_full = candidates_df.loc[lab_mask, CANDIDATE_FEATURE_COLUMNS].astype("float32")
    y_full = candidates_df.loc[lab_mask, CANDIDATE_TARGET_COLUMN].astype("int8").to_numpy()

    refit_t0 = time.perf_counter()
    best_model = HistGradientBoostingClassifier(**best_params)
    best_model.fit(X_full, y_full)
    refit_time = time.perf_counter() - refit_t0

    best_summary = {
        "best_trial_number": int(study.best_trial.number + 1),
        "best_oof_macro_f1": round(float(study.best_value), 6),
        "best_params": best_params,
        "total_trials": n_trials,
        "total_optimisation_time_s": round(total_opt_time, 4),
        "refit_time_s": round(refit_time, 4),
    }

    return trials_df, best_summary, best_model, best_oof_probs
