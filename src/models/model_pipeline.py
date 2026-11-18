"""Five-model training, GroupKFold evaluation, and observation-level map-matching metrics."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pathsnap.schemas.candidate_schema import (
    CANDIDATE_FEATURE_COLUMNS,
    CANDIDATE_SCHEMA_DTYPES,
    CANDIDATE_TARGET_COLUMN,
)
from pathsnap.graph.road_graph import TripRoadGraph


RANDOM_SEED = 42


def get_five_models(random_state: int = RANDOM_SEED) -> Dict[str, Tuple[str, str, Any]]:
    """Return dictionary mapping model_id to (Model Display Name, Algorithm Family, Estimator/Pipeline).

    All preprocessing (imputation, missing-value indicators, scaling) is encapsulated inside
    sklearn Pipelines so statistics are fitted strictly on training folds (zero trip leakage).
    """
    models: Dict[str, Tuple[str, str, Any]] = {
        "logistic_regression": (
            "Logistic Regression",
            "Linear & Probabilistic",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            max_iter=500,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        "distance_knn": (
            "Distance-Weighted KNN",
            "Instance-Based",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        KNeighborsClassifier(
                            n_neighbors=25,
                            weights="distance",
                            metric="manhattan",
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        "decision_tree": (
            "Decision Tree",
            "Tree-Based",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "clf",
                        DecisionTreeClassifier(
                            criterion="log_loss",
                            max_depth=8,
                            min_samples_leaf=30,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        "random_forest": (
            "Random Forest",
            "Bagging Ensemble",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=120,
                            max_depth=12,
                            min_samples_leaf=15,
                            max_features="sqrt",
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        "hist_gradient_boosting": (
            "HistGradientBoosting",
            "Gradient Boosting",
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=150,
                max_leaf_nodes=31,
                max_depth=8,
                min_samples_leaf=25,
                l2_regularization=1.0,
                random_state=random_state,
            ),
        ),
    }
    return models


def verify_candidate_contract(candidates_df: pd.DataFrame, observations_df: pd.DataFrame) -> Dict[str, Any]:
    """Verify all 20 candidate columns, 11 feature columns, label dtype, and label_eligible consistency."""
    expected_cand_cols = list(CANDIDATE_SCHEMA_DTYPES.keys())
    actual_cand_cols = list(candidates_df.columns)
    assert len(expected_cand_cols) == 20, "Contract requires 20 candidate columns"
    if actual_cand_cols != expected_cand_cols:
        raise ValueError(f"Candidate columns mismatch: {actual_cand_cols} vs {expected_cand_cols}")

    assert len(CANDIDATE_FEATURE_COLUMNS) == 11, "Contract requires 11 feature columns"
    for fcol in CANDIDATE_FEATURE_COLUMNS:
        if fcol not in candidates_df.columns:
            raise ValueError(f"Missing feature column {fcol}")

    # Ensure no ID columns leak into feature columns
    forbidden = {
        "trip_id",
        "partition",
        "observation_id",
        "timestamp_s",
        "candidate_rank",
        "candidate_arc_id",
        "from_node_id",
        "to_node_id",
        "label",
    }
    assert forbidden.isdisjoint(set(CANDIDATE_FEATURE_COLUMNS)), "Identifier leaked into features!"

    labeled_mask = candidates_df[CANDIDATE_TARGET_COLUMN].notna()
    labeled_cands = candidates_df.loc[labeled_mask]
    pos_count = int((labeled_cands[CANDIDATE_TARGET_COLUMN] == 1).sum())
    neg_count = int((labeled_cands[CANDIDATE_TARGET_COLUMN] == 0).sum())
    null_count = int((~labeled_mask).sum())

    aligned_obs = observations_df[observations_df["alignment_status"] == "aligned"]
    miss_count = int((aligned_obs["candidate_miss"] == True).sum())  # noqa: E712
    miss_rate = float(miss_count / len(aligned_obs)) if len(aligned_obs) > 0 else 0.0

    return {
        "num_trips": int(candidates_df["trip_id"].nunique()),
        "total_candidate_rows": int(len(candidates_df)),
        "labeled_candidate_rows": int(len(labeled_cands)),
        "null_label_candidate_rows": null_count,
        "positive_label_count": pos_count,
        "negative_label_count": neg_count,
        "positive_class_ratio": round(pos_count / max(len(labeled_cands), 1), 6),
        "total_observations": int(len(observations_df)),
        "aligned_observations": int(len(aligned_obs)),
        "label_eligible_observations": int(observations_df["label_eligible"].sum()),
        "candidate_miss_count": miss_count,
        "candidate_miss_rate": round(miss_rate, 6),
    }


def compute_observation_and_route_metrics(
    eval_cands: pd.DataFrame,
    eval_obs: pd.DataFrame,
    pred_probs: np.ndarray,
    graphs_by_trip: Dict[str, TripRoadGraph],
) -> Dict[str, float]:
    """Compute Observation-Level Top-1 Accuracy and Route Continuity on a set of trips."""
    df = eval_cands[
        ["trip_id", "observation_id", "candidate_arc_id", "distance_to_segment_m"]
    ].copy()
    df["pred_prob"] = pred_probs

    # Select highest-probability candidate per (trip_id, observation_id), breaking ties by shortest distance
    top1 = (
        df.sort_values(
            ["trip_id", "observation_id", "pred_prob", "distance_to_segment_m"],
            ascending=[True, True, False, True],
        )
        .groupby(["trip_id", "observation_id"], as_index=False)
        .first()
    )

    merged = eval_obs.merge(
        top1[["trip_id", "observation_id", "candidate_arc_id", "pred_prob"]],
        on=["trip_id", "observation_id"],
        how="left",
    ).sort_values(["trip_id", "observation_id"])

    # 1. Observation-level Top-1 Accuracy across aligned observations
    aligned_mask = merged["alignment_status"] == "aligned"
    if aligned_mask.any():
        pred_arcs = merged.loc[aligned_mask, "candidate_arc_id"].fillna(-999).to_numpy(dtype=np.int32)
        ref_arcs = merged.loc[aligned_mask, "reference_arc_id"].fillna(-888).to_numpy(dtype=np.int32)
        top1_acc_aligned = float(np.mean(pred_arcs == ref_arcs))
    else:
        top1_acc_aligned = float("nan")

    # 2. Observation-level Top-1 Accuracy across label_eligible observations
    elig_mask = merged["label_eligible"] == True  # noqa: E712
    if elig_mask.any():
        pred_arcs_e = merged.loc[elig_mask, "candidate_arc_id"].fillna(-999).to_numpy(dtype=np.int32)
        ref_arcs_e = merged.loc[elig_mask, "reference_arc_id"].fillna(-888).to_numpy(dtype=np.int32)
        top1_acc_eligible = float(np.mean(pred_arcs_e == ref_arcs_e))
    else:
        top1_acc_eligible = float("nan")

    # 3. Route continuity across consecutive observations per trip
    total_transitions = 0
    connected_transitions = 0
    for tid, grp in merged.groupby("trip_id", sort=False):
        graph = graphs_by_trip.get(str(tid))
        if graph is None or len(grp) <= 1:
            continue
        arcs_seq = grp["candidate_arc_id"].fillna(-1).to_numpy(dtype=np.int32)
        for i in range(1, len(arcs_seq)):
            u, v = int(arcs_seq[i - 1]), int(arcs_seq[i])
            total_transitions += 1
            if u >= 0 and v >= 0 and graph.are_connected(u, v):
                connected_transitions += 1

    route_continuity = (
        float(connected_transitions / total_transitions) if total_transitions > 0 else float("nan")
    )

    return {
        "obs_top1_accuracy_aligned": top1_acc_aligned,
        "obs_top1_accuracy_eligible": top1_acc_eligible,
        "route_continuity": route_continuity,
    }


def evaluate_five_models_group_cv(
    candidates_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    graphs_by_trip: Dict[str, TripRoadGraph],
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """Run GroupKFold cross-validation across all 5 models on labeled training candidates.

    Returns:
      - summary_df: Consolidated comparison table across the 5 models
      - fold_df: Detailed per-fold metrics table (`cv_results.csv`)
      - oof_probs_by_model: Out-of-fold predicted probabilities for every candidate row in `candidates_df`
    """
    unique_trips = candidates_df["trip_id"].unique()
    if len(unique_trips) < n_splits:
        raise ValueError(f"GroupKFold(n_splits={n_splits}) requires at least {n_splits} trips, got {len(unique_trips)}")

    # GroupKFold split at trip level so we can evaluate both labeled candidate metrics AND full-trip observation metrics
    trip_df = pd.DataFrame({"trip_id": unique_trips})
    gkf = GroupKFold(n_splits=n_splits)
    fold_trip_sets: List[Tuple[set, set]] = []
    for tr_idx, val_idx in gkf.split(trip_df, groups=trip_df["trip_id"]):
        tr_trips = set(trip_df.iloc[tr_idx]["trip_id"])
        val_trips = set(trip_df.iloc[val_idx]["trip_id"])
        assert tr_trips.isdisjoint(val_trips), "Trip leakage detected!"
        fold_trip_sets.append((tr_trips, val_trips))

    models = get_five_models(random_state=random_state)
    fold_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    oof_probs_by_model: Dict[str, np.ndarray] = {}

    # Also compute the Proximity Rank-1 Baseline (Nearest Segment) for scientific comparison
    rank1_probs = (1.0 / (1.0 + candidates_df["distance_to_segment_m"].to_numpy(dtype=np.float64))).clip(1e-6, 1 - 1e-6)
    oof_probs_by_model["nearest_segment_baseline"] = rank1_probs

    for model_key, (model_name, family, base_estimator) in models.items():
        from sklearn.base import clone

        oof_probs = np.full(len(candidates_df), np.nan, dtype=np.float64)
        total_train_time = 0.0
        total_infer_time = 0.0

        fold_metrics_list: List[Dict[str, float]] = []

        for fold_idx, (tr_trips, val_trips) in enumerate(fold_trip_sets):
            # Strictly filter training fold to labeled rows (label_eligible == True)
            tr_mask = candidates_df["trip_id"].isin(tr_trips) & candidates_df[CANDIDATE_TARGET_COLUMN].notna()
            val_all_mask = candidates_df["trip_id"].isin(val_trips)
            val_labeled_mask = val_all_mask & candidates_df[CANDIDATE_TARGET_COLUMN].notna()

            X_tr = candidates_df.loc[tr_mask, CANDIDATE_FEATURE_COLUMNS].astype("float32")
            y_tr = candidates_df.loc[tr_mask, CANDIDATE_TARGET_COLUMN].astype("int8").to_numpy()

            X_val_all = candidates_df.loc[val_all_mask, CANDIDATE_FEATURE_COLUMNS].astype("float32")
            X_val_lab = candidates_df.loc[val_labeled_mask, CANDIDATE_FEATURE_COLUMNS].astype("float32")
            y_val_lab = candidates_df.loc[val_labeled_mask, CANDIDATE_TARGET_COLUMN].astype("int8").to_numpy()

            est = clone(base_estimator)
            t0 = time.perf_counter()
            est.fit(X_tr, y_tr)
            tr_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            probs_val_all = est.predict_proba(X_val_all)[:, 1]
            inf_time = time.perf_counter() - t1

            oof_probs[val_all_mask.to_numpy()] = probs_val_all
            total_train_time += tr_time
            total_infer_time += inf_time

            # Extract probabilities for labeled subset
            labeled_in_val = candidates_df.loc[val_all_mask, CANDIDATE_TARGET_COLUMN].notna().to_numpy()
            probs_val_lab = probs_val_all[labeled_in_val]
            preds_val_lab = (probs_val_lab >= 0.5).astype(np.int8)

            acc = float(accuracy_score(y_val_lab, preds_val_lab))
            prec = float(precision_score(y_val_lab, preds_val_lab, zero_division=0))
            rec = float(recall_score(y_val_lab, preds_val_lab, zero_division=0))
            f1_bin = float(f1_score(y_val_lab, preds_val_lab, pos_label=1, zero_division=0))
            f1_mac = float(f1_score(y_val_lab, preds_val_lab, average="macro", zero_division=0))
            ll = float(log_loss(y_val_lab, np.clip(probs_val_lab, 1e-6, 1 - 1e-6), labels=[0, 1]))
            tn, fp, fn, tp = confusion_matrix(y_val_lab, preds_val_lab, labels=[0, 1]).ravel()

            val_obs_df = observations_df[observations_df["trip_id"].isin(val_trips)]
            val_cands_df = candidates_df.loc[val_all_mask]
            obs_metrics = compute_observation_and_route_metrics(
                val_cands_df, val_obs_df, probs_val_all, graphs_by_trip
            )

            f_row = {
                "model_key": model_key,
                "model_name": model_name,
                "algorithm_family": family,
                "fold": fold_idx + 1,
                "val_trips": ",".join(sorted(val_trips)),
                "train_labeled_rows": int(len(y_tr)),
                "val_labeled_rows": int(len(y_val_lab)),
                "accuracy": round(acc, 6),
                "precision_pos": round(prec, 6),
                "recall_pos": round(rec, 6),
                "f1_positive": round(f1_bin, 6),
                "macro_f1": round(f1_mac, 6),
                "log_loss": round(ll, 6),
                "obs_top1_accuracy": round(obs_metrics["obs_top1_accuracy_aligned"], 6),
                "obs_top1_accuracy_eligible": round(obs_metrics["obs_top1_accuracy_eligible"], 6),
                "route_continuity": round(obs_metrics["route_continuity"], 6),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "train_time_s": round(tr_time, 4),
                "inference_time_s": round(inf_time, 4),
            }
            fold_rows.append(f_row)
            fold_metrics_list.append(f_row)

        oof_probs_by_model[model_key] = oof_probs

        # Compute overall OOF metrics and mean/std across folds
        lab_mask = candidates_df[CANDIDATE_TARGET_COLUMN].notna().to_numpy()
        y_all_lab = candidates_df.loc[lab_mask, CANDIDATE_TARGET_COLUMN].astype("int8").to_numpy()
        oof_lab_probs = oof_probs[lab_mask]
        oof_lab_preds = (oof_lab_probs >= 0.5).astype(np.int8)

        oof_obs_metrics = compute_observation_and_route_metrics(
            candidates_df, observations_df, oof_probs, graphs_by_trip
        )
        tn_all, fp_all, fn_all, tp_all = confusion_matrix(y_all_lab, oof_lab_preds, labels=[0, 1]).ravel()

        f_df = pd.DataFrame(fold_metrics_list)
        summary_rows.append(
            {
                "model_key": model_key,
                "model_name": model_name,
                "algorithm_family": family,
                "evaluation_scope": f"Preliminary {len(unique_trips)}-Trip GroupKFold ({n_splits} folds)",
                "cv_macro_f1_mean": round(float(f_df["macro_f1"].mean()), 6),
                "cv_macro_f1_std": round(float(f_df["macro_f1"].std(ddof=0)), 6),
                "oof_macro_f1": round(float(f1_score(y_all_lab, oof_lab_preds, average="macro")), 6),
                "oof_positive_f1": round(float(f1_score(y_all_lab, oof_lab_preds, pos_label=1)), 6),
                "oof_accuracy": round(float(accuracy_score(y_all_lab, oof_lab_preds)), 6),
                "oof_precision": round(float(precision_score(y_all_lab, oof_lab_preds, zero_division=0)), 6),
                "oof_recall": round(float(recall_score(y_all_lab, oof_lab_preds, zero_division=0)), 6),
                "cv_log_loss_mean": round(float(f_df["log_loss"].mean()), 6),
                "oof_log_loss": round(float(log_loss(y_all_lab, np.clip(oof_lab_probs, 1e-6, 1 - 1e-6))), 6),
                "cv_obs_top1_accuracy_mean": round(float(f_df["obs_top1_accuracy"].mean()), 6),
                "oof_obs_top1_accuracy": round(oof_obs_metrics["obs_top1_accuracy_aligned"], 6),
                "oof_obs_top1_eligible": round(oof_obs_metrics["obs_top1_accuracy_eligible"], 6),
                "oof_route_continuity": round(oof_obs_metrics["route_continuity"], 6),
                "tn": int(tn_all),
                "fp": int(fp_all),
                "fn": int(fn_all),
                "tp": int(tp_all),
                "total_cv_train_time_s": round(total_train_time, 4),
                "total_cv_inference_time_s": round(total_infer_time, 4),
                "mean_fold_inference_time_s": round(total_infer_time / n_splits, 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    return summary_df, fold_df, oof_probs_by_model
