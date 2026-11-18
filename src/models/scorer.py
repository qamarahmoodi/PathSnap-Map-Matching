"""Reusable model loading and candidate scoring interface for Qamar (HMM/FastAPI) and Gokul (Evaluation)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd

from pathsnap.schemas.candidate_schema import (
    CANDIDATE_FEATURE_COLUMNS,
    CANDIDATE_ID_COLUMNS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "final_tuned_model.joblib"
DEFAULT_METADATA_PATH = REPO_ROOT / "artifacts" / "final_model_metadata.json"

# Stable minimum output columns guaranteed for Qamar's HMM/Viterbi and Gokul's evaluator
SCORER_REQUIRED_ID_COLUMNS: List[str] = [
    "trip_id",
    "observation_id",
    "candidate_arc_id",
    "candidate_rank",
]


@dataclass
class LoadedModelBundle:
    """Container holding the fitted estimator/pipeline and its verified feature/class metadata."""

    estimator: Any
    model_name: str
    model_key: str
    algorithm_family: str
    feature_columns: List[str]
    classes: List[int]
    positive_class: int
    positive_class_index: int
    metadata: Dict[str, Any]


def load_trained_model(
    artifact_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
) -> LoadedModelBundle:
    """Load the serialized final model pipeline and its metadata from disk.

    Args:
        artifact_path: Path to `.joblib` model file (defaults to `artifacts/final_tuned_model.joblib`).
        metadata_path: Path to `.json` metadata file (defaults to `artifacts/final_model_metadata.json`).

    Returns:
        LoadedModelBundle with verified feature order and positive-class index.
    """
    art_p = Path(artifact_path) if artifact_path is not None else DEFAULT_ARTIFACT_PATH
    meta_p = Path(metadata_path) if metadata_path is not None else DEFAULT_METADATA_PATH

    if not art_p.is_file():
        raise FileNotFoundError(f"Trained model artifact not found at: {art_p}")

    payload = joblib.load(art_p)
    if isinstance(payload, dict) and ("estimator" in payload or "model" in payload):
        estimator = payload.get("estimator") or payload.get("model")
        meta = payload.get("metadata", {})
    else:
        estimator = payload
        meta = {}

    if meta_p.is_file():
        with open(meta_p, "r", encoding="utf-8") as f:
            file_meta = json.load(f)
        meta = {**meta, **file_meta}

    feature_cols = list(meta.get("feature_columns", CANDIDATE_FEATURE_COLUMNS))
    classes_arr = getattr(estimator, "classes_", np.array([0, 1]))
    classes_list = [int(c) for c in classes_arr]
    pos_class = int(meta.get("positive_class", 1))
    if pos_class not in classes_list:
        raise ValueError(f"Positive class {pos_class} not found in estimator classes {classes_list}")
    pos_idx = classes_list.index(pos_class)

    return LoadedModelBundle(
        estimator=estimator,
        model_name=str(meta.get("model_name", type(estimator).__name__)),
        model_key=str(meta.get("model_key", "final_tuned_model")),
        algorithm_family=str(meta.get("algorithm_family", "Gradient Boosting")),
        feature_columns=feature_cols,
        classes=classes_list,
        positive_class=pos_class,
        positive_class_index=pos_idx,
        metadata=meta,
    )


def score_candidates(
    candidates_df: pd.DataFrame,
    model_or_bundle: Optional[Union[LoadedModelBundle, Any]] = None,
    artifact_path: Optional[Union[str, Path]] = None,
    prob_floor: float = 1e-6,
    retain_all_columns: bool = True,
) -> pd.DataFrame:
    """Score a candidate DataFrame and return calibrated probabilities preserving exact arc correspondence.

    Guarantees output columns:
      - `trip_id`: 8-char trip identifier
      - `observation_id`: 0-based observation index in `.track`
      - `candidate_arc_id`: 0-based row index in `<trip_id>.arcs`
      - `arc_id`: alias of `candidate_arc_id` for downstream convenience
      - `candidate_rank`: 1-based spatial distance rank
      - `predicted_probability`: P(label == 1 | features) in [prob_floor, 1 - prob_floor]
      - `pred_prob`: alias of `predicted_probability` for `decode_trip_viterbi` compatibility
      - `predicted_log_probability`: log(predicted_probability) for log-domain HMM/Viterbi emissions
      - `predicted_label`: binary prediction (1 if predicted_probability >= 0.5 else 0)
      - `is_obs_top1`: True for the highest-scoring candidate at each `(trip_id, observation_id)`
    """
    for req_col in SCORER_REQUIRED_ID_COLUMNS:
        if req_col not in candidates_df.columns:
            raise ValueError(f"Input candidates_df is missing required identifier column: '{req_col}'")

    if model_or_bundle is None:
        bundle = load_trained_model(artifact_path=artifact_path)
        estimator = bundle.estimator
        feature_cols = bundle.feature_columns
        pos_idx = bundle.positive_class_index
    elif isinstance(model_or_bundle, LoadedModelBundle):
        estimator = model_or_bundle.estimator
        feature_cols = model_or_bundle.feature_columns
        pos_idx = model_or_bundle.positive_class_index
    else:
        estimator = model_or_bundle
        feature_cols = list(CANDIDATE_FEATURE_COLUMNS)
        classes_arr = getattr(estimator, "classes_", np.array([0, 1]))
        pos_idx = list(classes_arr).index(1)

    for fcol in feature_cols:
        if fcol not in candidates_df.columns:
            raise ValueError(f"Input candidates_df is missing required feature column: '{fcol}'")

    if candidates_df.empty:
        base_cols = list(candidates_df.columns) if retain_all_columns else list(SCORER_REQUIRED_ID_COLUMNS)
        out = pd.DataFrame(columns=base_cols)
        out["arc_id"] = pd.Series(dtype="int32")
        out["predicted_probability"] = pd.Series(dtype="float64")
        out["pred_prob"] = pd.Series(dtype="float64")
        out["predicted_log_probability"] = pd.Series(dtype="float64")
        out["predicted_label"] = pd.Series(dtype="int8")
        out["is_obs_top1"] = pd.Series(dtype="bool")
        return out

    # Extract strictly the 11 feature columns in the exact order expected by the model
    X = candidates_df[feature_cols].astype("float32")
    raw_probs = estimator.predict_proba(X)[:, pos_idx]
    clipped_probs = np.clip(np.asarray(raw_probs, dtype=np.float64), prob_floor, 1.0 - prob_floor)
    log_probs = np.log(clipped_probs)
    pred_labels = (clipped_probs >= 0.5).astype(np.int8)

    if retain_all_columns:
        out = candidates_df.copy()
    else:
        keep_cols = [c for c in CANDIDATE_ID_COLUMNS if c in candidates_df.columns]
        if "distance_to_segment_m" in candidates_df.columns:
            keep_cols.append("distance_to_segment_m")
        if "label" in candidates_df.columns:
            keep_cols.append("label")
        out = candidates_df[keep_cols].copy()

    out["arc_id"] = out["candidate_arc_id"].astype(np.int32)
    out["predicted_probability"] = clipped_probs
    out["pred_prob"] = clipped_probs
    out["predicted_log_probability"] = log_probs
    out["predicted_label"] = pred_labels

    # Identify the top-1 highest probability candidate per (trip_id, observation_id), tie-breaking by distance/rank
    tie_col = "distance_to_segment_m" if "distance_to_segment_m" in out.columns else "candidate_rank"
    out["_row_pos"] = np.arange(len(out), dtype=np.int64)
    top1_rows = (
        out.sort_values(
            ["trip_id", "observation_id", "predicted_probability", tie_col, "candidate_arc_id"],
            ascending=[True, True, False, True, True],
        )
        .groupby(["trip_id", "observation_id"], as_index=False, sort=False)
        .first()["_row_pos"]
        .to_numpy()
    )
    is_top1 = np.zeros(len(out), dtype=bool)
    is_top1[top1_rows] = True
    out["is_obs_top1"] = is_top1
    out = out.drop(columns=["_row_pos"])

    return out
