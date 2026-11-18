"""Stable model-loading and candidate-scoring interface for Qamar (HMM/FastAPI) and Gokul (Evaluation)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

from models.scorer import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_METADATA_PATH,
    LoadedModelBundle,
    SCORER_REQUIRED_ID_COLUMNS,
    load_trained_model,
    score_candidates as _score_candidates_impl,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL_PATH = REPO_ROOT / "artifacts" / "final_model.joblib"


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model_bundle(
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
) -> LoadedModelBundle:
    """Load the serialized final model bundle and fitted preprocessing pipeline from disk.

    Defaults to `artifacts/final_model.joblib` (or falls back to `artifacts/final_tuned_model.joblib`).
    """
    if model_path is None:
        target_path = FINAL_MODEL_PATH if FINAL_MODEL_PATH.is_file() else DEFAULT_ARTIFACT_PATH
    else:
        target_path = Path(model_path)
    return load_trained_model(artifact_path=target_path, metadata_path=metadata_path)


def score_candidates(
    candidates_df: pd.DataFrame,
    model_bundle: Optional[Union[LoadedModelBundle, Any]] = None,
    model_or_bundle: Optional[Union[LoadedModelBundle, Any]] = None,
    model_path: Optional[Union[str, Path]] = None,
    prob_floor: float = 1e-6,
    retain_all_columns: bool = True,
) -> pd.DataFrame:
    """Validate candidate features, enforce exact feature-column order, and return calibrated probabilities.

    Preserves all original candidate identifiers (`trip_id`, `observation_id`, `candidate_arc_id`, `candidate_rank`)
    and appends:
      - `predicted_probability`: Calibrated positive-class probability P(label == 1 | X) in [prob_floor, 1 - prob_floor]
      - `pred_prob`: Alias for Viterbi decoder (`decode_trip_viterbi`)
      - `predicted_log_probability`: log(predicted_probability)
      - `predicted_label`: 1 if predicted_probability >= 0.5 else 0
      - `is_obs_top1`: True for the top-1 highest scoring candidate per `(trip_id, observation_id)`
    """
    active_bundle = model_bundle if model_bundle is not None else model_or_bundle
    if active_bundle is None:
        active_bundle = load_model_bundle(model_path=model_path)
    return _score_candidates_impl(
        candidates_df=candidates_df,
        model_or_bundle=active_bundle,
        prob_floor=prob_floor,
        retain_all_columns=retain_all_columns,
    )


__all__ = [
    "FINAL_MODEL_PATH",
    "LoadedModelBundle",
    "SCORER_REQUIRED_ID_COLUMNS",
    "compute_file_sha256",
    "load_model_bundle",
    "load_trained_model",
    "score_candidates",
]
