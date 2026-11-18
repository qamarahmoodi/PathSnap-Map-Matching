"""Model loading and candidate scoring adapter for PathSnap inference.

Owner: Qamar Ahmed (Technical Lead)

Loads Pranav's serialized `.joblib` model artifact once, validates feature schema
compatibility (`CANDIDATE_FEATURE_COLUMNS`), and scores candidate DataFrames
without using identifier or reference-route columns as features.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
import joblib
import numpy as np
import pandas as pd

from pathsnap.schemas.candidate_schema import CANDIDATE_FEATURE_COLUMNS


ENV_MODEL_PATH = "PATHSNAP_MODEL_PATH"
DEFAULT_PRIMARY_MODEL_REL = Path("artifacts") / "best_model.joblib"
DEFAULT_FALLBACK_MODEL_REL = Path("artifacts") / "phase1_pilot_tuned_hgb.joblib"


@dataclass
class ScoredCandidatesResult:
    """Output of CandidateModelScorer.score_candidates(...)."""

    scored_candidates: pd.DataFrame
    model_name: str
    model_path: str
    num_candidates: int


def resolve_model_path(model_path: Optional[str | Path] = None) -> Path:
    """Locate the trained model `.joblib` artifact.

    Resolution order:
    1. Explicit `model_path` argument (raises FileNotFoundError immediately if specified and missing).
    2. `PATHSNAP_MODEL_PATH` environment variable.
    3. `artifacts/best_model.joblib` (if exported by full 80-trip training).
    4. `artifacts/phase1_pilot_tuned_hgb.joblib` (committed Phase 1 tuned HistGradientBoosting artifact).
    """
    if model_path is not None:
        explicit_path = Path(model_path)
        if explicit_path.is_file():
            return explicit_path.resolve()
        raise FileNotFoundError(f"Explicitly requested model artifact not found: {explicit_path}")

    repo_root = Path(__file__).resolve().parents[3]
    candidates: List[Path] = []

    env_val = os.environ.get(ENV_MODEL_PATH)
    if env_val:
        candidates.append(Path(env_val))

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / DEFAULT_PRIMARY_MODEL_REL,
            repo_root / DEFAULT_PRIMARY_MODEL_REL,
            cwd / DEFAULT_FALLBACK_MODEL_REL,
            repo_root / DEFAULT_FALLBACK_MODEL_REL,
        ]
    )

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        f"No trained PathSnap model artifact found. Checked {ENV_MODEL_PATH}, "
        f"'{DEFAULT_PRIMARY_MODEL_REL}', and '{DEFAULT_FALLBACK_MODEL_REL}'."
    )


class CandidateModelScorer:
    """Reusable candidate-scoring component wrapping Pranav's exported scikit-learn estimator/pipeline."""

    def __init__(
        self,
        estimator: Any,
        model_path: str = "in_memory",
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(
                f"Loaded model '{type(estimator).__name__}' does not implement required 'predict_proba(X)' method."
            )
        self.estimator = estimator
        self.model_path = str(model_path)
        self.model_name = type(estimator).__name__
        self.feature_columns = list(feature_columns or CANDIDATE_FEATURE_COLUMNS)

        # Verify expected feature count if estimator exposes n_features_in_
        n_in = getattr(estimator, "n_features_in_", None)
        if n_in is not None and int(n_in) != len(self.feature_columns):
            raise ValueError(
                f"Model feature count mismatch: model expects {n_in} features, "
                f"contract specifies {len(self.feature_columns)} ({self.feature_columns})"
            )

    @classmethod
    def from_path(cls, model_path: Optional[str | Path] = None) -> "CandidateModelScorer":
        """Load a serialized `.joblib` model or dict bundle from disk."""
        resolved = resolve_model_path(model_path)
        loaded = joblib.load(resolved)

        # Support either direct sklearn estimator/Pipeline or a dict bundle {"model": ..., "feature_columns": ...}
        if isinstance(loaded, dict):
            if "model" not in loaded:
                raise ValueError(f"Model bundle dictionary at {resolved} is missing 'model' key")
            estimator = loaded["model"]
            feat_cols = loaded.get("feature_columns", CANDIDATE_FEATURE_COLUMNS)
        else:
            estimator = loaded
            feat_cols = CANDIDATE_FEATURE_COLUMNS

        return cls(estimator=estimator, model_path=str(resolved), feature_columns=feat_cols)

    def score_candidates(
        self,
        candidates_df: pd.DataFrame,
        prob_col: str = "pred_prob",
    ) -> ScoredCandidatesResult:
        """Validate candidate DataFrame schema, enforce exact feature order, and attach predicted probabilities.

        Preserves `trip_id`, `observation_id`, `candidate_arc_id`, `candidate_rank`, and all original columns,
        adding both `predicted_probability` and `prob_col` (`pred_prob`).
        """
        required_id_cols = ["trip_id", "observation_id", "candidate_arc_id", "candidate_rank"]
        missing_ids = [c for c in required_id_cols if c not in candidates_df.columns]
        if missing_ids:
            raise ValueError(f"Candidate DataFrame is missing required identifier columns: {missing_ids}")

        missing_feats = [c for c in self.feature_columns if c not in candidates_df.columns]
        if missing_feats:
            raise ValueError(f"Candidate DataFrame is missing required feature columns: {missing_feats}")

        out_df = candidates_df.copy()
        if out_df.empty:
            out_df["predicted_probability"] = pd.Series(dtype="float64")
            out_df[prob_col] = pd.Series(dtype="float64")
            return ScoredCandidatesResult(
                scored_candidates=out_df,
                model_name=self.model_name,
                model_path=self.model_path,
                num_candidates=0,
            )

        # Strictly extract only the 11 predictive feature columns in exact order as float32
        X = out_df[self.feature_columns].astype("float32")
        proba_matrix = np.asarray(self.estimator.predict_proba(X), dtype=np.float64)

        if proba_matrix.ndim != 2 or proba_matrix.shape[0] != len(out_df) or proba_matrix.shape[1] < 2:
            raise ValueError(
                f"Invalid predict_proba output shape {proba_matrix.shape} for {len(out_df)} candidates"
            )

        pos_probs = proba_matrix[:, 1]
        if not np.all(np.isfinite(pos_probs)):
            raise ValueError("Model returned non-finite (NaN/Inf) candidate probabilities")
        if np.any((pos_probs < -1e-6) | (pos_probs > 1.0 + 1e-6)):
            raise ValueError("Model returned probabilities outside [0.0, 1.0]")

        pos_probs = np.clip(pos_probs, 0.0, 1.0)
        out_df["predicted_probability"] = pos_probs
        out_df[prob_col] = pos_probs

        return ScoredCandidatesResult(
            scored_candidates=out_df,
            model_name=self.model_name,
            model_path=self.model_path,
            num_candidates=len(out_df),
        )
