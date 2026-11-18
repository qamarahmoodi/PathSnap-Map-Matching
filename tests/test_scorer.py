"""Tests for the reusable model loader and candidate scoring interface."""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from models.inference import compute_file_sha256, load_model_bundle, score_candidates
from models.model_pipeline import get_five_models
from models.scorer import LoadedModelBundle, load_trained_model
from pathsnap.schemas.candidate_schema import CANDIDATE_FEATURE_COLUMNS


def test_score_candidates_and_load_trained_model(tmp_path: Path):
    preview_cands = Path(__file__).resolve().parents[1] / "outputs" / "candidate_previews" / "candidates.parquet"
    df = pd.read_parquet(preview_cands)
    sub = df[df["trip_id"] == "00000001"].copy()
    lab = sub[sub["label"].notna()].copy()

    # Fit a quick Logistic Regression pipeline on a subset
    _, _, est = get_five_models(random_state=42)["logistic_regression"]
    X = lab[CANDIDATE_FEATURE_COLUMNS].astype("float32")
    y = lab["label"].astype("int8").to_numpy()
    est.fit(X, y)

    art_path = tmp_path / "test_model.joblib"
    meta_path = tmp_path / "test_meta.json"
    joblib.dump(
        {
            "estimator": est,
            "metadata": {
                "model_name": "Logistic Regression",
                "model_key": "logistic_regression",
                "algorithm_family": "Linear & Probabilistic",
                "feature_columns": list(CANDIDATE_FEATURE_COLUMNS),
                "positive_class": 1,
            },
        },
        art_path,
    )

    bundle = load_trained_model(artifact_path=art_path, metadata_path=meta_path)
    assert isinstance(bundle, LoadedModelBundle)
    assert bundle.feature_columns == list(CANDIDATE_FEATURE_COLUMNS)
    assert bundle.positive_class == 1

    scored = score_candidates(sub, model_or_bundle=bundle)
    assert len(scored) == len(sub)
    for col in [
        "trip_id",
        "observation_id",
        "candidate_arc_id",
        "arc_id",
        "candidate_rank",
        "predicted_probability",
        "pred_prob",
        "predicted_log_probability",
        "predicted_label",
        "is_obs_top1",
    ]:
        assert col in scored.columns

    # Exact candidate-to-arc correspondence preserved
    assert (scored["arc_id"] == sub["candidate_arc_id"]).all()
    assert (scored["observation_id"] == sub["observation_id"]).all()
    assert ((scored["predicted_probability"] > 0.0) & (scored["predicted_probability"] < 1.0)).all()
    assert np.allclose(np.log(scored["predicted_probability"]), scored["predicted_log_probability"])

    # Exactly 1 top-1 candidate per observation
    top1_counts = scored.groupby(["trip_id", "observation_id"])["is_obs_top1"].sum()
    assert (top1_counts == 1).all()

    # Test load_model_bundle and compute_file_sha256 from models.inference
    b2 = load_model_bundle(model_path=art_path, metadata_path=meta_path)
    assert b2.model_key == "logistic_regression"
    assert len(compute_file_sha256(art_path)) == 64

    # Test empty candidate DataFrame handling
    empty_scored = score_candidates(sub.iloc[0:0], model_bundle=b2)
    assert len(empty_scored) == 0
    assert "predicted_probability" in empty_scored.columns
