"""Tests for CandidateModelScorer, run_trip_inference, and isolated server-side evaluation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pathsnap.inference.inference_pipeline import run_trip_inference
from pathsnap.inference.model_scorer import CandidateModelScorer, resolve_model_path
from pathsnap.inference.server_evaluator import evaluate_trip_predictions_server_side
from pathsnap.schemas.candidate_schema import CANDIDATE_FEATURE_COLUMNS


class _InvalidEstimatorNoPredictProba:
    def predict(self, X):
        return np.zeros(len(X))


class _OutOfBoundsProbaEstimator:
    n_features_in_ = 11

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), -0.5), np.full(len(X), 1.5)])


def test_model_scorer_loads_actual_pranav_artifact_and_validates_schema():
    model_path = resolve_model_path()
    assert model_path.is_file()

    scorer = CandidateModelScorer.from_path(model_path)
    assert scorer.feature_columns == CANDIDATE_FEATURE_COLUMNS

    # Reject estimator without predict_proba
    with pytest.raises(TypeError, match="predict_proba"):
        CandidateModelScorer(_InvalidEstimatorNoPredictProba())


def test_model_scorer_rejects_missing_columns_and_invalid_probabilities():
    scorer = CandidateModelScorer.from_path()

    # Missing feature column
    bad_df = pd.DataFrame(
        {
            "trip_id": ["00000001"],
            "observation_id": [0],
            "candidate_arc_id": [10],
            "candidate_rank": [1],
        }
    )
    with pytest.raises(ValueError, match="missing required feature columns"):
        scorer.score_candidates(bad_df)

    # Out-of-bounds probabilities rejected
    bad_scorer = CandidateModelScorer(_OutOfBoundsProbaEstimator())
    valid_df = pd.DataFrame(
        {
            "trip_id": ["00000001"],
            "observation_id": np.array([0], dtype=np.int32),
            "candidate_arc_id": np.array([10], dtype=np.int32),
            "candidate_rank": np.array([1], dtype=np.int16),
            **{col: np.array([1.0], dtype=np.float32) for col in CANDIDATE_FEATURE_COLUMNS},
        }
    )
    with pytest.raises(ValueError, match="outside"):
        bad_scorer.score_candidates(valid_df)


def test_run_trip_inference_without_route_and_isolated_server_evaluation(monkeypatch):
    """Prove that run_trip_inference never calls load_route_file and works on actual student_test trip."""
    import pathsnap.data.loader as loader_mod

    route_called = {"count": 0}
    orig_load_route = loader_mod.load_route_file

    def _spy_load_route(path: Path):
        route_called["count"] += 1
        return orig_load_route(path)

    monkeypatch.setattr(loader_mod, "load_route_file", _spy_load_route)

    # 1. Execute route-free inference on student_test trip 00000007
    inf_out = run_trip_inference(trip_id="00000007", partition="student_test")
    assert route_called["count"] == 0, "run_trip_inference MUST NOT read .route!"

    assert inf_out.trip_id == "00000007"
    assert inf_out.partition == "student_test"
    assert len(inf_out.classifier_only_predictions) == inf_out.candidate_summary["num_observations"]
    assert len(inf_out.hmm_viterbi_predictions) == inf_out.candidate_summary["num_observations"]
    assert len(inf_out.predicted_route_arc_ids) > 0
    assert len(inf_out.predicted_route_segments_latlon) == len(inf_out.predicted_route_arc_ids)
    assert inf_out.timing_ms["total_pipeline_ms"] > 0.0

    # Verify required Gokul handoff columns in classifier_only_predictions & hmm_viterbi_predictions
    assert {
        "trip_id",
        "observation_id",
        "predicted_arc_id",
        "model_score",
        "prediction_status",
    }.issubset(inf_out.classifier_only_predictions.columns)

    assert {
        "trip_id",
        "observation_id",
        "predicted_arc_id",
        "decoded_score",
        "decoder_status",
        "gap_status",
    }.issubset(inf_out.hmm_viterbi_predictions.columns)

    # 2. Execute isolated server-side evaluation AFTER inference
    eval_res = evaluate_trip_predictions_server_side(inf_out)
    assert eval_res.reference_available is True
    assert "hmm_viterbi_top1_accuracy" in eval_res.metrics
    assert "classifier_only_top1_accuracy" in eval_res.metrics
    assert "hmm_viterbi_route_continuity" in eval_res.metrics
    # Ensure no raw reference route sequence or observation labels exist in eval_res.metrics
    for key in eval_res.metrics:
        assert "reference_route" not in key
        assert "ground_truth" not in key
        assert not isinstance(eval_res.metrics[key], (list, np.ndarray))
