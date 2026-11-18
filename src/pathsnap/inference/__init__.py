"""PathSnap production inference and isolated server-side evaluation package."""

from pathsnap.inference.inference_pipeline import (
    TripInferenceOutput,
    arcs_to_segments_latlon,
    run_trip_inference,
)
from pathsnap.inference.model_scorer import (
    CandidateModelScorer,
    ScoredCandidatesResult,
    resolve_model_path,
)
from pathsnap.inference.server_evaluator import (
    ServerEvaluationResult,
    compute_sequence_overlap_metrics,
    evaluate_trip_predictions_server_side,
)

__all__ = [
    "CandidateModelScorer",
    "ScoredCandidatesResult",
    "ServerEvaluationResult",
    "TripInferenceOutput",
    "arcs_to_segments_latlon",
    "compute_sequence_overlap_metrics",
    "evaluate_trip_predictions_server_side",
    "resolve_model_path",
    "run_trip_inference",
]
