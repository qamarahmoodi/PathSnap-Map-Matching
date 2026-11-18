"""Schemas, dataclasses, and formal data contracts for PathSnap candidate generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from pathsnap.graph.road_graph import TripRoadGraph


# Required columns and target pandas dtypes for candidates.parquet
CANDIDATE_SCHEMA_DTYPES: Dict[str, str] = {
    "trip_id": "string",
    "partition": "string",
    "observation_id": "int32",
    "timestamp_s": "float64",
    "candidate_rank": "int16",
    "candidate_arc_id": "int32",
    "from_node_id": "int32",
    "to_node_id": "int32",
    "distance_to_segment_m": "float32",
    "projection_fraction": "float32",
    "distance_to_start_m": "float32",
    "distance_to_end_m": "float32",
    "segment_length_m": "float32",
    "segment_bearing_deg": "float32",
    "gps_bearing_deg": "Float32",
    "bearing_difference_deg": "Float32",
    "gps_speed_mps": "Float32",
    "delta_time_s": "Float32",
    "is_fallback_candidate": "bool",
    "label": "Int8",
}

# Required columns and target pandas dtypes for observations.parquet
OBSERVATION_SCHEMA_DTYPES: Dict[str, str] = {
    "trip_id": "string",
    "partition": "string",
    "observation_id": "int32",
    "timestamp_s": "float64",
    "gps_longitude": "float64",
    "gps_latitude": "float64",
    "reference_arc_id": "Int32",
    "reference_route_position": "Int32",
    "alignment_distance_m": "Float32",
    "alignment_status": "string",
    "candidate_count": "int16",
    "candidate_miss": "boolean",
    "label_eligible": "bool",
}

# Explicit separation of feature columns vs identifier/metadata/target columns for Pranav's ML pipeline
CANDIDATE_ID_COLUMNS: List[str] = [
    "trip_id",
    "partition",
    "observation_id",
    "timestamp_s",
    "candidate_rank",
    "candidate_arc_id",
    "from_node_id",
    "to_node_id",
]

CANDIDATE_FEATURE_COLUMNS: List[str] = [
    "distance_to_segment_m",
    "projection_fraction",
    "distance_to_start_m",
    "distance_to_end_m",
    "segment_length_m",
    "segment_bearing_deg",
    "gps_bearing_deg",
    "bearing_difference_deg",
    "gps_speed_mps",
    "delta_time_s",
    "is_fallback_candidate",
]

CANDIDATE_TARGET_COLUMN: str = "label"

VALID_ALIGNMENT_STATUSES = frozenset({"aligned", "ambiguous", "unmatched", "not_available"})


@dataclass
class CandidateConfig:
    """Configuration for spatial projection, candidate search, feature engineering, and route alignment."""

    # Candidate generation parameters
    search_radius_m: float = 100.0
    max_candidates_k: int = 10
    enable_fallback_nearest: bool = True
    fallback_max_radius_m: float = 500.0

    # Feature engineering parameters
    min_displacement_for_bearing_m: float = 2.5
    max_valid_delta_time_s: float = 30.0
    max_plausible_speed_mps: float = 65.0  # ~234 km/h upper sanity bound

    # Reference route alignment parameters (dynamic programming)
    alignment_emission_sigma_m: float = 15.0
    alignment_bearing_weight: float = 5.0
    alignment_skip_penalty_per_arc: float = 1.5
    alignment_length_mismatch_weight: float = 0.25
    alignment_max_confident_distance_m: float = 50.0
    alignment_max_unmatched_distance_m: float = 120.0
    alignment_ambiguity_margin_m: float = 3.0


@dataclass
class CandidateGenerationResult:
    """Container returned by generate_trip_candidates(...)."""

    trip_id: str
    partition: str
    candidates: pd.DataFrame
    observations: pd.DataFrame
    graph: TripRoadGraph


@dataclass
class ReferenceAlignment:
    """Container returned by align_reference_route(...)."""

    trip_id: str
    observation_id: np.ndarray          # int32, shape (N_obs,)
    reference_arc_id: np.ndarray        # int32, shape (N_obs,), -1 when not aligned
    reference_route_position: np.ndarray  # int32, shape (N_obs,), -1 when not aligned
    alignment_distance_m: np.ndarray    # float32, shape (N_obs,), NaN when not available
    second_best_distance_m: np.ndarray  # float32, shape (N_obs,), distance to distinct competing route arc
    alignment_status: List[str]         # length N_obs, values in VALID_ALIGNMENT_STATUSES
    route_arc_ids: np.ndarray           # int32, shape (M_route,)
    diagnostics: Dict[str, float] = field(default_factory=dict)


def enforce_candidate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and cast a candidate DataFrame to the exact contract schema."""
    if df.empty:
        cols = {col: pd.Series(dtype=dtype) for col, dtype in CANDIDATE_SCHEMA_DTYPES.items()}
        return pd.DataFrame(cols)

    out = pd.DataFrame(index=df.index)
    for col, dtype in CANDIDATE_SCHEMA_DTYPES.items():
        if col not in df.columns:
            raise ValueError(f"Missing required candidate column: '{col}'")
        out[col] = df[col].astype(dtype)

    # Check uniqueness of (trip_id, observation_id, candidate_arc_id)
    dup_mask = out.duplicated(subset=["trip_id", "observation_id", "candidate_arc_id"])
    if dup_mask.any():
        raise ValueError(
            f"Duplicate (trip_id, observation_id, candidate_arc_id) rows detected: {int(dup_mask.sum())} duplicates"
        )
    return out


def enforce_observation_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and cast an observation DataFrame to the exact contract schema."""
    if df.empty:
        cols = {col: pd.Series(dtype=dtype) for col, dtype in OBSERVATION_SCHEMA_DTYPES.items()}
        return pd.DataFrame(cols)

    out = pd.DataFrame(index=df.index)
    for col, dtype in OBSERVATION_SCHEMA_DTYPES.items():
        if col not in df.columns:
            raise ValueError(f"Missing required observation column: '{col}'")
        out[col] = df[col].astype(dtype)

    invalid_statuses = set(out["alignment_status"].dropna().unique()) - VALID_ALIGNMENT_STATUSES
    if invalid_statuses:
        raise ValueError(f"Invalid alignment_status values: {invalid_statuses}")

    return out
