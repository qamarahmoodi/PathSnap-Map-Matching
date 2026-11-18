"""PathSnap Map-Matching and Candidate Generation package."""

from pathsnap.schemas.candidate_schema import (
    CANDIDATE_FEATURE_COLUMNS,
    CANDIDATE_ID_COLUMNS,
    CANDIDATE_SCHEMA_DTYPES,
    CANDIDATE_TARGET_COLUMN,
    OBSERVATION_SCHEMA_DTYPES,
    CandidateConfig,
    CandidateGenerationResult,
    ReferenceAlignment,
    enforce_candidate_schema,
    enforce_observation_schema,
)
from pathsnap.graph.road_graph import TripRoadGraph, build_trip_road_graph
from pathsnap.data.loader import (
    TripRawData,
    list_trips,
    load_arcs_file,
    load_nodes_file,
    load_route_file,
    load_track_file,
    load_trip,
    resolve_data_root,
    validate_trip_arrays,
)
from pathsnap.features.feature_engineering import (
    GPSKinematics,
    compute_candidate_bearing_difference,
    compute_gps_kinematics,
)
from pathsnap.spatial.road_network import TripRoadNetwork, build_trip_road_network
from pathsnap.spatial.candidate_generation import generate_trip_candidates
from pathsnap.alignment.route_alignment import align_reference_route
from pathsnap.alignment.label_generation import (
    LabelSummary,
    attach_alignment_labels,
    summarize_trip_labels,
)

__all__ = [
    "CANDIDATE_FEATURE_COLUMNS",
    "CANDIDATE_ID_COLUMNS",
    "CANDIDATE_SCHEMA_DTYPES",
    "CANDIDATE_TARGET_COLUMN",
    "OBSERVATION_SCHEMA_DTYPES",
    "CandidateConfig",
    "CandidateGenerationResult",
    "LabelSummary",
    "ReferenceAlignment",
    "TripRawData",
    "TripRoadGraph",
    "TripRoadNetwork",
    "align_reference_route",
    "attach_alignment_labels",
    "build_trip_road_graph",
    "build_trip_road_network",
    "enforce_candidate_schema",
    "enforce_observation_schema",
    "generate_trip_candidates",
    "list_trips",
    "load_arcs_file",
    "load_nodes_file",
    "load_route_file",
    "load_track_file",
    "load_trip",
    "resolve_data_root",
    "summarize_trip_labels",
    "validate_trip_arrays",
]
