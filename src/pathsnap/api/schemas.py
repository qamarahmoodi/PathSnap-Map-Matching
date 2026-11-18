"""Pydantic request and response schemas for PathSnap FastAPI backend (Laptop 1).

Owner: Qamar Ahmed (Technical Lead)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="API status ('ok' when ready, 'degraded' if model or data root missing)")
    ready: bool = Field(..., description="True only when both trained model and dataset root are valid and loaded")
    model_loaded: bool
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    data_root_available: bool
    active_partition: Optional[str] = None
    permitted_trip_count: int = 0
    error: Optional[str] = None


class TripOption(BaseModel):
    trip_id: str = Field(..., description="8-digit zero-padded trip identifier (e.g. '00000000')")
    display_label: str = Field(..., description="Safe human-readable display label for the UI dropdown")
    partition: str = Field(..., description="Partition name resolved on server")


class TripsListResponse(BaseModel):
    status: str = "ok"
    partition: str
    count: int
    trips: List[TripOption]


class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="Selected 8-digit trip_id from GET /trips")
    include_observation_records: bool = Field(
        default=True,
        description="Whether to include per-observation prediction records in the response",
    )


class ObservationPredictionRecord(BaseModel):
    observation_id: int
    timestamp_s: float
    predicted_arc_id: int
    classifier_only_arc_id: int
    decoded_score: Optional[float] = None
    classifier_only_score: Optional[float] = None
    decoder_status: str
    prediction_status: str
    gap_status: str
    transition_hops: int
    is_connected_step: bool


class PredictResponse(BaseModel):
    status: str = "ok"
    trip_id: str
    partition: str
    model_name: str
    predicted_route_arc_ids: List[int]
    classifier_only_route_arc_ids: List[int]
    predicted_route_segments_latlon: List[List[List[float]]]
    classifier_only_route_segments_latlon: List[List[List[float]]]
    gps_trajectory_latlon: List[List[float]]
    candidate_summary: Dict[str, Any]
    decoder_summary: Dict[str, Any]
    evaluation_metrics: Dict[str, Any]
    reference_evaluated_on_server: bool
    processing_time_ms: Dict[str, float]
    observation_predictions: List[ObservationPredictionRecord] = Field(default_factory=list)
