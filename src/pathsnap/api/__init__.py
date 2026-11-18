"""FastAPI backend package for PathSnap Laptop 1 deployment."""

from pathsnap.api.app import app, create_app
from pathsnap.api.schemas import (
    HealthResponse,
    ObservationPredictionRecord,
    PredictRequest,
    PredictResponse,
    TripOption,
    TripsListResponse,
)

__all__ = [
    "HealthResponse",
    "ObservationPredictionRecord",
    "PredictRequest",
    "PredictResponse",
    "TripOption",
    "TripsListResponse",
    "app",
    "create_app",
]
