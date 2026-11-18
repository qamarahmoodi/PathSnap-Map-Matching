"""Spatial projection, road network construction, and candidate generation."""
from pathsnap.spatial.projection import LocalTangentProjector, angular_difference_deg, compute_bearing_deg
from pathsnap.spatial.road_network import (
    TripRoadNetwork,
    build_trip_road_network,
    compute_point_to_segments_geometry,
)

__all__ = [
    "LocalTangentProjector",
    "TripRoadNetwork",
    "angular_difference_deg",
    "build_trip_road_network",
    "compute_bearing_deg",
    "compute_point_to_segments_geometry",
]
