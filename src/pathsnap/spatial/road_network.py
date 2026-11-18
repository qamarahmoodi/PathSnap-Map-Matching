"""Road-network geometry construction, spatial indexing, and vectorised point-to-segment metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import shapely
from shapely import STRtree

from pathsnap.graph.road_graph import TripRoadGraph, build_trip_road_graph
from pathsnap.spatial.projection import LocalTangentProjector, compute_bearing_deg


@dataclass
class TripRoadNetwork:
    """Metric road network with spatial index for fast candidate retrieval."""

    trip_id: str
    projector: LocalTangentProjector
    nodes_lon_lat: np.ndarray      # float64, shape (num_nodes, 2)
    nodes_xy_m: np.ndarray         # float64, shape (num_nodes, 2)
    arcs: np.ndarray               # int32, shape (num_arcs, 2) [from_node_id, to_node_id]
    start_xy_m: np.ndarray         # float64, shape (num_arcs, 2)
    end_xy_m: np.ndarray           # float64, shape (num_arcs, 2)
    seg_vec_xy_m: np.ndarray       # float64, shape (num_arcs, 2) = end_xy_m - start_xy_m
    segment_lengths_m: np.ndarray  # float32, shape (num_arcs,)
    segment_bearings_deg: np.ndarray  # float32, shape (num_arcs,)
    strtree: STRtree
    geometries: np.ndarray         # 1D array of shapely LineStrings in metric coordinates
    graph: TripRoadGraph


def build_trip_road_network(
    trip_id: str,
    nodes: np.ndarray,
    arcs: np.ndarray,
    projector: Optional[LocalTangentProjector] = None,
) -> TripRoadNetwork:
    """Construct projected road segments, lengths, bearings, STRtree index, and TripRoadGraph."""
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    arcs_arr = np.asarray(arcs, dtype=np.int32)
    num_nodes = nodes_arr.shape[0]

    if ((arcs_arr < 0) | (arcs_arr >= num_nodes)).any():
        raise ValueError(f"Trip {trip_id}: invalid node IDs in arcs (num_nodes={num_nodes})")

    if projector is None:
        projector = LocalTangentProjector.from_lon_lat(nodes_arr[:, 0], nodes_arr[:, 1])

    nodes_xy_m = projector.project_xy(nodes_arr)
    start_xy_m = nodes_xy_m[arcs_arr[:, 0]]
    end_xy_m = nodes_xy_m[arcs_arr[:, 1]]
    seg_vec_xy_m = end_xy_m - start_xy_m

    lengths_f64 = np.hypot(seg_vec_xy_m[:, 0], seg_vec_xy_m[:, 1])
    bearings_f64 = compute_bearing_deg(seg_vec_xy_m[:, 0], seg_vec_xy_m[:, 1])

    segment_lengths_m = lengths_f64.astype(np.float32)
    segment_bearings_deg = bearings_f64.astype(np.float32)

    # Create vectorised Shapely 2.0 LineStrings in metric coordinates for STRtree
    coords = np.stack([start_xy_m, end_xy_m], axis=1)  # shape (num_arcs, 2, 2)
    geometries = shapely.linestrings(coords)
    tree = STRtree(geometries)

    graph = build_trip_road_graph(
        trip_id=str(trip_id),
        arcs=arcs_arr,
        arc_lengths_m=segment_lengths_m,
        arc_bearings_deg=segment_bearings_deg,
    )

    return TripRoadNetwork(
        trip_id=str(trip_id),
        projector=projector,
        nodes_lon_lat=nodes_arr,
        nodes_xy_m=nodes_xy_m,
        arcs=arcs_arr,
        start_xy_m=start_xy_m,
        end_xy_m=end_xy_m,
        seg_vec_xy_m=seg_vec_xy_m,
        segment_lengths_m=segment_lengths_m,
        segment_bearings_deg=segment_bearings_deg,
        strtree=tree,
        geometries=geometries,
        graph=graph,
    )


def compute_point_to_segments_geometry(
    pts_xy_m: np.ndarray,
    start_xy_m: np.ndarray,
    end_xy_m: np.ndarray,
    seg_vec_xy_m: np.ndarray,
    seg_len_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised exact Euclidean point-to-segment projection in metres.

    Args:
        pts_xy_m: shape (..., 2)
        start_xy_m: shape (..., 2)
        end_xy_m: shape (..., 2)
        seg_vec_xy_m: shape (..., 2) = end_xy_m - start_xy_m
        seg_len_m: shape (...,)

    Returns:
        distance_to_segment_m: float32, perpendicular/clamped shortest distance from point to segment
        projection_fraction: float32 in [0, 1], normalized position along directed segment
        distance_to_start_m: float32, Euclidean distance from GPS point to segment start node
        distance_to_end_m: float32, Euclidean distance from GPS point to segment end node
    """
    w_xy = pts_xy_m - start_xy_m
    len_sq = seg_len_m.astype(np.float64) ** 2

    # Dot product w . v
    dot = w_xy[..., 0] * seg_vec_xy_m[..., 0] + w_xy[..., 1] * seg_vec_xy_m[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(len_sq > 1e-12, dot / len_sq, 0.0)
    frac_clamped = np.clip(frac, 0.0, 1.0)

    proj_x = start_xy_m[..., 0] + frac_clamped * seg_vec_xy_m[..., 0]
    proj_y = start_xy_m[..., 1] + frac_clamped * seg_vec_xy_m[..., 1]

    dist_seg = np.hypot(pts_xy_m[..., 0] - proj_x, pts_xy_m[..., 1] - proj_y)
    dist_start = np.hypot(w_xy[..., 0], w_xy[..., 1])
    dist_end = np.hypot(pts_xy_m[..., 0] - end_xy_m[..., 0], pts_xy_m[..., 1] - end_xy_m[..., 1])

    return (
        dist_seg.astype(np.float32),
        frac_clamped.astype(np.float32),
        dist_start.astype(np.float32),
        dist_end.astype(np.float32),
    )
