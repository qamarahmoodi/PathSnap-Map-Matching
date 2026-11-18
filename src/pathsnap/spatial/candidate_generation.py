"""Spatial candidate generation for GPS trajectories using Shapely STRtree."""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import shapely

from pathsnap.features.feature_engineering import (
    compute_candidate_bearing_difference,
    compute_gps_kinematics,
)
from pathsnap.schemas.candidate_schema import (
    CandidateConfig,
    CandidateGenerationResult,
    enforce_candidate_schema,
    enforce_observation_schema,
)
from pathsnap.spatial.road_network import (
    TripRoadNetwork,
    build_trip_road_network,
    compute_point_to_segments_geometry,
)


def generate_trip_candidates(
    trip_id: str,
    partition: str,
    track: pd.DataFrame,
    nodes: np.ndarray,
    arcs: np.ndarray,
    config: Optional[CandidateConfig] = None,
    road_network: Optional[TripRoadNetwork] = None,
) -> CandidateGenerationResult:
    """Generate top-K candidate road segments and spatial/kinematic features for a trip.

    Strictly operates WITHOUT `.route` (safe for both training and blind evaluation inference).
    """
    if config is None:
        config = CandidateConfig()

    trip_str = str(trip_id)
    part_str = str(partition)

    if road_network is None:
        road_network = build_trip_road_network(trip_id=trip_str, nodes=nodes, arcs=arcs)

    lon = track["longitude"].to_numpy(dtype=np.float64)
    lat = track["latitude"].to_numpy(dtype=np.float64)
    timestamps_s = track["timestamp_s"].to_numpy(dtype=np.float64)
    obs_ids = (
        track["observation_id"].to_numpy(dtype=np.int32)
        if "observation_id" in track.columns
        else np.arange(len(track), dtype=np.int32)
    )
    n_obs = len(track)

    pts_xy_m = road_network.projector.project_xy(np.column_stack([lon, lat]))
    kinematics = compute_gps_kinematics(pts_xy_m, timestamps_s, config)

    # Query STRtree within config.search_radius_m
    shapely_pts = shapely.points(pts_xy_m)
    pairs = road_network.strtree.query(
        shapely_pts,
        predicate="dwithin",
        distance=float(config.search_radius_m),
    )
    obs_idx_raw = pairs[0].astype(np.int32)
    arc_idx_raw = pairs[1].astype(np.int32)
    is_fallback_raw = np.zeros(obs_idx_raw.shape[0], dtype=bool)

    # Identify observations with 0 candidates within search_radius_m
    counts_within = np.bincount(obs_idx_raw, minlength=n_obs)
    missing_obs = np.flatnonzero(counts_within == 0).astype(np.int32)

    if missing_obs.size > 0 and config.enable_fallback_nearest and road_network.arcs.shape[0] > 0:
        nearest_res = road_network.strtree.query_nearest(
            shapely_pts[missing_obs],
            max_distance=float(config.fallback_max_radius_m),
            all_matches=False,
        )
        if nearest_res.shape[1] > 0:
            fb_obs = missing_obs[nearest_res[0].astype(np.int32)]
            fb_arc = nearest_res[1].astype(np.int32)
            obs_idx_raw = np.concatenate([obs_idx_raw, fb_obs])
            arc_idx_raw = np.concatenate([arc_idx_raw, fb_arc])
            is_fallback_raw = np.concatenate(
                [is_fallback_raw, np.ones(fb_obs.shape[0], dtype=bool)]
            )

    if obs_idx_raw.size == 0:
        empty_cands = enforce_candidate_schema(pd.DataFrame())
        obs_df = enforce_observation_schema(
            pd.DataFrame(
                {
                    "trip_id": [trip_str] * n_obs,
                    "partition": [part_str] * n_obs,
                    "observation_id": obs_ids,
                    "timestamp_s": timestamps_s,
                    "gps_longitude": lon,
                    "gps_latitude": lat,
                    "reference_arc_id": pd.Series([pd.NA] * n_obs, dtype="Int32"),
                    "reference_route_position": pd.Series([pd.NA] * n_obs, dtype="Int32"),
                    "alignment_distance_m": pd.Series([np.nan] * n_obs, dtype="Float32"),
                    "alignment_status": ["not_available"] * n_obs,
                    "candidate_count": np.zeros(n_obs, dtype=np.int16),
                    "candidate_miss": pd.Series([pd.NA] * n_obs, dtype="boolean"),
                    "label_eligible": np.zeros(n_obs, dtype=bool),
                }
            )
        )
        return CandidateGenerationResult(
            trip_id=trip_str,
            partition=part_str,
            candidates=empty_cands,
            observations=obs_df,
            graph=road_network.graph,
        )

    # Deduplicate (obs_idx, arc_idx) pairs if any exist
    pair_keys = obs_idx_raw.astype(np.int64) * int(road_network.arcs.shape[0] + 1) + arc_idx_raw.astype(np.int64)
    _, unique_indices = np.unique(pair_keys, return_index=True)
    obs_idx = obs_idx_raw[unique_indices]
    arc_idx = arc_idx_raw[unique_indices]
    is_fallback = is_fallback_raw[unique_indices]

    # Compute exact point-to-segment distances and projections
    dist_seg_m, proj_frac, dist_start_m, dist_end_m = compute_point_to_segments_geometry(
        pts_xy_m=pts_xy_m[obs_idx],
        start_xy_m=road_network.start_xy_m[arc_idx],
        end_xy_m=road_network.end_xy_m[arc_idx],
        seg_vec_xy_m=road_network.seg_vec_xy_m[arc_idx],
        seg_len_m=road_network.segment_lengths_m[arc_idx],
    )

    # Sort by (obs_idx ASC, distance_to_segment_m ASC, arc_idx ASC) for deterministic ranking
    order = np.lexsort((arc_idx, dist_seg_m, obs_idx))
    obs_idx = obs_idx[order]
    arc_idx = arc_idx[order]
    is_fallback = is_fallback[order]
    dist_seg_m = dist_seg_m[order]
    proj_frac = proj_frac[order]
    dist_start_m = dist_start_m[order]
    dist_end_m = dist_end_m[order]

    # Compute within-observation 1-based candidate rank
    unique_obs, first_idx, obs_counts = np.unique(
        obs_idx, return_index=True, return_counts=True
    )
    group_starts = np.repeat(first_idx, obs_counts)
    ranks_0based = np.arange(obs_idx.size, dtype=np.int32) - group_starts

    # Filter to top K candidates per observation
    keep_mask = ranks_0based < int(config.max_candidates_k)
    obs_idx = obs_idx[keep_mask]
    arc_idx = arc_idx[keep_mask]
    is_fallback = is_fallback[keep_mask]
    dist_seg_m = dist_seg_m[keep_mask]
    proj_frac = proj_frac[keep_mask]
    dist_start_m = dist_start_m[keep_mask]
    dist_end_m = dist_end_m[keep_mask]
    candidate_rank = (ranks_0based[keep_mask] + 1).astype(np.int16)

    # Assemble candidate features
    seg_len_m = road_network.segment_lengths_m[arc_idx]
    seg_brg_deg = road_network.segment_bearings_deg[arc_idx]
    gps_brg_deg = kinematics.gps_bearing_deg[obs_idx]
    brg_diff_deg = compute_candidate_bearing_difference(gps_brg_deg, seg_brg_deg)
    gps_spd_mps = kinematics.gps_speed_mps[obs_idx]
    dt_s = kinematics.delta_time_s[obs_idx]

    cand_df = pd.DataFrame(
        {
            "trip_id": pd.Series([trip_str] * obs_idx.size, dtype="string"),
            "partition": pd.Series([part_str] * obs_idx.size, dtype="string"),
            "observation_id": obs_ids[obs_idx],
            "timestamp_s": timestamps_s[obs_idx],
            "candidate_rank": candidate_rank,
            "candidate_arc_id": arc_idx.astype(np.int32),
            "from_node_id": road_network.arcs[arc_idx, 0].astype(np.int32),
            "to_node_id": road_network.arcs[arc_idx, 1].astype(np.int32),
            "distance_to_segment_m": dist_seg_m,
            "projection_fraction": proj_frac,
            "distance_to_start_m": dist_start_m,
            "distance_to_end_m": dist_end_m,
            "segment_length_m": seg_len_m,
            "segment_bearing_deg": seg_brg_deg,
            "gps_bearing_deg": pd.Series(gps_brg_deg, dtype="Float32"),
            "bearing_difference_deg": pd.Series(brg_diff_deg, dtype="Float32"),
            "gps_speed_mps": pd.Series(gps_spd_mps, dtype="Float32"),
            "delta_time_s": pd.Series(dt_s, dtype="Float32"),
            "is_fallback_candidate": is_fallback,
            "label": pd.Series([pd.NA] * obs_idx.size, dtype="Int8"),
        }
    )
    cand_df = enforce_candidate_schema(cand_df)

    # Assemble observation metadata table (all observations preserved)
    final_cand_counts = np.bincount(obs_idx, minlength=n_obs).astype(np.int16)
    obs_df = pd.DataFrame(
        {
            "trip_id": pd.Series([trip_str] * n_obs, dtype="string"),
            "partition": pd.Series([part_str] * n_obs, dtype="string"),
            "observation_id": obs_ids,
            "timestamp_s": timestamps_s,
            "gps_longitude": lon,
            "gps_latitude": lat,
            "reference_arc_id": pd.Series([pd.NA] * n_obs, dtype="Int32"),
            "reference_route_position": pd.Series([pd.NA] * n_obs, dtype="Int32"),
            "alignment_distance_m": pd.Series([np.nan] * n_obs, dtype="Float32"),
            "alignment_status": pd.Series(["not_available"] * n_obs, dtype="string"),
            "candidate_count": final_cand_counts,
            "candidate_miss": pd.Series([pd.NA] * n_obs, dtype="boolean"),
            "label_eligible": np.zeros(n_obs, dtype=bool),
        }
    )
    obs_df = enforce_observation_schema(obs_df)

    return CandidateGenerationResult(
        trip_id=trip_str,
        partition=part_str,
        candidates=cand_df,
        observations=obs_df,
        graph=road_network.graph,
    )
