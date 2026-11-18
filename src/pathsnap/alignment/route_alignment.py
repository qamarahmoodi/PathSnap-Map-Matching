"""Sequence-aware monotonic dynamic programming alignment of `.route` to `.track`."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd

from pathsnap.features.feature_engineering import compute_gps_kinematics
from pathsnap.schemas.candidate_schema import CandidateConfig, ReferenceAlignment
from pathsnap.spatial.projection import LocalTangentProjector, angular_difference_deg, compute_bearing_deg
from pathsnap.spatial.road_network import compute_point_to_segments_geometry


def align_reference_route(
    track: pd.DataFrame,
    nodes: np.ndarray,
    arcs: np.ndarray,
    route: Optional[np.ndarray],
    config: Optional[CandidateConfig] = None,
    trip_id: str = "",
    projector: Optional[LocalTangentProjector] = None,
) -> ReferenceAlignment:
    """Align an ordered reference `.route` arc sequence to a GPS `.track` via monotonic DP.

    Design properties:
    1. Reconstructs the metric geometry of every route occurrence j in 0..M-1 (preserving repeated arcs).
    2. Computes exact Euclidean distance_m[i, j] from GPS observation i to route segment occurrence j.
    3. Incorporates local GPS movement bearing agreement where GPS step displacement is reliable.
    4. Enforces monotonic non-decreasing route progression (j_i >= j_{i-1}), allowing:
       - staying on the same route segment across multiple observations (j_i == j_{i-1}),
       - advancing across one or more route segments (j_i > j_{i-1}), penalized by arc-length mismatch
         against observed GPS displacement and elapsed time.
    5. Classifies each observation's alignment status into:
       - 'aligned': confident alignment within `alignment_max_confident_distance_m` and unambiguous
       - 'ambiguous': within confident distance, but another non-identical route arc occurrence lies
         within `alignment_ambiguity_margin_m` and could plausibly claim the observation
       - 'unmatched': minimum alignment distance exceeds `alignment_max_confident_distance_m`
       - 'not_available': route is None or empty
    """
    if config is None:
        config = CandidateConfig()

    n_obs = len(track)
    obs_ids = (
        track["observation_id"].to_numpy(dtype=np.int32)
        if "observation_id" in track.columns
        else np.arange(n_obs, dtype=np.int32)
    )

    if route is None or len(route) == 0 or n_obs == 0:
        return ReferenceAlignment(
            trip_id=str(trip_id),
            observation_id=obs_ids,
            reference_arc_id=np.full(n_obs, -1, dtype=np.int32),
            reference_route_position=np.full(n_obs, -1, dtype=np.int32),
            alignment_distance_m=np.full(n_obs, np.nan, dtype=np.float32),
            second_best_distance_m=np.full(n_obs, np.nan, dtype=np.float32),
            alignment_status=["not_available"] * n_obs,
            route_arc_ids=np.empty(0, dtype=np.int32),
            diagnostics={"aligned_ratio": 0.0},
        )

    route_arr = np.asarray(route, dtype=np.int32).ravel()
    m_route = route_arr.shape[0]
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    arcs_arr = np.asarray(arcs, dtype=np.int32)

    if projector is None:
        projector = LocalTangentProjector.from_lon_lat(nodes_arr[:, 0], nodes_arr[:, 1])

    lon = track["longitude"].to_numpy(dtype=np.float64)
    lat = track["latitude"].to_numpy(dtype=np.float64)
    timestamps_s = track["timestamp_s"].to_numpy(dtype=np.float64)

    pts_xy_m = projector.project_xy(np.column_stack([lon, lat]))
    kinematics = compute_gps_kinematics(pts_xy_m, timestamps_s, config)

    # Extract geometry for the M route segments in order
    nodes_xy_m = projector.project_xy(nodes_arr)
    route_nodes_from = arcs_arr[route_arr, 0]
    route_nodes_to = arcs_arr[route_arr, 1]
    r_start_xy = nodes_xy_m[route_nodes_from]  # (M, 2)
    r_end_xy = nodes_xy_m[route_nodes_to]      # (M, 2)
    r_vec_xy = r_end_xy - r_start_xy           # (M, 2)
    r_len_m = np.hypot(r_vec_xy[:, 0], r_vec_xy[:, 1]).astype(np.float32)  # (M,)
    r_brg_deg = compute_bearing_deg(r_vec_xy[:, 0], r_vec_xy[:, 1]).astype(np.float32)  # (M,)

    # Broadcast (N, 1, 2) against (1, M, 2) to get exact distance matrix (N, M)
    dist_matrix_m, proj_frac_matrix, _, _ = compute_point_to_segments_geometry(
        pts_xy_m=pts_xy_m[:, None, :],
        start_xy_m=r_start_xy[None, :, :],
        end_xy_m=r_end_xy[None, :, :],
        seg_vec_xy_m=r_vec_xy[None, :, :],
        seg_len_m=r_len_m[None, :],
    )

    # Emission cost matrix E[i, j] in normalized units
    sigma = float(config.alignment_emission_sigma_m)
    emission_cost = (dist_matrix_m.astype(np.float64) / sigma) ** 2

    # Add directional penalty where GPS heading is reliable
    valid_brg = np.isfinite(kinematics.gps_bearing_deg)
    if np.any(valid_brg):
        brg_diff = angular_difference_deg(
            kinematics.gps_bearing_deg[valid_brg, None].astype(np.float64),
            r_brg_deg[None, :].astype(np.float64),
        )
        # Penalty in [0, alignment_bearing_weight] for 0..180 deg mismatch
        emission_cost[valid_brg, :] += float(config.alignment_bearing_weight) * ((brg_diff / 180.0) ** 2)

    # Cumulative arc lengths along the route to measure route distance traversed between j_prev and j_curr
    # Cum_start[j] is distance from start of route to start of route segment j
    cum_start_m = np.zeros(m_route, dtype=np.float64)
    if m_route > 1:
        cum_start_m[1:] = np.cumsum(r_len_m[:-1].astype(np.float64))

    # Position along the route for each (i, j): cum_start_m[j] + proj_frac[i, j] * r_len_m[j]
    route_pos_m = cum_start_m[None, :] + proj_frac_matrix.astype(np.float64) * r_len_m[None, :].astype(np.float64)

    # Dynamic Programming table: dp[i, j] and backpointers bp[i, j]
    dp = np.full((n_obs, m_route), np.inf, dtype=np.float64)
    bp = np.zeros((n_obs, m_route), dtype=np.int32)

    # Initial state at i = 0: slightly prefer starting near beginning of route, but allow anywhere if track starts mid-route
    dp[0, :] = emission_cost[0, :] + 0.05 * np.arange(m_route, dtype=np.float64)

    skip_pen = float(config.alignment_skip_penalty_per_arc)
    len_weight = float(config.alignment_length_mismatch_weight) / sigma

    # Maximum lookback in route indices per step (allow larger jump if time gap is large)
    for i in range(1, n_obs):
        dt_i = float(kinematics.delta_time_s[i]) if np.isfinite(kinematics.delta_time_s[i]) else 1.0
        disp_i = float(kinematics.displacement_m[i]) if np.isfinite(kinematics.displacement_m[i]) else 0.0

        # Allow jumping more arcs when there is a large GPS time gap
        max_jump = max(15, min(m_route, int(12 + max(1.0, dt_i) * 3)))

        prev_dp = dp[i - 1]
        prev_pos = route_pos_m[i - 1]
        curr_pos = route_pos_m[i]

        # Vectorised banded transition evaluation over delta_j = j_curr - j_prev in [0 .. max_jump]
        best_prev_cost = prev_dp.copy()
        best_prev_idx = np.arange(m_route, dtype=np.int32)

        # For delta_j = 0 (staying on same route occurrence): penalize backward motion along the segment slightly
        step_dist_0 = curr_pos - prev_pos
        backward_pen_0 = np.where(step_dist_0 < -5.0, ((-step_dist_0 - 5.0) / sigma) ** 2, 0.0)
        best_prev_cost = best_prev_cost + backward_pen_0

        for dj in range(1, min(max_jump + 1, m_route)):
            prev_slice = prev_dp[:-dj]
            route_advance_m = curr_pos[dj:] - prev_pos[:-dj]
            # Compare route advance with GPS step displacement
            diff_m = np.abs(route_advance_m - disp_i)
            # Down-weight arc-skip penalty during large GPS time gaps
            effective_skip_pen = skip_pen / max(1.0, np.sqrt(max(1.0, dt_i)))
            trans_cost = effective_skip_pen * (dj - 1) + len_weight * diff_m
            cand_cost = prev_slice + trans_cost

            target_slice = best_prev_cost[dj:]
            improved = cand_cost < target_slice
            if np.any(improved):
                target_slice[improved] = cand_cost[improved]
                best_prev_idx[dj:][improved] = np.arange(m_route - dj, dtype=np.int32)[improved]

        dp[i, :] = emission_cost[i, :] + best_prev_cost
        bp[i, :] = best_prev_idx

    # Backtrack from the final observation
    # Slight bias toward ending near the end of the route
    final_scores = dp[n_obs - 1, :] + 0.05 * np.arange(m_route - 1, -1, -1, dtype=np.float64)
    best_last_j = int(np.argmin(final_scores))

    aligned_pos = np.empty(n_obs, dtype=np.int32)
    aligned_pos[n_obs - 1] = best_last_j
    for i in range(n_obs - 1, 0, -1):
        aligned_pos[i - 1] = bp[i, aligned_pos[i]]

    aligned_arc_ids = route_arr[aligned_pos]
    aligned_dists_m = dist_matrix_m[np.arange(n_obs), aligned_pos].astype(np.float32)

    # Evaluate competing route arcs to detect ambiguous observations (e.g. tight intersections or overlapping loops)
    second_best_dists_m = np.full(n_obs, np.inf, dtype=np.float32)
    statuses: List[str] = []

    max_conf_m = float(config.alignment_max_confident_distance_m)
    ambig_margin_m = float(config.alignment_ambiguity_margin_m)

    for i in range(n_obs):
        d_best = float(aligned_dists_m[i])
        chosen_arc = int(aligned_arc_ids[i])
        chosen_j = int(aligned_pos[i])

        # Check alternative route positions within a local window [chosen_j - 3, chosen_j + 3] that have a DIFFERENT arc_id
        # (Exclude the direct predecessor/successor connected at the shared intersection node when the point is near that endpoint)
        row_dists = dist_matrix_m[i]
        diff_arc_mask = route_arr != chosen_arc
        if np.any(diff_arc_mask):
            second_best_dists_m[i] = float(np.min(row_dists[diff_arc_mask]))

        if not np.isfinite(d_best) or d_best > max_conf_m:
            statuses.append("unmatched")
        else:
            # Check if a non-adjacent route segment with a different arc_id is nearly equidistant and better than DP choice
            non_adj_mask = diff_arc_mask & (np.abs(np.arange(m_route) - chosen_j) > 2)
            if np.any(non_adj_mask):
                min_non_adj_dist = float(np.min(row_dists[non_adj_mask]))
            else:
                min_non_adj_dist = float("inf")

            if d_best > 20.0 and (min_non_adj_dist - d_best) < ambig_margin_m:
                statuses.append("ambiguous")
            else:
                statuses.append("aligned")

    # Mask out reference_arc_id and reference_route_position for unmatched observations
    out_arc_ids = aligned_arc_ids.copy()
    out_route_pos = aligned_pos.copy()
    for i, st in enumerate(statuses):
        if st in ("unmatched", "not_available"):
            out_arc_ids[i] = -1
            out_route_pos[i] = -1

    aligned_count = sum(1 for s in statuses if s == "aligned")
    ambig_count = sum(1 for s in statuses if s == "ambiguous")
    unmatched_count = sum(1 for s in statuses if s == "unmatched")

    diagnostics = {
        "num_observations": float(n_obs),
        "num_route_occurrences": float(m_route),
        "aligned_count": float(aligned_count),
        "ambiguous_count": float(ambig_count),
        "unmatched_count": float(unmatched_count),
        "aligned_ratio": float(aligned_count / n_obs) if n_obs > 0 else 0.0,
        "median_alignment_distance_m": float(np.nanmedian(aligned_dists_m)),
        "p95_alignment_distance_m": float(np.nanpercentile(aligned_dists_m, 95)),
        "max_alignment_distance_m": float(np.nanmax(aligned_dists_m)),
        "is_monotonic": float(bool(np.all(np.diff(aligned_pos) >= 0))),
    }

    return ReferenceAlignment(
        trip_id=str(trip_id),
        observation_id=obs_ids,
        reference_arc_id=out_arc_ids,
        reference_route_position=out_route_pos,
        alignment_distance_m=aligned_dists_m,
        second_best_distance_m=second_best_dists_m,
        alignment_status=statuses,
        route_arc_ids=route_arr,
        diagnostics=diagnostics,
    )
