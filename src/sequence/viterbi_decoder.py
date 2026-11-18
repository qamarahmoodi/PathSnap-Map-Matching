"""HMM and Viterbi sequential route reconstruction for PathSnap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from pathsnap.graph.road_graph import TripRoadGraph


@dataclass
class ViterbiConfig:
    """Configuration for HMM log-emission and topological log-transition scoring."""

    prob_floor: float = 1e-6
    emission_weight: float = 1.0
    same_arc_log_prob: float = -0.15          # Staying on the same road segment (hop == 0)
    direct_successor_log_prob: float = -0.60  # Moving to a directly connected outgoing arc (hop == 1)
    two_hop_log_prob: float = -2.20           # Skipping 1 intermediate arc across a 1s step (hop == 2)
    three_hop_log_prob: float = -3.80         # Skipping 2 intermediate arcs (hop == 3)
    disconnected_log_prob: float = -10.50     # Disconnected / invalid road-network transition (hop == -1)
    reverse_arc_penalty: float = -4.00        # Additional penalty for immediate U-turn onto reverse segment
    large_gap_threshold_s: float = 10.0       # Beyond this dt_s, relax transition penalty toward uniform
    reset_gap_threshold_s: float = 60.0       # Beyond this dt_s, treat transition as uniform reset
    max_hops: int = 3


@dataclass
class ViterbiDecodeResult:
    """Container for per-observation and route-level Viterbi decoding outputs."""

    trip_id: str
    observation_predictions: pd.DataFrame
    reconstructed_route_arcs: np.ndarray
    greedy_route_arcs: np.ndarray
    total_log_likelihood: float
    diagnostics: Dict[str, float]


def compute_log_transition_matrix(
    prev_arcs: np.ndarray,
    curr_arcs: np.ndarray,
    graph: TripRoadGraph,
    dt_s: float,
    cfg: ViterbiConfig,
) -> np.ndarray:
    """Compute shape (K_prev, K_curr) log-transition matrix between consecutive candidate sets."""
    k_prev = len(prev_arcs)
    k_curr = len(curr_arcs)
    trans = np.full((k_prev, k_curr), cfg.disconnected_log_prob, dtype=np.float64)

    if not np.isfinite(dt_s) or dt_s <= 0.0:
        dt_s = 1.0

    # If GPS dropout exceeds reset_gap_threshold_s, vehicle could have travelled anywhere: uniform transition
    if dt_s >= cfg.reset_gap_threshold_s:
        trans.fill(-np.log(max(k_curr, 1)))
        return trans

    # Relaxation factor for moderate GPS timestamp gaps (10s .. 60s)
    gap_relax = 1.0
    if dt_s > cfg.large_gap_threshold_s:
        gap_relax = max(0.25, 1.0 - (dt_s - cfg.large_gap_threshold_s) / cfg.reset_gap_threshold_s)

    endpoints = graph.arc_endpoints
    for i in range(k_prev):
        u_arc = int(prev_arcs[i])
        u_from, u_to = int(endpoints[u_arc, 0]), int(endpoints[u_arc, 1])
        for j in range(k_curr):
            v_arc = int(curr_arcs[j])
            if u_arc == v_arc:
                base_lp = cfg.same_arc_log_prob
            else:
                v_from, v_to = int(endpoints[v_arc, 0]), int(endpoints[v_arc, 1])
                if u_to == v_from:
                    base_lp = cfg.direct_successor_log_prob
                    # Penalize immediate U-turn (u_from -> u_to followed by u_to -> u_from)
                    if v_to == u_from:
                        base_lp += cfg.reverse_arc_penalty
                else:
                    hops = graph.shortest_hop_transition_type(u_arc, v_arc, max_hops=cfg.max_hops)
                    if hops == 2:
                        base_lp = cfg.two_hop_log_prob
                    elif hops == 3:
                        base_lp = cfg.three_hop_log_prob
                    else:
                        base_lp = cfg.disconnected_log_prob
            trans[i, j] = base_lp * gap_relax

    return trans


def decode_trip_viterbi(
    candidates_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    graph: TripRoadGraph,
    prob_col: str = "pred_prob",
    config: Optional[ViterbiConfig] = None,
) -> ViterbiDecodeResult:
    """Decode a trip's candidate emission probabilities into a globally coherent road-segment sequence.

    Handles:
      - Normal observations with 1..K candidates
      - Empty candidate sets (observations with 0 candidates: carries forward previous state or -1)
      - Disconnected transitions (via backoff log-probability `disconnected_log_prob`)
      - Large GPS timestamp gaps (via `large_gap_threshold_s` and `reset_gap_threshold_s`)
    """
    cfg = config or ViterbiConfig()
    obs_sorted = observations_df.sort_values("observation_id").reset_index(drop=True)
    n_obs = len(obs_sorted)
    trip_id = str(obs_sorted["trip_id"].iloc[0]) if n_obs > 0 else graph.trip_id

    if n_obs == 0:
        empty_df = pd.DataFrame(
            columns=[
                "trip_id",
                "observation_id",
                "timestamp_s",
                "greedy_arc_id",
                "viterbi_arc_id",
                "greedy_prob",
                "viterbi_prob",
                "transition_hops",
                "is_connected_step",
            ]
        )
        return ViterbiDecodeResult(
            trip_id=trip_id,
            observation_predictions=empty_df,
            reconstructed_route_arcs=np.empty(0, dtype=np.int32),
            greedy_route_arcs=np.empty(0, dtype=np.int32),
            total_log_likelihood=0.0,
            diagnostics={},
        )

    # Group candidates by observation_id
    cands_sorted = candidates_df.sort_values(["observation_id", "candidate_rank"]).reset_index(drop=True)
    obs_ids_arr = obs_sorted["observation_id"].to_numpy(dtype=np.int32)
    timestamps = obs_sorted["timestamp_s"].to_numpy(dtype=np.float64)

    # Build per-observation candidate arrays
    cand_arcs_by_step: List[np.ndarray] = []
    cand_probs_by_step: List[np.ndarray] = []
    cand_dists_by_step: List[np.ndarray] = []

    grouped = {
        int(oid): grp
        for oid, grp in cands_sorted.groupby("observation_id", sort=False)
    }

    for oid in obs_ids_arr:
        grp = grouped.get(int(oid))
        if grp is None or grp.empty:
            cand_arcs_by_step.append(np.empty(0, dtype=np.int32))
            cand_probs_by_step.append(np.empty(0, dtype=np.float64))
            cand_dists_by_step.append(np.empty(0, dtype=np.float64))
        else:
            cand_arcs_by_step.append(grp["candidate_arc_id"].to_numpy(dtype=np.int32))
            probs = np.clip(
                grp[prob_col].to_numpy(dtype=np.float64),
                cfg.prob_floor,
                1.0 - cfg.prob_floor,
            )
            cand_probs_by_step.append(probs)
            cand_dists_by_step.append(grp["distance_to_segment_m"].to_numpy(dtype=np.float64))

    # Viterbi forward recursion over steps with non-empty candidates
    dp_scores: List[np.ndarray] = []
    backpointers: List[np.ndarray] = []
    last_valid_step: int = -1

    for t in range(n_obs):
        arcs_t = cand_arcs_by_step[t]
        probs_t = cand_probs_by_step[t]
        k_t = len(arcs_t)

        if k_t == 0:
            dp_scores.append(np.empty(0, dtype=np.float64))
            backpointers.append(np.empty(0, dtype=np.int32))
            continue

        log_emit = cfg.emission_weight * np.log(probs_t)

        if last_valid_step < 0:
            dp_scores.append(log_emit)
            backpointers.append(np.full(k_t, -1, dtype=np.int32))
            last_valid_step = t
            continue

        prev_arcs = cand_arcs_by_step[last_valid_step]
        prev_dp = dp_scores[last_valid_step]
        dt_s = float(timestamps[t] - timestamps[last_valid_step])

        log_trans = compute_log_transition_matrix(prev_arcs, arcs_t, graph, dt_s, cfg)
        # total_score[i, j] = prev_dp[i] + log_trans[i, j]
        score_matrix = prev_dp[:, None] + log_trans
        best_prev_idx = np.argmax(score_matrix, axis=0).astype(np.int32)
        best_prev_score = score_matrix[best_prev_idx, np.arange(k_t)]

        dp_scores.append(best_prev_score + log_emit)
        backpointers.append(best_prev_idx)
        last_valid_step = t

    # Backtracking across valid steps
    valid_steps = [t for t in range(n_obs) if len(cand_arcs_by_step[t]) > 0]
    viterbi_arc_ids = np.full(n_obs, -1, dtype=np.int32)
    viterbi_probs = np.full(n_obs, np.nan, dtype=np.float64)
    greedy_arc_ids = np.full(n_obs, -1, dtype=np.int32)
    greedy_probs = np.full(n_obs, np.nan, dtype=np.float64)

    total_ll = 0.0
    if valid_steps:
        last_t = valid_steps[-1]
        best_state = int(np.argmax(dp_scores[last_t]))
        total_ll = float(dp_scores[last_t][best_state])

        chosen_state_by_valid_pos = [0] * len(valid_steps)
        chosen_state_by_valid_pos[-1] = best_state
        for pos in range(len(valid_steps) - 1, 0, -1):
            t_curr = valid_steps[pos]
            chosen_state_by_valid_pos[pos - 1] = int(backpointers[t_curr][chosen_state_by_valid_pos[pos]])

        for pos, t in enumerate(valid_steps):
            st = chosen_state_by_valid_pos[pos]
            viterbi_arc_ids[t] = int(cand_arcs_by_step[t][st])
            viterbi_probs[t] = float(cand_probs_by_step[t][st])

            # Greedy top-1: break ties by shortest distance_to_segment_m
            probs_t = cand_probs_by_step[t]
            dists_t = cand_dists_by_step[t]
            order = np.lexsort((dists_t, -probs_t))
            g_idx = int(order[0])
            greedy_arc_ids[t] = int(cand_arcs_by_step[t][g_idx])
            greedy_probs[t] = float(probs_t[g_idx])

        # Fill any empty-candidate observations by carrying forward the previous decoded arc
        for t in range(n_obs):
            if viterbi_arc_ids[t] == -1 and t > 0:
                viterbi_arc_ids[t] = viterbi_arc_ids[t - 1]
            if greedy_arc_ids[t] == -1 and t > 0:
                greedy_arc_ids[t] = greedy_arc_ids[t - 1]

    # Compute transition hops and route continuity for both Greedy and Viterbi
    viterbi_hops = np.zeros(n_obs, dtype=np.int32)
    viterbi_connected = np.ones(n_obs, dtype=bool)
    greedy_connected = np.ones(n_obs, dtype=bool)

    for t in range(1, n_obs):
        u_v, v_v = int(viterbi_arc_ids[t - 1]), int(viterbi_arc_ids[t])
        if u_v >= 0 and v_v >= 0:
            viterbi_hops[t] = graph.shortest_hop_transition_type(u_v, v_v, max_hops=cfg.max_hops)
            viterbi_connected[t] = graph.are_connected(u_v, v_v)
        else:
            viterbi_hops[t] = -1
            viterbi_connected[t] = False

        u_g, v_g = int(greedy_arc_ids[t - 1]), int(greedy_arc_ids[t])
        if u_g >= 0 and v_g >= 0:
            greedy_connected[t] = graph.are_connected(u_g, v_g)
        else:
            greedy_connected[t] = False

    obs_pred_df = pd.DataFrame(
        {
            "trip_id": obs_sorted["trip_id"].to_numpy(),
            "observation_id": obs_ids_arr,
            "timestamp_s": timestamps,
            "greedy_arc_id": greedy_arc_ids,
            "viterbi_arc_id": viterbi_arc_ids,
            "greedy_prob": greedy_probs,
            "viterbi_prob": viterbi_probs,
            "transition_hops": viterbi_hops,
            "is_connected_step": viterbi_connected,
            "is_greedy_connected_step": greedy_connected,
        }
    )

    # Deduplicate consecutive identical arcs to form chronological route sequence
    def _compress_route(arc_seq: np.ndarray) -> np.ndarray:
        valid = arc_seq[arc_seq >= 0]
        if len(valid) == 0:
            return np.empty(0, dtype=np.int32)
        keep = np.ones(len(valid), dtype=bool)
        keep[1:] = valid[1:] != valid[:-1]
        return valid[keep].astype(np.int32)

    viterbi_route = _compress_route(viterbi_arc_ids)
    greedy_route = _compress_route(greedy_arc_ids)

    viterbi_cont = float(np.mean(viterbi_connected[1:])) if n_obs > 1 else 1.0
    greedy_cont = float(np.mean(greedy_connected[1:])) if n_obs > 1 else 1.0
    multi_hop_reachable = float(np.mean(viterbi_hops[1:] >= 0)) if n_obs > 1 else 1.0

    diagnostics: Dict[str, float] = {
        "num_observations": float(n_obs),
        "empty_candidate_observations": float(n_obs - len(valid_steps)),
        "greedy_route_continuity": round(greedy_cont, 6),
        "viterbi_route_continuity": round(viterbi_cont, 6),
        "viterbi_multihop_reachability": round(multi_hop_reachable, 6),
        "changed_observations_count": float(np.sum(greedy_arc_ids != viterbi_arc_ids)),
    }

    # If reference_arc_id and alignment_status are present, compute aligned Top-1 accuracy for both
    if "alignment_status" in obs_sorted.columns and "reference_arc_id" in obs_sorted.columns:
        aligned_mask = obs_sorted["alignment_status"].to_numpy() == "aligned"
        if np.any(aligned_mask):
            ref_arcs = obs_sorted.loc[aligned_mask, "reference_arc_id"].to_numpy(dtype=np.int32)
            g_acc = float(np.mean(greedy_arc_ids[aligned_mask] == ref_arcs))
            v_acc = float(np.mean(viterbi_arc_ids[aligned_mask] == ref_arcs))
            diagnostics["greedy_top1_accuracy"] = round(g_acc, 6)
            diagnostics["viterbi_top1_accuracy"] = round(v_acc, 6)

    return ViterbiDecodeResult(
        trip_id=trip_id,
        observation_predictions=obs_pred_df,
        reconstructed_route_arcs=viterbi_route,
        greedy_route_arcs=greedy_route,
        total_log_likelihood=total_ll,
        diagnostics=diagnostics,
    )
