"""HMM and Viterbi sequential route reconstruction for PathSnap.

Owner: Qamar Ahmed (Technical Lead)

Hidden-State Definition:
  At each GPS observation step t (0 .. N_obs-1), the hidden state space S_t is the set of
  generated candidate road segments C_t = {c_1, ..., c_{K_t}}, each preserving its exact
  trip-specific `candidate_arc_id`, `candidate_rank`, and `predicted_probability`.

Emission Model:
  Primary input is the trained classifier's posterior probability P(y_{t,c}=1 | x_{t,c}).
  Probabilities are validated to lie in [0, 1], optionally normalized within each observation's
  candidate set (`normalize_emissions_within_obs`), clipped to [prob_floor, 1 - prob_floor],
  and converted to log-emissions: log_emit = emission_weight * log(p_{t,c}).

Transition Model:
  Uses the directed road graph (`TripRoadGraph`) and time step dt_s = t_curr - t_prev:
  - 0 hops (same directed arc u == v): `same_arc_log_prob`
  - 1 hop (direct outgoing successor to_node(u) == from_node(v)): `direct_successor_log_prob`
    (with an additional `reverse_arc_penalty` if v_to == u_from, i.e., immediate U-turn)
  - 2 hops (1 intermediate connector arc skipped): `two_hop_log_prob`
  - 3 hops (2 intermediate arcs skipped): `three_hop_log_prob`
  - Disconnected (unreachable within max_hops): `disconnected_log_prob`
  - Physical movement consistency: when `length_mismatch_penalty_per_m > 0` and GPS step displacement
    is available, penalizes large discrepancies between road progression and GPS movement.
  - GPS Gap Handling:
    * Normal step (dt_s <= large_gap_threshold_s): full topological transition weights (`gap_status="normal"`).
    * Moderate gap (large_gap_threshold_s < dt_s < reset_gap_threshold_s): linearly relaxes transition
      penalties (`gap_status="moderate_gap"`).
    * Reset gap (dt_s >= reset_gap_threshold_s): uniform transition log-probability across candidates
      so the sequence resets cleanly after long signal dropouts (`gap_status="reset_gap"`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from pathsnap.graph.road_graph import TripRoadGraph


@dataclass
class ViterbiConfig:
    """Configuration for HMM log-emission and topological log-transition scoring."""

    prob_floor: float = 1e-6
    emission_weight: float = 1.0
    normalize_emissions_within_obs: bool = False
    same_arc_log_prob: float = -0.15          # Staying on the same road segment (hop == 0)
    direct_successor_log_prob: float = -0.60  # Moving to a directly connected outgoing arc (hop == 1)
    two_hop_log_prob: float = -2.20           # Skipping 1 intermediate arc across a 1s step (hop == 2)
    three_hop_log_prob: float = -3.80         # Skipping 2 intermediate arcs (hop == 3)
    disconnected_log_prob: float = -10.50     # Disconnected / invalid road-network transition (hop == -1)
    reverse_arc_penalty: float = -4.00        # Additional penalty for immediate U-turn onto reverse segment
    length_mismatch_penalty_per_m: float = 0.0
    large_gap_threshold_s: float = 10.0       # Beyond this dt_s, relax transition penalty toward uniform
    reset_gap_threshold_s: float = 60.0       # Beyond this dt_s, treat transition as uniform reset
    max_hops: int = 3

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ViterbiConfig":
        """Load ViterbiConfig from `viterbi_decoder` section of a YAML configuration file."""
        path = Path(yaml_path)
        if not path.is_file():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        sec = data.get("viterbi_decoder", {})
        if not isinstance(sec, dict):
            return cls()
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in sec.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class ViterbiDecodeResult:
    """Container for per-observation and route-level Viterbi decoding outputs."""

    trip_id: str
    observation_predictions: pd.DataFrame
    reconstructed_route_arcs: np.ndarray
    greedy_route_arcs: np.ndarray
    total_log_likelihood: float
    diagnostics: Dict[str, float]


def classify_gap_status(dt_s: float, cfg: ViterbiConfig, is_first_step: bool = False) -> str:
    """Classify the temporal step into 'initial', 'normal', 'moderate_gap', or 'reset_gap'."""
    if is_first_step:
        return "initial"
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        return "normal"
    if dt_s >= cfg.reset_gap_threshold_s:
        return "reset_gap"
    if dt_s > cfg.large_gap_threshold_s:
        return "moderate_gap"
    return "normal"


def compute_log_transition_matrix(
    prev_arcs: np.ndarray,
    curr_arcs: np.ndarray,
    graph: TripRoadGraph,
    dt_s: float,
    cfg: ViterbiConfig,
    prev_proj_m: Optional[np.ndarray] = None,
    curr_proj_m: Optional[np.ndarray] = None,
    gps_disp_m: Optional[float] = None,
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
    arc_lengths = graph.arc_lengths_m

    for i in range(k_prev):
        u_arc = int(prev_arcs[i])
        u_from, u_to = int(endpoints[u_arc, 0]), int(endpoints[u_arc, 1])
        u_len = float(arc_lengths[u_arc])
        for j in range(k_curr):
            v_arc = int(curr_arcs[j])
            if u_arc == v_arc:
                base_lp = cfg.same_arc_log_prob
                # Optional movement consistency check along the same directed segment
                if (
                    cfg.length_mismatch_penalty_per_m > 0.0
                    and prev_proj_m is not None
                    and curr_proj_m is not None
                    and gps_disp_m is not None
                    and np.isfinite(gps_disp_m)
                ):
                    prog_m = float(curr_proj_m[j] - prev_proj_m[i])
                    if prog_m < -5.0:
                        base_lp -= cfg.length_mismatch_penalty_per_m * abs(prog_m)
            else:
                v_from, v_to = int(endpoints[v_arc, 0]), int(endpoints[v_arc, 1])
                if u_to == v_from:
                    base_lp = cfg.direct_successor_log_prob
                    # Penalize immediate U-turn (u_from -> u_to followed by u_to -> u_from)
                    if v_to == u_from:
                        base_lp += cfg.reverse_arc_penalty
                    elif (
                        cfg.length_mismatch_penalty_per_m > 0.0
                        and prev_proj_m is not None
                        and curr_proj_m is not None
                        and gps_disp_m is not None
                        and np.isfinite(gps_disp_m)
                    ):
                        prog_m = max(0.0, u_len - float(prev_proj_m[i])) + float(curr_proj_m[j])
                        base_lp -= cfg.length_mismatch_penalty_per_m * abs(prog_m - gps_disp_m)
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
      - Fallback-only candidate observations (`decoder_status="fallback_decoded"`)
      - Empty candidate sets (observations with 0 candidates: carries forward previous state with
        `decoder_status="carried_forward"`, or `decoder_status="unmatched"` if no valid state exists)
      - Disconnected transitions (via backoff log-probability `disconnected_log_prob`, flagged as
        `decoder_status="disconnected_bridge"`)
      - Large GPS timestamp gaps (`decoder_status="gap_reset"` when `dt_s >= reset_gap_threshold_s`)
      - Invariant to arbitrary row ordering of `candidates_df` (sorts deterministically by
        `(observation_id, candidate_rank, candidate_arc_id)`).
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
                "is_greedy_connected_step",
                "prediction_status",
                "decoder_status",
                "gap_status",
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

    obs_ids_arr = obs_sorted["observation_id"].to_numpy(dtype=np.int32)
    timestamps = obs_sorted["timestamp_s"].to_numpy(dtype=np.float64)

    # Sort candidates deterministically so permuted input rows never change output
    if not candidates_df.empty:
        if prob_col not in candidates_df.columns:
            raise ValueError(f"Missing required probability column '{prob_col}' in candidates_df")
        sort_cols = ["observation_id"]
        if "candidate_rank" in candidates_df.columns:
            sort_cols.append("candidate_rank")
        if "candidate_arc_id" in candidates_df.columns:
            sort_cols.append("candidate_arc_id")
        cands_sorted = candidates_df.sort_values(sort_cols).reset_index(drop=True)
        grouped = {
            int(oid): grp
            for oid, grp in cands_sorted.groupby("observation_id", sort=False)
        }
    else:
        grouped = {}

    cand_arcs_by_step: List[np.ndarray] = []
    cand_probs_by_step: List[np.ndarray] = []
    cand_raw_probs_by_step: List[np.ndarray] = []
    cand_dists_by_step: List[np.ndarray] = []
    cand_fallback_by_step: List[np.ndarray] = []
    cand_proj_m_by_step: List[np.ndarray] = []
    step_gps_disp_m: np.ndarray = np.full(n_obs, np.nan, dtype=np.float64)

    for idx_t, oid in enumerate(obs_ids_arr):
        grp = grouped.get(int(oid))
        if grp is None or grp.empty:
            cand_arcs_by_step.append(np.empty(0, dtype=np.int32))
            cand_probs_by_step.append(np.empty(0, dtype=np.float64))
            cand_raw_probs_by_step.append(np.empty(0, dtype=np.float64))
            cand_dists_by_step.append(np.empty(0, dtype=np.float64))
            cand_fallback_by_step.append(np.empty(0, dtype=bool))
            cand_proj_m_by_step.append(np.empty(0, dtype=np.float64))
        else:
            arcs_arr = grp["candidate_arc_id"].to_numpy(dtype=np.int32)
            if ((arcs_arr < 0) | (arcs_arr >= graph.num_arcs)).any():
                raise ValueError(
                    f"Trip {trip_id} obs {oid}: candidate_arc_id out of bounds for graph with {graph.num_arcs} arcs"
                )
            raw_p = grp[prob_col].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(raw_p)):
                raise ValueError(f"Trip {trip_id} obs {oid}: non-finite emission probabilities detected")
            if np.any((raw_p < -1e-6) | (raw_p > 1.0 + 1e-6)):
                raise ValueError(f"Trip {trip_id} obs {oid}: emission probabilities outside [0, 1]")

            clipped_p = np.clip(raw_p, cfg.prob_floor, 1.0 - cfg.prob_floor)
            if cfg.normalize_emissions_within_obs and len(clipped_p) > 0:
                norm_p = clipped_p / np.sum(clipped_p)
                clipped_p = np.clip(norm_p, cfg.prob_floor, 1.0 - cfg.prob_floor)

            dists = (
                grp["distance_to_segment_m"].to_numpy(dtype=np.float64)
                if "distance_to_segment_m" in grp.columns
                else np.zeros(len(arcs_arr), dtype=np.float64)
            )
            fb = (
                grp["is_fallback_candidate"].to_numpy(dtype=bool)
                if "is_fallback_candidate" in grp.columns
                else np.zeros(len(arcs_arr), dtype=bool)
            )
            if "projection_fraction" in grp.columns and "segment_length_m" in grp.columns:
                proj_m = (
                    grp["projection_fraction"].to_numpy(dtype=np.float64)
                    * grp["segment_length_m"].to_numpy(dtype=np.float64)
                )
            else:
                proj_m = np.zeros(len(arcs_arr), dtype=np.float64)

            if "gps_speed_mps" in grp.columns and "delta_time_s" in grp.columns:
                spd_val = grp["gps_speed_mps"].iloc[0]
                dt_val = grp["delta_time_s"].iloc[0]
                if pd.notna(spd_val) and pd.notna(dt_val):
                    step_gps_disp_m[idx_t] = float(spd_val) * float(dt_val)

            cand_arcs_by_step.append(arcs_arr)
            cand_probs_by_step.append(clipped_p)
            cand_raw_probs_by_step.append(np.clip(raw_p, 0.0, 1.0))
            cand_dists_by_step.append(dists)
            cand_fallback_by_step.append(fb)
            cand_proj_m_by_step.append(proj_m)

    # Viterbi forward recursion over steps with non-empty candidates
    dp_scores: List[np.ndarray] = []
    backpointers: List[np.ndarray] = []
    last_valid_step: int = -1

    for t in range(n_obs):
        arcs_t = cand_arcs_by_step[t]
        probs_t = cand_probs_by_step[t]
        dists_t = cand_dists_by_step[t]
        k_t = len(arcs_t)

        if k_t == 0:
            dp_scores.append(np.empty(0, dtype=np.float64))
            backpointers.append(np.empty(0, dtype=np.int32))
            continue

        # Tiny deterministic tie-breaker on shortest distance (1e-9 per metre) when probabilities are identical
        log_emit = cfg.emission_weight * np.log(probs_t) - 1e-9 * dists_t

        if last_valid_step < 0:
            dp_scores.append(log_emit)
            backpointers.append(np.full(k_t, -1, dtype=np.int32))
            last_valid_step = t
            continue

        prev_arcs = cand_arcs_by_step[last_valid_step]
        prev_dp = dp_scores[last_valid_step]
        dt_s = float(timestamps[t] - timestamps[last_valid_step])

        log_trans = compute_log_transition_matrix(
            prev_arcs=prev_arcs,
            curr_arcs=arcs_t,
            graph=graph,
            dt_s=dt_s,
            cfg=cfg,
            prev_proj_m=cand_proj_m_by_step[last_valid_step],
            curr_proj_m=cand_proj_m_by_step[t],
            gps_disp_m=float(step_gps_disp_m[t]),
        )
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
    chosen_is_fallback = np.zeros(n_obs, dtype=bool)
    greedy_is_fallback = np.zeros(n_obs, dtype=bool)

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
            viterbi_probs[t] = float(cand_raw_probs_by_step[t][st])
            chosen_is_fallback[t] = bool(cand_fallback_by_step[t][st])

            # Greedy top-1: break ties by shortest distance_to_segment_m, then lowest arc_id
            raw_probs_t = cand_raw_probs_by_step[t]
            dists_t = cand_dists_by_step[t]
            arcs_t = cand_arcs_by_step[t]
            order = np.lexsort((arcs_t, dists_t, -raw_probs_t))
            g_idx = int(order[0])
            greedy_arc_ids[t] = int(arcs_t[g_idx])
            greedy_probs[t] = float(raw_probs_t[g_idx])
            greedy_is_fallback[t] = bool(cand_fallback_by_step[t][g_idx])

        # Fill any empty-candidate observations by carrying forward the previous decoded arc
        for t in range(n_obs):
            if viterbi_arc_ids[t] == -1 and t > 0:
                viterbi_arc_ids[t] = viterbi_arc_ids[t - 1]
            if greedy_arc_ids[t] == -1 and t > 0:
                greedy_arc_ids[t] = greedy_arc_ids[t - 1]

    # Compute transition hops, route continuity, and explicit statuses per observation
    viterbi_hops = np.zeros(n_obs, dtype=np.int32)
    viterbi_connected = np.ones(n_obs, dtype=bool)
    greedy_connected = np.ones(n_obs, dtype=bool)
    prediction_statuses: List[str] = []
    decoder_statuses: List[str] = []
    gap_statuses: List[str] = []

    for t in range(n_obs):
        dt_step = float(timestamps[t] - timestamps[t - 1]) if t > 0 else np.nan
        g_stat = classify_gap_status(dt_step, cfg, is_first_step=(t == 0))
        gap_statuses.append(g_stat)

        if t > 0:
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
        else:
            if viterbi_arc_ids[0] < 0:
                viterbi_hops[0] = -1
                viterbi_connected[0] = False
                greedy_connected[0] = False

        # Classifier-only prediction_status
        if len(cand_arcs_by_step[t]) == 0:
            prediction_statuses.append("unmatched")
        elif greedy_is_fallback[t]:
            prediction_statuses.append("fallback_candidate")
        else:
            prediction_statuses.append("normal")

        # HMM/Viterbi decoder_status
        if len(cand_arcs_by_step[t]) == 0:
            if viterbi_arc_ids[t] >= 0:
                decoder_statuses.append("carried_forward")
            else:
                decoder_statuses.append("unmatched")
        elif g_stat == "reset_gap":
            decoder_statuses.append("gap_reset")
        elif chosen_is_fallback[t]:
            decoder_statuses.append("fallback_decoded")
        elif t > 0 and viterbi_hops[t] < 0:
            decoder_statuses.append("disconnected_bridge")
        else:
            decoder_statuses.append("normal")

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
            "prediction_status": prediction_statuses,
            "decoder_status": decoder_statuses,
            "gap_status": gap_statuses,
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

    viterbi_cont = float(np.mean(viterbi_connected[1:])) if n_obs > 1 else (1.0 if (n_obs == 1 and viterbi_arc_ids[0] >= 0) else 0.0)
    greedy_cont = float(np.mean(greedy_connected[1:])) if n_obs > 1 else (1.0 if (n_obs == 1 and greedy_arc_ids[0] >= 0) else 0.0)
    multi_hop_reachable = float(np.mean(viterbi_hops[1:] >= 0)) if n_obs > 1 else (1.0 if (n_obs == 1 and viterbi_arc_ids[0] >= 0) else 0.0)

    status_counts = pd.Series(decoder_statuses).value_counts()
    diagnostics: Dict[str, float] = {
        "num_observations": float(n_obs),
        "empty_candidate_observations": float(n_obs - len(valid_steps)),
        "normal_decoded_observations": float(status_counts.get("normal", 0)),
        "fallback_decoded_observations": float(status_counts.get("fallback_decoded", 0)),
        "gap_reset_observations": float(status_counts.get("gap_reset", 0)),
        "carried_forward_observations": float(status_counts.get("carried_forward", 0)),
        "disconnected_bridge_observations": float(status_counts.get("disconnected_bridge", 0)),
        "unmatched_observations": float(status_counts.get("unmatched", 0)),
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
