"""Production route-free inference pipeline for PathSnap.

Owner: Qamar Ahmed (Technical Lead)

Connects:
  1. Raw trip `.track`, `.nodes`, `.arcs` loading (`include_route=False` — NEVER reads `.route`).
  2. Array validation (`validate_trip_arrays`).
  3. Local WGS84 tangent-plane metric projection & `STRtree` road network construction.
  4. Causal GPS kinematics & top-K candidate generation (`generate_trip_candidates`).
  5. Trained ML candidate scoring (`CandidateModelScorer.score_candidates`).
  6. Sequential HMM + Viterbi decoding (`decode_trip_viterbi`).
  7. Route geometry conversion (`arcs_to_coord_segments` / `arcs_to_polyline_latlon`) for UI & mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from pathsnap.data.loader import load_trip, validate_trip_arrays
from pathsnap.graph.road_graph import TripRoadGraph
from pathsnap.inference.model_scorer import CandidateModelScorer
from pathsnap.schemas.candidate_schema import CandidateConfig
from pathsnap.spatial.candidate_generation import generate_trip_candidates
from pathsnap.spatial.road_network import build_trip_road_network
from sequence.viterbi_decoder import ViterbiConfig, ViterbiDecodeResult, decode_trip_viterbi


@dataclass
class TripInferenceOutput:
    """Complete route-free inference output for a single trip."""

    trip_id: str
    partition: str
    model_name: str
    model_path: str
    classifier_only_predictions: pd.DataFrame
    hmm_viterbi_predictions: pd.DataFrame
    observation_predictions: pd.DataFrame
    scored_candidates: pd.DataFrame
    observations_metadata: pd.DataFrame
    predicted_route_arc_ids: List[int]
    classifier_only_route_arc_ids: List[int]
    predicted_route_segments_latlon: List[List[List[float]]]
    classifier_only_route_segments_latlon: List[List[List[float]]]
    gps_trajectory_latlon: List[List[float]]
    timing_ms: Dict[str, float]
    candidate_summary: Dict[str, Any]
    decoder_summary: Dict[str, Any]
    graph: TripRoadGraph


def arcs_to_segments_latlon(
    arc_ids: np.ndarray | List[int],
    nodes_lon_lat: np.ndarray,
    arcs: np.ndarray,
) -> List[List[List[float]]]:
    """Convert an ordered sequence of `arc_id` integers into `[[[lat1, lon1], [lat2, lon2]], ...]` for UI mapping."""
    segments: List[List[List[float]]] = []
    num_arcs = arcs.shape[0]
    for aid in arc_ids:
        a_int = int(aid)
        if 0 <= a_int < num_arcs:
            u, v = int(arcs[a_int, 0]), int(arcs[a_int, 1])
            segments.append(
                [
                    [round(float(nodes_lon_lat[u, 1]), 6), round(float(nodes_lon_lat[u, 0]), 6)],
                    [round(float(nodes_lon_lat[v, 1]), 6), round(float(nodes_lon_lat[v, 0]), 6)],
                ]
            )
    return segments


def run_trip_inference(
    trip_id: str,
    partition: str = "student_test",
    data_root: Optional[str | Path] = None,
    scorer: Optional[CandidateModelScorer] = None,
    candidate_config: Optional[CandidateConfig] = None,
    viterbi_config: Optional[ViterbiConfig] = None,
) -> TripInferenceOutput:
    """Execute the complete PathSnap prediction pipeline for one trip WITHOUT reading `.route`."""
    t_total_start = time.perf_counter()

    if scorer is None:
        scorer = CandidateModelScorer.from_path()
    cand_cfg = candidate_config or CandidateConfig()
    vit_cfg = viterbi_config or ViterbiConfig()

    # 1. Load .track, .nodes, .arcs ONLY (strictly include_route=False)
    t0 = time.perf_counter()
    raw = load_trip(
        trip_id=str(trip_id),
        partition=partition,
        data_root=data_root,
        include_route=False,
    )
    assert raw.route is None, "SECURITY VIOLATION: .route must never be loaded in run_trip_inference!"
    validate_trip_arrays(raw.trip_id, raw.track, raw.nodes, raw.arcs, route=None)
    t_load_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Build metric road network & spatial index, then generate candidates & causal features
    t1 = time.perf_counter()
    net = build_trip_road_network(raw.trip_id, raw.nodes, raw.arcs)
    cand_res = generate_trip_candidates(
        trip_id=raw.trip_id,
        partition=raw.partition,
        track=raw.track,
        nodes=raw.nodes,
        arcs=raw.arcs,
        config=cand_cfg,
        road_network=net,
    )
    t_cand_ms = (time.perf_counter() - t1) * 1000.0

    # 3. Score candidates with trained ML model
    t2 = time.perf_counter()
    scored_res = scorer.score_candidates(cand_res.candidates, prob_col="pred_prob")
    scored_cands = scored_res.scored_candidates
    t_score_ms = (time.perf_counter() - t2) * 1000.0

    # 4. Run HMM + Viterbi sequential route reconstruction
    t3 = time.perf_counter()
    vit_res: ViterbiDecodeResult = decode_trip_viterbi(
        candidates_df=scored_cands,
        observations_df=cand_res.observations,
        graph=net.graph,
        prob_col="pred_prob",
        config=vit_cfg,
    )
    t_viterbi_ms = (time.perf_counter() - t3) * 1000.0

    # 5. Build standardized Classifier-Only and HMM/Viterbi DataFrames for Gokul & API
    t4 = time.perf_counter()
    obs_preds = vit_res.observation_predictions

    classifier_only_df = pd.DataFrame(
        {
            "trip_id": obs_preds["trip_id"],
            "observation_id": obs_preds["observation_id"],
            "timestamp_s": obs_preds["timestamp_s"],
            "predicted_arc_id": obs_preds["greedy_arc_id"],
            "model_score": obs_preds["greedy_prob"],
            "prediction_status": obs_preds["prediction_status"],
            "is_connected_step": obs_preds["is_greedy_connected_step"],
        }
    )

    hmm_viterbi_df = pd.DataFrame(
        {
            "trip_id": obs_preds["trip_id"],
            "observation_id": obs_preds["observation_id"],
            "timestamp_s": obs_preds["timestamp_s"],
            "predicted_arc_id": obs_preds["viterbi_arc_id"],
            "decoded_score": obs_preds["viterbi_prob"],
            "decoder_status": obs_preds["decoder_status"],
            "gap_status": obs_preds["gap_status"],
            "transition_hops": obs_preds["transition_hops"],
            "is_connected_step": obs_preds["is_connected_step"],
        }
    )

    # Convert decoded arc sequences into [lat, lon] segments for map rendering
    pred_route_arcs = [int(a) for a in vit_res.reconstructed_route_arcs]
    greedy_route_arcs = [int(a) for a in vit_res.greedy_route_arcs]
    pred_segments_latlon = arcs_to_segments_latlon(pred_route_arcs, raw.nodes, raw.arcs)
    greedy_segments_latlon = arcs_to_segments_latlon(greedy_route_arcs, raw.nodes, raw.arcs)

    gps_latlon = [
        [round(float(lat), 6), round(float(lon), 6)]
        for lon, lat in zip(raw.track["longitude"].to_numpy(), raw.track["latitude"].to_numpy())
    ]
    t_post_ms = (time.perf_counter() - t4) * 1000.0
    t_total_ms = (time.perf_counter() - t_total_start) * 1000.0

    timing_ms = {
        "data_loading_ms": round(t_load_ms, 3),
        "candidate_generation_ms": round(t_cand_ms, 3),
        "model_scoring_ms": round(t_score_ms, 3),
        "viterbi_decoding_ms": round(t_viterbi_ms, 3),
        "geometry_serialization_ms": round(t_post_ms, 3),
        "total_pipeline_ms": round(t_total_ms, 3),
    }

    n_obs = len(cand_res.observations)
    n_cands = len(scored_cands)
    fb_cands = int(scored_cands["is_fallback_candidate"].sum()) if n_cands > 0 else 0
    zero_cand_obs = int((cand_res.observations["candidate_count"] == 0).sum())

    candidate_summary = {
        "num_observations": n_obs,
        "num_road_nodes": int(raw.nodes.shape[0]),
        "num_road_arcs": int(raw.arcs.shape[0]),
        "total_candidates": n_cands,
        "avg_candidates_per_observation": round(n_cands / max(n_obs, 1), 3),
        "fallback_candidates_count": fb_cands,
        "zero_candidate_observations": zero_cand_obs,
        "search_radius_m": float(cand_cfg.search_radius_m),
        "max_candidates_k": int(cand_cfg.max_candidates_k),
    }

    decoder_summary = dict(vit_res.diagnostics)
    decoder_summary["predicted_route_arc_count"] = len(pred_route_arcs)
    decoder_summary["classifier_only_route_arc_count"] = len(greedy_route_arcs)
    decoder_summary["total_log_likelihood"] = round(float(vit_res.total_log_likelihood), 4)

    return TripInferenceOutput(
        trip_id=raw.trip_id,
        partition=raw.partition,
        model_name=scorer.model_name,
        model_path=scorer.model_path,
        classifier_only_predictions=classifier_only_df,
        hmm_viterbi_predictions=hmm_viterbi_df,
        observation_predictions=obs_preds,
        scored_candidates=scored_cands,
        observations_metadata=cand_res.observations,
        predicted_route_arc_ids=pred_route_arcs,
        classifier_only_route_arc_ids=greedy_route_arcs,
        predicted_route_segments_latlon=pred_segments_latlon,
        classifier_only_route_segments_latlon=greedy_segments_latlon,
        gps_trajectory_latlon=gps_latlon,
        timing_ms=timing_ms,
        candidate_summary=candidate_summary,
        decoder_summary=decoder_summary,
        graph=net.graph,
    )
