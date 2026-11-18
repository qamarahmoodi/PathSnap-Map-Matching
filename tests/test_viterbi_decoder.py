"""Deterministic unit tests for HMM/Viterbi decoder on synthetic directed graphs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.graph.road_graph import build_trip_road_graph
from sequence.viterbi_decoder import ViterbiConfig, decode_trip_viterbi


def _make_synthetic_linear_and_parallel_graph():
    """Create a small directed road network:
    Main connected chain:
      Arc 0: 0 -> 1
      Arc 1: 1 -> 2
      Arc 2: 2 -> 3
    Disconnected parallel distractor road:
      Arc 3: 10 -> 11 (not connected to 0, 1, 2)
    """
    arcs = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [10, 11],
        ],
        dtype=np.int32,
    )
    lengths = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32)
    bearings = np.array([90.0, 90.0, 90.0, 90.0], dtype=np.float32)
    return build_trip_road_graph("SYNTH_01", arcs, lengths, bearings)


def test_viterbi_overrides_disconnected_noisy_spike():
    """At observation 1, distractor Arc 3 has slightly higher ML probability (0.60 vs 0.45),
    but is topologically disconnected from Arc 0 (at t=0) and Arc 2 (at t=2).
    Greedy classifier picks [0, 3, 2] (continuity 0.0), whereas Viterbi picks [0, 1, 2] (continuity 1.0).
    """
    graph = _make_synthetic_linear_and_parallel_graph()

    obs_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01", "SYNTH_01", "SYNTH_01"],
            "observation_id": np.array([0, 1, 2], dtype=np.int32),
            "timestamp_s": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "alignment_status": ["aligned", "aligned", "aligned"],
            "reference_arc_id": np.array([0, 1, 2], dtype=np.int32),
        }
    )

    cands_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 6,
            "observation_id": np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
            "candidate_rank": np.array([1, 2, 1, 2, 1, 2], dtype=np.int16),
            "candidate_arc_id": np.array([0, 3, 3, 1, 2, 3], dtype=np.int32),
            "distance_to_segment_m": np.array([3.0, 15.0, 5.0, 7.0, 4.0, 18.0], dtype=np.float32),
            "pred_prob": np.array([0.92, 0.08, 0.60, 0.45, 0.90, 0.10], dtype=np.float64),
        }
    )

    res = decode_trip_viterbi(cands_df, obs_df, graph, prob_col="pred_prob")
    preds = res.observation_predictions

    assert list(preds["greedy_arc_id"]) == [0, 3, 2]
    assert list(preds["viterbi_arc_id"]) == [0, 1, 2]
    assert list(res.reconstructed_route_arcs) == [0, 1, 2]
    assert res.diagnostics["greedy_route_continuity"] == 0.0
    assert res.diagnostics["viterbi_route_continuity"] == 1.0
    assert res.diagnostics["viterbi_top1_accuracy"] == 1.0


def test_viterbi_handles_empty_candidates_and_large_time_gaps():
    """Verify that an observation with 0 candidates does not crash and carries forward previous state,
    and that a huge timestamp gap (>60s) allows transitioning to a new disconnected subgraph cleanly.
    """
    graph = _make_synthetic_linear_and_parallel_graph()

    obs_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 4,
            "observation_id": np.array([0, 1, 2, 3], dtype=np.int32),
            "timestamp_s": np.array([1.0, 2.0, 3.0, 120.0], dtype=np.float64),  # 117s gap before obs 3
        }
    )

    # Observation 1 has NO candidates in cands_df
    cands_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 4,
            "observation_id": np.array([0, 2, 3, 3], dtype=np.int32),
            "candidate_rank": np.array([1, 1, 1, 2], dtype=np.int16),
            "candidate_arc_id": np.array([0, 1, 3, 2], dtype=np.int32),
            "distance_to_segment_m": np.array([2.0, 3.0, 2.0, 25.0], dtype=np.float32),
            "pred_prob": np.array([0.95, 0.90, 0.85, 0.15], dtype=np.float64),
        }
    )

    res = decode_trip_viterbi(cands_df, obs_df, graph, prob_col="pred_prob", config=ViterbiConfig())
    preds = res.observation_predictions

    # Obs 1 carried forward from Obs 0 (Arc 0), Obs 2 is Arc 1, Obs 3 after 117s gap transitions to Arc 3
    assert list(preds["viterbi_arc_id"]) == [0, 0, 1, 3]
    assert res.diagnostics["empty_candidate_observations"] == 1.0
