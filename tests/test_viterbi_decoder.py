"""Deterministic unit tests for HMM/Viterbi decoder covering all 16 required test scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    Reverse arc for U-turn check:
      Arc 4: 1 -> 0 (reverse of Arc 0)
    """
    arcs = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [10, 11],
            [1, 0],
        ],
        dtype=np.int32,
    )
    lengths = np.array([50.0, 50.0, 50.0, 50.0, 50.0], dtype=np.float32)
    bearings = np.array([90.0, 90.0, 90.0, 90.0, 270.0], dtype=np.float32)
    return build_trip_road_graph("SYNTH_01", arcs, lengths, bearings)


def test_01_single_observation_with_multiple_candidates():
    graph = _make_synthetic_linear_and_parallel_graph()
    obs_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"],
            "observation_id": np.array([0], dtype=np.int32),
            "timestamp_s": np.array([1.0], dtype=np.float64),
        }
    )
    cands_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 3,
            "observation_id": np.array([0, 0, 0], dtype=np.int32),
            "candidate_rank": np.array([1, 2, 3], dtype=np.int16),
            "candidate_arc_id": np.array([0, 1, 3], dtype=np.int32),
            "distance_to_segment_m": np.array([5.0, 8.0, 12.0], dtype=np.float32),
            "pred_prob": np.array([0.20, 0.85, 0.10], dtype=np.float64),
        }
    )
    res = decode_trip_viterbi(cands_df, obs_df, graph)
    assert list(res.observation_predictions["viterbi_arc_id"]) == [1]
    assert list(res.observation_predictions["greedy_arc_id"]) == [1]
    assert list(res.reconstructed_route_arcs) == [1]


def test_02_single_candidate_at_each_observation():
    graph = _make_synthetic_linear_and_parallel_graph()
    obs_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 3,
            "observation_id": np.array([0, 1, 2], dtype=np.int32),
            "timestamp_s": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        }
    )
    cands_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 3,
            "observation_id": np.array([0, 1, 2], dtype=np.int32),
            "candidate_rank": np.array([1, 1, 1], dtype=np.int16),
            "candidate_arc_id": np.array([0, 1, 2], dtype=np.int32),
            "distance_to_segment_m": np.array([2.0, 2.5, 3.0], dtype=np.float32),
            "pred_prob": np.array([0.80, 0.75, 0.90], dtype=np.float64),
        }
    )
    res = decode_trip_viterbi(cands_df, obs_df, graph)
    assert list(res.observation_predictions["viterbi_arc_id"]) == [0, 1, 2]
    assert res.diagnostics["viterbi_route_continuity"] == 1.0


def test_03_and_04_viterbi_overrides_disconnected_noisy_spike():
    """Tests both (3) clearly connected sequence and (4) disconnected consecutive-road transition override."""
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


def test_05_valid_multihop_road_transition():
    """At t=1, Arc 1 (intermediate short connector) is skipped, so transition is Arc 0 -> Arc 2 (2 hops) vs Arc 0 -> Arc 3 (disconnected)."""
    graph = _make_synthetic_linear_and_parallel_graph()
    obs_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 2,
            "observation_id": np.array([0, 1], dtype=np.int32),
            "timestamp_s": np.array([1.0, 2.0], dtype=np.float64),
        }
    )
    cands_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 3,
            "observation_id": np.array([0, 1, 1], dtype=np.int32),
            "candidate_rank": np.array([1, 1, 2], dtype=np.int16),
            "candidate_arc_id": np.array([0, 3, 2], dtype=np.int32),
            "distance_to_segment_m": np.array([2.0, 4.0, 5.0], dtype=np.float32),
            "pred_prob": np.array([0.95, 0.55, 0.45], dtype=np.float64),
        }
    )
    res = decode_trip_viterbi(cands_df, obs_df, graph)
    preds = res.observation_predictions
    assert list(preds["viterbi_arc_id"]) == [0, 2]
    assert int(preds["transition_hops"].iloc[1]) == 2
    assert res.diagnostics["viterbi_multihop_reachability"] == 1.0


def test_06_and_07_empty_candidates_and_large_time_gaps():
    """Tests (6) large GPS time gap (>60s reset) and (7) empty candidate set carry-forward."""
    graph = _make_synthetic_linear_and_parallel_graph()
    obs_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 4,
            "observation_id": np.array([0, 1, 2, 3], dtype=np.int32),
            "timestamp_s": np.array([1.0, 2.0, 3.0, 120.0], dtype=np.float64),
        }
    )
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
    assert list(preds["viterbi_arc_id"]) == [0, 0, 1, 3]
    assert preds["decoder_status"].iloc[1] == "carried_forward"
    assert preds["decoder_status"].iloc[3] == "gap_reset"
    assert preds["gap_status"].iloc[3] == "reset_gap"
    assert res.diagnostics["empty_candidate_observations"] == 1.0


def test_08_fallback_candidate_status_tracking():
    graph = _make_synthetic_linear_and_parallel_graph()
    obs_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 2,
            "observation_id": np.array([0, 1], dtype=np.int32),
            "timestamp_s": np.array([1.0, 2.0], dtype=np.float64),
        }
    )
    cands_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 2,
            "observation_id": np.array([0, 1], dtype=np.int32),
            "candidate_rank": np.array([1, 1], dtype=np.int16),
            "candidate_arc_id": np.array([0, 1], dtype=np.int32),
            "distance_to_segment_m": np.array([4.0, 145.0], dtype=np.float32),
            "is_fallback_candidate": [False, True],
            "pred_prob": np.array([0.90, 0.65], dtype=np.float64),
        }
    )
    res = decode_trip_viterbi(cands_df, obs_df, graph)
    preds = res.observation_predictions
    assert list(preds["prediction_status"]) == ["normal", "fallback_candidate"]
    assert list(preds["decoder_status"]) == ["normal", "fallback_decoded"]
    assert res.diagnostics["fallback_decoded_observations"] == 1.0


def test_09_zero_or_near_zero_emission_probability_numerical_stability():
    graph = _make_synthetic_linear_and_parallel_graph()
    obs_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 2,
            "observation_id": np.array([0, 1], dtype=np.int32),
            "timestamp_s": np.array([1.0, 2.0], dtype=np.float64),
        }
    )
    cands_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 4,
            "observation_id": np.array([0, 0, 1, 1], dtype=np.int32),
            "candidate_rank": np.array([1, 2, 1, 2], dtype=np.int16),
            "candidate_arc_id": np.array([0, 3, 1, 3], dtype=np.int32),
            "distance_to_segment_m": np.array([2.0, 20.0, 3.0, 25.0], dtype=np.float32),
            "pred_prob": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        }
    )
    res = decode_trip_viterbi(cands_df, obs_df, graph)
    assert np.isfinite(res.total_log_likelihood)
    assert list(res.observation_predictions["viterbi_arc_id"]) == [0, 1]


def test_10_and_11_repeated_road_arcs_and_stationary_observations():
    """Tests (10) repeated road arcs on a loop and (11) stationary GPS observations staying on the same arc."""
    graph = _make_synthetic_linear_and_parallel_graph()
    obs_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 5,
            "observation_id": np.array([0, 1, 2, 3, 4], dtype=np.int32),
            "timestamp_s": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        }
    )
    # Vehicle stays stationary on Arc 0 for 3 steps, moves to Arc 4 (1->0), then returns to Arc 0 (repeated arc in route!)
    cands_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 5,
            "observation_id": np.array([0, 1, 2, 3, 4], dtype=np.int32),
            "candidate_rank": np.array([1, 1, 1, 1, 1], dtype=np.int16),
            "candidate_arc_id": np.array([0, 0, 0, 4, 0], dtype=np.int32),
            "distance_to_segment_m": np.array([1.0, 1.1, 1.0, 2.0, 1.5], dtype=np.float32),
            "pred_prob": np.array([0.9, 0.9, 0.9, 0.85, 0.9], dtype=np.float64),
        }
    )
    res = decode_trip_viterbi(cands_df, obs_df, graph)
    # Consecutive identical arcs [0, 0, 0] compress to single 0, but later return to 0 is preserved: [0, 4, 0]
    assert list(res.observation_predictions["viterbi_arc_id"]) == [0, 0, 0, 4, 0]
    assert list(res.reconstructed_route_arcs) == [0, 4, 0]
    assert res.diagnostics["viterbi_route_continuity"] == 1.0


def test_12_no_valid_path_between_consecutive_observations():
    """When only disconnected candidates exist at t=1, decoder uses backoff penalty and marks status 'disconnected_bridge'."""
    graph = _make_synthetic_linear_and_parallel_graph()
    obs_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 2,
            "observation_id": np.array([0, 1], dtype=np.int32),
            "timestamp_s": np.array([1.0, 2.0], dtype=np.float64),
        }
    )
    cands_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 2,
            "observation_id": np.array([0, 1], dtype=np.int32),
            "candidate_rank": np.array([1, 1], dtype=np.int16),
            "candidate_arc_id": np.array([0, 3], dtype=np.int32),  # Arc 0 (0->1) and Arc 3 (10->11) are disconnected
            "distance_to_segment_m": np.array([2.0, 3.0], dtype=np.float32),
            "pred_prob": np.array([0.9, 0.9], dtype=np.float64),
        }
    )
    res = decode_trip_viterbi(cands_df, obs_df, graph)
    preds = res.observation_predictions
    assert list(preds["viterbi_arc_id"]) == [0, 3]
    assert preds["decoder_status"].iloc[1] == "disconnected_bridge"
    assert not bool(preds["is_connected_step"].iloc[1])


def test_13_14_15_16_tied_scores_permuted_candidates_ordering_and_determinism():
    """Tests (13) tied candidate scores, (14) permuted candidate order, (15) observation ordering, and (16) determinism."""
    graph = _make_synthetic_linear_and_parallel_graph()
    # Provide observations out of chronological row order to verify observation order preservation
    obs_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 3,
            "observation_id": np.array([2, 0, 1], dtype=np.int32),
            "timestamp_s": np.array([3.0, 1.0, 2.0], dtype=np.float64),
        }
    )
    # Connected chain 0 -> 1 -> 2 has identical emission probabilities (0.50) and identical transition hop count (1 hop each)
    # to an alternative 1-hop chain or same-arc path when tested at single step, or here both paths have equal transition cost:
    # Compare two 1-hop chains with tied probabilities: (0 -> 1 -> 2) vs (0 -> 4 -> 0) where 0->1->2 has shorter distances
    cands_df = pd.DataFrame(
        {
            "trip_id": ["SYNTH_01"] * 5,
            "observation_id": np.array([0, 1, 1, 2, 2], dtype=np.int32),
            "candidate_rank": np.array([1, 2, 1, 2, 1], dtype=np.int16),
            "candidate_arc_id": np.array([0, 4, 1, 0, 2], dtype=np.int32),
            "distance_to_segment_m": np.array([3.0, 9.0, 4.0, 10.0, 2.0], dtype=np.float32),
            "pred_prob": np.array([0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float64),
        }
    )

    res_a = decode_trip_viterbi(cands_df, obs_df, graph)
    # Shuffle candidate rows completely
    cands_shuffled = cands_df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    res_b = decode_trip_viterbi(cands_shuffled, obs_df, graph)

    assert list(res_a.observation_predictions["observation_id"]) == [0, 1, 2]
    assert list(res_a.observation_predictions["viterbi_arc_id"]) == [0, 1, 2]
    assert list(res_a.observation_predictions["greedy_arc_id"]) == [0, 1, 2]
    assert res_a.observation_predictions.equals(res_b.observation_predictions)
    assert np.array_equal(res_a.reconstructed_route_arcs, res_b.reconstructed_route_arcs)
