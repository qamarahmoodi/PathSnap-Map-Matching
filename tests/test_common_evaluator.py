"""Unit tests for Gokul's Common Evaluation & Measurement Framework."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.common_evaluator import (
    compute_candidate_classification_metrics,
    compute_route_arc_f1,
    compute_route_continuity,
    evaluate_observation_predictions,
)
from pathsnap.graph.road_graph import build_trip_road_graph


def test_candidate_classification_metrics_perfect():
    y_true = [1, 0, 1, 0, 0]
    y_score = [0.9, 0.1, 0.8, 0.2, 0.05]
    m = compute_candidate_classification_metrics(y_true, y_score, threshold=0.5)

    assert m.total_candidate_rows == 5
    assert m.true_positives == 2
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.true_negatives == 3
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.positive_f1 == 1.0
    assert m.macro_f1 == 1.0
    assert m.confusion_matrix == [[3, 0], [0, 2]]
    assert m.log_loss is not None
    assert m.log_loss < 0.25


def test_candidate_classification_metrics_zero_division():
    # All true negative, all predicted negative -> precision/recall should be safely 0.0
    y_true = [0, 0, 0]
    y_score = [0.1, 0.2, 0.3]
    m = compute_candidate_classification_metrics(y_true, y_score, threshold=0.5)

    assert m.true_positives == 0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.accuracy == 1.0
    assert m.negative_f1 == 1.0
    assert m.macro_f1 == 0.5


def test_route_arc_f1():
    p_arcs = [10, 11, 12, 13]
    r_arcs = [11, 12, 14, 15]
    # Intersect: {11, 12} (len 2). |P|=4, |R|=4
    prec, rec, f1 = compute_route_arc_f1(p_arcs, r_arcs)
    assert prec == 0.5
    assert rec == 0.5
    assert f1 == 0.5


def test_route_continuity_synthetic_graph():
    # Arcs: 0 (node 0->1), 1 (node 1->2), 2 (node 2->3), 3 (node 10->11, disconnected)
    arcs = np.array([[0, 1], [1, 2], [2, 3], [10, 11]], dtype=np.int32)
    lengths = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32)
    bearings = np.array([90.0, 90.0, 90.0, 90.0], dtype=np.float32)
    g = build_trip_road_graph("TEST_TRIP", arcs, lengths, bearings)

    # Sequence: 0 -> 0 (same, 0-hop), 0 -> 1 (direct, 1-hop), 1 -> 3 (disconnected)
    seq = [0, 0, 1, 3]
    cont_rate, multi_rate, hops = compute_route_continuity(seq, g, max_hops=3)

    assert len(hops) == 3
    assert hops[0] == 0  # same arc
    assert hops[1] == 1  # 1-hop connected
    assert hops[2] == -1  # disconnected
    assert abs(cont_rate - 2 / 3) < 1e-5
    assert abs(multi_rate - 2 / 3) < 1e-5


def test_candidate_miss_separation_in_observation_evaluation():
    # 3 observations. Obs 1 has candidate miss.
    arcs = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    lengths = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    bearings = np.array([90.0, 90.0, 90.0], dtype=np.float32)
    g = build_trip_road_graph("TEST_TRIP", arcs, lengths, bearings)

    preds = [0, 1, 2]
    refs = [0, 99, 2]  # Arc 99 is ground truth for obs 1, but not in its candidates!
    statuses = ["aligned", "aligned", "aligned"]
    cand_by_obs = [
        [0, 1],
        [1, 2],  # does NOT contain 99 -> Candidate Miss!
        [2, 0],
    ]
    ref_route = [0, 99, 2]

    res = evaluate_observation_predictions(
        trip_id="TEST_TRIP",
        prediction_type="test",
        predicted_arc_ids=preds,
        reference_arc_ids=refs,
        alignment_statuses=statuses,
        candidate_arcs_by_obs=cand_by_obs,
        road_graph=g,
        reference_route_arcs=ref_route,
        inference_time_ms=15.0,
    )

    assert res.total_observations == 3
    assert res.aligned_observations == 3
    assert res.candidate_miss_count == 1
    assert abs(res.candidate_miss_rate - 1 / 3) < 1e-5
    assert res.top1_correct_count == 2
    assert abs(res.top1_accuracy_overall - 2 / 3) < 1e-5
    # Eligible accuracy: 2 eligible obs (obs 0 and obs 2), both correct -> 100%!
    assert res.top1_accuracy_eligible == 1.0
