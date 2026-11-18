"""Tests for monotonic sequence-aware reference route alignment."""

import numpy as np

from pathsnap.alignment.route_alignment import align_reference_route
from pathsnap.data.loader import load_trip
from pathsnap.schemas.candidate_schema import CandidateConfig, VALID_ALIGNMENT_STATUSES


def test_align_reference_route_monotonicity_and_residuals():
    for trip_id in ["00000001", "00000002", "00000003"]:
        raw = load_trip(trip_id, partition="train", include_route=True)
        cfg = CandidateConfig()
        alignment = align_reference_route(
            track=raw.track,
            nodes=raw.nodes,
            arcs=raw.arcs,
            route=raw.route,
            config=cfg,
            trip_id=raw.trip_id,
        )

        n_obs = len(raw.track)
        assert len(alignment.observation_id) == n_obs
        assert len(alignment.alignment_status) == n_obs
        assert set(alignment.alignment_status).issubset(VALID_ALIGNMENT_STATUSES)

        # Verify monotonicity of route positions on aligned/ambiguous observations
        valid_pos = alignment.reference_route_position[alignment.reference_route_position >= 0]
        assert len(valid_pos) > 0
        assert np.all(np.diff(valid_pos) >= 0), f"Non-monotonic alignment in trip {trip_id}"

        # Verify reference_arc_id matches route[reference_route_position]
        for i in range(n_obs):
            pos = alignment.reference_route_position[i]
            arc = alignment.reference_arc_id[i]
            if pos >= 0:
                assert arc == raw.route[pos]
            else:
                assert arc == -1

        # Verify median alignment distance is small (< 15m on real urban GPS tracks)
        assert alignment.diagnostics["median_alignment_distance_m"] < 15.0
        assert alignment.diagnostics["aligned_ratio"] > 0.85


def test_align_reference_route_without_route():
    raw = load_trip("00000001", partition="train", include_route=False)
    alignment = align_reference_route(
        track=raw.track,
        nodes=raw.nodes,
        arcs=raw.arcs,
        route=None,
        trip_id=raw.trip_id,
    )
    assert all(s == "not_available" for s in alignment.alignment_status)
    assert (alignment.reference_arc_id == -1).all()
