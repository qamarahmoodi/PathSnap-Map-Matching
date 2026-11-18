"""Tests for PathSnap data loader and raw dataset integrity."""

import numpy as np
import pytest

from pathsnap.data.loader import (
    list_trips,
    load_trip,
    resolve_data_root,
    validate_trip_arrays,
)


def test_resolve_data_root_and_partitions():
    root = resolve_data_root()
    assert (root / "train").is_dir()
    assert (root / "student_test").is_dir()

    train_trips = list_trips("train", root)
    test_trips = list_trips("student_test", root)
    assert len(train_trips) == 80
    assert len(test_trips) == 10
    # Verify leading zeros preserved
    assert all(len(t) == 8 and t.startswith("0") for t in train_trips)
    assert all(len(t) == 8 and t.startswith("0") for t in test_trips)


def test_load_trip_with_and_without_route():
    raw_with = load_trip("00000001", partition="train", include_route=True)
    assert raw_with.trip_id == "00000001"
    assert raw_with.partition == "train"
    assert list(raw_with.track.columns) == ["observation_id", "longitude", "latitude", "timestamp_s"]
    assert len(raw_with.track) > 0
    assert raw_with.nodes.ndim == 2 and raw_with.nodes.shape[1] == 2
    assert raw_with.arcs.ndim == 2 and raw_with.arcs.shape[1] == 2
    assert raw_with.route is not None and raw_with.route.ndim == 1

    # Strict inference mode without route
    raw_no_route = load_trip("00000001", partition="train", include_route=False)
    assert raw_no_route.route is None

    # Validate arrays
    stats = validate_trip_arrays(
        raw_with.trip_id, raw_with.track, raw_with.nodes, raw_with.arcs, raw_with.route
    )
    assert stats["non_finite_gps"] == 0
    assert stats["invalid_lon_lat"] == 0
    assert stats["invalid_arc_nodes"] == 0
    assert stats["invalid_route_arcs"] == 0
    # Verify coordinate order is (longitude ~ 8.5E, latitude ~ 49.8N)
    assert 5.0 < stats["min_lon"] < 15.0
    assert 45.0 < stats["min_lat"] < 55.0
