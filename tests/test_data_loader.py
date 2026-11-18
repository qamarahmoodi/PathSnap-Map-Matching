"""Tests for PathSnap data loader and raw dataset integrity."""

import numpy as np
import pytest

from pathsnap.data.loader import (
    list_trips,
    load_trip,
    resolve_data_root,
    validate_trip_arrays,
)
from pathsnap.alignment.label_generation import attach_alignment_labels
from pathsnap.alignment.route_alignment import align_reference_route
from pathsnap.schemas.candidate_schema import (
    CANDIDATE_FEATURE_COLUMNS,
    CANDIDATE_ID_COLUMNS,
    CANDIDATE_SCHEMA_DTYPES,
    OBSERVATION_SCHEMA_DTYPES,
    CandidateConfig,
)
from pathsnap.spatial.candidate_generation import generate_trip_candidates
import data_loader as gokul_loader


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


def test_gokul_data_loader_integration_with_candidate_pipeline():
    """Verify Gokul's src/data_loader.py output dict works directly with Qamar's candidate pipeline."""
    root = resolve_data_root()
    cfg = CandidateConfig(search_radius_m=100.0, max_candidates_k=10)

    for trip_id in ["00000001", "00000003", "00000005"]:
        trip_dir = root / "train" / trip_id
        d = gokul_loader.load_trip(trip_dir)

        # Verify expected shapes from Gokul's loader
        assert set(d.keys()) == {"track", "nodes", "arcs", "route"}
        assert d["track"].ndim == 2 and d["track"].shape[1] == 3
        assert d["nodes"].ndim == 2 and d["nodes"].shape[1] == 2
        assert d["arcs"].ndim == 2 and d["arcs"].shape[1] == 2
        assert d["route"].ndim == 1

        # Pass Gokul's numpy arrays directly to generate_trip_candidates (without .route)
        res_gokul = generate_trip_candidates(
            trip_id=trip_dir.name,
            partition="train",
            track=d["track"],
            nodes=d["nodes"],
            arcs=d["arcs"],
            config=cfg,
        )
        assert res_gokul.trip_id == trip_id
        assert len(res_gokul.observations) == d["track"].shape[0]
        assert list(res_gokul.candidates.columns) == list(CANDIDATE_SCHEMA_DTYPES.keys())
        assert list(res_gokul.observations.columns) == list(OBSERVATION_SCHEMA_DTYPES.keys())
        assert set(CANDIDATE_FEATURE_COLUMNS).issubset(res_gokul.candidates.columns)
        assert set(CANDIDATE_ID_COLUMNS).issubset(res_gokul.candidates.columns)

        # Pass Gokul's numpy arrays directly to align_reference_route and attach_alignment_labels
        align_gokul = align_reference_route(
            track=d["track"],
            nodes=d["nodes"],
            arcs=d["arcs"],
            route=d["route"],
            config=cfg,
            trip_id=trip_dir.name,
        )
        labeled_gokul = attach_alignment_labels(res_gokul, align_gokul)

        # Compare against Qamar's loader output to confirm 100% bit-for-bit agreement
        raw_qamar = load_trip(trip_id, partition="train", data_root=root, include_route=True)
        res_qamar = generate_trip_candidates(
            trip_id=raw_qamar.trip_id,
            partition=raw_qamar.partition,
            track=raw_qamar.track,
            nodes=raw_qamar.nodes,
            arcs=raw_qamar.arcs,
            config=cfg,
        )
        align_qamar = align_reference_route(
            track=raw_qamar.track,
            nodes=raw_qamar.nodes,
            arcs=raw_qamar.arcs,
            route=raw_qamar.route,
            config=cfg,
            trip_id=raw_qamar.trip_id,
        )
        labeled_qamar = attach_alignment_labels(res_qamar, align_qamar)

        assert labeled_gokul.candidates.equals(labeled_qamar.candidates)
        assert labeled_gokul.observations.equals(labeled_qamar.observations)
