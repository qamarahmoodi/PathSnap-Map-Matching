"""Tests for spatial candidate generation and feature engineering."""

import numpy as np
import pandas as pd

from pathsnap.data.loader import load_trip
from pathsnap.schemas.candidate_schema import (
    CANDIDATE_FEATURE_COLUMNS,
    CANDIDATE_SCHEMA_DTYPES,
    OBSERVATION_SCHEMA_DTYPES,
    CandidateConfig,
)
from pathsnap.spatial.candidate_generation import generate_trip_candidates


def test_generate_trip_candidates_schema_and_invariants():
    raw = load_trip("00000001", partition="train", include_route=False)
    cfg = CandidateConfig(search_radius_m=100.0, max_candidates_k=10)
    res = generate_trip_candidates(
        trip_id=raw.trip_id,
        partition=raw.partition,
        track=raw.track,
        nodes=raw.nodes,
        arcs=raw.arcs,
        config=cfg,
    )

    cands = res.candidates
    obs = res.observations

    # Check exact columns
    assert list(cands.columns) == list(CANDIDATE_SCHEMA_DTYPES.keys())
    assert list(obs.columns) == list(OBSERVATION_SCHEMA_DTYPES.keys())

    # Check all observations are retained
    assert len(obs) == len(raw.track)
    assert (obs["trip_id"] == "00000001").all()

    # Check uniqueness of (trip_id, observation_id, candidate_arc_id)
    assert not cands.duplicated(subset=["trip_id", "observation_id", "candidate_arc_id"]).any()

    # Check candidate_rank is 1..K and sorted by distance_to_segment_m
    for obs_id, grp in cands.groupby("observation_id"):
        ranks = grp["candidate_rank"].to_numpy()
        dists = grp["distance_to_segment_m"].to_numpy()
        assert np.array_equal(ranks, np.arange(1, len(grp) + 1))
        assert len(grp) <= cfg.max_candidates_k
        assert np.all(np.diff(dists) >= -1e-5)

    # Check projection_fraction in [0, 1] and positive lengths
    assert (cands["projection_fraction"] >= 0.0).all()
    assert (cands["projection_fraction"] <= 1.0).all()
    assert (cands["segment_length_m"] >= 0.0).all()
    assert (cands["segment_bearing_deg"] >= 0.0).all()
    assert (cands["segment_bearing_deg"] < 360.0).all()

    # Check first observation has null delta_time_s, gps_speed_mps, gps_bearing_deg, bearing_difference_deg
    obs0_cands = cands[cands["observation_id"] == 0]
    assert obs0_cands["delta_time_s"].isna().all()
    assert obs0_cands["gps_speed_mps"].isna().all()
    assert obs0_cands["gps_bearing_deg"].isna().all()
    assert obs0_cands["bearing_difference_deg"].isna().all()

    # Without attach_alignment_labels, all labels are null and alignment_status is not_available
    assert cands["label"].isna().all()
    assert (obs["alignment_status"] == "not_available").all()
    assert (~obs["label_eligible"]).all()
