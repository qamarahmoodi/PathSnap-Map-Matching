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


def test_gps_kinematics_gap_and_nonpositive_dt_handling():
    """Verify that positive GPS gaps > 30s retain delta_time_s while masking speed/bearing,
    and non-positive dt <= 0 masks delta_time_s, speed, and bearing to NaN.
    """
    from pathsnap.features.feature_engineering import compute_gps_kinematics

    pts_xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],    # obs 1: normal 1s step, 10m east (speed=10m/s, bearing=90 deg)
            [11.0, 0.0],    # obs 2: 1s step, 1m east (<2.5m stationary jitter -> bearing NaN, speed=1m/s)
            [200.0, 0.0],   # obs 3: 45s gap (>30s -> delta_time_s=45.0, speed=NaN, bearing=NaN)
            [205.0, 0.0],   # obs 4: duplicate timestamp dt=0 (-> delta_time_s=NaN, speed=NaN, bearing=NaN)
        ],
        dtype=np.float64,
    )
    ts = np.array([1.0, 2.0, 3.0, 48.0, 48.0], dtype=np.float64)
    kin = compute_gps_kinematics(pts_xy, ts, CandidateConfig())

    # Obs 0: all NaN
    assert np.isnan(kin.delta_time_s[0])
    assert np.isnan(kin.gps_speed_mps[0])
    assert np.isnan(kin.gps_bearing_deg[0])

    # Obs 1: normal 1s step
    assert np.isclose(kin.delta_time_s[1], 1.0)
    assert np.isclose(kin.gps_speed_mps[1], 10.0)
    assert np.isclose(kin.gps_bearing_deg[1], 90.0)

    # Obs 2: stationary jitter (< 2.5m) -> speed finite, bearing NaN
    assert np.isclose(kin.delta_time_s[2], 1.0)
    assert np.isclose(kin.gps_speed_mps[2], 1.0)
    assert np.isnan(kin.gps_bearing_deg[2])

    # Obs 3: large positive gap (45s > 30s) -> delta_time_s retained (45.0), speed & bearing NaN
    assert np.isclose(kin.delta_time_s[3], 45.0)
    assert np.isnan(kin.gps_speed_mps[3])
    assert np.isnan(kin.gps_bearing_deg[3])

    # Obs 4: duplicate timestamp (dt == 0) -> delta_time_s, speed, and bearing all NaN
    assert np.isnan(kin.delta_time_s[4])
    assert np.isnan(kin.gps_speed_mps[4])
    assert np.isnan(kin.gps_bearing_deg[4])
