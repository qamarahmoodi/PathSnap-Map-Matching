"""Tests for candidate label attachment and candidate-miss handling."""

import numpy as np
import pandas as pd

from pathsnap.alignment.label_generation import attach_alignment_labels, summarize_trip_labels
from pathsnap.alignment.route_alignment import align_reference_route
from pathsnap.data.loader import load_trip
from pathsnap.schemas.candidate_schema import CandidateConfig
from pathsnap.spatial.candidate_generation import generate_trip_candidates


def test_attach_alignment_labels_real_trip():
    raw = load_trip("00000001", partition="train", include_route=True)
    cfg = CandidateConfig(search_radius_m=100.0, max_candidates_k=10)
    cand_res = generate_trip_candidates(
        trip_id=raw.trip_id,
        partition=raw.partition,
        track=raw.track,
        nodes=raw.nodes,
        arcs=raw.arcs,
        config=cfg,
    )
    alignment = align_reference_route(
        track=raw.track,
        nodes=raw.nodes,
        arcs=raw.arcs,
        route=raw.route,
        config=cfg,
        trip_id=raw.trip_id,
    )
    labeled_res = attach_alignment_labels(cand_res, alignment)
    summary = summarize_trip_labels(labeled_res)

    obs = labeled_res.observations
    cands = labeled_res.candidates

    # Every label_eligible observation must have exactly 1 positive candidate (label == 1)
    eligible_obs_ids = set(obs.loc[obs["label_eligible"], "observation_id"])
    ineligible_obs_ids = set(obs.loc[~obs["label_eligible"], "observation_id"])

    el_cands = cands[cands["observation_id"].isin(eligible_obs_ids)]
    inel_cands = cands[cands["observation_id"].isin(ineligible_obs_ids)]

    assert el_cands["label"].notna().all()
    pos_per_obs = el_cands.groupby("observation_id")["label"].apply(lambda s: (s == 1).sum())
    assert (pos_per_obs == 1).all()

    # Every non-eligible observation (including candidate misses and ambiguous/unmatched) MUST have null labels
    assert inel_cands["label"].isna().all()

    # Check candidate miss rate formula
    assert summary.aligned_observations == summary.label_eligible_observations + summary.candidate_miss_count
    if summary.aligned_observations > 0:
        expected_rate = summary.candidate_miss_count / summary.aligned_observations
        assert np.isclose(summary.candidate_miss_rate, expected_rate)


def test_candidate_miss_never_creates_false_negatives():
    """Force a tiny search radius / K=1 so some aligned observations miss their reference arc,
    and verify those observations receive label=<NA> across all candidates rather than all-zeros.
    """
    raw = load_trip("00000001", partition="train", include_route=True)
    cfg = CandidateConfig(search_radius_m=15.0, max_candidates_k=1, enable_fallback_nearest=True)
    cand_res = generate_trip_candidates(
        trip_id=raw.trip_id,
        partition=raw.partition,
        track=raw.track,
        nodes=raw.nodes,
        arcs=raw.arcs,
        config=cfg,
    )
    alignment = align_reference_route(
        track=raw.track,
        nodes=raw.nodes,
        arcs=raw.arcs,
        route=raw.route,
        config=cfg,
        trip_id=raw.trip_id,
    )
    labeled_res = attach_alignment_labels(cand_res, alignment)
    obs = labeled_res.observations
    cands = labeled_res.candidates

    miss_obs_ids = set(obs.loc[obs["candidate_miss"] == True, "observation_id"])  # noqa: E712
    assert len(miss_obs_ids) > 0, "Expected at least one candidate miss with K=1"

    miss_cands = cands[cands["observation_id"].isin(miss_obs_ids)]
    assert miss_cands["label"].isna().all(), "Candidate miss rows must have null labels, not 0!"
