"""Attach reference alignment labels and candidate-miss flags to CandidateGenerationResult."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

from pathsnap.schemas.candidate_schema import (
    CandidateGenerationResult,
    ReferenceAlignment,
    enforce_candidate_schema,
    enforce_observation_schema,
)


@dataclass
class LabelSummary:
    """Summary metrics for candidate label attachment and candidate miss rate."""

    trip_id: str
    total_observations: int
    aligned_observations: int
    ambiguous_observations: int
    unmatched_observations: int
    not_available_observations: int
    label_eligible_observations: int
    candidate_miss_count: int
    candidate_miss_rate: float
    total_candidate_rows: int
    labeled_candidate_rows: int
    positive_candidate_rows: int
    negative_candidate_rows: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "trip_id": self.trip_id,
            "total_observations": self.total_observations,
            "aligned_observations": self.aligned_observations,
            "ambiguous_observations": self.ambiguous_observations,
            "unmatched_observations": self.unmatched_observations,
            "not_available_observations": self.not_available_observations,
            "label_eligible_observations": self.label_eligible_observations,
            "candidate_miss_count": self.candidate_miss_count,
            "candidate_miss_rate": self.candidate_miss_rate,
            "total_candidate_rows": self.total_candidate_rows,
            "labeled_candidate_rows": self.labeled_candidate_rows,
            "positive_candidate_rows": self.positive_candidate_rows,
            "negative_candidate_rows": self.negative_candidate_rows,
        }


def attach_alignment_labels(
    candidate_result: CandidateGenerationResult,
    alignment: ReferenceAlignment,
) -> CandidateGenerationResult:
    """Attach reference route labels to `candidate_result` without mutating inputs in-place.

    Rules enforced from the contract:
    1. Only observations with `alignment_status == 'aligned'` are considered confidently aligned.
    2. For a confidently aligned observation:
       - If `reference_arc_id` is present among the observation's generated candidates:
         * matching candidate receives `label = 1`
         * all other candidates for that observation receive `label = 0`
         * `candidate_miss = False`, `label_eligible = True`
       - If `reference_arc_id` is ABSENT from the generated candidate set:
         * `candidate_miss = True`, `label_eligible = False`
         * ALL candidate rows for that observation keep `label = <NA>` (never false negatives!)
    3. For `ambiguous`, `unmatched`, or `not_available` observations:
       - `candidate_miss = <NA>`, `label_eligible = False`
       - ALL candidate rows for that observation keep `label = <NA>`.
    """
    obs_df = candidate_result.observations.copy()
    cand_df = candidate_result.candidates.copy()

    n_obs = len(obs_df)
    if len(alignment.observation_id) != n_obs:
        raise ValueError(
            f"Observation count mismatch: candidate_result has {n_obs}, alignment has {len(alignment.observation_id)}"
        )

    statuses = list(alignment.alignment_status)
    ref_arcs = alignment.reference_arc_id
    ref_pos = alignment.reference_route_position
    align_dists = alignment.alignment_distance_m

    # Populate reference_arc_id and reference_route_position with nullable Int32
    obs_df["reference_arc_id"] = pd.Series(
        [int(a) if a >= 0 else pd.NA for a in ref_arcs],
        index=obs_df.index,
        dtype="Int32",
    )
    obs_df["reference_route_position"] = pd.Series(
        [int(p) if p >= 0 else pd.NA for p in ref_pos],
        index=obs_df.index,
        dtype="Int32",
    )
    obs_df["alignment_distance_m"] = pd.Series(
        align_dists, index=obs_df.index, dtype="Float32"
    )
    obs_df["alignment_status"] = pd.Series(
        statuses, index=obs_df.index, dtype="string"
    )

    # Determine which observations have their reference_arc_id inside cand_df
    if not cand_df.empty:
        cand_obs_ids = cand_df["observation_id"].to_numpy(dtype=np.int32)
        cand_arc_ids = cand_df["candidate_arc_id"].to_numpy(dtype=np.int32)
        # Map each candidate row to the aligned reference_arc_id of its observation
        obs_id_to_idx = np.zeros(int(obs_df["observation_id"].max()) + 1, dtype=np.int32)
        obs_id_to_idx[obs_df["observation_id"].to_numpy(dtype=np.int32)] = np.arange(n_obs, dtype=np.int32)
        row_obs_idx = obs_id_to_idx[cand_obs_ids]

        row_ref_arc = ref_arcs[row_obs_idx]
        row_is_aligned = np.array([statuses[idx] == "aligned" for idx in row_obs_idx], dtype=bool)
        row_matches_ref = row_is_aligned & (cand_arc_ids == row_ref_arc)

        # Count matches per observation index
        obs_has_hit = np.bincount(
            row_obs_idx[row_matches_ref], minlength=n_obs
        ) > 0
    else:
        obs_has_hit = np.zeros(n_obs, dtype=bool)
        row_obs_idx = np.empty(0, dtype=np.int32)
        row_matches_ref = np.empty(0, dtype=bool)

    cand_miss_list = []
    label_eligible_arr = np.zeros(n_obs, dtype=bool)

    for i in range(n_obs):
        st = statuses[i]
        if st == "aligned":
            if obs_has_hit[i]:
                cand_miss_list.append(False)
                label_eligible_arr[i] = True
            else:
                cand_miss_list.append(True)
                label_eligible_arr[i] = False
        else:
            cand_miss_list.append(pd.NA)
            label_eligible_arr[i] = False

    obs_df["candidate_miss"] = pd.Series(cand_miss_list, index=obs_df.index, dtype="boolean")
    obs_df["label_eligible"] = pd.Series(label_eligible_arr, index=obs_df.index, dtype="bool")

    # Assign candidate labels only for label_eligible observations
    if not cand_df.empty:
        row_eligible = label_eligible_arr[row_obs_idx]
        labels = pd.Series([pd.NA] * len(cand_df), index=cand_df.index, dtype="Int8")
        labels[row_eligible & row_matches_ref] = 1
        labels[row_eligible & (~row_matches_ref)] = 0
        cand_df["label"] = labels

    cand_df = enforce_candidate_schema(cand_df)
    obs_df = enforce_observation_schema(obs_df)

    return CandidateGenerationResult(
        trip_id=candidate_result.trip_id,
        partition=candidate_result.partition,
        candidates=cand_df,
        observations=obs_df,
        graph=candidate_result.graph,
    )


def summarize_trip_labels(result: CandidateGenerationResult) -> LabelSummary:
    """Compute exact candidate miss rate and label distribution statistics for a labeled trip."""
    obs = result.observations
    cands = result.candidates

    total_obs = len(obs)
    st_counts = obs["alignment_status"].value_counts()
    aligned_obs = int(st_counts.get("aligned", 0))
    ambig_obs = int(st_counts.get("ambiguous", 0))
    unmatched_obs = int(st_counts.get("unmatched", 0))
    na_obs = int(st_counts.get("not_available", 0))

    eligible_obs = int(obs["label_eligible"].sum())
    miss_count = int((obs["candidate_miss"] == True).sum())  # noqa: E712
    miss_rate = float(miss_count / aligned_obs) if aligned_obs > 0 else float("nan")

    total_cands = len(cands)
    if total_cands > 0:
        valid_lbl = cands["label"].dropna()
        labeled_cands = len(valid_lbl)
        pos_cands = int((valid_lbl == 1).sum())
        neg_cands = int((valid_lbl == 0).sum())
    else:
        labeled_cands = 0
        pos_cands = 0
        neg_cands = 0

    return LabelSummary(
        trip_id=result.trip_id,
        total_observations=total_obs,
        aligned_observations=aligned_obs,
        ambiguous_observations=ambig_obs,
        unmatched_observations=unmatched_obs,
        not_available_observations=na_obs,
        label_eligible_observations=eligible_obs,
        candidate_miss_count=miss_count,
        candidate_miss_rate=miss_rate,
        total_candidate_rows=total_cands,
        labeled_candidate_rows=labeled_cands,
        positive_candidate_rows=pos_cands,
        negative_candidate_rows=neg_cands,
    )
