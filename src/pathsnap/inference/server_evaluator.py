"""Isolated server-side evaluation module for PathSnap.

Owner: Qamar Ahmed (Technical Lead)

SECURITY & INTEGRITY GUARANTEE:
  - This module executes strictly AFTER `run_trip_inference(...)` has produced predictions.
  - It reads `<trip_id>.route` internally on Laptop 1, aligns or compares predictions against
    the reference route, and returns ONLY aggregate numerical evaluation metrics.
  - It NEVER returns reference arc IDs, reference route geometry, or per-observation ground-truth
    labels to the client/UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from pathsnap.alignment.label_generation import attach_alignment_labels, summarize_trip_labels
from pathsnap.alignment.route_alignment import align_reference_route
from pathsnap.data.loader import load_route_file, load_trip, resolve_data_root
from pathsnap.inference.inference_pipeline import TripInferenceOutput
from pathsnap.schemas.candidate_schema import CandidateConfig, CandidateGenerationResult


@dataclass
class ServerEvaluationResult:
    """Safe, aggregate-only server-side evaluation metrics returned to the UI."""

    trip_id: str
    reference_available: bool
    evaluation_time_ms: float
    metrics: Dict[str, Any]


def compute_sequence_overlap_metrics(pred_arcs: List[int], ref_arcs: np.ndarray) -> Dict[str, float]:
    """Compute set-level and ordered subsequence-level similarity between predicted route and `.route`."""
    pred_set = set(int(x) for x in pred_arcs if int(x) >= 0)
    ref_set = set(int(x) for x in ref_arcs if int(x) >= 0)

    if not pred_set and not ref_set:
        return {"route_jaccard_similarity": 1.0, "route_arc_precision": 1.0, "route_arc_recall": 1.0, "route_arc_f1": 1.0}
    if not ref_set:
        return {"route_jaccard_similarity": 0.0, "route_arc_precision": 0.0, "route_arc_recall": 0.0, "route_arc_f1": 0.0}

    inter = len(pred_set & ref_set)
    union = len(pred_set | ref_set)
    jaccard = inter / max(union, 1)
    prec = inter / max(len(pred_set), 1)
    rec = inter / max(len(ref_set), 1)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "route_jaccard_similarity": round(float(jaccard), 6),
        "route_arc_precision": round(float(prec), 6),
        "route_arc_recall": round(float(rec), 6),
        "route_arc_f1": round(float(f1), 6),
    }


def evaluate_trip_predictions_server_side(
    inference_output: TripInferenceOutput,
    data_root: Optional[str | Path] = None,
    candidate_config: Optional[CandidateConfig] = None,
) -> ServerEvaluationResult:
    """Compute server-side evaluation metrics after prediction is complete, without leaking `.route`."""
    t0 = time.perf_counter()
    root = resolve_data_root(data_root)
    trip_dir = root / inference_output.partition / inference_output.trip_id
    route_path = trip_dir / f"{inference_output.trip_id}.route"

    # Always include topological continuity metrics (which do not require .route)
    base_metrics: Dict[str, Any] = {
        "classifier_only_route_continuity": inference_output.decoder_summary.get("greedy_route_continuity"),
        "hmm_viterbi_route_continuity": inference_output.decoder_summary.get("viterbi_route_continuity"),
        "hmm_viterbi_multihop_reachability": inference_output.decoder_summary.get("viterbi_multihop_reachability"),
        "changed_observations_by_hmm": int(inference_output.decoder_summary.get("changed_observations_count", 0)),
    }

    if not route_path.is_file():
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ServerEvaluationResult(
            trip_id=inference_output.trip_id,
            reference_available=False,
            evaluation_time_ms=round(elapsed_ms, 3),
            metrics=base_metrics,
        )

    # Read .route internally on the server
    route_arr = load_route_file(route_path)
    raw = load_trip(
        trip_id=inference_output.trip_id,
        partition=inference_output.partition,
        data_root=root,
        include_route=False,
    )

    cfg = candidate_config or CandidateConfig()
    alignment = align_reference_route(
        track=raw.track,
        nodes=raw.nodes,
        arcs=raw.arcs,
        route=route_arr,
        config=cfg,
        trip_id=raw.trip_id,
    )

    temp_cand_res = CandidateGenerationResult(
        trip_id=inference_output.trip_id,
        partition=inference_output.partition,
        candidates=inference_output.scored_candidates,
        observations=inference_output.observations_metadata,
        graph=inference_output.graph,
    )
    labeled_res = attach_alignment_labels(temp_cand_res, alignment)
    lbl_summary = summarize_trip_labels(labeled_res)

    obs_labeled = labeled_res.observations.sort_values("observation_id").reset_index(drop=True)
    obs_preds = inference_output.observation_predictions.sort_values("observation_id").reset_index(drop=True)

    aligned_mask = (obs_labeled["alignment_status"] == "aligned").to_numpy()
    eligible_mask = obs_labeled["label_eligible"].to_numpy(dtype=bool)

    greedy_arcs = obs_preds["greedy_arc_id"].to_numpy(dtype=np.int32)
    viterbi_arcs = obs_preds["viterbi_arc_id"].to_numpy(dtype=np.int32)
    ref_arcs_obs = obs_labeled["reference_arc_id"].fillna(-888).to_numpy(dtype=np.int32)

    if np.any(aligned_mask):
        greedy_top1_aligned = float(np.mean(greedy_arcs[aligned_mask] == ref_arcs_obs[aligned_mask]))
        viterbi_top1_aligned = float(np.mean(viterbi_arcs[aligned_mask] == ref_arcs_obs[aligned_mask]))
    else:
        greedy_top1_aligned = float("nan")
        viterbi_top1_aligned = float("nan")

    if np.any(eligible_mask):
        greedy_top1_eligible = float(np.mean(greedy_arcs[eligible_mask] == ref_arcs_obs[eligible_mask]))
        viterbi_top1_eligible = float(np.mean(viterbi_arcs[eligible_mask] == ref_arcs_obs[eligible_mask]))
    else:
        greedy_top1_eligible = float("nan")
        viterbi_top1_eligible = float("nan")

    # Candidate-level classification metrics on label_eligible candidates
    cands_lab = labeled_res.candidates
    valid_lbl_mask = cands_lab["label"].notna()
    if valid_lbl_mask.any() and "predicted_probability" in inference_output.scored_candidates.columns:
        y_true = cands_lab.loc[valid_lbl_mask, "label"].astype("int8").to_numpy()
        y_prob = inference_output.scored_candidates.loc[valid_lbl_mask, "predicted_probability"].to_numpy(dtype=np.float64)
        y_pred = (y_prob >= 0.5).astype(np.int8)
        cand_acc = round(float(accuracy_score(y_true, y_pred)), 6)
        cand_prec = round(float(precision_score(y_true, y_pred, zero_division=0)), 6)
        cand_rec = round(float(recall_score(y_true, y_pred, zero_division=0)), 6)
        cand_macro_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6)
        cand_pos_f1 = round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 6)
        cand_log_loss = round(float(log_loss(y_true, np.clip(y_prob, 1e-6, 1.0 - 1e-6), labels=[0, 1])), 6)
    else:
        cand_acc = cand_prec = cand_rec = cand_macro_f1 = cand_pos_f1 = cand_log_loss = None

    greedy_seq_metrics = compute_sequence_overlap_metrics(
        inference_output.classifier_only_route_arc_ids, route_arr
    )
    viterbi_seq_metrics = compute_sequence_overlap_metrics(
        inference_output.predicted_route_arc_ids, route_arr
    )

    metrics: Dict[str, Any] = {
        **base_metrics,
        "aligned_observations": int(lbl_summary.aligned_observations),
        "label_eligible_observations": int(lbl_summary.label_eligible_observations),
        "candidate_miss_count": int(lbl_summary.candidate_miss_count),
        "candidate_miss_rate": round(float(lbl_summary.candidate_miss_rate), 6)
        if np.isfinite(lbl_summary.candidate_miss_rate)
        else None,
        "classifier_only_top1_accuracy": round(greedy_top1_aligned, 6)
        if np.isfinite(greedy_top1_aligned)
        else None,
        "hmm_viterbi_top1_accuracy": round(viterbi_top1_aligned, 6)
        if np.isfinite(viterbi_top1_aligned)
        else None,
        "top1_accuracy_improvement_delta": round(viterbi_top1_aligned - greedy_top1_aligned, 6)
        if (np.isfinite(viterbi_top1_aligned) and np.isfinite(greedy_top1_aligned))
        else None,
        "classifier_only_top1_accuracy_eligible": round(greedy_top1_eligible, 6)
        if np.isfinite(greedy_top1_eligible)
        else None,
        "hmm_viterbi_top1_accuracy_eligible": round(viterbi_top1_eligible, 6)
        if np.isfinite(viterbi_top1_eligible)
        else None,
        "route_continuity_improvement_delta": round(
            float(base_metrics["hmm_viterbi_route_continuity"] - base_metrics["classifier_only_route_continuity"]),
            6,
        ),
        "candidate_macro_f1": cand_macro_f1,
        "candidate_positive_f1": cand_pos_f1,
        "candidate_accuracy": cand_acc,
        "candidate_precision": cand_prec,
        "candidate_recall": cand_rec,
        "candidate_log_loss": cand_log_loss,
        "classifier_only_route_jaccard": greedy_seq_metrics["route_jaccard_similarity"],
        "hmm_viterbi_route_jaccard": viterbi_seq_metrics["route_jaccard_similarity"],
        "classifier_only_route_arc_f1": greedy_seq_metrics["route_arc_f1"],
        "hmm_viterbi_route_arc_f1": viterbi_seq_metrics["route_arc_f1"],
    }

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return ServerEvaluationResult(
        trip_id=inference_output.trip_id,
        reference_available=True,
        evaluation_time_ms=round(elapsed_ms, 3),
        metrics=metrics,
    )
