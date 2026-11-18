"""Extract and cache full candidate and observation Parquet datasets across all 80 train and 10 student_test trips."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pathsnap.alignment.label_generation import attach_alignment_labels, summarize_trip_labels
from pathsnap.alignment.route_alignment import align_reference_route
from pathsnap.data.loader import list_trips, load_trip, resolve_data_root
from pathsnap.schemas.candidate_schema import CandidateConfig
from pathsnap.spatial.candidate_generation import generate_trip_candidates
from pathsnap.spatial.road_network import build_trip_road_network


def extract_partition_candidates(
    partition: str,
    data_root: Path,
    cfg: CandidateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
    """Generate and align candidates for all trips in a partition (`train` or `student_test`)."""
    trip_ids = list_trips(partition, data_root)
    cand_frames: List[pd.DataFrame] = []
    obs_frames: List[pd.DataFrame] = []
    summaries: List[Dict[str, object]] = []

    t0 = time.perf_counter()
    for idx, tid in enumerate(trip_ids, start=1):
        t_trip = time.perf_counter()
        raw = load_trip(tid, partition=partition, data_root=data_root, include_route=True)
        net = build_trip_road_network(raw.trip_id, raw.nodes, raw.arcs)
        c_res = generate_trip_candidates(
            trip_id=raw.trip_id,
            partition=raw.partition,
            track=raw.track,
            nodes=raw.nodes,
            arcs=raw.arcs,
            config=cfg,
            road_network=net,
        )
        align = align_reference_route(
            track=raw.track,
            nodes=raw.nodes,
            arcs=raw.arcs,
            route=raw.route,
            config=cfg,
            trip_id=raw.trip_id,
            projector=net.projector,
        )
        labeled = attach_alignment_labels(c_res, align)
        summ = summarize_trip_labels(labeled)
        s_dict = summ.to_dict()
        s_dict["partition"] = partition
        s_dict["median_align_dist_m"] = round(float(align.diagnostics["median_alignment_distance_m"]), 3)
        s_dict["p95_align_dist_m"] = round(float(align.diagnostics["p95_alignment_distance_m"]), 3)
        s_dict["is_monotonic"] = bool(align.diagnostics["is_monotonic"])
        s_dict["extraction_time_s"] = round(time.perf_counter() - t_trip, 3)
        summaries.append(s_dict)

        cand_frames.append(labeled.candidates)
        obs_frames.append(labeled.observations)

        if idx % 10 == 0 or idx == len(trip_ids):
            print(
                f"[{partition}] Processed {idx}/{len(trip_ids)} trips "
                f"(elapsed={time.perf_counter() - t0:.1f}s)"
            )

    all_cands = pd.concat(cand_frames, ignore_index=True)
    all_obs = pd.concat(obs_frames, ignore_index=True)
    return all_cands, all_obs, summaries


def main() -> None:
    data_root = resolve_data_root()
    out_dir = REPO_ROOT / "outputs" / "full_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = REPO_ROOT / "results" / "phase2"
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = CandidateConfig(search_radius_m=100.0, max_candidates_k=10, enable_fallback_nearest=True)

    t_start = time.perf_counter()
    train_cands, train_obs, train_summs = extract_partition_candidates("train", data_root, cfg)
    train_cands.to_parquet(out_dir / "train_candidates.parquet", index=False)
    train_obs.to_parquet(out_dir / "train_observations.parquet", index=False)

    test_cands, test_obs, test_summs = extract_partition_candidates("student_test", data_root, cfg)
    test_cands.to_parquet(out_dir / "student_test_candidates.parquet", index=False)
    test_obs.to_parquet(out_dir / "student_test_observations.parquet", index=False)

    all_summs_df = pd.DataFrame(train_summs + test_summs)
    all_summs_df.to_csv(results_dir / "full_90trip_candidate_summary.csv", index=False)

    digest = {
        "train_trips": int(train_cands["trip_id"].nunique()),
        "train_observations": int(len(train_obs)),
        "train_aligned_observations": int((train_obs["alignment_status"] == "aligned").sum()),
        "train_label_eligible_observations": int(train_obs["label_eligible"].sum()),
        "train_candidate_misses": int((train_obs.loc[train_obs["alignment_status"] == "aligned", "candidate_miss"] == True).sum()),  # noqa: E712
        "train_total_candidates": int(len(train_cands)),
        "train_labeled_candidates": int(train_cands["label"].notna().sum()),
        "train_positive_candidates": int((train_cands["label"] == 1).sum()),
        "train_negative_candidates": int((train_cands["label"] == 0).sum()),
        "student_test_trips": int(test_cands["trip_id"].nunique()),
        "student_test_observations": int(len(test_obs)),
        "student_test_aligned_observations": int((test_obs["alignment_status"] == "aligned").sum()),
        "student_test_label_eligible_observations": int(test_obs["label_eligible"].sum()),
        "student_test_candidate_misses": int((test_obs.loc[test_obs["alignment_status"] == "aligned", "candidate_miss"] == True).sum()),  # noqa: E712
        "student_test_total_candidates": int(len(test_cands)),
        "student_test_labeled_candidates": int(test_cands["label"].notna().sum()),
        "student_test_positive_candidates": int((test_cands["label"] == 1).sum()),
        "student_test_negative_candidates": int((test_cands["label"] == 0).sum()),
        "total_extraction_time_s": round(time.perf_counter() - t_start, 2),
    }
    digest["train_candidate_miss_rate"] = round(
        digest["train_candidate_misses"] / max(digest["train_aligned_observations"], 1), 6
    )
    digest["student_test_candidate_miss_rate"] = round(
        digest["student_test_candidate_misses"] / max(digest["student_test_aligned_observations"], 1), 6
    )

    with open(results_dir / "full_dataset_extraction_digest.json", "w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2)

    print("Full 90-trip candidate extraction digest:", json.dumps(digest, indent=2))


if __name__ == "__main__":
    main()
