"""Run full PathSnap dataset audit, candidate sensitivity analysis, preview generation, and Folium mapping."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import folium
import numpy as np
import pandas as pd

# Ensure src is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pathsnap.alignment.label_generation import attach_alignment_labels, summarize_trip_labels
from pathsnap.alignment.route_alignment import align_reference_route
from pathsnap.data.loader import list_trips, load_trip, resolve_data_root, validate_trip_arrays
from pathsnap.schemas.candidate_schema import CandidateConfig
from pathsnap.spatial.candidate_generation import generate_trip_candidates
from pathsnap.spatial.road_network import build_trip_road_network


def generate_dataset_inventory(data_root: Path, out_csv: Path) -> pd.DataFrame:
    """Audit all 80 train and 10 student_test trips and save dataset_inventory.csv."""
    rows: List[Dict[str, object]] = []
    for partition in ["train", "student_test"]:
        trip_ids = list_trips(partition, data_root)
        for tid in trip_ids:
            raw = load_trip(tid, partition=partition, data_root=data_root, include_route=True)
            stats = validate_trip_arrays(raw.trip_id, raw.track, raw.nodes, raw.arcs, raw.route)
            stats["partition"] = partition
            rows.append(stats)

    df = pd.DataFrame(rows)
    # Reorder columns cleanly
    first_cols = ["trip_id", "partition", "num_observations", "num_nodes", "num_arcs", "num_route_arcs"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def run_sensitivity_analysis(
    data_root: Path,
    sample_trip_ids: List[str],
    out_csv: Path,
) -> pd.DataFrame:
    """Evaluate candidate recall / miss rate vs search_radius_m and max_candidates_k on sample training trips."""
    radii = [30.0, 50.0, 75.0, 100.0, 150.0]
    k_values = [3, 5, 8, 10, 15, 20]

    # Preload trips, networks, and reference alignments once per trip
    trip_cache = {}
    for tid in sample_trip_ids:
        raw = load_trip(tid, partition="train", data_root=data_root, include_route=True)
        net = build_trip_road_network(raw.trip_id, raw.nodes, raw.arcs)
        align = align_reference_route(
            track=raw.track,
            nodes=raw.nodes,
            arcs=raw.arcs,
            route=raw.route,
            trip_id=raw.trip_id,
            projector=net.projector,
        )
        trip_cache[tid] = (raw, net, align)

    results = []
    for r_m in radii:
        # Generate candidates with max(k_values) once per radius, then slice by candidate_rank <= k
        max_k = max(k_values)
        cfg_r = CandidateConfig(search_radius_m=r_m, max_candidates_k=max_k, enable_fallback_nearest=True)
        r_results_by_trip = {}
        t0 = time.perf_counter()
        for tid, (raw, net, align) in trip_cache.items():
            c_res = generate_trip_candidates(
                trip_id=raw.trip_id,
                partition=raw.partition,
                track=raw.track,
                nodes=raw.nodes,
                arcs=raw.arcs,
                config=cfg_r,
                road_network=net,
            )
            r_results_by_trip[tid] = (c_res, align)
        elapsed_r = time.perf_counter() - t0

        for k in k_values:
            total_obs = 0
            total_aligned = 0
            total_ambig = 0
            total_unmatched = 0
            total_misses = 0
            total_cand_rows = 0
            total_pos = 0

            for tid, (c_res, align) in r_results_by_trip.items():
                sub_cands = c_res.candidates[c_res.candidates["candidate_rank"] <= k].copy()
                sub_obs = c_res.observations.copy()
                # Recompute candidate_count for cap k
                if not sub_cands.empty:
                    cnts = np.bincount(
                        sub_cands["observation_id"].to_numpy(dtype=np.int32),
                        minlength=len(sub_obs),
                    ).astype(np.int16)
                else:
                    cnts = np.zeros(len(sub_obs), dtype=np.int16)
                sub_obs["candidate_count"] = cnts

                from pathsnap.schemas.candidate_schema import CandidateGenerationResult
                sub_res = CandidateGenerationResult(
                    trip_id=c_res.trip_id,
                    partition=c_res.partition,
                    candidates=sub_cands,
                    observations=sub_obs,
                    graph=c_res.graph,
                )
                labeled = attach_alignment_labels(sub_res, align)
                summ = summarize_trip_labels(labeled)

                total_obs += summ.total_observations
                total_aligned += summ.aligned_observations
                total_ambig += summ.ambiguous_observations
                total_unmatched += summ.unmatched_observations
                total_misses += summ.candidate_miss_count
                total_cand_rows += summ.total_candidate_rows
                total_pos += summ.positive_candidate_rows

            miss_rate = total_misses / total_aligned if total_aligned > 0 else float("nan")
            recall = 1.0 - miss_rate if np.isfinite(miss_rate) else float("nan")
            avg_cands_per_obs = total_cand_rows / total_obs if total_obs > 0 else 0.0

            results.append(
                {
                    "search_radius_m": r_m,
                    "max_candidates_k": k,
                    "num_trips": len(sample_trip_ids),
                    "total_observations": total_obs,
                    "aligned_observations": total_aligned,
                    "ambiguous_observations": total_ambig,
                    "unmatched_observations": total_unmatched,
                    "candidate_miss_count": total_misses,
                    "candidate_miss_rate": round(miss_rate, 6),
                    "candidate_recall": round(recall, 6),
                    "total_candidate_rows": total_cand_rows,
                    "avg_candidates_per_obs": round(avg_cands_per_obs, 3),
                    "positive_class_ratio": round(total_pos / total_cand_rows, 5) if total_cand_rows > 0 else 0.0,
                    "radius_query_time_s": round(elapsed_r, 3),
                }
            )

    df_sens = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sens.to_csv(out_csv, index=False)
    return df_sens


def create_folium_alignment_map(
    raw,
    net,
    labeled_res,
    alignment,
    out_html: Path,
) -> None:
    """Create an interactive Folium map showing raw GPS points, nearby road network, reference route, and aligned segments."""
    lon = raw.track["longitude"].to_numpy()
    lat = raw.track["latitude"].to_numpy()
    center_lat = float(np.mean(lat))
    center_lon = float(np.mean(lon))

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # 1. Layer: Candidate Road Network Subset (all arcs that appeared as top-3 candidates or in route)
    top_cands = labeled_res.candidates[labeled_res.candidates["candidate_rank"] <= 3]
    cand_arc_ids = np.unique(top_cands["candidate_arc_id"].to_numpy(dtype=np.int32))
    route_arc_ids = np.unique(raw.route) if raw.route is not None else np.empty(0, dtype=np.int32)
    relevant_arcs = np.unique(np.concatenate([cand_arc_ids, route_arc_ids]))

    fg_net = folium.FeatureGroup(name="Candidate & Nearby Road Network", show=True)
    for aid in relevant_arcs:
        u, v = int(raw.arcs[aid, 0]), int(raw.arcs[aid, 1])
        coords = [
            [float(raw.nodes[u, 1]), float(raw.nodes[u, 0])],
            [float(raw.nodes[v, 1]), float(raw.nodes[v, 0])],
        ]
        folium.PolyLine(
            coords,
            color="#888888",
            weight=2,
            opacity=0.5,
            tooltip=f"Arc {aid} ({u} -> {v}), len={net.segment_lengths_m[aid]:.1f}m",
        ).add_to(fg_net)
    fg_net.add_to(m)

    # 2. Layer: Supplied Reference Route (.route)
    if raw.route is not None:
        fg_route = folium.FeatureGroup(name="Supplied Reference Route (.route)", show=True)
        for pos_idx, aid in enumerate(raw.route):
            u, v = int(raw.arcs[aid, 0]), int(raw.arcs[aid, 1])
            coords = [
                [float(raw.nodes[u, 1]), float(raw.nodes[u, 0])],
                [float(raw.nodes[v, 1]), float(raw.nodes[v, 0])],
            ]
            folium.PolyLine(
                coords,
                color="#16a34a",
                weight=5,
                opacity=0.85,
                tooltip=f"Reference Route Pos {pos_idx}: Arc {aid} ({u}->{v})",
            ).add_to(fg_route)
        fg_route.add_to(m)

    # 3. Layer: Raw GPS Observations colored by alignment status
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
    obs_df = labeled_res.observations
    color_map = {
        "aligned": "#2563eb",
        "ambiguous": "#d97706",
        "unmatched": "#dc2626",
        "not_available": "#6b7280",
    }
    # Plot trajectory polyline lightly
    traj_coords = [[float(lat[i]), float(lon[i])] for i in range(len(lat))]
    folium.PolyLine(traj_coords, color="#3b82f6", weight=2, opacity=0.4, dash_array="4").add_to(fg_gps)

    # Plot every 3rd point + all non-aligned or missed points so the HTML remains fast and responsive
    for i in range(len(obs_df)):
        row = obs_df.iloc[i]
        st = str(row["alignment_status"])
        is_miss = bool(row["candidate_miss"]) if pd.notna(row["candidate_miss"]) else False
        if i % 3 != 0 and st == "aligned" and not is_miss:
            continue
        c = "#9333ea" if is_miss else color_map.get(st, "#2563eb")
        ref_a = row["reference_arc_id"]
        dist_a = row["alignment_distance_m"]
        folium.CircleMarker(
            location=[float(row["gps_latitude"]), float(row["gps_longitude"])],
            radius=4 if not is_miss else 6,
            color=c,
            fill=True,
            fill_opacity=0.85,
            tooltip=(
                f"Obs {int(row['observation_id'])} | t={row['timestamp_s']:.0f}s | "
                f"status={st} | ref_arc={ref_a} | align_dist={dist_a:.1f}m | "
                f"cands={int(row['candidate_count'])} | miss={row['candidate_miss']}"
            ),
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def main() -> None:
    data_root = resolve_data_root()
    print(f"Resolved dataset root: {data_root}")

    # 1. Full dataset inventory (all 80 train + 10 student_test trips)
    inv_path = REPO_ROOT / "data" / "dataset_inventory.csv"
    t0 = time.perf_counter()
    inv_df = generate_dataset_inventory(data_root, inv_path)
    print(f"Generated dataset_inventory.csv ({len(inv_df)} trips) in {time.perf_counter() - t0:.2f}s")

    # 2. Candidate generation & alignment across 10 representative training trips
    sample_train_trips = list_trips("train", data_root)[:10]
    sens_path = REPO_ROOT / "data" / "sensitivity_analysis.csv"
    t0 = time.perf_counter()
    sens_df = run_sensitivity_analysis(data_root, sample_train_trips, sens_path)
    print(f"Generated sensitivity_analysis.csv in {time.perf_counter() - t0:.2f}s")

    # 3. Generate candidate_previews (candidates.parquet, observations.parquet, CSV samples, and Folium map)
    preview_dir = REPO_ROOT / "outputs" / "candidate_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    cfg = CandidateConfig(search_radius_m=100.0, max_candidates_k=10)
    cand_frames = []
    obs_frames = []
    trip_summaries = []

    for idx, tid in enumerate(sample_train_trips[:5]):
        raw = load_trip(tid, partition="train", data_root=data_root, include_route=True)
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
        labeled_res = attach_alignment_labels(c_res, align)
        summ = summarize_trip_labels(labeled_res)
        d_dict = summ.to_dict()
        d_dict["median_align_dist_m"] = round(align.diagnostics["median_alignment_distance_m"], 3)
        d_dict["p95_align_dist_m"] = round(align.diagnostics["p95_alignment_distance_m"], 3)
        d_dict["is_monotonic"] = bool(align.diagnostics["is_monotonic"])
        trip_summaries.append(d_dict)

        cand_frames.append(labeled_res.candidates)
        obs_frames.append(labeled_res.observations)

        if idx == 0:
            map_path = preview_dir / f"trip_{tid}_alignment_map.html"
            create_folium_alignment_map(raw, net, labeled_res, align, map_path)
            print(f"Saved interactive Folium map to {map_path}")

    all_cands = pd.concat(cand_frames, ignore_index=True)
    all_obs = pd.concat(obs_frames, ignore_index=True)

    all_cands.to_parquet(preview_dir / "candidates.parquet", index=False)
    all_obs.to_parquet(preview_dir / "observations.parquet", index=False)
    all_cands.head(200).to_csv(preview_dir / "candidates_preview_200.csv", index=False)
    all_obs.head(200).to_csv(preview_dir / "observations_preview_200.csv", index=False)

    pd.DataFrame(trip_summaries).to_csv(preview_dir / "alignment_validation_summary.csv", index=False)

    # Save JSON digest of actual dataset observations for documentation
    digest = {
        "total_trips": int(len(inv_df)),
        "train_trips": int((inv_df["partition"] == "train").sum()),
        "student_test_trips": int((inv_df["partition"] == "student_test").sum()),
        "total_gps_observations": int(inv_df["num_observations"].sum()),
        "train_gps_observations": int(inv_df.loc[inv_df["partition"] == "train", "num_observations"].sum()),
        "student_test_gps_observations": int(inv_df.loc[inv_df["partition"] == "student_test", "num_observations"].sum()),
        "min_obs_per_trip": int(inv_df["num_observations"].min()),
        "median_obs_per_trip": float(inv_df["num_observations"].median()),
        "max_obs_per_trip": int(inv_df["num_observations"].max()),
        "min_arcs_per_trip": int(inv_df["num_arcs"].min()),
        "median_arcs_per_trip": float(inv_df["num_arcs"].median()),
        "max_arcs_per_trip": int(inv_df["num_arcs"].max()),
        "overall_lon_range": [round(float(inv_df["min_lon"].min()), 6), round(float(inv_df["max_lon"].max()), 6)],
        "overall_lat_range": [round(float(inv_df["min_lat"].min()), 6), round(float(inv_df["max_lat"].max()), 6)],
        "mean_exact_1s_ratio": round(float(inv_df["exact_1s_ratio"].mean()), 6),
        "total_non_finite_gps": int(inv_df["non_finite_gps"].sum()),
        "total_invalid_arc_nodes": int(inv_df["invalid_arc_nodes"].sum()),
        "total_invalid_route_arcs": int(inv_df["invalid_route_arcs"].sum()),
        "total_disconnected_route_transitions": int(inv_df["disconnected_route_transitions"].sum()),
        "trips_with_gaps_gt_5s": int((inv_df["gap_gt_5s_count"] > 0).sum()),
        "trips_with_gaps_gt_30s": int((inv_df["gap_gt_30s_count"] > 0).sum()),
        "max_time_gap_s": float(inv_df["max_dt_s"].max()),
        "trips_with_isolated_nodes": int((inv_df["isolated_nodes_count"] > 0).sum()),
    }
    with open(preview_dir / "dataset_audit_digest.json", "w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2)

    print("Audit digest:", json.dumps(digest, indent=2))


if __name__ == "__main__":
    main()
