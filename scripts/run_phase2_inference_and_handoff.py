"""Execute Phase 2 route-free inference and isolated evaluation across all 10 student_test trips.

Owner: Qamar Ahmed (Technical Lead)

Generates:
  - results/phase2/student_test_trip_metrics.csv
  - results/phase2/student_test_aggregate_summary.json
  - results/phase2/classifier_only_predictions_sample.csv  (for Gokul's evaluation handoff)
  - results/phase2/hmm_viterbi_predictions_sample.csv      (for Gokul's evaluation handoff)
  - results/phase2/trip_00000000_student_test_map.html     (Interactive Folium map on held-out student_test)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import folium
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pathsnap.data.loader import list_trips, load_trip, resolve_data_root
from pathsnap.inference.inference_pipeline import run_trip_inference
from pathsnap.inference.model_scorer import CandidateModelScorer
from pathsnap.inference.server_evaluator import evaluate_trip_predictions_server_side
from pathsnap.schemas.candidate_schema import CandidateConfig
from sequence.viterbi_decoder import ViterbiConfig


def create_student_test_folium_map(
    raw,
    inf_out,
    out_html: Path,
) -> None:
    """Create an interactive Folium map on a held-out student_test trip comparing Raw GPS, Reference Route, Greedy, and HMM/Viterbi."""
    lon = raw.track["longitude"].to_numpy()
    lat = raw.track["latitude"].to_numpy()
    m = folium.Map(
        location=[float(np.mean(lat)), float(np.mean(lon))],
        zoom_start=14,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Positron (Street Map)",
        subdomains="abcd",
        max_zoom=20,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri World Street Map",
    ).add_to(m)

    if raw.route is not None:
        fg_ref = folium.FeatureGroup(name="Supplied Reference Route (.route)", show=True)
        for aid in raw.route:
            u, v = int(raw.arcs[aid, 0]), int(raw.arcs[aid, 1])
            folium.PolyLine(
                [[float(raw.nodes[u, 1]), float(raw.nodes[u, 0])], [float(raw.nodes[v, 1]), float(raw.nodes[v, 0])]],
                color="#16a34a",
                weight=6,
                opacity=0.75,
                tooltip=f"Reference Arc {aid}",
            ).add_to(fg_ref)
        fg_ref.add_to(m)

    fg_greedy = folium.FeatureGroup(name="Classifier-Only (Greedy Top-1)", show=False)
    for seg in inf_out.classifier_only_route_segments_latlon:
        folium.PolyLine(
            seg,
            color="#f59e0b",
            weight=4,
            opacity=0.85,
            dash_array="6",
            tooltip="Classifier-Only Predicted Segment",
        ).add_to(fg_greedy)
    fg_greedy.add_to(m)

    fg_vit = folium.FeatureGroup(name="Classifier + HMM/Viterbi Reconstructed Route", show=True)
    for seg in inf_out.predicted_route_segments_latlon:
        folium.PolyLine(
            seg,
            color="#dc2626",
            weight=4,
            opacity=0.9,
            tooltip="HMM/Viterbi Predicted Segment",
        ).add_to(fg_vit)
    fg_vit.add_to(m)

    fg_gps = folium.FeatureGroup(name="Raw GPS Trajectory (.track)", show=True)
    folium.PolyLine(
        inf_out.gps_trajectory_latlon,
        color="#2563eb",
        weight=2,
        opacity=0.55,
        tooltip="Raw GPS Trajectory",
    ).add_to(fg_gps)
    fg_gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def main() -> None:
    t0_all = time.perf_counter()
    data_root = resolve_data_root()
    scorer = CandidateModelScorer.from_path()
    cand_cfg = CandidateConfig()
    vit_cfg = ViterbiConfig.from_yaml(REPO_ROOT / "configs" / "default.yaml")

    out_dir = REPO_ROOT / "results" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    student_test_trips = list_trips("student_test", data_root)
    print(f"Running Phase 2 Inference & Server Evaluation on {len(student_test_trips)} student_test trips...")

    trip_rows: List[Dict[str, object]] = []
    clf_only_frames: List[pd.DataFrame] = []
    hmm_frames: List[pd.DataFrame] = []

    for idx, tid in enumerate(student_test_trips):
        inf_out = run_trip_inference(
            trip_id=tid,
            partition="student_test",
            data_root=data_root,
            scorer=scorer,
            candidate_config=cand_cfg,
            viterbi_config=vit_cfg,
        )
        eval_res = evaluate_trip_predictions_server_side(
            inference_output=inf_out,
            data_root=data_root,
            candidate_config=cand_cfg,
        )

        row = {
            "trip_id": tid,
            "partition": "student_test",
            "model_name": inf_out.model_name,
            "num_observations": inf_out.candidate_summary["num_observations"],
            "num_road_arcs": inf_out.candidate_summary["num_road_arcs"],
            "total_candidates": inf_out.candidate_summary["total_candidates"],
            "avg_candidates_per_obs": inf_out.candidate_summary["avg_candidates_per_observation"],
            "fallback_candidates_count": inf_out.candidate_summary["fallback_candidates_count"],
            "aligned_observations": eval_res.metrics.get("aligned_observations"),
            "candidate_miss_count": eval_res.metrics.get("candidate_miss_count"),
            "candidate_miss_rate": eval_res.metrics.get("candidate_miss_rate"),
            "classifier_only_top1_accuracy": eval_res.metrics.get("classifier_only_top1_accuracy"),
            "hmm_viterbi_top1_accuracy": eval_res.metrics.get("hmm_viterbi_top1_accuracy"),
            "top1_accuracy_improvement_delta": eval_res.metrics.get("top1_accuracy_improvement_delta"),
            "classifier_only_route_continuity": eval_res.metrics.get("classifier_only_route_continuity"),
            "hmm_viterbi_route_continuity": eval_res.metrics.get("hmm_viterbi_route_continuity"),
            "route_continuity_improvement_delta": eval_res.metrics.get("route_continuity_improvement_delta"),
            "hmm_viterbi_multihop_reachability": eval_res.metrics.get("hmm_viterbi_multihop_reachability"),
            "candidate_macro_f1": eval_res.metrics.get("candidate_macro_f1"),
            "candidate_log_loss": eval_res.metrics.get("candidate_log_loss"),
            "classifier_only_route_arc_f1": eval_res.metrics.get("classifier_only_route_arc_f1"),
            "hmm_viterbi_route_arc_f1": eval_res.metrics.get("hmm_viterbi_route_arc_f1"),
            "changed_observations_by_hmm": eval_res.metrics.get("changed_observations_by_hmm"),
            "normal_decoded_obs": int(inf_out.decoder_summary.get("normal_decoded_observations", 0)),
            "fallback_decoded_obs": int(inf_out.decoder_summary.get("fallback_decoded_observations", 0)),
            "gap_reset_obs": int(inf_out.decoder_summary.get("gap_reset_observations", 0)),
            "disconnected_bridge_obs": int(inf_out.decoder_summary.get("disconnected_bridge_observations", 0)),
            "data_loading_ms": inf_out.timing_ms["data_loading_ms"],
            "candidate_generation_ms": inf_out.timing_ms["candidate_generation_ms"],
            "model_scoring_ms": inf_out.timing_ms["model_scoring_ms"],
            "viterbi_decoding_ms": inf_out.timing_ms["viterbi_decoding_ms"],
            "total_pipeline_ms": inf_out.timing_ms["total_pipeline_ms"],
            "server_evaluation_ms": eval_res.evaluation_time_ms,
        }
        trip_rows.append(row)
        print(
            f"  Trip {tid}: obs={row['num_observations']} | "
            f"Greedy Top-1={row['classifier_only_top1_accuracy']:.4f} -> "
            f"Viterbi Top-1={row['hmm_viterbi_top1_accuracy']:.4f} "
            f"(delta={row['top1_accuracy_improvement_delta']:+.4f}) | "
            f"Continuity={row['classifier_only_route_continuity']:.4f} -> "
            f"{row['hmm_viterbi_route_continuity']:.4f} | "
            f"time={row['total_pipeline_ms']:.1f}ms"
        )

        if idx < 2:
            clf_only_frames.append(inf_out.classifier_only_predictions.head(150))
            hmm_frames.append(inf_out.hmm_viterbi_predictions.head(150))

        if idx == 0:
            raw_0 = load_trip(tid, partition="student_test", data_root=data_root, include_route=True)
            map_path = out_dir / f"trip_{tid}_student_test_map.html"
            create_student_test_folium_map(raw_0, inf_out, map_path)
            print(f"  Saved student_test interactive Folium map to {map_path}")

    df_metrics = pd.DataFrame(trip_rows)
    df_metrics.to_csv(out_dir / "student_test_trip_metrics.csv", index=False)

    pd.concat(clf_only_frames, ignore_index=True).to_csv(
        out_dir / "classifier_only_predictions_sample.csv", index=False
    )
    pd.concat(hmm_frames, ignore_index=True).to_csv(
        out_dir / "hmm_viterbi_predictions_sample.csv", index=False
    )

    # Compute observation-weighted & trip-mean aggregates across all 10 student_test trips
    tot_obs = int(df_metrics["num_observations"].sum())
    tot_aligned = int(df_metrics["aligned_observations"].sum())
    tot_misses = int(df_metrics["candidate_miss_count"].sum())
    weighted_greedy_top1 = float(
        (df_metrics["classifier_only_top1_accuracy"] * df_metrics["aligned_observations"]).sum() / tot_aligned
    )
    weighted_viterbi_top1 = float(
        (df_metrics["hmm_viterbi_top1_accuracy"] * df_metrics["aligned_observations"]).sum() / tot_aligned
    )

    agg_summary = {
        "partition": "student_test",
        "num_trips": len(df_metrics),
        "model_name": scorer.model_name,
        "model_path": scorer.model_path,
        "total_observations": tot_obs,
        "total_aligned_observations": tot_aligned,
        "total_candidates_generated": int(df_metrics["total_candidates"].sum()),
        "overall_candidate_miss_count": tot_misses,
        "overall_candidate_miss_rate": round(tot_misses / max(tot_aligned, 1), 6),
        "mean_trip_classifier_only_top1_accuracy": round(float(df_metrics["classifier_only_top1_accuracy"].mean()), 6),
        "mean_trip_hmm_viterbi_top1_accuracy": round(float(df_metrics["hmm_viterbi_top1_accuracy"].mean()), 6),
        "mean_trip_top1_accuracy_delta": round(float(df_metrics["top1_accuracy_improvement_delta"].mean()), 6),
        "weighted_classifier_only_top1_accuracy": round(weighted_greedy_top1, 6),
        "weighted_hmm_viterbi_top1_accuracy": round(weighted_viterbi_top1, 6),
        "weighted_top1_accuracy_delta": round(weighted_viterbi_top1 - weighted_greedy_top1, 6),
        "mean_trip_classifier_only_continuity": round(float(df_metrics["classifier_only_route_continuity"].mean()), 6),
        "mean_trip_hmm_viterbi_continuity": round(float(df_metrics["hmm_viterbi_route_continuity"].mean()), 6),
        "mean_trip_continuity_delta": round(float(df_metrics["route_continuity_improvement_delta"].mean()), 6),
        "mean_trip_multihop_reachability": round(float(df_metrics["hmm_viterbi_multihop_reachability"].mean()), 6),
        "mean_trip_candidate_macro_f1": round(float(df_metrics["candidate_macro_f1"].mean()), 6),
        "mean_trip_candidate_log_loss": round(float(df_metrics["candidate_log_loss"].mean()), 6),
        "mean_trip_classifier_only_route_arc_f1": round(float(df_metrics["classifier_only_route_arc_f1"].mean()), 6),
        "mean_trip_hmm_viterbi_route_arc_f1": round(float(df_metrics["hmm_viterbi_route_arc_f1"].mean()), 6),
        "trips_improved_top1_by_hmm": int((df_metrics["top1_accuracy_improvement_delta"] > 0).sum()),
        "trips_improved_continuity_by_hmm": int((df_metrics["route_continuity_improvement_delta"] > 0).sum()),
        "total_inference_pipeline_ms": round(float(df_metrics["total_pipeline_ms"].sum()), 2),
        "mean_inference_pipeline_ms_per_trip": round(float(df_metrics["total_pipeline_ms"].mean()), 2),
        "mean_model_scoring_ms_per_trip": round(float(df_metrics["model_scoring_ms"].mean()), 2),
        "mean_viterbi_decoding_ms_per_trip": round(float(df_metrics["viterbi_decoding_ms"].mean()), 2),
        "wall_clock_elapsed_s": round(time.perf_counter() - t0_all, 3),
    }

    (out_dir / "student_test_aggregate_summary.json").write_text(
        json.dumps(agg_summary, indent=2), encoding="utf-8"
    )
    print("Aggregate student_test summary:", json.dumps(agg_summary, indent=2))


if __name__ == "__main__":
    main()
