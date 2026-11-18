"""Execute Phase 1 ML Experimentation (5 Models), Optuna Bayesian Tuning, and HMM/Viterbi Verification."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import folium
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.model_pipeline import (
    RANDOM_SEED,
    evaluate_five_models_group_cv,
    verify_candidate_contract,
)
from optimization.bayesian_tuning import run_bayesian_tuning_hgb
from pathsnap.data.loader import load_trip, resolve_data_root
from pathsnap.graph.road_graph import TripRoadGraph, build_trip_road_graph
from pathsnap.spatial.road_network import build_trip_road_network
from sequence.viterbi_decoder import ViterbiConfig, decode_trip_viterbi


def run_synthetic_hmm_benchmarks() -> List[Dict[str, object]]:
    """Run deterministic synthetic directed-graph tests to verify Viterbi behaviour independently."""
    rows: List[Dict[str, object]] = []

    # Case 1: Parallel distractor road spike override
    arcs1 = np.array([[0, 1], [1, 2], [2, 3], [10, 11]], dtype=np.int32)
    lengths1 = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32)
    bearings1 = np.array([90.0, 90.0, 90.0, 90.0], dtype=np.float32)
    g1 = build_trip_road_graph("SYNTH_PARALLEL_SPIKE", arcs1, lengths1, bearings1)

    obs1 = pd.DataFrame(
        {
            "trip_id": ["SYNTH_PARALLEL_SPIKE"] * 3,
            "observation_id": np.array([0, 1, 2], dtype=np.int32),
            "timestamp_s": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "alignment_status": ["aligned", "aligned", "aligned"],
            "reference_arc_id": np.array([0, 1, 2], dtype=np.int32),
        }
    )
    cands1 = pd.DataFrame(
        {
            "trip_id": ["SYNTH_PARALLEL_SPIKE"] * 6,
            "observation_id": np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
            "candidate_rank": np.array([1, 2, 1, 2, 1, 2], dtype=np.int16),
            "candidate_arc_id": np.array([0, 3, 3, 1, 2, 3], dtype=np.int32),
            "distance_to_segment_m": np.array([3.0, 15.0, 5.0, 7.0, 4.0, 18.0], dtype=np.float32),
            "pred_prob": np.array([0.92, 0.08, 0.60, 0.45, 0.90, 0.10], dtype=np.float64),
        }
    )
    res1 = decode_trip_viterbi(cands1, obs1, g1, prob_col="pred_prob")
    rows.append(
        {
            "test_type": "synthetic_deterministic",
            "trip_or_case_id": "SYNTH_PARALLEL_SPIKE",
            "model_source": "Synthetic Emission Spike",
            "num_observations": 3,
            "aligned_observations": 3,
            "greedy_top1_accuracy": res1.diagnostics["greedy_top1_accuracy"],
            "viterbi_top1_accuracy": res1.diagnostics["viterbi_top1_accuracy"],
            "top1_accuracy_delta": round(
                res1.diagnostics["viterbi_top1_accuracy"] - res1.diagnostics["greedy_top1_accuracy"], 6
            ),
            "greedy_route_continuity": res1.diagnostics["greedy_route_continuity"],
            "viterbi_route_continuity": res1.diagnostics["viterbi_route_continuity"],
            "continuity_delta": round(
                res1.diagnostics["viterbi_route_continuity"] - res1.diagnostics["greedy_route_continuity"], 6
            ),
            "changed_observations_count": int(res1.diagnostics["changed_observations_count"]),
        }
    )

    # Case 2: Empty candidate set + large GPS gap (>60s) recovery
    obs2 = pd.DataFrame(
        {
            "trip_id": ["SYNTH_GAP_AND_EMPTY"] * 4,
            "observation_id": np.array([0, 1, 2, 3], dtype=np.int32),
            "timestamp_s": np.array([1.0, 2.0, 3.0, 120.0], dtype=np.float64),
            "alignment_status": ["aligned", "aligned", "aligned", "aligned"],
            "reference_arc_id": np.array([0, 0, 1, 3], dtype=np.int32),
        }
    )
    cands2 = pd.DataFrame(
        {
            "trip_id": ["SYNTH_GAP_AND_EMPTY"] * 4,
            "observation_id": np.array([0, 2, 3, 3], dtype=np.int32),
            "candidate_rank": np.array([1, 1, 1, 2], dtype=np.int16),
            "candidate_arc_id": np.array([0, 1, 3, 2], dtype=np.int32),
            "distance_to_segment_m": np.array([2.0, 3.0, 2.0, 25.0], dtype=np.float32),
            "pred_prob": np.array([0.95, 0.90, 0.85, 0.15], dtype=np.float64),
        }
    )
    res2 = decode_trip_viterbi(cands2, obs2, g1, prob_col="pred_prob")
    rows.append(
        {
            "test_type": "synthetic_deterministic",
            "trip_or_case_id": "SYNTH_GAP_AND_EMPTY",
            "model_source": "Synthetic Empty-Cand + 117s Gap",
            "num_observations": 4,
            "aligned_observations": 4,
            "greedy_top1_accuracy": res2.diagnostics["greedy_top1_accuracy"],
            "viterbi_top1_accuracy": res2.diagnostics["viterbi_top1_accuracy"],
            "top1_accuracy_delta": round(
                res2.diagnostics["viterbi_top1_accuracy"] - res2.diagnostics["greedy_top1_accuracy"], 6
            ),
            "greedy_route_continuity": res2.diagnostics["greedy_route_continuity"],
            "viterbi_route_continuity": res2.diagnostics["viterbi_route_continuity"],
            "continuity_delta": round(
                res2.diagnostics["viterbi_route_continuity"] - res2.diagnostics["greedy_route_continuity"], 6
            ),
            "changed_observations_count": int(res2.diagnostics["changed_observations_count"]),
        }
    )
    return rows


def create_phase1_plots(
    model_comp_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    hmm_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate clean diagnostic plots for Phase 1 evaluation."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model Comparison: Macro F1 and Observation Top-1 Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = model_comp_df["model_name"].tolist()
    y_pos = np.arange(len(names))

    axes[0].barh(y_pos, model_comp_df["oof_macro_f1"], color="#2563eb", alpha=0.85, label="OOF Macro F1")
    axes[0].barh(
        y_pos,
        model_comp_df["oof_obs_top1_accuracy"],
        color="#16a34a",
        alpha=0.55,
        label="Obs Top-1 Accuracy",
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names)
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_xlabel("Score")
    axes[0].set_title("5-Model Comparison: Macro F1 & Top-1 Accuracy (5-Trip Pilot)")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    axes[1].barh(y_pos, model_comp_df["oof_log_loss"], color="#dc2626", alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(names)
    axes[1].set_xlabel("Out-of-Fold Log Loss (Lower is Better)")
    axes[1].set_title("5-Model Calibration: Out-of-Fold Log Loss")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_chart.png", dpi=160)
    plt.close()

    # 2. Bayesian Optimisation Progression Plot
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    trial_nums = trials_df["trial_number"].to_numpy()
    f1_vals = trials_df["oof_macro_f1"].to_numpy()
    best_so_far = np.maximum.accumulate(f1_vals)
    ax.plot(trial_nums, f1_vals, "o--", color="#6b7280", label="Trial OOF Macro F1")
    ax.plot(trial_nums, best_so_far, "s-", color="#2563eb", linewidth=2, label="Best-So-Far OOF Macro F1")
    ax.set_xlabel("Optuna TPE Trial Number")
    ax.set_ylabel("Out-of-Fold Macro F1")
    ax.set_title("Bayesian Hyperparameter Optimisation (Optuna TPE — Preliminary 5-Trip CV)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "bayesian_tuning_progression.png", dpi=160)
    plt.close()


def create_viterbi_folium_map(
    raw,
    net,
    obs_df: pd.DataFrame,
    viterbi_res,
    out_html: Path,
) -> None:
    """Create an interactive Folium map comparing Raw GPS, Reference Route, Greedy Top-1, and HMM/Viterbi Route."""
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
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors",
        name="OpenStreetMap",
        referrer_policy="origin",
    ).add_to(m)

    # 1. Reference Route (.route)
    if raw.route is not None:
        fg_ref = folium.FeatureGroup(name="Reference Route (.route)", show=True)
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

    # 2. Classifier-Only (Greedy Top-1) Route
    fg_greedy = folium.FeatureGroup(name="Classifier-Only (Greedy Top-1)", show=False)
    for aid in viterbi_res.greedy_route_arcs:
        u, v = int(raw.arcs[aid, 0]), int(raw.arcs[aid, 1])
        folium.PolyLine(
            [[float(raw.nodes[u, 1]), float(raw.nodes[u, 0])], [float(raw.nodes[v, 1]), float(raw.nodes[v, 0])]],
            color="#f59e0b",
            weight=4,
            opacity=0.85,
            dash_array="6",
            tooltip=f"Greedy Predicted Arc {aid}",
        ).add_to(fg_greedy)
    fg_greedy.add_to(m)

    # 3. Classifier + HMM/Viterbi Reconstructed Route
    fg_vit = folium.FeatureGroup(name="Classifier + HMM/Viterbi Decoded Route", show=True)
    for aid in viterbi_res.reconstructed_route_arcs:
        u, v = int(raw.arcs[aid, 0]), int(raw.arcs[aid, 1])
        folium.PolyLine(
            [[float(raw.nodes[u, 1]), float(raw.nodes[u, 0])], [float(raw.nodes[v, 1]), float(raw.nodes[v, 0])]],
            color="#dc2626",
            weight=4,
            opacity=0.9,
            tooltip=f"HMM/Viterbi Decoded Arc {aid}",
        ).add_to(fg_vit)
    fg_vit.add_to(m)

    # 4. Raw GPS Trajectory
    fg_gps = folium.FeatureGroup(name="Raw GPS Trajectory (.track)", show=True)
    folium.PolyLine(
        [[float(lat[i]), float(lon[i])] for i in range(len(lat))],
        color="#2563eb",
        weight=2,
        opacity=0.5,
        tooltip="Raw GPS Trajectory",
    ).add_to(fg_gps)
    fg_gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def main() -> None:
    t0_all = time.perf_counter()
    data_root = resolve_data_root()
    preview_dir = REPO_ROOT / "outputs" / "candidate_previews"
    phase1_dir = REPO_ROOT / "results" / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    cands_path = preview_dir / "candidates.parquet"
    obs_path = preview_dir / "observations.parquet"
    if not cands_path.exists() or not obs_path.exists():
        raise FileNotFoundError(
            "Missing candidates.parquet or observations.parquet. Run scripts/run_audit_and_previews.py first."
        )

    cands_df = pd.read_parquet(cands_path)
    obs_df = pd.read_parquet(obs_path)

    # 1. Verify contract & save candidate_generation_summary.csv + label_distribution.csv
    contract_stats = verify_candidate_contract(cands_df, obs_df)
    print("Verified candidate contract:", json.dumps(contract_stats, indent=2))

    # Copy/augment per-trip candidate generation summary
    align_val_csv = preview_dir / "alignment_validation_summary.csv"
    cand_gen_summary_df = pd.read_csv(align_val_csv, dtype={"trip_id": str})
    cand_gen_summary_df.to_csv(phase1_dir / "candidate_generation_summary.csv", index=False)

    # Build detailed label_distribution.csv per trip + aggregate row
    label_rows = []
    for tid, grp_c in cands_df.groupby("trip_id", sort=True):
        grp_o = obs_df[obs_df["trip_id"] == tid]
        pos_c = int((grp_c["label"] == 1).sum())
        neg_c = int((grp_c["label"] == 0).sum())
        null_c = int(grp_c["label"].isna().sum())
        tot_c = len(grp_c)
        lab_c = pos_c + neg_c
        aligned_o = int((grp_o["alignment_status"] == "aligned").sum())
        ambig_o = int((grp_o["alignment_status"] == "ambiguous").sum())
        unm_o = int((grp_o["alignment_status"] == "unmatched").sum())
        miss_o = int((grp_o.loc[grp_o["alignment_status"] == "aligned", "candidate_miss"] == True).sum())  # noqa: E712
        label_rows.append(
            {
                "trip_id": str(tid),
                "total_observations": int(len(grp_o)),
                "aligned_observations": aligned_o,
                "ambiguous_observations": ambig_o,
                "unmatched_observations": unm_o,
                "label_eligible_observations": int(grp_o["label_eligible"].sum()),
                "candidate_miss_count": miss_o,
                "candidate_miss_rate": round(miss_o / max(aligned_o, 1), 6),
                "total_candidate_rows": tot_c,
                "labeled_candidate_rows": lab_c,
                "null_label_candidate_rows": null_c,
                "positive_label_count": pos_c,
                "negative_label_count": neg_c,
                "positive_class_ratio_among_labeled": round(pos_c / max(lab_c, 1), 6),
            }
        )

    # Add aggregate TOTAL row
    label_rows.append(
        {
            "trip_id": "ALL_5_PREVIEW_TRIPS",
            "total_observations": contract_stats["total_observations"],
            "aligned_observations": contract_stats["aligned_observations"],
            "ambiguous_observations": int((obs_df["alignment_status"] == "ambiguous").sum()),
            "unmatched_observations": int((obs_df["alignment_status"] == "unmatched").sum()),
            "label_eligible_observations": contract_stats["label_eligible_observations"],
            "candidate_miss_count": contract_stats["candidate_miss_count"],
            "candidate_miss_rate": contract_stats["candidate_miss_rate"],
            "total_candidate_rows": contract_stats["total_candidate_rows"],
            "labeled_candidate_rows": contract_stats["labeled_candidate_rows"],
            "null_label_candidate_rows": contract_stats["null_label_candidate_rows"],
            "positive_label_count": contract_stats["positive_label_count"],
            "negative_label_count": contract_stats["negative_label_count"],
            "positive_class_ratio_among_labeled": contract_stats["positive_class_ratio"],
        }
    )
    label_dist_df = pd.DataFrame(label_rows)
    label_dist_df.to_csv(phase1_dir / "label_distribution.csv", index=False)

    # 2. Load TripRoadGraph for each of the 5 preview trips
    graphs_by_trip: Dict[str, TripRoadGraph] = {}
    raw_by_trip = {}
    net_by_trip = {}
    for tid in sorted(cands_df["trip_id"].unique()):
        raw = load_trip(str(tid), partition="train", data_root=data_root, include_route=True)
        net = build_trip_road_network(raw.trip_id, raw.nodes, raw.arcs)
        graphs_by_trip[str(tid)] = net.graph
        raw_by_trip[str(tid)] = raw
        net_by_trip[str(tid)] = net

    # 3. Execute 5-Model Experimentation with 5-Fold GroupKFold
    print("Running 5-Model GroupKFold CV across the 5 representative training trips...")
    model_comp_df, cv_results_df, oof_probs_by_model = evaluate_five_models_group_cv(
        cands_df, obs_df, graphs_by_trip, n_splits=5, random_state=RANDOM_SEED
    )
    model_comp_df.to_csv(phase1_dir / "model_comparison.csv", index=False)
    cv_results_df.to_csv(phase1_dir / "cv_results.csv", index=False)
    print(model_comp_df[["model_name", "oof_macro_f1", "oof_log_loss", "oof_obs_top1_accuracy", "oof_route_continuity"]])

    # 4. Execute Genuine Optuna TPE Bayesian Hyperparameter Optimisation (8 trials)
    print("Running Optuna TPE Bayesian Hyperparameter Optimisation for HistGradientBoosting...")
    trials_df, best_bayes_summary, tuned_hgb_model, best_oof_probs = run_bayesian_tuning_hgb(
        cands_df, obs_df, graphs_by_trip, n_trials=8, n_splits=5, random_state=RANDOM_SEED
    )
    trials_df.to_csv(phase1_dir / "bayesian_trials.csv", index=False)
    with open(phase1_dir / "bayesian_best_summary.json", "w", encoding="utf-8") as f:
        json.dump(best_bayes_summary, f, indent=2)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_hgb_model, artifacts_dir / "phase1_pilot_tuned_hgb.joblib")
    print("Bayesian Tuning Summary:", json.dumps(best_bayes_summary, indent=2))

    # 5. Execute HMM / Viterbi Route Reconstruction Benchmarks (Synthetic + Real OOF ML Probabilities)
    print("Running HMM/Viterbi decoding on synthetic cases and real OOF ML candidate probabilities...")
    hmm_rows = run_synthetic_hmm_benchmarks()

    # Decode each of the 5 preview trips using out-of-fold probabilities from Tuned HistGradientBoosting and Logistic Regression
    cands_with_probs = cands_df.copy()
    cands_with_probs["pred_prob_hgb"] = best_oof_probs
    cands_with_probs["pred_prob_lr"] = oof_probs_by_model["logistic_regression"]

    viterbi_cfg = ViterbiConfig()
    for model_label, pcol in [
        ("Tuned HistGradientBoosting (OOF)", "pred_prob_hgb"),
        ("Logistic Regression (OOF)", "pred_prob_lr"),
    ]:
        tot_obs_m = 0
        tot_aligned_m = 0
        greedy_correct_m = 0
        viterbi_correct_m = 0
        tot_steps_m = 0
        greedy_conn_m = 0
        viterbi_conn_m = 0
        tot_changed_m = 0

        for tid in sorted(graphs_by_trip.keys()):
            sub_c = cands_with_probs[cands_with_probs["trip_id"] == tid]
            sub_o = obs_df[obs_df["trip_id"] == tid]
            res_v = decode_trip_viterbi(sub_c, sub_o, graphs_by_trip[tid], prob_col=pcol, config=viterbi_cfg)
            diag = res_v.diagnostics

            aligned_cnt = int((sub_o["alignment_status"] == "aligned").sum())
            hmm_rows.append(
                {
                    "test_type": "real_trip_oof_pilot",
                    "trip_or_case_id": str(tid),
                    "model_source": model_label,
                    "num_observations": int(len(sub_o)),
                    "aligned_observations": aligned_cnt,
                    "greedy_top1_accuracy": diag.get("greedy_top1_accuracy", float("nan")),
                    "viterbi_top1_accuracy": diag.get("viterbi_top1_accuracy", float("nan")),
                    "top1_accuracy_delta": round(
                        diag.get("viterbi_top1_accuracy", 0.0) - diag.get("greedy_top1_accuracy", 0.0), 6
                    ),
                    "greedy_route_continuity": diag["greedy_route_continuity"],
                    "viterbi_route_continuity": diag["viterbi_route_continuity"],
                    "continuity_delta": round(
                        diag["viterbi_route_continuity"] - diag["greedy_route_continuity"], 6
                    ),
                    "changed_observations_count": int(diag["changed_observations_count"]),
                }
            )

            # Save interactive Folium map for Trip 00000001 with Tuned HGB + Viterbi
            if tid == "00000001" and "HistGradientBoosting" in model_label:
                map_out = phase1_dir / "trip_00000001_viterbi_vs_greedy_map.html"
                create_viterbi_folium_map(raw_by_trip[tid], net_by_trip[tid], sub_o, res_v, map_out)
                print(f"Saved Viterbi vs Greedy Folium map to {map_out}")

            tot_obs_m += len(sub_o)
            tot_aligned_m += aligned_cnt
            greedy_correct_m += int(round(diag.get("greedy_top1_accuracy", 0.0) * aligned_cnt))
            viterbi_correct_m += int(round(diag.get("viterbi_top1_accuracy", 0.0) * aligned_cnt))
            steps_t = max(len(sub_o) - 1, 0)
            tot_steps_m += steps_t
            greedy_conn_m += int(round(diag["greedy_route_continuity"] * steps_t))
            viterbi_conn_m += int(round(diag["viterbi_route_continuity"] * steps_t))
            tot_changed_m += int(diag["changed_observations_count"])

        g_acc_agg = greedy_correct_m / max(tot_aligned_m, 1)
        v_acc_agg = viterbi_correct_m / max(tot_aligned_m, 1)
        g_cont_agg = greedy_conn_m / max(tot_steps_m, 1)
        v_cont_agg = viterbi_conn_m / max(tot_steps_m, 1)
        hmm_rows.append(
            {
                "test_type": "real_trip_oof_aggregate",
                "trip_or_case_id": "ALL_5_PREVIEW_TRIPS",
                "model_source": model_label,
                "num_observations": tot_obs_m,
                "aligned_observations": tot_aligned_m,
                "greedy_top1_accuracy": round(g_acc_agg, 6),
                "viterbi_top1_accuracy": round(v_acc_agg, 6),
                "top1_accuracy_delta": round(v_acc_agg - g_acc_agg, 6),
                "greedy_route_continuity": round(g_cont_agg, 6),
                "viterbi_route_continuity": round(v_cont_agg, 6),
                "continuity_delta": round(v_cont_agg - g_cont_agg, 6),
                "changed_observations_count": tot_changed_m,
            }
        )

    hmm_df = pd.DataFrame(hmm_rows)
    hmm_df.to_csv(phase1_dir / "hmm_test_results.csv", index=False)

    # 6. Create diagnostic visualisations
    create_phase1_plots(model_comp_df, trials_df, hmm_df, phase1_dir)
    print(f"Phase 1 ML + Bayesian + HMM pipeline finished in {time.perf_counter() - t0_all:.2f}s")


if __name__ == "__main__":
    main()
