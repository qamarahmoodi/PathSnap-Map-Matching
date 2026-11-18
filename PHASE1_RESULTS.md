# PathSnap — Phase 1 Executed Results & Evidence Summary

**Generated On:** 2026-11-18  
**Branch:** `feature/pranav-ml`  

---

## Explicit Scope Separation Across the Six Phase 1 Experiments

To maintain strict scientific clarity, our Phase 1 evidence consists of **six distinct, clearly scoped experiments**:
1. **Experiment A — Full 90-Trip Raw Dataset Audit (`80 train` + `10 student_test` trips, `227,577` GPS observations):** Complete structural, coordinate, road-graph, route-loop, and timestamp-gap verification (`data/dataset_inventory.csv`, `outputs/dataset_inventory.csv`).
2. **Experiment B — 10-Trip Candidate Sensitivity Analysis (`00000001`–`00000012` in `train`, `23,178` observations, `22,193` aligned):** Grid evaluation over search radii $R \in \{30, 50, 75, 100, 150\}\text{ m}$ and rank caps $K \in \{3, 5, 8, 10, 15, 20\}$ (`data/sensitivity_analysis.csv`).
3. **Experiment C — 5-Trip Candidate Extraction & Monotonic DP Alignment Preview (`00000001`–`00000005` in `train`, `11,488` observations, `98,636` candidate rows):** Full Parquet generation (`outputs/candidate_previews/candidates.parquet`, `observations.parquet`) and interactive alignment mapping (`results/phase1/candidate_generation_summary.csv`, `results/phase1/label_distribution.csv`).
4. **Experiment D — 5-Trip Preliminary Five-Model `GroupKFold` Pilot (`92,299` `label_eligible` candidate rows across `5` folds):** Out-of-fold cross-validation of our 5 shortlisted classifiers (`results/phase1/model_comparison.csv`, `results/phase1/cv_results.csv`). *Note: This is a 5-trip pilot, not the full 80-trip evaluation.*
5. **Experiment E — 5-Trip Preliminary Optuna TPE Bayesian Optimisation Pilot (`8` trials of 5-fold `GroupKFold`):** Demonstrated genuine Bayesian search on `HistGradientBoostingClassifier` (`results/phase1/bayesian_trials.csv`). *Note: Because trials are selected on the same 5-trip CV folds, unbiased holdout generalisation is deferred to the 10-trip `student_test` partition in Phase 2.*
6. **Experiment F — Synthetic Unit Benchmarks + 5-Trip Out-of-Fold HMM/Viterbi Decoding (`11,488` observations):** Verified `decode_trip_viterbi` on deterministic synthetic graphs (`tests/test_viterbi_decoder.py`) and out-of-fold ML probabilities from the 5-trip pilot (`results/phase1/hmm_test_results.csv`).

---

## 1. Full 90-Trip Dataset Audit Results (Experiment A)

- **Source Files:** [`data/dataset_inventory.csv`](data/dataset_inventory.csv), [`outputs/dataset_inventory.csv`](outputs/dataset_inventory.csv), [`outputs/partition_summary.txt`](outputs/partition_summary.txt), [`outputs/candidate_previews/dataset_audit_digest.json`](outputs/candidate_previews/dataset_audit_digest.json)
- **Visualisations:**
  - [`outputs/gps_observations_per_trip.png`](outputs/gps_observations_per_trip.png)
  - [`outputs/max_timestamp_gap_per_trip.png`](outputs/max_timestamp_gap_per_trip.png)
  - [`outputs/road_nodes_per_trip.png`](outputs/road_nodes_per_trip.png)
  - [`outputs/road_arcs_per_trip.png`](outputs/road_arcs_per_trip.png)
  - [`outputs/trip_map.html`](outputs/trip_map.html) (Interactive Folium map of raw GPS + road network + reference route)

| Partition | Trips | Total GPS Obs (`.track`) | Total Road Nodes (`.nodes`) | Total Road Arcs (`.arcs`) | Total Route Arcs (`.route`) | Corrupt / Non-Finite Rows | Trips with Repeated Route Arcs | Disconnected Route Steps | Max Timestamp Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | `80` | `208,417` | `1,933,485` | `3,876,904` | `31,336` | `0` | `9 / 80` (`67` repeats) | `13 / 31,256` (`99.935%` conn.) | `153,064.0 s` |
| `student_test` | `10` | `19,160` | `90,689` | `185,041` | `3,477` | `0` | `1 / 10` (`255` repeats) | `1 / 3,467` (`99.973%` conn.) | `210.0 s` |
| **Total** | **`90`** | **`227,577`** | **`2,024,174`** | **`4,061,945`** | **`34,813`** | **`0`** | **`10 / 90` (`322` repeats)** | **`14 / 34,723`** | **`153,064.0 s`** |

---

## 2. Candidate Generation & Alignment Verification (Experiments B & C)

- **Source Files:** [`data/sensitivity_analysis.csv`](data/sensitivity_analysis.csv) (10 trips), [`results/phase1/candidate_generation_summary.csv`](results/phase1/candidate_generation_summary.csv) (5 trips), [`results/phase1/label_distribution.csv`](results/phase1/label_distribution.csv) (5 trips)
- **Interactive Alignment Map:** [`outputs/candidate_previews/trip_00000001_alignment_map.html`](outputs/candidate_previews/trip_00000001_alignment_map.html)

| Trip ID | Total Obs | `aligned` Obs | `ambiguous` Obs | `unmatched` Obs | `label_eligible` Obs | Candidate Misses | **Candidate Miss Rate** | Total Cand Rows | Labeled Cand Rows | Positive (`label=1`) | Negative (`label=0`) | Null (`<NA>`) | Median / P95 Align Dist | Monotonic? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000001` | `1,315` | `1,315` | `0` | `0` | `1,313` | `2` | `0.1521%` | `12,343` | `12,323` | `1,313` | `11,010` | `20` | `4.428 m / 12.565 m` | `True` |
| `00000002` | `7,243` | `6,373` | `2` | `868` | `6,368` | `5` | `0.0785%` | `65,333` | `59,156` | `6,368` | `52,788` | `6,177` | `5.916 m / 242.414 m` | `True` |
| `00000003` | `821` | `821` | `0` | `0` | `815` | `6` | `0.7308%` | `8,089` | `8,029` | `815` | `7,214` | `60` | `4.472 m / 14.012 m` | `True` |
| `00000004` | `1,064` | `1,064` | `0` | `0` | `1,056` | `8` | `0.7519%` | `8,593` | `8,513` | `1,056` | `7,457` | `80` | `8.861 m / 29.503 m` | `True` |
| `00000005` | `1,045` | `1,045` | `0` | `0` | `1,045` | `0` | `0.0000%` | `4,278` | `4,278` | `1,045` | `3,233` | `0` | `4.565 m / 13.260 m` | `True` |
| **Total (5 Trips)** | **`11,488`** | **`10,618`** | **`2`** | **`868`** | **`10,597`** | **`21`** | **`0.1978%`** | **`98,636`** | **`92,299`** | **`10,597`** | **`81,702`** | **`6,337`** | — | **`True`** |

---

## 3. Preliminary 5-Model Cross-Validation Results (Experiment D — 5-Trip Pilot)

- **Source Files:** [`results/phase1/model_comparison.csv`](results/phase1/model_comparison.csv), [`results/phase1/cv_results.csv`](results/phase1/cv_results.csv)
- **Visualisation:** [`results/phase1/model_comparison_chart.png`](results/phase1/model_comparison_chart.png)
- **Baseline Reference:** Naive Nearest-Segment (`candidate_rank == 1`) achieves **`52.89%` (`0.528913`)** Observation Top-1 Accuracy across the `10,618` aligned observations (`76.12%` on Trip 1, `39.38%` on Trip 2, `72.84%` on Trip 3, `48.50%` on Trip 4, `94.83%` on Trip 5).

| Model | Algorithm Family | Mean Fold Macro F1 ($\pm$ SD) | OOF Macro F1 | OOF Positive F1 | OOF Accuracy | OOF Precision | OOF Recall | Mean Fold Log Loss | OOF Log Loss | Mean Fold Obs Top-1 Acc | OOF Obs Top-1 Acc (`aligned` / `eligible`) | Greedy Route Continuity | Total CV Train / Infer Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | `0.7838 ± 0.0797` | `0.7274` | `0.5093` | `0.9019` | `0.5980` | `0.4435` | `0.2112` | `0.2378` | **`0.7749`** | **`0.6500` / `0.6513`** | **`0.8677`** | `1.166s / 0.039s` |
| **Distance-Weighted KNN** | Instance-Based | `0.7960 ± 0.1018` | `0.7411` | `0.5287` | `0.9153` | `0.7316` | `0.4139` | `0.2735` | `0.3836` | `0.7077` | `0.5911` / `0.5922` | `0.8320` | `1.717s / 24.084s` |
| **Decision Tree** | Tree-Based | `0.6893 ± 0.0701` | `0.7016` | `0.4604` | `0.8964` | `0.5729` | `0.3848` | `0.5259` | `0.6159` | `0.5856` | `0.5199` / `0.5209` | `0.7714` | `4.626s / 0.029s` |
| **Random Forest** | Bagging Ensemble | `0.7961 ± 0.1105` | `0.7396` | `0.5251` | **`0.9163`** | **`0.7537`** | `0.4029` | **`0.1894`** | **`0.2102`** | `0.7389` | `0.6338` / `0.6351` | `0.8415` | `10.419s / 0.419s` |
| **HistGradientBoosting** | Gradient Boosting | **`0.8013 ± 0.0974`** | **`0.7445`** | **`0.5382`** | `0.9110` | `0.6658` | **`0.4516`** | `0.2217` | `0.2759` | `0.7174` | `0.5962` / `0.5973` | `0.8329` | `5.393s / 0.205s` |

---

## 4. Preliminary Optuna TPE Bayesian Tuning Results (Experiment E — 5-Trip Pilot)

- **Source Files:** [`results/phase1/bayesian_trials.csv`](results/phase1/bayesian_trials.csv), [`results/phase1/bayesian_best_summary.json`](results/phase1/bayesian_best_summary.json)
- **Visualisation:** [`results/phase1/bayesian_tuning_progression.png`](results/phase1/bayesian_tuning_progression.png)
- **Best Trial (#7 vs Enqueued Baseline Trial #1 on 5-Trip `GroupKFold`):**
  - `learning_rate = 0.073068`, `max_iter = 120`, `max_leaf_nodes = 31`, `max_depth = 12`, `min_samples_leaf = 41`, `l2_regularization = 4.052611`
  - **OOF Macro F1:** `0.744416` (Trial #1) $\to$ **`0.745228`** (Trial #7)
  - **OOF Positive F1:** `0.538072` (Trial #1) $\to$ **`0.539661`** (Trial #7)
  - **OOF Log Loss:** `0.274761` (Trial #1) $\to$ **`0.272779`** (Trial #7; and `0.247645` in high-regularisation Trial #6)
  - **Greedy Route Continuity:** `0.832883` (Trial #1) $\to$ **`0.838544`** (Trial #7)
  - *Note:* In-sample CV selection optimism applies; independent holdout evaluation on `student_test` is scheduled for Phase 2.

---

## 5. HMM & Viterbi Sequential Decoding Results (Experiment F — Synthetic + 5-Trip OOF Pilot)

- **Source File:** [`results/phase1/hmm_test_results.csv`](results/phase1/hmm_test_results.csv)
- **Interactive Folium Comparison Map:** [`results/phase1/trip_00000001_viterbi_vs_greedy_map.html`](results/phase1/trip_00000001_viterbi_vs_greedy_map.html)

| Test Type | Trip / Case ID | Emission Source | Obs Count (`Aligned`) | Greedy Top-1 Acc | **HMM/Viterbi Top-1 Acc** | **Top-1 Delta** | Greedy Continuity | **HMM/Viterbi Continuity** | **Continuity Delta** | Changed Obs |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `synthetic_deterministic` | `SYNTH_PARALLEL_SPIKE` | Synthetic Emission Spike | `3 (3)` | `66.67%` | **`100.00%`** | **`+33.33%`** | `0.00%` | **`100.00%`** | **`+100.00%`** | `1` |
| `synthetic_deterministic` | `SYNTH_GAP_AND_EMPTY` | Empty-Cand + 117s Gap | `4 (4)` | `100.00%` | **`100.00%`** | `0.00%` | `66.67%` | **`66.67%`** | `0.00%` | `0` |
| `real_trip_oof_pilot` | `00000001` | Tuned HGB (OOF) | `1,315 (1,315)` | `75.36%` | **`93.31%`** | **`+17.95%`** | `75.42%` | **`96.80%`** | **`+21.39%`** | `289` |
| `real_trip_oof_pilot` | `00000002` | Tuned HGB (OOF) | `7,243 (6,373)` | **`47.45%`** | `43.75%` | `-3.70%` | `83.87%` | **`96.09%`** | **`+12.22%`** | `1,836` |
| `real_trip_oof_pilot` | `00000003` | Tuned HGB (OOF) | `821 (821)` | `76.49%` | **`81.73%`** | **`+5.24%`** | `82.93%` | **`98.17%`** | **`+15.24%`** | `111` |
| `real_trip_oof_pilot` | `00000004` | Tuned HGB (OOF) | `1,064 (1,064)` | `56.39%` | **`63.63%`** | **`+7.24%`** | `80.62%` | **`85.61%`** | **`+4.99%`** | `116` |
| `real_trip_oof_pilot` | `00000005` | Tuned HGB (OOF) | `1,045 (1,045)` | `98.18%` | **`99.33%`** | **`+1.15%`** | `98.37%` | **`100.00%`** | **`+1.63%`** | `12` |
| `real_trip_oof_aggregate` | `ALL_5_PREVIEW_TRIPS` | Tuned HGB (OOF) | `11,488 (10,618)` | `59.04%` | **`60.28%`** | **`+1.24%`** | `83.85%` | **`95.71%`** | **`+11.85%`** | `2,364` |
| `real_trip_oof_pilot` | `00000001` | Logistic Regression (OOF) | `1,315 (1,315)` | `84.87%` | **`94.14%`** | **`+9.28%`** | `83.79%` | **`97.72%`** | **`+13.93%`** | `193` |
| `real_trip_oof_pilot` | `00000002` | Logistic Regression (OOF) | `7,243 (6,373)` | `52.24%` | **`79.73%`** | **`+27.49%`** | `86.68%` | **`96.67%`** | **`+10.00%`** | `3,006` |
| `real_trip_oof_pilot` | `00000003` | Logistic Regression (OOF) | `821 (821)` | `77.47%` | **`92.33%`** | **`+14.86%`** | `77.20%` | **`99.15%`** | **`+21.95%`** | `156` |
| `real_trip_oof_pilot` | `00000004` | Logistic Regression (OOF) | `1,064 (1,064)` | `76.88%` | **`87.78%`** | **`+10.90%`** | `88.33%` | **`94.73%`** | **`+6.40%`** | `160` |
| `real_trip_oof_pilot` | `00000005` | Logistic Regression (OOF) | `1,045 (1,045)` | `95.98%` | **`96.94%`** | **`+0.96%`** | `97.89%` | **`100.00%`** | **`+2.11%`** | `12` |
| `real_trip_oof_aggregate` | `ALL_5_PREVIEW_TRIPS` | **Logistic Regression (OOF)** | `11,488 (10,618)` | `65.00%` | **`84.99%`** | **`+19.98%`** | `86.84%` | **`97.09%`** | **`+10.25%`** | `3,527` |
