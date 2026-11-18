# PathSnap — Phase 1 Executed Results & Evidence Summary

**Generated On:** 2026-11-18  
**Branch:** `feature/pranav-ml`  
**Scope Declaration:** All dataset audit results cover the **full 90 supplied trips (`80 train` + `10 student_test`)**. Candidate sensitivity results cover **10 representative training trips (`23,178` observations)**. Preliminary 5-model cross-validation, Optuna TPE Bayesian optimisation, and HMM/Viterbi decoding results were executed on the **5-trip preview dataset (`00000001`–`00000005`, `11,488` observations, `98,636` candidate rows, `92,299` labeled rows)** and are explicitly reported as a **5-trip Phase 1 pilot**.

---

## 1. Full 90-Trip Dataset Audit Results

- **Source Files:** [`data/dataset_inventory.csv`](data/dataset_inventory.csv), [`outputs/dataset_inventory.csv`](outputs/dataset_inventory.csv), [`outputs/partition_summary.txt`](outputs/partition_summary.txt), [`outputs/candidate_previews/dataset_audit_digest.json`](outputs/candidate_previews/dataset_audit_digest.json)
- **Visualisations:**
  - [`outputs/gps_observations_per_trip.png`](outputs/gps_observations_per_trip.png)
  - [`outputs/max_timestamp_gap_per_trip.png`](outputs/max_timestamp_gap_per_trip.png)
  - [`outputs/road_nodes_per_trip.png`](outputs/road_nodes_per_trip.png)
  - [`outputs/road_arcs_per_trip.png`](outputs/road_arcs_per_trip.png)
  - [`outputs/trip_map.html`](outputs/trip_map.html) (Interactive Folium map of raw GPS + road network + reference route)

| Partition | Trips | Total GPS Obs (`.track`) | Total Road Nodes (`.nodes`) | Total Road Arcs (`.arcs`) | Total Route Arcs (`.route`) | Corrupt / Non-Finite Rows | Mean Route Connectivity | Max Timestamp Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | `80` | `208,417` | `1,933,485` | `3,876,904` | `31,336` | `0` | `99.935%` | `153,064.0 s` |
| `student_test` | `10` | `19,160` | `90,689` | `185,041` | `3,477` | `0` | `99.973%` | `210.0 s` |
| **Total** | **`90`** | **`227,577`** | **`2,024,174`** | **`4,061,945`** | **`34,813`** | **`0`** | **`99.939%`** | **`153,064.0 s`** |

---

## 2. Candidate Generation & Alignment Verification (5-Trip Preview)

- **Source Files:** [`results/phase1/candidate_generation_summary.csv`](results/phase1/candidate_generation_summary.csv), [`results/phase1/label_distribution.csv`](results/phase1/label_distribution.csv), [`data/sensitivity_analysis.csv`](data/sensitivity_analysis.csv)
- **Interactive Alignment Map:** [`outputs/candidate_previews/trip_00000001_alignment_map.html`](outputs/candidate_previews/trip_00000001_alignment_map.html)

| Trip ID | Total Obs | `aligned` Obs | `ambiguous` Obs | `unmatched` Obs | `label_eligible` Obs | Candidate Misses | **Candidate Miss Rate** | Total Cand Rows | Labeled Cand Rows | Positive (`label=1`) | Negative (`label=0`) | Null (`<NA>`) | Median Align Dist (m) | Monotonic? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000001` | `1,315` | `1,315` | `0` | `0` | `1,313` | `2` | `0.1521%` | `12,343` | `12,323` | `1,313` | `11,010` | `20` | `4.428 m` | `True` |
| `00000002` | `7,243` | `6,373` | `2` | `868` | `6,368` | `5` | `0.0785%` | `65,333` | `59,156` | `6,368` | `52,788` | `6,177` | `5.916 m` | `True` |
| `00000003` | `821` | `821` | `0` | `0` | `815` | `6` | `0.7308%` | `8,089` | `8,029` | `815` | `7,214` | `60` | `4.472 m` | `True` |
| `00000004` | `1,064` | `1,064` | `0` | `0` | `1,056` | `8` | `0.7519%` | `8,593` | `8,513` | `1,056` | `7,457` | `80` | `8.861 m` | `True` |
| `00000005` | `1,045` | `1,045` | `0` | `0` | `1,045` | `0` | `0.0000%` | `4,278` | `4,278` | `1,045` | `3,233` | `0` | `4.565 m` | `True` |
| **Total (5 Trips)** | **`11,488`** | **`10,618`** | **`2`** | **`868`** | **`10,597`** | **`21`** | **`0.1978%`** | **`98,636`** | **`92,299`** | **`10,597`** | **`81,702`** | **`6,337`** | — | **`True`** |

---

## 3. Preliminary 5-Model Cross-Validation Results (5-Trip Pilot)

- **Source Files:** [`results/phase1/model_comparison.csv`](results/phase1/model_comparison.csv), [`results/phase1/cv_results.csv`](results/phase1/cv_results.csv)
- **Visualisation:** [`results/phase1/model_comparison_chart.png`](results/phase1/model_comparison_chart.png)

| Model | Algorithm Family | Mean Fold Macro F1 ($\pm$ SD) | OOF Macro F1 | OOF Positive F1 | OOF Accuracy | OOF Precision | OOF Recall | Mean Fold Log Loss | OOF Log Loss | Mean Fold Obs Top-1 Acc | OOF Obs Top-1 Acc | Greedy Route Continuity | Total CV Train / Infer Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | `0.7838 ± 0.0797` | `0.7274` | `0.5093` | `0.9019` | `0.5980` | `0.4435` | `0.2112` | `0.2378` | **`0.7749`** | **`0.6500`** | **`0.8677`** | `1.166s / 0.039s` |
| **Distance-Weighted KNN** | Instance-Based | `0.7960 ± 0.1018` | `0.7411` | `0.5287` | `0.9153` | `0.7316` | `0.4139` | `0.2735` | `0.3836` | `0.7077` | `0.5911` | `0.8320` | `1.717s / 24.084s` |
| **Decision Tree** | Tree-Based | `0.6893 ± 0.0701` | `0.7016` | `0.4604` | `0.8964` | `0.5729` | `0.3848` | `0.5259` | `0.6159` | `0.5856` | `0.5199` | `0.7714` | `4.626s / 0.029s` |
| **Random Forest** | Bagging Ensemble | `0.7961 ± 0.1105` | `0.7396` | `0.5251` | **`0.9163`** | **`0.7537`** | `0.4029` | **`0.1894`** | **`0.2102`** | `0.7389` | `0.6338` | `0.8415` | `10.419s / 0.419s` |
| **HistGradientBoosting** | Gradient Boosting | **`0.8013 ± 0.0974`** | **`0.7445`** | **`0.5382`** | `0.9110` | `0.6658` | **`0.4516`** | `0.2217` | `0.2759` | `0.7174` | `0.5962` | `0.8329` | `5.393s / 0.205s` |

---

## 4. Preliminary Optuna TPE Bayesian Tuning Results (5-Trip Pilot)

- **Source Files:** [`results/phase1/bayesian_trials.csv`](results/phase1/bayesian_trials.csv), [`results/phase1/bayesian_best_summary.json`](results/phase1/bayesian_best_summary.json)
- **Visualisation:** [`results/phase1/bayesian_tuning_progression.png`](results/phase1/bayesian_tuning_progression.png)
- **Best Trial (#7 vs Untuned Trial #1):**
  - `learning_rate = 0.073068`, `max_iter = 120`, `max_leaf_nodes = 31`, `max_depth = 12`, `min_samples_leaf = 41`, `l2_regularization = 4.052611`
  - **OOF Macro F1:** `0.744416` (Untuned) $\to$ **`0.745228`** (Tuned)
  - **OOF Positive F1:** `0.538072` (Untuned) $\to$ **`0.539661`** (Tuned)
  - **OOF Log Loss:** `0.274761` (Untuned) $\to$ **`0.272779`** (Tuned; and `0.247645` in high-regularisation Trial #6)
  - **Greedy Route Continuity:** `0.832883` (Untuned) $\to$ **`0.838544`** (Tuned)

---

## 5. HMM & Viterbi Sequential Decoding Results (Synthetic + 5-Trip Pilot)

- **Source File:** [`results/phase1/hmm_test_results.csv`](results/phase1/hmm_test_results.csv)
- **Interactive Folium Comparison Map:** [`results/phase1/trip_00000001_viterbi_vs_greedy_map.html`](results/phase1/trip_00000001_viterbi_vs_greedy_map.html)

| Test Type | Trip / Case ID | Emission Source | Obs Count | Greedy Top-1 Acc | **HMM/Viterbi Top-1 Acc** | **Top-1 Delta** | Greedy Continuity | **HMM/Viterbi Continuity** | **Continuity Delta** | Changed Obs |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `synthetic_deterministic` | `SYNTH_PARALLEL_SPIKE` | Synthetic Emission Spike | `3` | `66.67%` | **`100.00%`** | **`+33.33%`** | `0.00%` | **`100.00%`** | **`+100.00%`** | `1` |
| `synthetic_deterministic` | `SYNTH_GAP_AND_EMPTY` | Empty-Cand + 117s Gap | `4` | `100.00%` | **`100.00%`** | `0.00%` | `66.67%` | **`66.67%`** | `0.00%` | `0` |
| `real_trip_oof_pilot` | `00000001` | Tuned HGB (OOF) | `1,315` | `75.36%` | **`93.31%`** | **`+17.95%`** | `75.42%` | **`96.80%`** | **`+21.39%`** | `289` |
| `real_trip_oof_pilot` | `00000003` | Tuned HGB (OOF) | `821` | `76.49%` | **`81.73%`** | **`+5.24%`** | `82.93%` | **`98.17%`** | **`+15.24%`** | `111` |
| `real_trip_oof_pilot` | `00000004` | Tuned HGB (OOF) | `1,064` | `56.39%` | **`63.63%`** | **`+7.24%`** | `80.62%` | **`85.61%`** | **`+4.99%`** | `116` |
| `real_trip_oof_pilot` | `00000005` | Tuned HGB (OOF) | `1,045` | `98.18%` | **`99.33%`** | **`+1.15%`** | `98.37%` | **`100.00%`** | **`+1.63%`** | `12` |
| `real_trip_oof_aggregate` | `ALL_5_PREVIEW_TRIPS` | Tuned HGB (OOF) | `11,488` | `59.04%` | **`60.28%`** | **`+1.24%`** | `83.85%` | **`95.71%`** | **`+11.85%`** | `2,364` |
| `real_trip_oof_aggregate` | `ALL_5_PREVIEW_TRIPS` | **Logistic Regression (OOF)** | `11,488` | `65.00%` | **`84.99%`** | **`+19.98%`** | `86.84%` | **`97.09%`** | **`+10.25%`** | `3,527` |
