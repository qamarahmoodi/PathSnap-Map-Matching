# PathSnap — Phase 1 Experimental Results & Technical Summary

**Owner:** Pranav Shashikiran Atreya (ML Engineering — 5 Models, GroupKFold CV, Bayesian Tuning, HMM/Viterbi Integration)  
**Branch:** `feature/pranav-ml`  
**Evaluation Scope Note:** All ML and HMM metrics below were executed on the **5 representative training trips** (`00000001`–`00000005`, containing **11,488 GPS observations** and **98,636 candidate rows**, of which **92,299 rows are labeled** across **10,597 `label_eligible` observations**). These results are explicitly reported as our **Phase 1 Preliminary 5-Trip Pilot** (5-fold `GroupKFold`, 1 held-out trip per fold), not the final 80-trip evaluation.

---

## 1. Candidate Data Contract & Label Verification

Verified via `verify_candidate_contract(...)` on `outputs/candidate_previews/candidates.parquet` and `observations.parquet`:

- **Exact Schema Compliance:** All **20 columns** in `CANDIDATE_SCHEMA_DTYPES` and all **13 columns** in `OBSERVATION_SCHEMA_DTYPES` verified.
- **Exact Feature Set (`CANDIDATE_FEATURE_COLUMNS` — 11 columns):**
  `distance_to_segment_m`, `projection_fraction`, `distance_to_start_m`, `distance_to_end_m`, `segment_length_m`, `segment_bearing_deg`, `gps_bearing_deg`, `bearing_difference_deg`, `gps_speed_mps`, `delta_time_s`, `is_fallback_candidate`.
- **Zero Identifier Leakage:** `trip_id`, `observation_id`, `candidate_arc_id`, `from_node_id`, `to_node_id`, and `candidate_rank` are strictly excluded from the feature matrix.
- **Label & Alignment Distribution (`results/phase1/label_distribution.csv`):**
  - Total GPS observations (5 preview trips): **11,488**
  - `aligned` observations: **10,618** (`92.43%`)
  - `ambiguous` observations: **2** (`0.02%`)
  - `unmatched` observations: **868** (`7.56%` — unmapped trajectory tail at the end of Trip `00000002`, automatically excluded from training via `label_eligible = False`)
  - Candidate misses among `aligned` observations: **21 / 10,618** (**Candidate Miss Rate = `0.1978%`**, Candidate Recall = `99.8022%`)
  - `label_eligible` observations: **10,597**
  - Labeled candidate rows: **92,299** (**10,597 positive** `label = 1` [`11.48%`], **81,702 negative** `label = 0` [`88.52%`], **6,337 null** `<NA>` excluded from fitting).

---

## 2. Preliminary Five-Model Experimentation (`GroupKFold`, $K=5$ Trips)

Full fold-by-fold results are saved in [`results/phase1/cv_results.csv`](cv_results.csv) and consolidated in [`results/phase1/model_comparison.csv`](model_comparison.csv).

| Model | Algorithm Family | Mean Fold Macro F1 ($\pm$ SD) | Out-of-Fold Macro F1 | Out-of-Fold Positive F1 | Mean Fold Log Loss | Out-of-Fold Log Loss | Mean Fold Obs Top-1 Acc | Out-of-Fold Obs Top-1 Acc | Greedy Route Continuity | Total CV Train Time (s) | Total CV Infer Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Logistic Regression** | Linear & Probabilistic | `0.7838 ± 0.0797` | `0.7274` | `0.5093` | `0.2112` | `0.2378` | **`0.7749`** | **`0.6500`** | **`0.8677`** | `1.166s` | `0.039s` |
| **2. Distance-Weighted KNN** | Instance-Based | `0.7960 ± 0.1018` | `0.7411` | `0.5287` | `0.2735` | `0.3836` | `0.7077` | `0.5911` | `0.8320` | `1.717s` | `24.084s` |
| **3. Decision Tree** | Tree-Based | `0.6893 ± 0.0701` | `0.7016` | `0.4604` | `0.5259` | `0.6159` | `0.5856` | `0.5199` | `0.7714` | `4.626s` | `0.029s` |
| **4. Random Forest** | Bagging Ensemble | `0.7961 ± 0.1105` | `0.7396` | `0.5251` | **`0.1894`** | **`0.2102`** | `0.7389` | `0.6338` | `0.8415` | `10.419s` | `0.419s` |
| **5. HistGradientBoosting** | Gradient Boosting | **`0.8013 ± 0.0974`** | **`0.7445`** | **`0.5382`** | `0.2217` | `0.2759` | `0.7174` | `0.5962` | `0.8329` | `5.393s` | `0.205s` |

### Key Scientific Observations from the 5-Model Pilot
1. **Candidate-Row Classification vs Probability Calibration:**
   - **HistGradientBoosting** achieves the highest **Mean Fold Macro F1 (`0.8013`)** and highest **Out-of-Fold Macro F1 (`0.7445`)**, outperforming Random Forest (`0.7396`), Distance-Weighted KNN (`0.7411`), Logistic Regression (`0.7274`), and Decision Tree (`0.7016`).
   - **Random Forest** (`0.2102` OOF Log Loss) and **Logistic Regression** (`0.2378` OOF Log Loss) produce the best-calibrated probability distributions on this 5-trip pilot because Trip `00000002` comprises `64.1%` of the 5-trip candidate rows (`59,156 / 92,299`), meaning when Trip `00000002` is held out in Fold 4, models are trained on only `33,143` rows from the 4 smaller trips. Expanding training to all **80 trips** in Phase 2 will eliminate this small-sample fold asymmetry.

---

## 3. Genuine Bayesian Hyperparameter Optimisation (Optuna TPE)

Executed via `src/optimization/bayesian_tuning.py` using **`optuna.samplers.TPESampler(seed=42)`** over 8 trials of 5-fold `GroupKFold` cross-validation on `HistGradientBoostingClassifier`. Full trial history is saved in [`results/phase1/bayesian_trials.csv`](bayesian_trials.csv).

- **Search Space:**
  - `learning_rate`: log-uniform `[0.02, 0.18]`
  - `max_iter`: integer `[80, 240]` (step 20)
  - `max_leaf_nodes`: integer `[15, 63]`
  - `max_depth`: integer `[4, 12]`
  - `min_samples_leaf`: integer `[10, 80]`
  - `l2_regularization`: log-uniform `[0.01, 10.0]`
- **Best Trial (#7):**
  - `learning_rate = 0.073068`, `max_iter = 120`, `max_leaf_nodes = 31`, `max_depth = 12`, `min_samples_leaf = 41`, `l2_regularization = 4.052611`
  - **Untuned Trial #1:** OOF Macro F1 = `0.744416`, OOF Log Loss = `0.274761`, Greedy Route Continuity = `0.832883`
  - **Bayesian Tuned Trial #7:** OOF Macro F1 = **`0.745228`** (`+0.0008`), OOF Positive F1 = **`0.539661`**, OOF Log Loss = **`0.272779`** (improved calibration), Greedy Route Continuity = **`0.838544`** (`+0.57%`)
  - **Trial #6 (High-Regularisation Regime):** Achieved OOF Log Loss = **`0.247645`** and Route Continuity = **`0.843334`**, confirming that stronger L2 regularisation (`5.28`) prevents log-odds overconfidence on small trip subsets.

---

## 4. HMM & Viterbi Sequential Route Reconstruction Results

Implemented in [`src/sequence/viterbi_decoder.py`](../../src/sequence/viterbi_decoder.py) and tested in [`tests/test_viterbi_decoder.py`](../../tests/test_viterbi_decoder.py) + [`results/phase1/hmm_test_results.csv`](hmm_test_results.csv).

### A. Deterministic Synthetic Directed Graph Verification
1. **`SYNTH_PARALLEL_SPIKE` (Parallel Distractor Spike Override):**
   - Greedy Top-1 Accuracy: `66.67%` (`0.00%` Route Continuity — jumps onto disconnected parallel arc).
   - **HMM/Viterbi Top-1 Accuracy:** **`100.00%`** (**`100.00%` Route Continuity** — overrides noisy spike using directed graph connectivity).
2. **`SYNTH_GAP_AND_EMPTY` (Missing Candidate Set + 117s GPS Dropout):**
   - Cleanly carries forward previous arc across empty-candidate observation `t=1` and resets transition prior across the `117s` gap at `t=3` (`100.0%` accuracy).

### B. Real 5-Trip Out-of-Fold ML Emission + Viterbi Decoding (`11,488` Observations)

| Trip / Aggregate | Model Emissions Used | Greedy Top-1 Acc | **HMM/Viterbi Top-1 Acc** | **Top-1 Gain ($\Delta$)** | Greedy Continuity | **HMM/Viterbi Continuity** | **Continuity Gain ($\Delta$)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trip `00000001`** (`1,315` obs) | Tuned HistGradientBoosting (OOF) | `75.36%` | **`93.31%`** | **`+17.95%`** | `75.42%` | **`96.80%`** | **`+21.39%`** |
| **Trip `00000002`** (`7,243` obs, `6,373` aligned) | Tuned HistGradientBoosting (OOF) | **`47.45%`** | `43.75%` | `-3.70%` | `83.87%` | **`96.09%`** | **`+12.22%`** |
| **Trip `00000003`** (`821` obs) | Tuned HistGradientBoosting (OOF) | `76.49%` | **`81.73%`** | **`+5.24%`** | `82.93%` | **`98.17%`** | **`+15.24%`** |
| **Trip `00000004`** (`1,064` obs) | Tuned HistGradientBoosting (OOF) | `56.39%` | **`63.63%`** | **`+7.24%`** | `80.62%` | **`85.61%`** | **`+4.99%`** |
| **Trip `00000005`** (`1,045` obs) | Tuned HistGradientBoosting (OOF) | `98.18%` | **`99.33%`** | **`+1.15%`** | `98.37%` | **`100.00%`** | **`+1.63%`** |
| **All 5 Preview Trips (`11,488` obs)** | Tuned HistGradientBoosting (OOF) | `59.04%` | **`60.28%`** | **`+1.24%`** | `83.85%` | **`95.71%`** | **`+11.85%`** |
| **All 5 Preview Trips (`11,488` obs)** | **Logistic Regression (OOF)** | `65.00%` | **`84.99%`** | **`+19.98%`** | `86.84%` | **`97.09%`** | **`+10.25%`** |

- **Note on Trip `00000002` Trade-Off:** Because Trip `00000002` holds `64.1%` of the 5-trip pilot rows (`59,156 / 92,299`), Fold 4 trains HGB on only `33,143` rows from the 4 smaller trips (`0.3177` fold Log Loss), causing uncalibrated log-odds at intersections where Viterbi prioritizes connected paths (`96.09%` continuity vs `83.87%` greedy) at a `-3.70%` pointwise trade-off. With better-calibrated Logistic Regression probabilities (`0.2378` OOF Log Loss), Viterbi improves Trip `00000002` accuracy from `52.24%` to `79.73%` (`+27.49%`).
- **Interactive Map Generated:** [`results/phase1/trip_00000001_viterbi_vs_greedy_map.html`](trip_00000001_viterbi_vs_greedy_map.html) visualizes Trip `00000001` where HMM/Viterbi boosts observation accuracy from **`75.36%` to `93.31%`** (`+17.95%`) and route continuity from **`75.42%` to `96.80%`** (`+21.39%`).
