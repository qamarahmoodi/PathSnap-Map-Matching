# PathSnap — Phase 2 Technical Report Contribution (Pranav Shashikiran Atreya)

**Author:** Pranav Shashikiran Atreya (**PES2UG24CS912**)  
**Scope:** Full 80-Trip Five-Model Experimentation & Comparison, 10-Trip `student_test` Holdout Evaluation, 25-Trial Optuna TPE Bayesian Hyperparameter Optimisation, Final Model Scorer Handover, HMM/Viterbi Integration Results, and Laptop 2 Evaluator UI.

---

## 1. Full-Scale Dataset & Candidate Extraction Summary

All 80 training trips (`train`) and 10 development-testing trips (`student_test`) were processed using `CandidateConfig(search_radius_m=100.0, max_candidates_k=10, enable_fallback_nearest=True)` and aligned via `align_reference_route(...)`. Full per-trip statistics are recorded in [`results/phase2/full_90trip_candidate_summary.csv`](../results/phase2/full_90trip_candidate_summary.csv) and [`results/phase2/full_dataset_extraction_digest.json`](../results/phase2/full_dataset_extraction_digest.json). The 10 reserved `evaluator_test` trips were never accessed or inspected.

| Metric | Training Partition (`train`) | Student-Testing (`student_test`) | Combined Development (`90` Trips) |
| :--- | :---: | :---: | :---: |
| **Trips** | **80** | **10** | **90** |
| **Total GPS Observations (`.track`)** | **208,417** | **19,160** | **227,577** |
| **Confidently `aligned` Observations** | **184,721** (`88.63%`) | **19,103** (`99.70%`) | **203,824** |
| **Candidate Misses on `aligned` Observations** | **1,250** (`0.6767%`) | **29** (`0.1518%`) | **1,279** (`0.6275%`) |
| **`label_eligible` Observations (`aligned` & hit)** | **183,471** | **19,074** | **202,545** |
| **Total Generated Candidate Rows** | **1,615,009** | **157,851** | **1,772,860** |
| **Labeled Candidate Rows (`label_eligible == True`)** | **1,484,716** | **157,066** | **1,641,782** |
| **Positive Candidate Rows (`label = 1`)** | **183,471** (`12.357%`) | **19,074** (`12.144%`) | **202,545** (`12.337%`) |
| **Negative Candidate Rows (`label = 0`)** | **1,301,245** (`87.643%`) | **137,992** (`87.856%`) | **1,439,237** (`87.663%`) |
| **Null-Label Candidate Rows (`<NA>`, excluded from fit)** | **130,293** | **785** | **131,078** |
| **Naive Nearest-Segment (`candidate_rank == 1`) Top-1 Acc** | **`53.924%`** (`54.291%` eligible) | **`52.384%`** (`52.464%` eligible) | — |

---

## 2. Section 3.1 — Consolidated Five-Model Comparison Table

All five models were evaluated using 5-fold trip-grouped cross-validation (`GroupKFold(n_splits=5)` by `trip_id`, 16 trips per fold) across the **80 training trips (`1,484,716` labeled rows)**, and then fitted on all 80 training trips and evaluated on the **10 held-out `student_test` trips (`157,066` labeled rows, `19,103` aligned observations)**. Full data is saved in [`results/phase2/model_comparison_80train_10test.csv`](../results/phase2/model_comparison_80train_10test.csv) and [`results/phase2/cv_80trip_fold_results.csv`](../results/phase2/cv_80trip_fold_results.csv). Visualisations are saved in [`results/phase2/five_model_comparison_full.png`](../results/phase2/five_model_comparison_full.png) and [`results/phase2/student_test_confusion_matrices.png`](../results/phase2/student_test_confusion_matrices.png).

| Model | Algorithm Family | Key Configuration | 80-Trip CV Macro F1 ($\pm$ SD) | Student-Test Top-1 Accuracy (`aligned`) | Student-Test Route Continuity | Student-Test Macro F1-Score | Student-Test Log Loss | Student-Test Inference Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Logistic Regression** | Linear & Probabilistic | `L2, C=1.0, lbfgs, StandardScaler + SimpleImputer(median, add_indicator=True)` | `0.8451 ± 0.0341` (`0.8425` OOF) | `78.54%` (`0.785426`) | `89.08%` (`0.890809`) | `0.8527` (`0.852650`) | `0.1574` (`0.157376`) | **`0.0159 s`** (`2.70s` train) |
| **2. Distance-Weighted KNN** | Instance-Based | `n_neighbors=25, weights='distance', metric='manhattan', Subsampled(35k) + StandardScaler` | `0.8173 ± 0.0404` (`0.8148` OOF) | `76.41%` (`0.764121`) | `87.12%` (`0.871227`) | `0.8382` (`0.838170`) | `0.1858` (`0.185754`) | `32.0640 s` (`1.26s` train) |
| **3. Decision Tree** | Tree-Based | `criterion='log_loss', max_depth=8, min_samples_leaf=30, SimpleImputer` | `0.8410 ± 0.0440` (`0.8385` OOF) | `77.87%` (`0.778674`) | `89.19%` (`0.891906`) | `0.8595` (`0.859529`) | `0.1470` (`0.147047`) | `0.0186 s` (`16.13s` train) |
| **4. Random Forest** | Bagging Ensemble | `n_estimators=100, max_depth=12, min_samples_leaf=15, max_features='sqrt', max_samples=0.25` | `0.8451 ± 0.0416` (`0.8421` OOF) | **`79.85%`** (`0.798461`) | **`89.91%`** (`0.899112`) | `0.8643` (`0.864341`) | `0.1417` (`0.141702`) | `0.1285 s` (`16.38s` train) |
| **5. HistGradientBoosting** | Gradient Boosting | `lr=0.08, max_iter=150, max_leaf_nodes=31, max_depth=8, min_samples_leaf=25, l2=1.0, native NaN` | **`0.8481 ± 0.0430`** (**`0.8455`** OOF) | `79.22%` (`0.792179`) | `89.70%` (`0.897023`) | **`0.8656`** (`0.865577`) | **`0.1366`** (`0.136627`) | `0.1909 s` (`7.23s` train) |

*(Additional Student-Test candidate classification details: Accuracy / Positive-Class Precision / Positive-Class Recall / Positive-Class F1 / Confusion Matrix `[TN, FP, FN, TP]`):*
- **Logistic Regression:** Acc=`0.9375`, Prec=`0.7464`, Rec=`0.7353`, Pos-F1=`0.7408`, CM=`[133226, 4766, 5048, 14026]`
- **Distance-Weighted KNN:** Acc=`0.9374`, Prec=`0.8073`, Rec=`0.6360`, Pos-F1=`0.7115`, CM=`[135096, 2896, 6943, 12131]`
- **Decision Tree:** Acc=`0.9414`, Prec=`0.7729`, Rec=`0.7327`, Pos-F1=`0.7523`, CM=`[133886, 4106, 5098, 13976]`
- **Random Forest:** Acc=`0.9460`, Prec=`0.8280`, Rec=`0.7008`, Pos-F1=`0.7591`, CM=`[135215, 2777, 5707, 13367]`
- **HistGradientBoosting:** Acc=`0.9447`, Prec=`0.7965`, Rec=`0.7312`, Pos-F1=`0.7625`, CM=`[134429, 3563, 5127, 13947]`

---

## 3. Section 3.2 — Compulsory Per-Model Subsections (Questions A, B, C for All 5 Models)

### Model 1: Logistic Regression (`Linear & Probabilistic Family`)
- **A1. Why suitable for PathSnap:** Candidate road probability decays monotonically as perpendicular distance (`distance_to_segment_m`) and heading mismatch (`bearing_difference_deg`) increase. Logistic Regression directly models the log-odds $\log \frac{P(y=1\mid\mathbf{x})}{1-P(y=1\mid\mathbf{x})} = \mathbf{w}^\top\mathbf{x} + b$ as a convex linear combination of standardized features, producing smooth, well-calibrated probabilities ideal for downstream HMM emission scoring.
- **A2. Major strengths and limitations:**
  - *Strengths:* Extremely fast training (`2.70 s` on 1.48M rows) and sub-20ms inference (`0.0159 s` for 157,851 test candidates); zero overfitting; boosts Observation Top-1 accuracy by `+26.16%` over raw nearest-segment snapping (`52.38%` $\to$ `78.54%`).
  - *Limitations:* Cannot capture conditional nonlinear interactions (for example, `bearing_difference_deg` is highly informative when `gps_speed_mps > 5 m/s` on arterial roads, but noisy when speed is low near intersections).
- **B5. Important hyperparameters and initial values:** `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `max_iter=500`, `random_state=42`, preceded by `SimpleImputer(strategy='median', add_indicator=True)` and `StandardScaler()`.
- **B6. Loss function / optimisation objective:** L2-regularised binary cross-entropy (Log Loss).
- **C7 & C8. Performance on `student_test` dataset:** Macro F1 = **`0.8527`** (`Positive F1 = 0.7408`, `Accuracy = 93.75%`, `Precision = 74.64%`, `Recall = 73.53%`), Observation-level Top-1 Accuracy = **`78.54%`** (`78.66%` on `label_eligible`), Route Continuity = **`89.08%`**, and Log Loss = **`0.1574`**.
- **C9. Evidence of overfitting or underfitting:** **Mild underfitting (high bias, zero variance):** Training Macro F1 (`0.8449`), 80-Trip CV Macro F1 (`0.8451 ± 0.0341`), and Student-Test Macro F1 (`0.8527`) are virtually identical (train-to-CV gap `-0.0002`), showing zero overfitting, while its lower Macro F1 and higher Log Loss (`0.1574` vs `0.1366` for HGB) reflect linear boundary bias at intersections.

### Model 2: Distance-Weighted K-Nearest Neighbours (`Instance-Based Family`)
- **A1. Why suitable for PathSnap:** Tests whether non-parametric local proximity in standardized 11D geometric/kinematic feature space (weighted by $w_j = 1 / \|\mathbf{x} - \mathbf{x}_j\|_1$) can classify road candidates without global functional assumptions.
- **A2. Major strengths and limitations:**
  - *Strengths:* High positive-class precision (`80.73%` on `student_test`) when a candidate has near-identical historical geometric/kinematic matches.
  - *Limitations:* $O(N_{\text{train}} \cdot N_{\text{query}})$ tree-search complexity makes inference **`2,000x` slower** than linear/tree models (`32.06 s` on `student_test` even with `35,000`-row deterministic subsampling per fold, vs `0.016 s` for Logistic Regression), and boundary sparsity in 15D imputed+indicator space degrades probability calibration (`0.1858` Log Loss) and recall (`63.60%`).
- **B5. Important hyperparameters and initial values:** `n_neighbors=25`, `weights='distance'`, `metric='manhattan'`, `max_train_samples=35000`, `random_state=42`, inside `SimpleImputer(median, add_indicator=True)` + `StandardScaler()`.
- **B6. Loss function / optimisation objective:** Non-parametric lazy learning; predicts inverse Manhattan-distance weighted neighbour class proportions.
- **C7 & C8. Performance on `student_test` dataset:** Macro F1 = **`0.8382`** (`Positive F1 = 0.7115`, `Accuracy = 93.74%`, `Precision = 80.73%`, `Recall = 63.60%`), Observation-level Top-1 Accuracy = **`76.41%`**, Route Continuity = **`87.12%`**, and Log Loss = **`0.1858`**.
- **C9. Evidence of overfitting or underfitting:** Moderate local-neighbourhood variance (`Train Macro F1 = 0.8338`, `80-Trip CV Macro F1 = 0.8173 ± 0.0404`, `Student-Test Macro F1 = 0.8382`) combined with curse-of-dimensionality distance concentration in 15D feature space, making KNN unsuitable for low-latency server deployment.

### Model 3: Decision Tree Classifier (`Tree-Based Family`)
- **A1. Why suitable for PathSnap:** Captures axis-aligned hierarchical rules that condition directional features on kinematic state (e.g., splitting on `bearing_difference_deg` only when `gps_speed_mps` is valid and above stationary thresholds, or treating endpoint projections `projection_fraction in {0, 1}` differently from mid-segment projections).
- **A2. Major strengths and limitations:**
  - *Strengths:* Fast inference (`0.0186 s` on `student_test`), transparent decision rules, and a `+0.0104` Log Loss improvement over Logistic Regression (`0.1470` vs `0.1574`) by capturing nonlinear speed/bearing interactions.
  - *Limitations:* Axis-aligned step splits create coarse piecewise-constant leaf probabilities (`max 2^8 = 256` distinct probability values at `max_depth=8`), yielding lower Observation Top-1 ranking resolution (`77.87%`) than ensembles (`79.85%` RF, `79.22%` HGB).
- **B5. Important hyperparameters and initial values:** `criterion='log_loss'`, `max_depth=8`, `min_samples_leaf=30` (enforced to smooth leaf frequency probabilities and prevent zero-probability leaves), `random_state=42`, with `SimpleImputer(median, add_indicator=True)`.
- **B6. Loss function / optimisation objective:** Greedy recursive binary splitting minimizing Shannon entropy / binary Log Loss (`criterion='log_loss'`).
- **C7 & C8. Performance on `student_test` dataset:** Macro F1 = **`0.8595`** (`Positive F1 = 0.7523`, `Accuracy = 94.14%`, `Precision = 77.29%`, `Recall = 73.27%`), Observation-level Top-1 Accuracy = **`77.87%`**, Route Continuity = **`89.19%`**, and Log Loss = **`0.1470`**.
- **C9. Evidence of overfitting or underfitting:** Well-regularised by `max_depth=8` and `min_samples_leaf=30`: Training Macro F1 is `0.8527` vs `0.8410` on 80-Trip CV (a small `+0.0117` gap) and `0.8595` on `student_test`, avoiding the severe leaf memorization of unpruned trees while slightly underfitting smooth diagonal boundaries.

### Model 4: Random Forest Classifier (`Bagging Ensemble Family`)
- **A1. Why suitable for PathSnap:** Averages predictions across $100$ decorrelated bootstrap trees trained on random feature subsets (`max_features='sqrt'`), smoothing the hard step boundaries of a single Decision Tree into continuous, well-calibrated probability gradients around complex intersections.
- **A2. Major strengths and limitations:**
  - *Strengths:* Achieves the highest greedy **Student-Test Observation Top-1 Accuracy (`79.85%`)** and highest greedy **Route Continuity (`89.91%`)** among untuned models, with strong precision (`82.80%`) and low Log Loss (`0.1417`).
  - *Limitations:* Slower training (`70.60 s` in CV vs `28.68 s` for HGB) and requires explicit median imputation for nullable kinematic columns rather than native missing-value split routing.
- **B5. Important hyperparameters and initial values:** `n_estimators=100`, `max_depth=12`, `min_samples_leaf=15`, `max_features='sqrt'`, `max_samples=0.25`, `random_state=42`, wrapped with `SimpleImputer(median, add_indicator=True)`.
- **B6. Loss function / optimisation objective:** Bootstrap aggregation of Gini-impurity decision trees with ensemble probability averaging $\hat{P}(y=1\mid\mathbf{x}) = \frac{1}{B}\sum_{b=1}^B \hat{P}_b(y=1\mid\mathbf{x})$.
- **C7 & C8. Performance on `student_test` dataset:** Macro F1 = **`0.8643`** (`Positive F1 = 0.7591`, `Accuracy = 94.60%`, `Precision = 82.80%`, `Recall = 70.08%`), Observation-level Top-1 Accuracy = **`79.85%`**, Route Continuity = **`89.91%`**, and Log Loss = **`0.1417`**.
- **C9. Evidence of overfitting or underfitting:** Minimal overfitting (`Train Macro F1 = 0.8618` vs `80-Trip CV Macro F1 = 0.8451` and `Student-Test Macro F1 = 0.8643`), demonstrating strong variance reduction over the single Decision Tree (`+0.0198` Top-1 accuracy gain on `student_test`).

### Model 5: HistGradientBoosting Classifier (`Gradient Boosting Family` — Selected Model)
- **A1. Why suitable for PathSnap:** Sequentially fits additive histogram-binned regression trees to the negative gradients of binary log-loss. Crucially, `HistGradientBoostingClassifier` **learns optimal split directions for `NaN` values natively** in `gps_bearing_deg`, `bearing_difference_deg`, `gps_speed_mps`, and `delta_time_s` (separating stationary GPS jitter and first-observation points from moving vehicle steps without median-imputation distortion).
- **A2. Major strengths and limitations:**
  - *Strengths:* Achieves the **highest 80-Trip CV Macro F1 (`0.8481 ± 0.0430`, `0.8455` OOF)**, **highest 80-Trip OOF Route Continuity (`90.10%`)**, **highest Student-Test Macro F1 (`0.8656`)**, **highest Student-Test Positive F1 (`0.7625`)**, and **lowest Student-Test Log Loss (`0.1366`)**, while training **`2.5x` faster** than Random Forest (`7.23 s` vs `16.38 s`).
  - *Limitations:* Requires regularisation (`max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`, and learning-rate shrinkage) to prevent late-iteration overconfidence on noisy training trajectories.
- **B5. Important hyperparameters and initial values:** `learning_rate=0.08`, `max_iter=150`, `max_leaf_nodes=31`, `max_depth=8`, `min_samples_leaf=25`, `l2_regularization=1.0`, `random_state=42`.
- **B6. Loss function / optimisation objective:** Direct sequential minimisation of Binary Cross-Entropy (Log Loss) via second-order Newton-Raphson histogram gradient boosting.
- **C7 & C8. Performance on `student_test` dataset:** Macro F1 = **`0.8656`** (`Positive F1 = 0.7625`, `Accuracy = 94.47%`, `Precision = 79.65%`, `Recall = 73.12%`), Observation-level Top-1 Accuracy = **`79.22%`** (`79.34%` eligible), Route Continuity = **`89.70%`**, and Log Loss = **`0.1366`**.
- **C9. Evidence of overfitting or underfitting:** Mild, well-controlled generalisation gap (`Train Macro F1 = 0.8825` vs `80-Trip CV Macro F1 = 0.8481` and `Student-Test Macro F1 = 0.8656`). Inspection of the real iteration-wise loss curve ([`results/phase2/hgb_iterative_loss_curve.png`](../results/phase2/hgb_iterative_loss_curve.png) & [`results/phase2/hgb_iteration_loss_history.csv`](../results/phase2/hgb_iteration_loss_history.csv)) confirms that both training and `student_test` Log Loss decrease monotonically without divergence (`student_test` Log Loss drops from `0.3358` at iteration 1 to `0.1386` at iteration 200).

---

## 4. Final Model Selection Justification

Based strictly on **80-trip training cross-validation evidence** (`results/phase2/model_comparison_80train_10test.csv`), we selected **`HistGradientBoostingClassifier`** for Bayesian hyperparameter optimisation and final HMM/Viterbi integration:
1. **Highest 80-Trip CV Macro F1 & Positive F1:** `0.8481` Mean Fold Macro F1 (`0.8455` pooled OOF Macro F1, `0.7271` OOF Positive F1), outperforming Random Forest (`0.8451` / `0.8421`), Logistic Regression (`0.8451` / `0.8425`), Decision Tree (`0.8410` / `0.8385`), and KNN (`0.8173` / `0.8148`).
2. **Superior Probability Calibration for HMM Emissions:** Achieves the lowest Log Loss on `student_test` (`0.1366`) and second-lowest on 80-trip CV (`0.1549`), which is essential because the downstream Viterbi decoder sums $\log P(y_{i,c}=1\mid\mathbf{x}_{i,c})$ across hundreds of time steps.
3. **Native Missing-Value Routing & Compact Artifact Size:** Natively handles `<NA>` kinematic features without an external imputer/scaler wrapper, producing a compact **`433 KB`** production artifact (`artifacts/final_model.joblib`) with fast inference (`0.19 s` for all 10 test trips = `19 ms/trip`).

---

## 5. Section 4 — Bayesian Hyperparameter Optimisation (25 Trials on 80 Training Trips)

Executed in [`src/optimization/bayesian_tuning.py`](../src/optimization/bayesian_tuning.py) using **Optuna's Tree-structured Parzen Estimator (`TPESampler(seed=42, n_startup_trials=3)`)** across **25 completed trials** of 5-fold `GroupKFold` cross-validation on the 80 training trips (`1,484,716` labeled rows). Full trial logs are saved in [`results/phase2/bayesian_trials_80trip.csv`](../results/phase2/bayesian_trials_80trip.csv) and [`results/phase2/bayesian_best_summary_80trip.json`](../results/phase2/bayesian_best_summary_80trip.json), with the progression chart in [`results/phase2/bayesian_tuning_80trip_progression.png`](../results/phase2/bayesian_tuning_80trip_progression.png).

1. **Hyperparameters & Search Distributions:**
   - `learning_rate`: log-uniform $\in [0.02, 0.18]$
   - `max_iter`: integer $\in [80, 240]$ (step $20$)
   - `max_leaf_nodes`: integer $\in [15, 63]$
   - `max_depth`: integer $\in [4, 12]$
   - `min_samples_leaf`: integer $\in [10, 80]$
   - `l2_regularization`: log-uniform $\in [0.01, 10.0]$
2. **Optimisation Objective:** Maximize 80-trip Out-of-Fold (`GroupKFold`, $K=5$) Macro F1-score (`oof_macro_f1`).
3. **Number of Trials & Computational Cost:** **25 trials completed** in **`1,108.17 s` (`18.47 min`)** (~`44.3 s` per 5-fold 1.48M-row trial), plus **`9.07 s`** to refit the best configuration on all `1,484,716` labeled training rows.
4. **Best Hyperparameters Found (Trial #16):**
   - `learning_rate = 0.031547`, `max_iter = 200`, `max_leaf_nodes = 17`, `max_depth = 9`, `min_samples_leaf = 40`, `l2_regularization = 0.663513` (a smaller, more regularised tree structure `max_leaf_nodes=17, min_samples_leaf=40` paired with a lower learning rate `0.0315` over `200` trees).
5. **Untuned vs. Bayesian-Tuned Comparison ([`results/phase2/tuned_vs_untuned_comparison.csv`](../results/phase2/tuned_vs_untuned_comparison.csv)):**

| Metric | Untuned HistGradientBoosting (`lr=0.08, leaves=31, iter=150`) | **Bayesian-Tuned HistGradientBoosting (Trial #16)** | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **80-Trip CV Macro F1 (Mean across Folds)** | `0.848113` | **`0.850489`** | **`+0.002376`** |
| **80-Trip Out-of-Fold Macro F1** | `0.845540` | **`0.848206`** | **`+0.002666`** |
| **80-Trip Out-of-Fold Positive F1** | `0.727077` | **`0.731910`** | **`+0.004833`** |
| **80-Trip Out-of-Fold Log Loss** | `0.157044` | **`0.154300`** | **`-0.002744` (Better)** |
| **80-Trip Out-of-Fold Obs Top-1 Accuracy** | `75.3255%` | **`75.7911%`** | **`+0.4656%`** |
| **10-Trip `student_test` Macro F1** | `0.865577` | **`0.865544`** | `-0.000033` (Tied) |
| **10-Trip `student_test` Positive-Class F1** | `0.762464` | **`0.762555`** | **`+0.000091`** |
| **10-Trip `student_test` Positive Recall** | `73.1205%` | **`73.4770%`** | **`+0.3565%`** |
| **10-Trip `student_test` Obs Top-1 Accuracy (Greedy)** | `79.2179%` | `79.0975%` (**`88.7243%` with Viterbi**) | — |
| **10-Trip `student_test` Log Loss** | `0.136627` | `0.138567` | `+0.001940` |
| **10-Trip `student_test` Inference Time** | `0.1869 s` | `0.2193 s` (`21.9 ms / trip`) | `+0.0324 s` |

- **Scientific Analysis:** On the 80-trip training cross-validation, Bayesian tuning improved **every single metric** (`+0.0027` OOF Macro F1, `+0.0048` Positive F1, `-0.0027` lower OOF Log Loss, and `+0.47%` higher OOF Top-1 Accuracy) by reducing `max_leaf_nodes` from `31` to `17` and increasing `min_samples_leaf` from `25` to `40`. On the 10-trip `student_test` holdout, the tuned model achieves higher positive-class recall (`73.48%` vs `73.12%`) and higher Positive F1 (`0.7626`), while virtually tying on Macro F1 (`0.8655` vs `0.8656`).

---

## 6. Section 5 — Classifier-Only (Greedy) vs. Classifier + HMM/Viterbi on All 10 `student_test` Trips

Using the Bayesian-Tuned HistGradientBoosting emission probabilities (`results/phase2/predictions/student_test_predictions_tuned_hgb.parquet`), we evaluated `decode_trip_viterbi` across all **10 `student_test` trips (`19,160` observations, `19,103` aligned)**. Full results are saved in [`results/phase2/student_test_hmm_viterbi_comparison.csv`](../results/phase2/student_test_hmm_viterbi_comparison.csv) and plotted in [`results/phase2/student_test_greedy_vs_viterbi.png`](../results/phase2/student_test_greedy_vs_viterbi.png).

| Student-Test Trip ID | Total Obs (`Aligned`) | Classifier-Only (Greedy) Top-1 Acc | **Classifier + HMM/Viterbi Top-1 Acc** | **Top-1 Accuracy Gain ($\Delta$)** | Greedy Route Continuity | **HMM/Viterbi Route Continuity** | **Continuity Gain ($\Delta$)** | Observations Corrected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`00000000`** | `2,503 (2,503)` | `93.17%` | **`95.45%`** | **`+2.28%`** | `95.00%` | **`99.52%`** | **`+4.52%`** | `121` |
| **`00000007`** | `1,354 (1,354)` | `90.25%` | **`94.17%`** | **`+3.91%`** | `99.41%` | **`99.93%`** | **`+0.52%`** | `119` |
| **`00000009`** | `1,049 (1,030)` | `73.20%` | **`82.43%`** | **`+9.22%`** | `83.78%` | **`96.76%`** | **`+12.98%`** | `180` |
| **`00000040`** | `2,995 (2,995)` | `88.41%` | **`93.36%`** | **`+4.94%`** | `95.02%` | **`99.30%`** | **`+4.28%`** | `274` |
| **`00000052`** | `262 (262)` | `90.46%` | **`96.18%`** | **`+5.73%`** | `93.49%` | **`97.32%`** | **`+3.83%`** | `16` |
| **`00000056`** | `3,958 (3,925)` | `79.39%` | **`88.97%`** | **`+9.58%`** | `86.00%` | **`97.32%`** | **`+11.32%`** | `564` |
| **`00000065`** | `768 (768)` | `77.86%` | **`87.50%`** | **`+9.64%`** | `87.48%` | **`98.70%`** | **`+11.21%`** | `127` |
| **`00000076`** | `1,486 (1,484)` | `76.89%` | **`88.68%`** | **`+11.79%`** | `84.38%` | **`97.85%`** | **`+13.47%`** | `245` |
| **`00000077`** | `3,911 (3,908)` | `61.00%` | **`79.91%`** | **`+18.91%`** | `87.72%` | **`98.62%`** | **`+10.90%`** | `1,339` |
| **`00000091`** | `874 (874)` | `77.57%` | **`89.82%`** | **`+12.24%`** | `83.73%` | **`98.17%`** | **`+14.43%`** | `152` |
| **All 10 `student_test` Trips** | **`19,160 (19,103)`** | **`79.10%`** | **`88.72%`** | **`+9.63%`** | **`89.70%`** | **`98.47%`** | **`+8.77%`** | **`3,137`** |

> **Key Takeaway:** Unlike the 5-trip Phase 1 pilot (where Fold 4 trained on only 4 small trips), once `HistGradientBoosting` is trained on all **80 training trips (`1,484,716` rows)**, its emission probabilities are well-calibrated (`0.1386` Log Loss), and **HMM/Viterbi decoding improves Observation Top-1 Accuracy and Route Continuity on all 10 out of 10 `student_test` trips**, raising overall Top-1 Accuracy from **`79.10%` to `88.72%` (`+9.63%`, or `+36.34%` over raw nearest-segment snapping [`52.38%`])** and Route Continuity from **`89.70%` to `98.47%` (`+8.77%`)**!

---

## 7. Section 7 — Laptop 2 Evaluator UI & Model Scorer Handover

1. **Frozen Model Scorer Interface ([`src/models/inference.py`](../src/models/inference.py) & [`src/models/scorer.py`](../src/models/scorer.py)):**
   - Serialized final artifact: [`artifacts/final_model.joblib`](../artifacts/final_model.joblib) (`432,867` bytes, SHA-256 `6f9e74088a7689b3ec8f6e9f3e992d7f0fea42804deb68546fbd1531c88748b1`) + [`artifacts/final_model_metadata.json`](../artifacts/final_model_metadata.json).
   - Provides `load_model_bundle()` and `score_candidates(candidates_df, model_bundle)` preserving `trip_id`, `observation_id`, `candidate_arc_id`, `arc_id`, `candidate_rank`, and outputting `predicted_probability`, `pred_prob`, `predicted_log_probability`, `predicted_label`, and `is_obs_top1`.
2. **Laptop 2 Evaluator UI ([`src/ui/app.py`](../src/ui/app.py) & [`src/ui/api_client.py`](../src/ui/api_client.py)):**
   - Connects over the LAN to Laptop 1's FastAPI server via `PATHSNAP_API_BASE_URL` (`GET /health`, `GET /trips`, `POST /predict`).
   - Strictly client-side: never loads local models, never reads dataset files, and automatically strips any accidental reference-route fields from responses (`test_ui_live_http_client_strips_forbidden_ground_truth_fields`).
   - Supports both **Live API Mode** and a clearly badged **Mock Contract Preview Mode** (`PATHSNAP_UI_MOCK_MODE=true`).
