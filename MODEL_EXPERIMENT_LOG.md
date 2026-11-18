# PathSnap — Model Experimentation & Bayesian Optimisation Log

**Owner:** Pranav Shashikiran Atreya  
**Random Seed:** `42`  
**Cross-Validation Scheme:** `GroupKFold(n_splits=5)` grouped strictly by `trip_id`  
**Phase 1 Dataset Scope:** Preliminary 5-Trip Pilot (`00000001`, `00000002`, `00000003`, `00000004`, `00000005` — `11,488` GPS observations, `98,636` total candidates, `92,299` `label_eligible` labeled candidates: `10,597` positive [`11.48%`], `81,702` negative [`88.52%`]).

---

## Experiment 1: Five-Model Baseline Comparison (`results/phase1/model_comparison.csv`)

### Feature Set (`CANDIDATE_FEATURE_COLUMNS` — 11 features)
`distance_to_segment_m`, `projection_fraction`, `distance_to_start_m`, `distance_to_end_m`, `segment_length_m`, `segment_bearing_deg`, `gps_bearing_deg`, `bearing_difference_deg`, `gps_speed_mps`, `delta_time_s`, `is_fallback_candidate`.

### Model Configurations & Measured Results

| # | Model Name | Family | Pipeline / Configuration | Mean Fold Macro F1 | OOF Macro F1 | OOF Pos F1 | OOF Precision | OOF Recall | OOF Log Loss | Mean Fold Obs Top-1 Acc | OOF Obs Top-1 Acc | Greedy Continuity | Confusion Matrix `(TN, FP, FN, TP)` | Train / Infer Time (s) |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Logistic Regression** | Linear & Probabilistic | `SimpleImputer(median, add_indicator=True)` + `StandardScaler()` + `LogisticRegression(C=1.0, max_iter=500, random_state=42)` | `0.7838` | `0.7274` | `0.5093` | `0.5980` | `0.4435` | `0.2378` | **`0.7749`** | **`0.6500`** | **`0.8677`** | `(78542, 3160, 5897, 4700)` | `1.17s / 0.04s` |
| 2 | **Distance-Weighted KNN** | Instance-Based | `SimpleImputer(median, add_indicator=True)` + `StandardScaler()` + `KNeighborsClassifier(n_neighbors=25, weights='distance', metric='manhattan')` | `0.7960` | `0.7411` | `0.5287` | `0.7316` | `0.4139` | `0.3836` | `0.7077` | `0.5911` | `0.8320` | `(80093, 1609, 6211, 4386)` | `1.72s / 24.08s` |
| 3 | **Decision Tree** | Tree-Based | `SimpleImputer(median, add_indicator=True)` + `DecisionTreeClassifier(criterion='log_loss', max_depth=8, min_samples_leaf=30, random_state=42)` | `0.6893` | `0.7016` | `0.4604` | `0.5729` | `0.3848` | `0.6159` | `0.5856` | `0.5199` | `0.7714` | `(78662, 3040, 6519, 4078)` | `4.63s / 0.03s` |
| 4 | **Random Forest** | Bagging Ensemble | `SimpleImputer(median, add_indicator=True)` + `RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=15, max_features='sqrt', random_state=42)` | `0.7961` | `0.7396` | `0.5251` | **`0.7537`** | `0.4029` | **`0.2102`** | `0.7389` | `0.6338` | `0.8415` | `(80307, 1395, 6328, 4269)` | `10.42s / 0.42s` |
| 5 | **HistGradientBoosting** | Gradient Boosting | `HistGradientBoostingClassifier(lr=0.08, max_iter=150, max_leaf_nodes=31, max_depth=8, min_samples_leaf=25, l2=1.0, random_state=42)` | **`0.8013`** | **`0.7445`** | **`0.5382`** | `0.6658` | **`0.4516`** | `0.2759` | `0.7174` | `0.5962` | `0.8329` | `(79300, 2402, 5811, 4786)` | `5.39s / 0.20s` |

---

## Experiment 2: Optuna TPE Bayesian Hyperparameter Optimisation (`results/phase1/bayesian_trials.csv`)

- **Algorithm:** `optuna.samplers.TPESampler(seed=42, n_startup_trials=3)`
- **Target Model:** `HistGradientBoostingClassifier`
- **Objective:** Maximize 5-fold `GroupKFold` Out-of-Fold Macro F1-score on labeled training candidates.
- **Total Compute Time (8 Trials + Final Refit):** `50.13s`

| Trial | `learning_rate` | `max_iter` | `max_leaf_nodes` | `max_depth` | `min_samples_leaf` | `l2_regularization` | Mean Fold Macro F1 | **OOF Macro F1** | OOF Pos F1 | OOF Log Loss | Greedy Continuity | Time (s) |
| :-: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 (Baseline) | `0.080000` | `140` | `31` | `8` | `25` | `1.000000` | `0.8011` | `0.744416` | `0.5381` | `0.2748` | `0.8329` | `5.52s` |
| 2 | `0.045544` | `240` | `50` | `9` | `21` | `0.029375` | `0.7985` | `0.742336` | `0.5346` | `0.2911` | `0.8347` | `10.92s` |
| 3 | `0.022722` | `220` | `44` | `10` | `11` | `8.123245` | `0.7958` | `0.742306` | `0.5338` | `0.2525` | `0.8412` | `10.31s` |
| 4 | `0.157043` | `80` | `25` | `10` | `13` | `0.743122` | `0.7985` | `0.740871` | `0.5322` | `0.2791` | `0.8383` | `3.03s` |
| 5 | `0.129547` | `180` | `16` | `7` | `55` | `0.158764` | `0.7988` | `0.740501` | `0.5310` | `0.2814` | `0.8328` | `4.97s` |
| 6 | `0.054427` | `80` | `41` | `8` | `27` | `5.277649` | `0.7937` | `0.743182` | `0.5350` | **`0.2476`** | **`0.8433`** | `3.89s` |
| **7 (Best)** | **`0.073068`** | **`120`** | **`31`** | **`12`** | **`41`** | **`4.052611`** | **`0.8010`** | **`0.745228`** | **`0.5397`** | `0.2728` | `0.8385` | `4.62s` |
| 8 | `0.135547` | `120` | `59` | `11` | `41` | `0.852608` | `0.7894` | `0.733025` | `0.5174` | `0.3046` | `0.8271` | `5.82s` |

---

## Experiment 3: HMM & Viterbi Sequential Decoding (`results/phase1/hmm_test_results.csv`)

- **Decoder:** `decode_trip_viterbi(...)` (`src/sequence/viterbi_decoder.py`)
- **Transition Scoring (`ViterbiConfig`):**
  - `0` hops (stay on same directed arc): `log P = -0.15`
  - `1` hop (direct outgoing successor `to_node(u) == from_node(v)`): `log P = -0.60` (`-4.00` penalty if immediate U-turn `v_to == u_from`)
  - `2` hops (skipped short intermediate segment): `log P = -2.20`
  - `3` hops: `log P = -3.80`
  - Disconnected (`>3` hops): `log P = -10.50`
  - Timestamp gap relaxation: linear decay between `10s` and `60s`, uniform reset above `60s`.
- **Key Findings:**
  - On Trip `00000001` (`1,315` observations), Viterbi decoding on Tuned HGB out-of-fold probabilities boosts **Observation Top-1 Accuracy from `75.36%` to `93.31%` (`+17.95%`)** and **Route Continuity from `75.42%` to `96.80%` (`+21.39%`)**, correcting `289` noisy pointwise assignments.
  - On Trip `00000002` (`7,243` observations, `6,373` aligned), Tuned HGB out-of-fold probabilities are less calibrated (`0.3177` fold Log Loss because Fold 4 trains on only `33,143` rows from the 4 smaller trips); Viterbi increases **Route Continuity from `83.87%` to `96.09%` (`+12.22%`)** while pointwise Top-1 accuracy shifts from `47.45%` to `43.75%` (`-3.70%`) where the decoder commits to connected alternative branches.
  - Across all 5 preview trips (`11,488` observations), Viterbi decoding on well-calibrated **Logistic Regression (OOF)** probabilities (`0.2378` Log Loss) increases **Observation Top-1 Accuracy on every trip**, raising the 5-trip aggregate from **`65.00%` to `84.99%` (`+19.98%`)** and **Route Continuity from `86.84%` to `97.09%` (`+10.25%`)**.
