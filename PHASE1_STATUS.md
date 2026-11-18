# PathSnap — Phase 1 Status & Milestone Checklist

**Branch:** `feature/pranav-ml` (synced with `origin/main` commit `4ba8528`)  
**Phase 1 Evaluation Readiness:** **READY (10/10 Pytest Tests Passing, All Phase 1 CSVs, Plots & Folium Maps Generated)**

---

## 1. Completed Phase 1 Milestones

- [x] **Team Integration Verified (`origin/main`):**
  - Integrated L A Gokul's dataset ingestion (`src/data_loader.py`), integrity auditing (`src/eda.py`), and static/Folium visualisations (`src/visualization.py`, `outputs/trip_map.html`).
  - Integrated Qamar Ahmed's WGS84 tangent projection, `STRtree` spatial candidate generation, 11 causal features, monotonic DP route alignment, and `TripRoadGraph` (`src/pathsnap/`).
- [x] **Editable Package & Full Automated Test Suite (`10/10 Passed`):**
  - Installed `pathsnap` in `.venv` via `python -m pip install -e . --no-deps`.
  - All 8 existing unit tests + 2 new deterministic Viterbi unit tests (`tests/test_viterbi_decoder.py`) pass (`10 passed in 8.73s`).
- [x] **Candidate Preview Generation & Schema Audit (`scripts/run_audit_and_previews.py`):**
  - Executed across all 80 training and 10 student-testing trips for inventory (`data/dataset_inventory.csv`), 10-trip sensitivity analysis (`data/sensitivity_analysis.csv`), and 5-trip Parquet candidate/observation extraction (`outputs/candidate_previews/candidates.parquet`, `observations.parquet`).
  - Confirmed 20 candidate columns, 11 predictive features (`CANDIDATE_FEATURE_COLUMNS`), `Int8` nullable labels, `0.1978%` candidate miss rate (`21 / 10,618` aligned observations), and zero identifier leakage.
- [x] **Preliminary Five-Model Experiment (`src/models/model_pipeline.py`):**
  - Trained and evaluated **Logistic Regression**, **Distance-Weighted KNN**, **Decision Tree**, **Random Forest**, and **HistGradientBoosting** using 5-fold `GroupKFold(n_splits=5)` grouped by `trip_id` across the 5 representative training trips (`92,299` labeled rows).
  - Saved full results to `results/phase1/model_comparison.csv` and `results/phase1/cv_results.csv`.
- [x] **Genuine Bayesian Hyperparameter Optimisation (`src/optimization/bayesian_tuning.py`):**
  - Executed 8-trial **Optuna `TPESampler`** Bayesian search over `HistGradientBoostingClassifier` using 5-fold `GroupKFold` (`results/phase1/bayesian_trials.csv`, saved pilot artifact `artifacts/phase1_pilot_tuned_hgb.joblib`).
- [x] **HMM / Viterbi Route Reconstruction (`src/sequence/viterbi_decoder.py`):**
  - Implemented log-domain Viterbi decoder integrating ML emission probabilities with `TripRoadGraph` topological transitions (`0`-hop same arc, `1`-hop direct successor, `2..3`-hop bridge, disconnected backoff, U-turn penalty, empty candidate carry-forward, and large GPS gap relaxation).
  - Verified on both synthetic test cases and real 5-trip out-of-fold ML probabilities (`results/phase1/hmm_test_results.csv`), improving Trip `00000001` HGB Top-1 accuracy from **`75.36%` to `93.31%`** (`+17.95%`) and Logistic Regression 5-trip aggregate Top-1 accuracy from **`65.00%` to `84.99%`** (`+19.98%`) with **`97.09%` route continuity**.
- [x] **Interactive Folium Maps Ready for Demo:**
  - `outputs/trip_map.html` (Gokul's raw trajectory + road network + reference route overview)
  - `outputs/candidate_previews/trip_00000001_alignment_map.html` (Qamar's candidate network + monotonic reference alignment status map)
  - `results/phase1/trip_00000001_viterbi_vs_greedy_map.html` (Pranav's Classifier-Only Greedy vs Classifier + HMM/Viterbi reconstructed route comparison map)

---

## 2. Remaining Work for Phase 2 (Post-Lunch Final Submission)

1. **Full 80-Trip Candidate Extraction & Training:** Run candidate generation across all **80 training trips** and **10 `student_test` trips**, then re-run the 5-model `GroupKFold` comparison on the complete 80-trip training set.
2. **Extended Bayesian Optimisation & Calibration:** Run a 25–40 trial Optuna TPE search on the full 80-trip dataset and evaluate Untuned vs. Tuned models on the 10 held-out `student_test` trips.
3. **Two-Laptop FastAPI Backend & Evaluator UI Integration:** Wire the exported full-80-trip model and `decode_trip_viterbi(...)` into Qamar's FastAPI endpoint for live 10-trip evaluator testing.
