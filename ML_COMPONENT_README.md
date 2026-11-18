# PathSnap — ML, Bayesian Optimisation & HMM/Viterbi Component README

**Component Owner:** Pranav Shashikiran Atreya  
**Branch:** `feature/pranav-ml`

---

## 1. Architecture & Module Layout

- [`src/models/model_pipeline.py`](src/models/model_pipeline.py):
  - `verify_candidate_contract(candidates_df, observations_df)`: Validates all 20 candidate columns, 11 `CANDIDATE_FEATURE_COLUMNS`, label eligibility rules, and candidate miss rates.
  - `get_five_models(random_state=42)`: Constructs leak-free `sklearn` pipelines for **Logistic Regression**, **Distance-Weighted KNN**, **Decision Tree**, **Random Forest**, and **HistGradientBoosting**.
  - `evaluate_five_models_group_cv(...)`: Runs trip-grouped `GroupKFold(n_splits=5)` cross-validation, computing candidate-level metrics (`Accuracy`, `Precision`, `Recall`, `Macro F1`, `Positive F1`, `Log Loss`, `Confusion Matrix`) and observation/sequence metrics (`Observation Top-1 Accuracy`, `Route Continuity`).
- [`src/optimization/bayesian_tuning.py`](src/optimization/bayesian_tuning.py):
  - `run_bayesian_tuning_hgb(...)`: Runs genuine **Optuna TPE (`TPESampler`)** Bayesian hyperparameter optimisation over `HistGradientBoostingClassifier` using `GroupKFold` by `trip_id`.
- [`src/sequence/viterbi_decoder.py`](src/sequence/viterbi_decoder.py):
  - `decode_trip_viterbi(candidates_df, observations_df, graph, prob_col, config)`: Log-domain HMM/Viterbi route reconstruction consuming ML candidate probabilities and `TripRoadGraph` connectivity.
- [`scripts/run_phase1_ml_pipeline.py`](scripts/run_phase1_ml_pipeline.py):
  - End-to-end executable script that verifies the candidate data, runs the 5-model `GroupKFold` experiment, runs Optuna TPE Bayesian optimisation, executes synthetic and real-trip HMM/Viterbi decoding, and saves all CSVs, PNG charts, and interactive Folium HTML maps into `results/phase1/`.

---

## 2. Exact Commands to Reproduce Everything

From `C:\Users\Pranav S Atreya\PES\sem5\ML\hackathon\PathSnap-Map-Matching` in PowerShell:

```powershell
# 1. Point to the raw dataset root
$env:PATHSNAP_DATA_ROOT="C:\Users\Pranav S Atreya\PES\sem5\ML\hackathon\Student folder\Student folder"

# 2. Run the full automated pytest suite safely (11 tests: data loader, spatial/kinematics, alignment, labels, graph, Viterbi)
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider

# 3. Generate dataset inventory (90 trips), sensitivity analysis (10 trips), and 5-trip Parquet previews + alignment map
.\.venv\Scripts\python.exe scripts/run_audit_and_previews.py

# 4. Run the 5-model GroupKFold experiment, Optuna TPE Bayesian optimisation, and HMM/Viterbi route reconstruction
.\.venv\Scripts\python.exe scripts/run_phase1_ml_pipeline.py
```

---

## 3. Generated Phase 1 Evidence Files (`results/phase1/`)

- [`results/phase1/candidate_generation_summary.csv`](results/phase1/candidate_generation_summary.csv)
- [`results/phase1/label_distribution.csv`](results/phase1/label_distribution.csv)
- [`results/phase1/model_comparison.csv`](results/phase1/model_comparison.csv)
- [`results/phase1/cv_results.csv`](results/phase1/cv_results.csv)
- [`results/phase1/bayesian_trials.csv`](results/phase1/bayesian_trials.csv)
- [`results/phase1/hmm_test_results.csv`](results/phase1/hmm_test_results.csv)
- [`results/phase1/phase1_summary.md`](results/phase1/phase1_summary.md)
- [`results/phase1/model_comparison_chart.png`](results/phase1/model_comparison_chart.png)
- [`results/phase1/bayesian_tuning_progression.png`](results/phase1/bayesian_tuning_progression.png)
- [`results/phase1/trip_00000001_viterbi_vs_greedy_map.html`](results/phase1/trip_00000001_viterbi_vs_greedy_map.html)
