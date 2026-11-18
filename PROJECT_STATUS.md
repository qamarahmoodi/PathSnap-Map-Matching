# PathSnap — Project Status & Integration Tracker

**Last Updated:** 2026-11-18  
**Active Branch:** `feature/pranav-ml`  
**Component Owner:** Pranav Shashikiran Atreya (ML Experimentation, Bayesian Optimisation, Model Comparison, HMM/Viterbi Coordination)

---

## 1. Completed Phase 1 Tasks

- [x] **Workspace & Materials Audit:**
  - Verified local workspace `C:\Users\Pranav S Atreya\PES\sem5\ML\hackathon`.
  - Audited all five authoritative PDF guidelines in `PathSnap_student_materials/`:
    1. `Main Guidelines.pdf`
    2. `Models and Training Guidelines.pdf`
    3. `Deliverables.pdf`
    4. `Assessment and Plagiarism Policy.pdf`
    5. `Dataset Exploration.pdf`
- [x] **Repository & Branch Setup:**
  - Cloned `https://github.com/qamarahmoodi/PathSnap-Map-Matching.git` into `C:\Users\Pranav S Atreya\PES\sem5\ML\hackathon\PathSnap-Map-Matching`.
  - Verified initial repository contents (`README.md` only, clean working tree on `main`).
  - Created and checked out dedicated feature branch `feature/pranav-ml` without nested `.git` directories.
- [x] **Dataset Verification & Schema Audit:**
  - Inspected `Student folder/Student folder/` containing `train` (80 trips) and `student_test` (10 trips).
  - Confirmed all 90 supplied trips have all 4 files (`.track`, `.nodes`, `.arcs`, `.route`) with zero missing files or column-count anomalies.
  - Confirmed the 10 reserved `evaluator_test` trips (`00000015`, `00000017`, `00000023`, `00000030`, `00000046`, `00000050`, `00000061`, `00000074`, `00000082`, `00000085`) are held back for live evaluation and are not in the student archive.
- [x] **Python Environment & Dependencies:**
  - Verified Python `3.13.1` and created isolated virtual environment at `PathSnap-Map-Matching\.venv`.
  - Authored `requirements.txt` and `.gitignore`.
  - Installed all required ML, Bayesian optimisation, spatial/graph, visualisation, and API libraries inside `.venv`.
- [x] **Modular Project Structure & Documentation:**
  - Created standard modular folders: `src/{data,features,models,evaluation,optimization,sequence,utils}`, `scripts/`, `configs/`, `artifacts/`, `results/`, `reports/`, `tests/`.
  - Created `ML_REQUIREMENTS.md` (authoritative requirement checklist for ML, Bayesian tuning, and HMM).
  - Created `CANDIDATE_DATA_SCHEMA.md` (`PROVISIONAL` interface specification for candidate generation and label alignment).

---

## 2. Outstanding Requirements (Next Phases)

1. **Phase 2 — Candidate Generation & Label Alignment Confirmation (with Qamar & Gokul):**
   - Finalize `CANDIDATE_DATA_SCHEMA.md` with Qamar Ahmed.
   - Confirm exact spatial search radius ($R_{\text{search}}$), maximum candidates per GPS observation ($K_{\max}$), and the `.route`-to-`.track` reference label alignment algorithm.
2. **Phase 3 — Five-Model Experimentation & Comparison:**
   - Implement group-aware cross-validation (`GroupKFold` by `trip_id`) on the 80 training trips with leakage-free preprocessing pipelines (`StandardScaler` / `Imputer` fit strictly on training folds).
   - Train and evaluate at least 5 distinct models across different algorithm families (e.g., Logistic Regression, Random Forest, Extra Trees / KNN / MLP, LightGBM, XGBoost / CatBoost).
   - Log CV metrics and 10-trip `student_test` metrics (`Accuracy`, `Precision`, `Recall`, `Macro F1`, `Binary F1`, `Log Loss`, `Observation Top-1 Accuracy`, `Candidate Miss Rate`, `Route Continuity`, `Training Time`, `Inference Time`).
   - Generate mandatory comparison plots, confusion matrices, epoch loss curves (for iterative models), and consolidated table (`results/model_comparison.csv`).
3. **Phase 4 — Bayesian Hyperparameter Optimisation:**
   - Run Bayesian hyperparameter optimisation (Optuna TPE) on the selected top-performing model using group-aware CV on the 80 training trips.
   - Save full trial history (`results/bayesian_trials.csv`), best parameters (`configs/best_params.yaml`), refit model artifact (`artifacts/best_model.joblib`), and Untuned vs. Tuned comparison on `student_test`.
4. **Phase 5 — HMM & Viterbi Route Reconstruction Integration:**
   - Coordinate with Qamar to integrate calibrated model emission probabilities with road-network topological transitions (`.arcs`, `.nodes`).
   - Handle missing candidates, disconnected transitions, and large GPS time gaps ($\Delta t > 1\text{ s}$).
   - Evaluate and compare **Classifier-Only** vs. **Classifier + HMM/Viterbi** on `student_test` and generate interactive Folium comparison maps.
5. **Phase 6 — FastAPI Backend, Two-Laptop Evaluation & Technical Report:**
   - Export model and preprocessing artifacts for Laptop 1's FastAPI inference server.
   - Verify end-to-end latency and logs for all 10 trips.
   - Compile actual experimental tables, figures, and compulsory model answers into the final consolidated PDF report.

---

## 3. Current Blockers & Integration Dependencies

| Dependency / Blocker | Owner | Impact | Required Action |
| :--- | :--- | :--- | :--- |
| **1. Candidate Generation & `.route` Label Alignment Methodology** | Qamar Ahmed & Pranav S Atreya | **Hard Blocker for Model Training:** Because `.track` ($T$ rows) and `.route` ($R$ rows) are not 1-to-1 aligned and `arc_id` is trip-local, supervised training cannot start until the exact candidate table & reference label assignment method are agreed upon. | Qamar to review `CANDIDATE_DATA_SCHEMA.md` and confirm candidate radius $R$, max candidates $K$, feature list, and how `.route` is aligned to each GPS observation `obs_idx`. |
| **2. Shared Raw Loader & EDA Utilities** | L A Gokul & Qamar Ahmed | Prevents duplicate file parsers for `.track`, `.nodes`, `.arcs`, `.route` and ensures consistent coordinate projection (lon/lat to metric meters). | Agree on shared `src/data/` loader interface and coordinate projection method (e.g., local UTM / equirectangular metric projection vs. geodesic distance). |
| **3. HMM / Viterbi Module Ownership & API Contract** | Qamar Ahmed & Pranav S Atreya | Prevents duplicate HMM implementations and ensures classifier emission probabilities match the decoder's expected input format. | Confirm whether HMM/Viterbi lives in `src/sequence/` and agree on the function signature between the classifier output and Viterbi decoder. |
