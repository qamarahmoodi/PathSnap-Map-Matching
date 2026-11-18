# PathSnap — Authoritative Guideline Compliance Matrix (Phase 1 Checkpoint)

**Purpose:** Tracks exact compliance against all five official hackathon PDF documents, clearly separating **Completed & Verified in Phase 1** from **Scheduled for Phase 2 (Post-Lunch Final Submission)**.

---

## 1. Dataset Exploration & Preprocessing (`Dataset Exploration.pdf`)

| Guideline Requirement | Phase 1 Status | Code & Evidence Reference |
| :--- | :--- | :--- |
| Verify 80 `train` and 10 `student_test` trips (`evaluator_test` untouched) | **COMPLETED (90 Trips Audited)** | `src/pathsnap/data/loader.py`, `src/eda.py`, [`data/dataset_inventory.csv`](data/dataset_inventory.csv), [`outputs/partition_summary.txt`](outputs/partition_summary.txt) |
| Understand `.track`, `.nodes`, `.arcs`, `.route` & never align `.track` row $i$ with `.route` row $i$ | **COMPLETED** | [`PHASE1_VIVA_NOTES.md`](PHASE1_VIVA_NOTES.md) §1, `tests/test_data_loader.py` |
| Treat `arc_id` and `node_id` as trip-specific local indices (never global classes) | **COMPLETED** | `CANDIDATE_ID_COLUMNS` excluded from `CANDIDATE_FEATURE_COLUMNS` in [`src/pathsnap/schemas/candidate_schema.py`](src/pathsnap/schemas/candidate_schema.py) |
| Proper metric coordinate projection (do not use raw Euclidean degrees) | **COMPLETED** | `LocalTangentProjector` (WGS84 prime-vertical $N(\phi)$ & meridional $M(\phi)$ curvature) in [`src/pathsnap/spatial/projection.py`](src/pathsnap/spatial/projection.py) |
| Analyse missing values, sampling intervals, timestamp gaps, and `.route` limitations | **COMPLETED** | [`outputs/candidate_previews/dataset_audit_digest.json`](outputs/candidate_previews/dataset_audit_digest.json), [`outputs/max_timestamp_gap_per_trip.png`](outputs/max_timestamp_gap_per_trip.png), Trip `00000002` `868` unmatched tail points identified |
| Interactive Folium geographical visualisations of raw GPS, road network, and route | **COMPLETED (3 Interactive HTML Maps)** | [`outputs/trip_map.html`](outputs/trip_map.html), [`outputs/candidate_previews/trip_00000001_alignment_map.html`](outputs/candidate_previews/trip_00000001_alignment_map.html), [`results/phase1/trip_00000001_viterbi_vs_greedy_map.html`](results/phase1/trip_00000001_viterbi_vs_greedy_map.html) |

---

## 2. Candidate Generation, Feature Engineering & Label Alignment (`Models and Training Guidelines.pdf`)

| Guideline Requirement | Phase 1 Status | Code & Evidence Reference |
| :--- | :--- | :--- |
| Spatial candidate search (`STRtree`), radius $R$, top-$K$, and nearest-neighbour fallback | **COMPLETED & SENSITIVITY TESTED** | [`src/pathsnap/spatial/candidate_generation.py`](src/pathsnap/spatial/candidate_generation.py), [`data/sensitivity_analysis.csv`](data/sensitivity_analysis.csv) ($R=100\text{ m}, K=10 \implies 0.27\%$ miss rate) |
| Extract 11 causal geometric & kinematic candidate features (`CANDIDATE_FEATURE_COLUMNS`) | **COMPLETED & UNIT TESTED** | [`src/pathsnap/features/feature_engineering.py`](src/pathsnap/features/feature_engineering.py), `tests/test_candidate_generation.py` |
| Monotonic dynamic programming alignment of `.route` to `.track` | **COMPLETED & UNIT TESTED** | [`src/pathsnap/alignment/route_alignment.py`](src/pathsnap/alignment/route_alignment.py), `tests/test_route_alignment.py` |
| Candidate miss handling (`candidate_miss == True` receives `label = <NA>`, never false `0`) | **COMPLETED & UNIT TESTED** | [`src/pathsnap/alignment/label_generation.py`](src/pathsnap/alignment/label_generation.py), `tests/test_label_generation.py`, [`results/phase1/label_distribution.csv`](results/phase1/label_distribution.csv) |

---

## 3. Five-Model Experimentation, Bayesian Optimisation & HMM/Viterbi (`Models and Training Guidelines.pdf` & `Deliverables.pdf`)

| Guideline Requirement | Phase 1 Status | Phase 2 Remaining Scope | Evidence Reference |
| :--- | :--- | :--- | :--- |
| Shortlist & implement 5 distinct ML models from different algorithm families | **COMPLETED (5-Trip Pilot Executed)** | Run full **80-trip** `train` CV + **10-trip** `student_test` evaluation | [`docs/MODEL_SHORTLIST.md`](docs/MODEL_SHORTLIST.md), [`src/models/model_pipeline.py`](src/models/model_pipeline.py), [`results/phase1/model_comparison.csv`](results/phase1/model_comparison.csv) |
| Leak-free `GroupKFold` by `trip_id` + in-fold `SimpleImputer` and `StandardScaler` | **COMPLETED & VERIFIED** | Reuse exact pipeline on all 80 training trips | [`src/models/model_pipeline.py`](src/models/model_pipeline.py), [`results/phase1/cv_results.csv`](results/phase1/cv_results.csv) |
| Genuine Bayesian hyperparameter optimisation (`Optuna TPE`) | **COMPLETED (8-Trial 5-Trip Pilot; in-sample CV selection optimism noted)** | Run 25–40 trials on full 80 training trips & compare Untuned vs Tuned on held-out `student_test` | [`src/optimization/bayesian_tuning.py`](src/optimization/bayesian_tuning.py), [`results/phase1/bayesian_trials.csv`](results/phase1/bayesian_trials.csv) |
| HMM & Viterbi sequential route reconstruction (`TripRoadGraph` + log-Viterbi) | **COMPLETED (Synthetic + 5-Trip OOF Pilot; Trip `00000002` HGB vs LR calibration trade-off documented)** | Evaluate on 10 `student_test` trips with final 80-trip calibrated model | [`src/sequence/viterbi_decoder.py`](src/sequence/viterbi_decoder.py), `tests/test_viterbi_decoder.py`, [`results/phase1/hmm_test_results.csv`](results/phase1/hmm_test_results.csv) |
| FastAPI Server (Laptop 1), Evaluator UI (Laptop 2), and Final Consolidated PDF Report | **DEFERRED TO PHASE 2 (Per Scope Restriction)** | Implement FastAPI endpoint, UI, two-laptop test, and final PDF report | Scheduled for Phase 2 |
