# PathSnap — Distributed GPS Map-Matching & Route Reconstruction

**Course:** UE24CS352A Machine Learning Hackathon (PES University)  
**Team Members:**
- **Qamar Ahmed (Technical Lead):** Spatial projection (`LocalTangentProjector`), `Shapely 2.x STRtree` candidate generation, causal feature engineering, monotonic DP reference alignment, `TripRoadGraph`.
- **Pranav Shashikiran Atreya (ML Engineering):** Five-model `GroupKFold` training & comparison pipeline, Optuna TPE Bayesian hyperparameter optimisation, HMM & Viterbi route reconstruction.
- **L A Gokul (Data Engineering & Visualisation):** Raw dataset ingestion, 90-trip integrity auditing, exploratory data analysis, Folium interactive maps.

---

## 1. Repository Structure (Aligned with Official Deliverables)

```text
PathSnap-Map-Matching/
├── configs/
│   └── default.yaml                           # Candidate search, feature, and alignment configuration
├── data/
│   ├── README.md                              # External dataset location & partition guide
│   ├── dataset_inventory.csv                  # Full 90-trip audit (80 train + 10 student_test)
│   └── sensitivity_analysis.csv               # Candidate recall vs. radius R and top-K sensitivity
├── docs/
│   ├── CANDIDATE_DATA_CONTRACT.md             # Formal schema for candidates.parquet, observations.parquet & TripRoadGraph
│   ├── MODEL_SHORTLIST.md                     # 5 ML model families, hyperparameters & probability outputs
│   └── PREPROCESSING_PIPELINE.md              # Step-by-step preprocessing, projection & DP alignment guide
├── src/
│   ├── data_loader.py                         # Raw NumPy array loader (.track, .nodes, .arcs, .route)
│   ├── eda.py                                 # Dataset inventory & integrity audit script
│   ├── visualization.py                       # Static EDA histograms & Folium map generator
│   ├── pathsnap/
│   │   ├── data/loader.py                     # Typed loader & array validator (strict route separation)
│   │   ├── spatial/projection.py              # WGS84 ellipsoidal tangent-plane metric projection
│   │   ├── spatial/road_network.py            # Road segment construction & Shapely 2.x STRtree index
│   │   ├── spatial/candidate_generation.py    # Top-K candidate search & fallback nearest recovery
│   │   ├── features/feature_engineering.py    # 11 causal geometric & kinematic candidate features
│   │   ├── alignment/route_alignment.py       # Monotonic dynamic programming .route-to-.track aligner
│   │   ├── alignment/label_generation.py      # Candidate label attachment & miss-rate calculator
│   │   ├── graph/road_graph.py                # TripRoadGraph with O(1) outgoing successor lookup
│   │   └── schemas/candidate_schema.py        # Authoritative column dtypes & feature/ID separation
│   ├── models/model_pipeline.py               # 5 ML models + leak-free 5-fold GroupKFold evaluation
│   ├── optimization/bayesian_tuning.py        # Optuna TPE Bayesian hyperparameter optimisation
│   └── sequence/viterbi_decoder.py            # Log-domain HMM + Viterbi route reconstruction decoder
├── scripts/
│   ├── run_audit_and_previews.py              # Generates inventory, sensitivity CSV, previews & alignment map
│   └── run_phase1_ml_pipeline.py              # Runs 5-model GroupKFold CV, Optuna TPE & Viterbi decoding
├── outputs/
│   ├── gps_observations_per_trip.png          # EDA distribution of GPS trajectory lengths
│   ├── max_timestamp_gap_per_trip.png         # EDA distribution of maximum timestamp gaps
│   ├── road_nodes_per_trip.png                # EDA distribution of road network node counts
│   ├── road_arcs_per_trip.png                 # EDA distribution of road network arc counts
│   ├── partition_summary.txt                  # Partition-level summary statistics (train vs student_test)
│   ├── trip_map.html                          # Interactive Folium map (Raw GPS + Road Network + Reference Route)
│   └── candidate_previews/                    # Sample candidate/observation CSVs & alignment Folium map
├── results/phase1/
│   ├── model_comparison.csv                   # Consolidated 5-model comparison table
│   ├── cv_results.csv                         # Fold-by-fold GroupKFold metrics
│   ├── bayesian_trials.csv                    # Optuna TPE Bayesian optimisation trial log
│   ├── bayesian_best_summary.json             # Best hyperparameter configuration & timing
│   ├── hmm_test_results.csv                   # Classifier-only Greedy vs. Classifier + HMM/Viterbi benchmarks
│   ├── model_comparison_chart.png             # Visual comparison of Macro F1, Top-1 Accuracy & Log Loss
│   ├── bayesian_tuning_progression.png        # Optuna TPE optimisation progression plot
│   ├── phase1_summary.md                      # Consolidated experimental summary report
│   └── trip_00000001_viterbi_vs_greedy_map.html # Interactive Folium map: Greedy vs. HMM/Viterbi route
├── artifacts/
│   └── phase1_pilot_tuned_hgb.joblib          # Persisted tuned HistGradientBoosting pipeline artifact
└── tests/                                     # 11 automated unit & integration tests (pytest)
```

---

## 2. Installation & Setup

1. Install dependencies and register the `pathsnap` package:
   ```powershell
   python -m pip install -r requirements.txt
   python -m pip install -e . --no-deps
   ```
2. Configure the external dataset root (defaults to `../Student folder` containing `train/` and `student_test/`):
   ```powershell
   $env:PATHSNAP_DATA_ROOT="..\Student folder"
   ```

---

## 3. Reproducing All Outputs & Tests

```powershell
# 1. Run the automated test suite (11 tests covering loader, projection, candidates, DP alignment, labels, graph, and Viterbi)
python -m pytest -v

# 2. Run dataset audit (90 trips), candidate sensitivity analysis (10 trips), and candidate previews
python scripts/run_audit_and_previews.py

# 3. Run 5-model GroupKFold CV, Optuna TPE Bayesian tuning, and HMM/Viterbi route reconstruction
python scripts/run_phase1_ml_pipeline.py
```
