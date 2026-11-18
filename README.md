# PathSnap — Distributed GPS Map-Matching & Route Reconstruction

**Course:** UE24CS352A Machine Learning Hackathon (PES University)
**Team Members:**
- **Qamar Ahmed (PES2UG24CS904) (Technical Lead):** Spatial projection (`LocalTangentProjector`), `Shapely 2.x STRtree` candidate generation, causal feature engineering, monotonic DP reference alignment, `TripRoadGraph`, HMM & topological Viterbi decoder, route-free inference engine, and FastAPI backend server.
- **Pranav Shashikiran Atreya (PES2UG24CS912) (ML Engineering):** Five-model `GroupKFold` training & comparison pipeline, Optuna TPE Bayesian hyperparameter optimization, model serialization (`artifacts/phase1_pilot_tuned_hgb.joblib`), and Evaluator UI.
- **L A Gokul (PES2UG24CS246) (Common Evaluation, Visualizations & Reporting):** Common evaluation & measurement framework (`common_evaluator.py`), 9 publication-grade comparison figures, reusable 5-layer Folium map generation pipeline (`map_generator.py`), 90-trip dataset integrity & EDA evidence, and consolidated technical report assembly (`PathSnap_Consolidated_Technical_Report.pdf`).

---

## 1. Repository Structure

```text
PathSnap-Map-Matching/
├── configs/
│   └── default.yaml                           # Candidate search, feature, alignment, and HMM configuration
├── data/
│   ├── README.md                              # External dataset location & partition guide
│   ├── dataset_inventory.csv                  # Full 90-trip audit (80 train + 10 student_test)
│   └── sensitivity_analysis.csv               # Candidate recall vs. radius R and top-K sensitivity
├── docs/
│   ├── CANDIDATE_DATA_CONTRACT.md             # Formal schema for candidates & observations
│   ├── MODEL_SHORTLIST.md                     # 5 ML model families, hyperparameters & probability outputs
│   ├── PREPROCESSING_PIPELINE.md              # Preprocessing, projection & DP alignment guide
│   ├── QAMAR_HMM_HANDOFF.md                   # HMM transition model, emission floor & gap handling
│   ├── QAMAR_API_HANDOFF.md                   # FastAPI endpoints, request/response schemas & LAN setup
│   └── PHASE2_QAMAR_RESULTS.md                # 10-trip student_test inference benchmarks
├── src/
│   ├── data_loader.py                         # Raw NumPy array loader (.track, .nodes, .arcs, .route)
│   ├── eda.py                                 # Dataset inventory & integrity audit script
│   ├── quality_analysis.py                    # 90-trip deep data-quality analysis script
│   ├── visualization.py                       # Static EDA histograms & Folium map generator
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── common_evaluator.py                # Gokul: Common Evaluation & Measurement Framework
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── map_generator.py                   # Gokul: Reusable multi-layer Folium map pipeline
│   ├── pathsnap/
│   │   ├── data/loader.py                     # Typed loader & array validator (strict route separation)
│   │   ├── spatial/projection.py              # WGS84 ellipsoidal tangent-plane metric projection
│   │   ├── spatial/road_network.py            # Road segment construction & Shapely 2.x STRtree index
│   │   ├── spatial/candidate_generation.py    # Top-K candidate search & fallback nearest recovery
│   │   ├── features/feature_engineering.py    # 11 causal geometric & kinematic candidate features
│   │   ├── alignment/route_alignment.py       # Monotonic dynamic programming .route-to-.track aligner
│   │   ├── alignment/label_generation.py      # Candidate label attachment & miss-rate calculator
│   │   ├── graph/road_graph.py                # TripRoadGraph with O(1) outgoing successor lookup
│   │   ├── inference/
│   │   │   ├── inference_pipeline.py          # Route-free end-to-end inference engine
│   │   │   ├── model_scorer.py                # Serialized model loader & candidate feature scorer
│   │   │   └── server_evaluator.py            # Ground-truth isolated server-side evaluator
│   │   └── api/
│   │       ├── app.py                         # FastAPI backend service (endpoints: /health, /trips, /predict)
│   │       └── schemas.py                     # Strict Pydantic v2 schemas
│   ├── models/model_pipeline.py               # 5 ML models + leak-free 5-fold GroupKFold evaluation
│   ├── optimization/bayesian_tuning.py        # Optuna TPE Bayesian hyperparameter optimization
│   └── sequence/viterbi_decoder.py            # Log-domain HMM + Viterbi route reconstruction decoder
├── scripts/
│   ├── run_audit_and_previews.py              # Generates inventory, sensitivity CSV, previews & alignment map
│   ├── run_phase1_ml_pipeline.py              # Runs 5-model GroupKFold CV, Optuna TPE & Viterbi decoding
│   ├── run_phase2_inference_and_handoff.py    # Runs route-free inference & benchmarks across student_test
│   ├── generate_phase2_visualizations.py      # Gokul: Generates all 9 publication-grade model result figures
│   └── build_consolidated_report.py           # Gokul: Assembles final consolidated technical report PDF
├── outputs/
│   ├── visualizations/                        # 9 high-resolution comparison charts (Figures 1 to 9)
│   ├── maps/                                  # Interactive Folium HTML maps (Trip 00000000, 00000009)
│   ├── evaluation/                            # Machine-readable evaluation summary JSON & Markdown report
│   ├── data_quality_report.csv                # 90-trip missing value and anomaly analysis
│   └── dataset_inventory.csv                  # Trip-by-trip counts for GPS, nodes, arcs, and bounds
├── results/
│   ├── phase1/                                # Phase 1 model comparison, CV results & Optuna trials
│   └── phase2/                                # Phase 2 student_test trip metrics & sample prediction CSVs
├── artifacts/
│   └── phase1_pilot_tuned_hgb.joblib          # Exported tuned HistGradientBoosting model artifact
├── reports/
│   └── PathSnap_Consolidated_Technical_Report.pdf # Final publication-grade consolidated technical report
└── tests/                                     # 31 automated unit & integration tests (all passing)
```

---

## 2. Installation & Environment Setup

```powershell
# 1. Install all required dependencies
pip install -r requirements.txt reportlab

# 2. (Optional) Configure dataset root if located outside default paths
$env:PATHSNAP_DATA_ROOT="C:\Users\lgoku\Downloads\PathSnap_Dataset\Student folder"
```

---

## 3. How to Run & Reproduce All Results

### A. Run Automated Unit & Integration Tests (31 Tests)
```powershell
python -m pytest -v
```
*Result: 31 passed in ~19 seconds.*

### B. Run the Common Evaluation Framework
```powershell
python src/evaluation/common_evaluator.py
```
*Outputs:*
- `outputs/evaluation/common_evaluation_summary.json` (Machine-readable metrics)
- `outputs/evaluation/common_evaluation_report.md` (Human-readable markdown summary)

### C. Generate All 9 Model Result Visualizations
```powershell
python scripts/generate_phase2_visualizations.py
```
*Outputs in `outputs/visualizations/`:*
1. `fig1_cv_macro_f1_comparison.png` — 5-Fold Cross-Validation Macro F1 comparison with error bars.
2. `fig2_student_test_top1_comparison.png` — Observation Top-1 Accuracy across models & HMM.
3. `fig3_student_test_macro_f1_comparison.png` — Out-of-fold candidate Macro F1 comparison.
4. `fig4_log_loss_comparison.png` — Candidate probability Log Loss comparison.
5. `fig5_final_model_comparison.png` — Consolidated multi-metric performance comparison.
6. `fig6_confusion_matrix_best_model.png` — Candidate classification confusion matrix (92,300 rows).
7. `fig7_tuned_vs_untuned_comparison.png` — Untuned vs Optuna-tuned HistGradientBoosting impact.
8. `fig8_classifier_only_vs_hmm_comparison.png` — Trip-by-trip Top-1 accuracy and route continuity gains.
9. `fig9_bayesian_tuning_progression.png` — Optuna TPE optimization objective progression over 8 trials.

### D. Generate Interactive Folium Maps
```powershell
# Generate interactive map for student_test trip 00000000
python src/visualization/map_generator.py --trip-id 00000000

# Generate interactive map for urban clutter trip 00000009
python src/visualization/map_generator.py --trip-id 00000009
```
*Outputs in `outputs/maps/`:*
- `outputs/maps/trip_00000000_map.html`
- `outputs/maps/trip_00000009_map.html`

### E. Compile the Consolidated Technical Report PDF
```powershell
python scripts/build_consolidated_report.py
```
*Output:*
- `reports/PathSnap_Consolidated_Technical_Report.pdf` (710 KB publication-grade document covering Sections A to F).

---

## 4. Distributed Multi-Laptop Deployment (Laptop 1 & Laptop 2)

PathSnap supports distributed evaluation across two machines on the same local network:

### Laptop 1 (Backend Server)
Runs the FastAPI route-free map-matching engine:
```powershell
python -m uvicorn pathsnap.api.app:app --app-dir src --host 0.0.0.0 --port 8000
```
- Health check: `curl http://localhost:8000/health`
- List permitted trips: `curl http://localhost:8000/trips?partition=student_test`
- Predict trip route: `POST http://localhost:8000/predict` with JSON body `{"trip_id": "00000000"}`

### Laptop 2 (Client / Evaluator UI)
Connects to Laptop 1 across LAN Wi-Fi:
- Set Laptop 1 IP address: e.g. `http://192.168.1.50:8000`
- Sends prediction requests to `http://192.168.1.50:8000/predict`
- Displays interactive Leaflet maps and evaluates route accuracy without exposing reference ground truth.

---

## 5. Summary of Key Experimental Results

| Metric | Classifier-Only (Greedy) | Classifier + HMM/Viterbi | Empirical Gain (&Delta;) |
|---|---:|---:|---:|
| **Observation Top-1 Accuracy** | 63.57% | **73.62%** | **+10.05%** |
| **Route Continuity (1-hop Transitions)** | 83.82% | **95.11%** | **+11.29%** |
| **Multi-Hop Reachability (3-hop)** | N/A | **99.82%** | **Consistent Topology** |
| **Route Arc F1 Overlap** | 0.6395 | **0.8207** | **+18.12%** |
| **Trips Improved by HMM** | Baseline | **10 / 10 Trips (100%)** | Universal Gain |
| **Candidate Miss Rate** | 0.15% | 0.15% | **99.85% Recall** (R=100m, K=10) |
| **Total Inference Latency** | 345 ms / trip | **3,221 ms / trip** | **~1.5 ms / GPS point** |
