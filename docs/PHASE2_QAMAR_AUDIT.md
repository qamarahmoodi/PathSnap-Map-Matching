# PathSnap Phase 2 — Qamar Ahmed Technical Audit & Implementation Plan

**Owner:** Qamar Ahmed (Technical Lead)
**Date:** 2026-11-18
**Base Commit on `main`:** `6610904` (`12 passed` in existing test suite)

---

## 1. Current Implementation Status & Usable Phase 1 Modules

All Phase 1 spatial, feature, alignment, graph, and initial ML/HMM modules are intact, clean, and verified on `main`:

| Module / File | Owner | Current Status | Key Verified Capabilities |
|---|---|---|---|
| `src/pathsnap/schemas/candidate_schema.py` | Qamar | **Production-Ready** | Defines `CANDIDATE_SCHEMA_DTYPES` (20 cols), `OBSERVATION_SCHEMA_DTYPES` (13 cols), `CANDIDATE_FEATURE_COLUMNS` (11 cols), `CANDIDATE_ID_COLUMNS` (8 cols), `CandidateConfig`, `CandidateGenerationResult`, `ReferenceAlignment`, and schema validators. |
| `src/pathsnap/data/loader.py` | Qamar | **Production-Ready** | Resolves `PATHSNAP_DATA_ROOT` or `../Student folder`, preserves 8-digit zero-padded trip IDs (`"00000000"`–`"00000099"`), supports `include_route=False` for strict route-free inference, and validates arrays (`validate_trip_arrays`). |
| `src/data_loader.py`, `src/eda.py`, `src/visualization.py` | Gokul | **Production-Ready (Preserved)** | Gokul's raw NumPy array loader (`load_trip`), 90-trip EDA inventory, and Folium/matplotlib visualisations. Fully compatible with `generate_trip_candidates`. |
| `src/pathsnap/spatial/projection.py` | Qamar | **Production-Ready** | `LocalTangentProjector` using WGS84 ellipsoidal radii of curvature $N(\phi_0)$ and $M(\phi_0)$ around trip centroid (`< 1.7 cm` median error vs geodesic). |
| `src/pathsnap/spatial/road_network.py` | Qamar | **Production-Ready** | Builds `TripRoadNetwork`, metric segment lengths/bearings, `Shapely 2.x STRtree`, and vectorised Euclidean orthogonal projections (`compute_point_to_segments_geometry`). |
| `src/pathsnap/features/feature_engineering.py` | Qamar | **Production-Ready** | Causal kinematics (`compute_gps_kinematics`) with stationary jitter (`< 2.5 m`), long gap (`> 30 s` retains `delta_time_s` while masking speed/bearing), and non-positive `dt <= 0` safeguards. |
| `src/pathsnap/spatial/candidate_generation.py` | Qamar | **Production-Ready** | `generate_trip_candidates(...)` operating strictly without `.route`, performing `STRtree` radius search (`R=100 m`), fallback nearest search (`<= 500 m`, `is_fallback_candidate=True`), deduplication, and top-$K$ (`K=10`) ranking. |
| `src/pathsnap/alignment/route_alignment.py` & `label_generation.py` | Qamar | **Production-Ready** | Monotonic dynamic programming aligner (`align_reference_route`) and label generator (`attach_alignment_labels`) keeping candidate misses (`candidate_miss=True`) as `label=<NA>, label_eligible=False`. |
| `src/pathsnap/graph/road_graph.py` | Qamar | **Production-Ready** | `TripRoadGraph` with `O(1)` `outgoing_arcs_by_node` lookup, `successor_arcs`, `are_connected`, and `shortest_hop_transition_type`. |
| `src/models/model_pipeline.py` & `src/optimization/bayesian_tuning.py` | Pranav | **Preserved** | 5-model `GroupKFold` training pipeline, Optuna TPE Bayesian optimisation, and exported model artifact `artifacts/phase1_pilot_tuned_hgb.joblib`. |
| `src/sequence/viterbi_decoder.py` | Qamar & Pranav | **Needs Phase 2 Enhancements** | Implements log-domain Viterbi decoding (`decode_trip_viterbi`) over `TripRoadGraph`, but needs richer physical movement/length transition consistency, explicit per-observation `decoder_status` / `gap_status` / `prediction_status` flags, and configurable normalization. |

---

## 2. Existing Tests & Baseline Execution Results

Running `python -m pytest -q` on `main` (`6610904`):
- `tests/test_data_loader.py` (3 tests: partition discovery, `include_route=False`, and Gokul loader $\leftrightarrow$ Qamar candidate pipeline bit-for-bit integration) — **PASSED**
- `tests/test_candidate_generation.py` (2 tests: candidate schema invariants + GPS kinematics gap/stationary handling) — **PASSED**
- `tests/test_route_alignment.py` (2 tests: monotonic DP residuals + inference without route) — **PASSED**
- `tests/test_label_generation.py` (2 tests: real trip label attachment + candidate miss never creating false negatives) — **PASSED**
- `tests/test_graph.py` (1 test: `TripRoadGraph` successors & route connectivity) — **PASSED**
- `tests/test_viterbi_decoder.py` (2 tests: parallel distractor spike override + empty candidate & 117s gap handling) — **PASSED**
- **Total:** **`12 passed in 28.05s`**

---

## 3. Identified Phase 2 Gaps (Missing Functionality to Implement)

1. **Candidate-Scoring Model Loader & Adapter (`src/pathsnap/inference/model_scorer.py`):**
   - Need a dedicated, singleton-cached `CandidateModelScorer` that loads Pranav's serialized `.joblib` model from a configurable path (`PATHSNAP_MODEL_PATH`, defaulting to `artifacts/best_model.joblib` with fallback to `artifacts/phase1_pilot_tuned_hgb.joblib`), validates `CANDIDATE_FEATURE_COLUMNS` and dtypes, handles both `predict_proba` and `decision_function` (or mock scoring for synthetic tests), validates output probability bounds $[0, 1]$, and returns scored candidates preserving `(trip_id, observation_id, candidate_arc_id, candidate_rank, predicted_probability)`.
2. **Enhanced HMM & Viterbi Decoder (`src/sequence/viterbi_decoder.py` and `src/pathsnap/inference/`):**
   - Must explicitly output per-observation statuses required for Gokul's evaluation and Pranav's UI:
     - `prediction_status` (for Classifier-Only): `"normal"`, `"fallback_candidate"`, `"unmatched"`
     - `decoder_status` (for HMM/Viterbi): `"normal"`, `"fallback_decoded"`, `"gap_reset"`, `"carried_forward"`, `"disconnected_bridge"`, `"unmatched"`
     - `gap_status`: `"normal"`, `"moderate_gap"`, `"reset_gap"`, `"initial"`
   - Incorporate optional within-observation emission normalization and physical road-length vs. GPS-displacement compatibility alongside `TripRoadGraph` hop topology.
   - Must pass all 16 comprehensive Viterbi test cases (single observation, single candidate, connected chain, disconnected transition, multi-hop transition, large time gap, empty candidate set, fallback candidate, near-zero emission probability, repeated arcs, stationary observations, no valid path, tied candidates, permuted candidate row order, observation order preservation, determinism).
3. **Production Route-Free Inference Pipeline (`src/pathsnap/inference/inference_pipeline.py`):**
   - End-to-end function `run_trip_inference(...)` that takes `trip_id`, loads `.track`, `.nodes`, `.arcs` with `include_route=False` (never touching `.route`), generates candidates, scores them with `CandidateModelScorer`, runs `decode_trip_viterbi`, converts decoded arc sequences into GeoJSON/coordinate polyline geometries (`[[lat, lon], ...]`), and produces structured Classifier-Only and HMM/Viterbi prediction outputs plus timing breakdowns.
4. **Isolated Server-Side Evaluator (`src/pathsnap/inference/server_evaluator.py`):**
   - Strictly separated module `evaluate_trip_predictions_server_side(...)` that runs **only after** `run_trip_inference(...)` completes.
   - Reads `<trip_id>.route` internally on the server, computes permitted aggregate metrics (Observation-level Top-1 Accuracy, Route Continuity, Multi-hop Reachability, Sequence Jaccard/F1 overlap, and Classifier-Only vs. HMM comparison deltas), and returns **only aggregate metrics**—never returning reference arc IDs, reference route geometry, or per-observation ground-truth labels.
5. **FastAPI Backend for Laptop 1 (`src/pathsnap/api/app.py` & `src/pathsnap/api/schemas.py`):**
   - `GET /health`: Reports API status, whether the model artifact is loaded, data root availability, and configuration readiness.
   - `GET /trips`: Returns the 10 configured evaluation trip IDs (discovered safely from `<data_root>/<eval_partition>` or configurable manifest, defaulting cleanly to the 10 `evaluator_test` IDs when present or the 10 `student_test` IDs `00000000`..`00000091` in development mode) with safe display labels and zero ground-truth exposure.
   - `POST /predict`: Validates `trip_id` against the permitted allowlist, logs every pipeline stage visibly to the Laptop 1 console, runs `run_trip_inference(...)` (without `.route`), runs `evaluate_trip_predictions_server_side(...)` if reference data exists on Laptop 1, and returns a compact, UI-ready JSON response containing predicted arc IDs, coordinate geometry for mapping, observation-level prediction records, candidate/decoder status summaries, timing breakdown, and aggregate evaluation metrics.
   - Configurable CORS, host (`0.0.0.0`), and port (`8000`) for Laptop 1 $\leftrightarrow$ Laptop 2 LAN communication.
6. **Gokul & Pranav Handoff Exports + Full 10-Trip `student_test` Benchmark:**
   - Script `scripts/run_phase2_inference_and_handoff.py` to generate Classifier-Only vs. HMM/Viterbi prediction tables and runtime benchmarks across all **10 `student_test` trips** (`19,160` observations) for Gokul's evaluation framework and Pranav's UI integration.
   - Complete Phase 2 documentation: `docs/PHASE2_QAMAR_AUDIT.md`, `docs/PHASE2_QAMAR_ARCHITECTURE.md`, `docs/QAMAR_HMM_HANDOFF.md`, `docs/QAMAR_API_HANDOFF.md`, and `docs/PHASE2_QAMAR_RESULTS.md`.

---

## 4. Planned File Modifications & Additions

- **Created:**
  - `docs/PHASE2_QAMAR_AUDIT.md`
  - `src/pathsnap/inference/__init__.py`
  - `src/pathsnap/inference/model_scorer.py`
  - `src/pathsnap/inference/inference_pipeline.py`
  - `src/pathsnap/inference/server_evaluator.py`
  - `src/pathsnap/api/__init__.py`
  - `src/pathsnap/api/schemas.py`
  - `src/pathsnap/api/app.py`
  - `scripts/run_phase2_inference_and_handoff.py`
  - `tests/test_api.py`
  - `tests/test_inference_and_evaluator.py`
  - `docs/PHASE2_QAMAR_ARCHITECTURE.md`
  - `docs/QAMAR_HMM_HANDOFF.md`
  - `docs/QAMAR_API_HANDOFF.md`
  - `docs/PHASE2_QAMAR_RESULTS.md`
- **Extended (Backward-Compatible):**
  - `src/sequence/viterbi_decoder.py` (add `decoder_status`, `gap_status`, `prediction_status`, physical distance transition compatibility, and emission normalization options while keeping existing return fields 100% intact)
  - `tests/test_viterbi_decoder.py` (expand from 2 to 16 comprehensive Viterbi unit tests)
  - `configs/default.yaml`, `requirements.txt`, `pyproject.toml` (add `viterbi`, `inference`, and `api` configuration sections plus `fastapi`, `uvicorn`, `httpx`)
- **Untouched Teammate Files:**
  - `src/data_loader.py`, `src/eda.py`, `src/visualization.py` (Gokul)
  - `src/models/model_pipeline.py`, `src/optimization/bayesian_tuning.py` (Pranav)
