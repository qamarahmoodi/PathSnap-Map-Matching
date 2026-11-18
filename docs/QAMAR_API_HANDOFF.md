# PathSnap Phase 2 — FastAPI Backend & Two-Laptop Deployment Guide

**Owner:** Qamar Ahmed (Technical Lead — Laptop 1 Backend)
**Target Consumer:** Pranav Shashikiran Atreya (Laptop 2 Evaluator UI)
**Implementation:** [`src/pathsnap/api/app.py`](../src/pathsnap/api/app.py), [`src/pathsnap/api/schemas.py`](../src/pathsnap/api/schemas.py)

---

## 1. Laptop 1 Startup & LAN Configuration

### Step 1: Find Laptop 1's Local Network IP Address (Windows PowerShell)
On **Laptop 1** (connected to the same Wi-Fi / hotspot as Laptop 2), run:

```powershell
ipconfig | Select-String "IPv4"
```
Note the LAN IPv4 address (e.g., `192.168.1.42` or `10.x.x.x`).

### Step 2: Configure Environment Variables & Start the FastAPI Server on Laptop 1
From the repository root on **Laptop 1**:

```powershell
# Optional: explicitly set dataset root, evaluation partition, and model artifact path
$env:PATHSNAP_DATA_ROOT="..\Student folder"
$env:PATHSNAP_EVAL_PARTITION="student_test"   # Use "evaluator_test" when the 10 reserved evaluator trips are mounted
$env:PATHSNAP_MODEL_PATH="artifacts/phase1_pilot_tuned_hgb.joblib"
$env:PATHSNAP_CORS_ORIGINS="*"

# Start FastAPI with Uvicorn bound to 0.0.0.0:8000 for LAN access from Laptop 2
python -m uvicorn pathsnap.api.app:app --app-dir src --host 0.0.0.0 --port 8000
```

### Step 3: Verify Connectivity from Laptop 2
On **Laptop 2**, replace `<LAPTOP1_IP>` with Laptop 1's IPv4 address:

```powershell
curl http://<LAPTOP1_IP>:8000/health
curl http://<LAPTOP1_IP>:8000/trips
```

Set Laptop 2's UI API base URL to:
```text
http://<LAPTOP1_IP>:8000
```

---

## 2. Endpoint Reference & Exact JSON Schemas

### A. `GET /health`
Verifies that the FastAPI server is online, the trained `.joblib` model is loaded in memory, and the permitted trip directory is accessible.

**Example Response (`200 OK`):**
```json
{
  "status": "ok",
  "ready": true,
  "model_loaded": true,
  "model_name": "HistGradientBoostingClassifier",
  "model_path": "C:\\Users\\Qamar\\Downloads\\PathSnap-Map-Matching\\artifacts\\phase1_pilot_tuned_hgb.joblib",
  "data_root_available": true,
  "active_partition": "student_test",
  "permitted_trip_count": 10,
  "error": null
}
```

---

### B. `GET /trips`
Returns the 10 permitted evaluation trip options for Pranav's dropdown selector. **Never exposes reference routes or ground-truth data.**

**Example Response (`200 OK`):**
```json
{
  "status": "ok",
  "partition": "student_test",
  "count": 10,
  "trips": [
    {
      "trip_id": "00000000",
      "display_label": "Evaluation Trip 1 (00000000)",
      "partition": "student_test"
    },
    {
      "trip_id": "00000007",
      "display_label": "Evaluation Trip 2 (00000007)",
      "partition": "student_test"
    }
  ]
}
```

---

### C. `POST /predict`
Runs the complete route-free prediction pipeline (`run_trip_inference`) followed by isolated server-side evaluation (`evaluate_trip_predictions_server_side`).

**Request Body (`application/json`):**
```json
{
  "trip_id": "00000007",
  "include_observation_records": true
}
```

**Response Body (`200 OK`):**
```json
{
  "status": "ok",
  "trip_id": "00000007",
  "partition": "student_test",
  "model_name": "HistGradientBoostingClassifier",
  "predicted_route_arc_ids": [412, 413, 819, 820],
  "classifier_only_route_arc_ids": [412, 991, 413, 819, 820],
  "predicted_route_segments_latlon": [
    [[49.851234, 8.591234], [49.851890, 8.591950]]
  ],
  "classifier_only_route_segments_latlon": [
    [[49.851234, 8.591234], [49.851890, 8.591950]]
  ],
  "gps_trajectory_latlon": [
    [49.851210, 8.591200],
    [49.851500, 8.591600]
  ],
  "candidate_summary": {
    "num_observations": 1354,
    "num_road_nodes": 2061,
    "num_road_arcs": 3899,
    "total_candidates": 7792,
    "avg_candidates_per_observation": 5.755,
    "fallback_candidates_count": 0,
    "zero_candidate_observations": 0,
    "search_radius_m": 100.0,
    "max_candidates_k": 10
  },
  "decoder_summary": {
    "num_observations": 1354.0,
    "normal_decoded_observations": 1337.0,
    "fallback_decoded_observations": 0.0,
    "gap_reset_observations": 0.0,
    "carried_forward_observations": 0.0,
    "disconnected_bridge_observations": 17.0,
    "greedy_route_continuity": 0.875831,
    "viterbi_route_continuity": 0.971175,
    "viterbi_multihop_reachability": 0.987435,
    "changed_observations_count": 236.0
  },
  "evaluation_metrics": {
    "classifier_only_route_continuity": 0.875831,
    "hmm_viterbi_route_continuity": 0.971175,
    "hmm_viterbi_multihop_reachability": 0.987435,
    "changed_observations_by_hmm": 236,
    "aligned_observations": 1354,
    "label_eligible_observations": 1348,
    "candidate_miss_count": 6,
    "candidate_miss_rate": 0.004431,
    "classifier_only_top1_accuracy": 0.720089,
    "hmm_viterbi_top1_accuracy": 0.858936,
    "top1_accuracy_improvement_delta": 0.138848,
    "route_continuity_improvement_delta": 0.095344,
    "candidate_macro_f1": 0.7812,
    "candidate_log_loss": 0.2195,
    "classifier_only_route_jaccard": 0.5412,
    "hmm_viterbi_route_jaccard": 0.7681,
    "classifier_only_route_arc_f1": 0.7023,
    "hmm_viterbi_route_arc_f1": 0.8688
  },
  "reference_evaluated_on_server": true,
  "processing_time_ms": {
    "data_loading_ms": 18.4,
    "candidate_generation_ms": 85.2,
    "model_scoring_ms": 95.5,
    "viterbi_decoding_ms": 1612.9,
    "geometry_serialization_ms": 3.1,
    "total_pipeline_ms": 1815.1,
    "server_evaluation_ms": 310.4,
    "total_request_ms": 2126.0
  },
  "observation_predictions": [
    {
      "observation_id": 0,
      "timestamp_s": 1.0,
      "predicted_arc_id": 412,
      "classifier_only_arc_id": 412,
      "decoded_score": 0.912345,
      "classifier_only_score": 0.912345,
      "decoder_status": "normal",
      "prediction_status": "normal",
      "gap_status": "initial",
      "transition_hops": 0,
      "is_connected_step": true
    }
  ]
}
```

---

## 3. Visible Console Logging on Laptop 1

During evaluation, faculty evaluators require seeing live API logs on Laptop 1's terminal when Laptop 2 triggers a prediction. Every `POST /predict` call prints structured timestamped logs to `stdout`:

```text
[14:41:36] [LAPTOP-1 API] [INFO] ============================================================
[14:41:36] [LAPTOP-1 API] [INFO] POST /predict received | requested trip_id='00000007'
[14:41:36] [LAPTOP-1 API] [INFO] Starting route-free inference pipeline for trip_id='00000007' (partition='student_test')
[14:41:38] [LAPTOP-1 API] [INFO] Candidate generation & scoring complete | obs=1354 | candidates=7792 | model=HistGradientBoostingClassifier | score_ms=95.47
[14:41:38] [LAPTOP-1 API] [INFO] HMM/Viterbi decoding complete | route_arcs=124 | viterbi_continuity=0.9712 | decode_ms=1612.96
[14:41:39] [LAPTOP-1 API] [INFO] Server-side evaluation complete | ref_available=True | top1_acc=0.858936 | eval_ms=310.42
[14:41:39] [LAPTOP-1 API] [INFO] POST /predict succeeded for trip_id='00000007' in 2126.00 ms
[14:41:39] [LAPTOP-1 API] [INFO] ============================================================
```

---

## 4. Error Handling Contract

- **`404 Not Found`:** Returned if `trip_id` is not in the configured permitted evaluation allowlist or if trip files are missing.
- **`422 Unprocessable Entity`:** Returned automatically by FastAPI/Pydantic if the request JSON body is malformed.
- **`503 Service Unavailable`:** Returned if the model artifact is missing/unloaded or if `PATHSNAP_DATA_ROOT` is unavailable.
- **`500 Internal Server Error`:** Returned with a descriptive message if an unexpected runtime error occurs during inference.
