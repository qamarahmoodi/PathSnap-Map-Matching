# PathSnap — Two-Laptop API Contract & Model Scorer Integration Specification

**Authors:** Pranav Shashikiran Atreya (ML Scorer & Laptop 2 Evaluator UI), Qamar Ahmed (Laptop 1 FastAPI Backend & HMM/Viterbi)  
**Status:** Scorer Interface Frozen & Ready (`src/models/inference.py`); FastAPI Response Schema Proposed (`PROPOSED — Awaiting Qamar's Live Backend Commit`).

---

## Part 1: Frozen Model-Loading & Candidate-Scoring Interface (For Laptop 1 Backend)

Qamar's FastAPI server and HMM/Viterbi pipeline on **Laptop 1** can load and score the final trained classifier in two lines of Python without importing any internal training code:

```python
from models.inference import load_model_bundle, score_candidates

# 1. Load once at FastAPI startup (loads artifacts/final_model.joblib + metadata)
model_bundle = load_model_bundle()

# 2. Score any candidate DataFrame returned by generate_trip_candidates(...)
scored_df = score_candidates(cand_result.candidates, model_bundle=model_bundle)
```

### Guaranteed Output Schema of `score_candidates(candidates_df, model_bundle)`
- **Row Count & Order:** Identical to input `candidates_df` (1-to-1 candidate-to-arc correspondence preserved).
- **Retained Identifier Columns:**
  - `trip_id` (`string`)
  - `observation_id` (`int32`)
  - `candidate_arc_id` (`int32`) — original 0-based row index in `<trip_id>.arcs`
  - `arc_id` (`int32`) — exact alias of `candidate_arc_id`
  - `candidate_rank` (`int16`)
- **Appended Prediction Columns:**
  - `predicted_probability` (`float64` $\in [10^{-6}, 1 - 10^{-6}]$): Calibrated positive-class probability $P(\text{label} = 1 \mid \mathbf{x}_{i,c})$.
  - `pred_prob` (`float64`): Exact alias of `predicted_probability` so `decode_trip_viterbi(scored_df, ...)` works out-of-the-box.
  - `predicted_log_probability` (`float64`): $\log P(\text{label} = 1 \mid \mathbf{x}_{i,c})$ for log-domain Viterbi emission scoring.
  - `predicted_label` (`int8`): Binary prediction (`1` if `predicted_probability >= 0.5` else `0`).
  - `is_obs_top1` (`bool`): `True` for the highest-scoring candidate at each `(trip_id, observation_id)` (Classifier-Only greedy baseline).

---

## Part 2: Two-Laptop Network & CORS Configuration

| Component | Machine | Default Port | Configurable Environment Variable |
| :--- | :--- | :---: | :--- |
| **FastAPI Backend + Model + HMM/Viterbi** | **Laptop 1 (Qamar)** | `8000` | Bind to `0.0.0.0:8000` (`uvicorn app:app --host 0.0.0.0 --port 8000`) |
| **Evaluator User Interface** | **Laptop 2 (Pranav)** | `8501` | `PATHSNAP_API_BASE_URL="http://<LAPTOP_1_LAN_IP>:8000"`, `PATHSNAP_UI_PORT="8501"` |

### Required CORS Configuration on Laptop 1 (`FastAPI`)
Because Laptop 2 runs on a separate physical machine on the local Wi-Fi/LAN, Qamar's FastAPI app must enable `CORSMiddleware` allowing:
- `allow_origins=["*"]` (or specifically `http://localhost:8501`, `http://127.0.0.1:8501`, and `http://<LAPTOP_2_LAN_IP>:8501`)
- `allow_methods=["GET", "POST", "OPTIONS"]`
- `allow_headers=["*"]`

---

## Part 3: Official HTTP Endpoints & JSON Schemas (`PROPOSED` for Laptop 1)

### 1. `GET /health`
Returns server status and confirms that the trained model artifact is loaded.
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "Bayesian-Tuned HistGradientBoostingClassifier",
  "decoder": "HMM + Viterbi"
}
```

### 2. `GET /trips`
Returns the list of 10 evaluation trip options available on Laptop 1. The UI supports either a list of trip objects or a list of trip ID strings:
```json
{
  "trips": [
    {
      "trip_id": "00000015",
      "partition": "evaluator_test",
      "num_observations": 1420,
      "label": "Trip 00000015 (1,420 GPS pts)"
    }
  ]
}
```

### 3. `POST /predict`
- **Request Body:**
```json
{
  "trip_id": "00000015"
}
```
- **Response Body (Strictly Server-Side Evaluation — Never Exposes `.route` or Ground-Truth Labels):**
```json
{
  "trip_id": "00000015",
  "model_name": "Bayesian-Tuned HistGradientBoostingClassifier",
  "decoder": "HMM + Viterbi",
  "num_observations": 1420,
  "predicted_arc_ids": [12, 12, 12, 48, 48, 103],
  "reconstructed_route_arcs": [12, 48, 103],
  "predicted_route_segments_latlon": [
    [[49.84405, 8.59520], [49.84444, 8.59529]],
    [[49.84444, 8.59529], [49.84488, 8.59535]]
  ],
  "gps_points_latlon": [
    [49.84405, 8.59520],
    [49.84444, 8.59529]
  ],
  "evaluation_metrics": {
    "observation_top1_accuracy": 0.9142,
    "macro_f1": 0.8580,
    "route_continuity": 0.9825,
    "candidate_miss_rate": 0.0018
  },
  "inference_time_ms": 145.2
}
```

> **Security & Academic Integrity Safeguard in `src/ui/api_client.py`:**  
> Even if a backend response accidentally includes keys named `reference_route`, `reference_arc_ids`, `ground_truth`, `labels`, or `route`, `PathSnapAPIClient.predict_trip(...)` automatically strips them before rendering so ground-truth labels can never be displayed on Laptop 2.
