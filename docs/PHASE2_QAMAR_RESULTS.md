# PathSnap Phase 2 — Measured Inference, HMM/Viterbi & API Benchmark Results

**Owner:** Qamar Ahmed (Technical Lead)
**Evaluation Partition:** Held-out `student_test` partition (**10 trips**, **19,160 GPS observations**, **157,851 candidate rows**)
**Model Artifact Used:** `artifacts/final_tuned_model.joblib` (Full 80-Trip Optuna TPE Bayesian-Tuned `HistGradientBoostingClassifier`)
**Source Data:** [`results/phase2/student_test_trip_metrics.csv`](../results/phase2/student_test_trip_metrics.csv), [`results/phase2/student_test_aggregate_summary.json`](../results/phase2/student_test_aggregate_summary.json)

---

## 1. Executive Summary (`student_test` — All 10 Trips)

Across all **10 held-out `student_test` trips (`19,160` observations, `19,103` confidently aligned observations)**:
- **Candidate Miss Rate:** Only **`29 / 19,103` (`0.1518%`)**, achieving **`99.8482%` candidate recall** at $R=100\text{ m}, K=10$.
- **HMM/Viterbi Improves Top-1 Accuracy on 10 / 10 Trips (`100%` of trips):**
  - **Mean Trip Observation Top-1 Accuracy:** Jumps from **`80.82%` (Classifier-Only)** to **`89.63%` (Classifier + HMM/Viterbi)** (`+8.81%` mean trip gain; **`79.10%` $\to$ `88.72%`** observation-weighted across all `19,103` aligned points, and up to **`+19.06%`** on Trip `00000077`).
- **HMM/Viterbi Improves Route Continuity on 10 / 10 Trips (`100%` of trips):**
  - **Mean Trip Route Continuity:** Jumps from **`89.60%` (Classifier-Only)** to **`98.27%` (Classifier + HMM/Viterbi)** (`+8.67%`), with **`99.92%` 3-hop topological reachability**.
- **Sequence-Level Route Arc F1 Overlap:**
  - **Mean Route Arc F1:** Jumps from **`0.8060` (Classifier-Only)** to **`0.9388` (Classifier + HMM/Viterbi)** (`+13.29%`).

---

## 2. Trip-by-Trip `student_test` Comparison Table

| Trip ID | GPS Obs | Road Arcs | Candidates | Cand Miss Rate | Classifier-Only Top-1 Acc | **HMM/Viterbi Top-1 Acc** | **Top-1 Gain ($\Delta$)** | Classifier-Only Continuity | **HMM/Viterbi Continuity** | **Continuity Gain ($\Delta$)** | Route Arc F1 (Greedy $\to$ HMM) | Total Inference Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `00000000` | 2,503 | 18,196 | 13,602 | `0.16%` | `93.17%` | **`95.49%`** | **`+2.32%`** | `95.00%` | **`99.52%`** | **`+4.52%`** | `0.858` $\to$ **`0.955`** | `3,626.9 ms` |
| `00000007` | 1,354 | 4,369 | 7,792 | `0.00%` | `90.25%` | **`94.17%`** | **`+3.91%`** | `99.41%` | **`99.93%`** | **`+0.52%`** | `0.903` $\to$ **`0.958`** | `1,817.1 ms` |
| `00000009` | 1,049 | 13,106 | 10,374 | `0.78%` | `73.20%` | **`82.23%`** | **`+9.03%`** | `83.78%` | **`96.28%`** | **`+12.50%`** | `0.692` $\to$ **`0.892`** | `1,932.6 ms` |
| `00000040` | 2,995 | 27,026 | 18,485 | `0.00%` | `88.41%` | **`93.26%`** | **`+4.84%`** | `95.02%` | **`99.37%`** | **`+4.34%`** | `0.782` $\to$ **`0.952`** | `3,861.6 ms` |
| `00000052` | 262 | 3,577 | 2,487 | `0.00%` | `90.46%` | **`96.18%`** | **`+5.73%`** | `93.49%` | **`97.32%`** | **`+3.83%`** | `0.880` $\to$ **`0.946`** | `532.9 ms` |
| `00000056` | 3,958 | 27,631 | 36,612 | `0.08%` | `79.39%` | **`88.89%`** | **`+9.50%`** | `86.00%` | **`97.27%`** | **`+11.27%`** | `0.865` $\to$ **`0.958`** | `6,698.8 ms` |
| `00000065` | 768 | 13,184 | 7,611 | `0.00%` | `77.86%` | **`87.37%`** | **`+9.51%`** | `87.48%` | **`98.57%`** | **`+11.08%`** | `0.807` $\to$ **`0.958`** | `1,468.9 ms` |
| `00000076` | 1,486 | 56,027 | 14,776 | `0.61%` | `76.89%` | **`88.61%`** | **`+11.73%`** | `84.38%` | **`97.71%`** | **`+13.33%`** | `0.736` $\to$ **`0.922`** | `2,869.4 ms` |
| `00000077` | 3,911 | 9,622 | 37,658 | `0.13%` | `61.00%` | **`80.07%`** | **`+19.06%`** | `87.72%` | **`98.59%`** | **`+10.87%`** | `0.785` $\to$ **`0.916`** | `6,921.0 ms` |
| `00000091` | 874 | 12,303 | 8,454 | `0.00%` | `77.57%` | **`90.05%`** | **`+12.47%`** | `83.73%` | **`98.17%`** | **`+14.43%`** | `0.750` $\to$ **`0.931`** | `1,544.0 ms` |
| **Mean / Total** | **19,160** | **185,041** | **157,851** | **`0.15%`** | **`80.82%`** | **`89.63%`** | **`+8.81%`** | **`89.60%`** | **`98.27%`** | **`+8.67%`** | **`0.806` $\to$ `0.939`** | **`3,127.3 ms` / trip** |

---

## 3. Pipeline Latency Breakdown (`student_test` Mean per Trip)

- **Raw Data Loading & Validation (`.track`, `.nodes`, `.arcs`):** `86.7 ms`
- **Metric Projection, `STRtree` Construction & Candidate/Feature Generation:** `184.3 ms`
- **Trained Model Scoring (`HistGradientBoostingClassifier.predict_proba`):** `81.9 ms`
- **HMM + Viterbi Topological Decoding (`decode_trip_viterbi`):** `2,830.3 ms` (`~1.48 ms` per GPS observation)
- **Total Route-Free Inference Pipeline (`run_trip_inference`):** **`3,127.3 ms` (`3.13 s` per trip)**

---

## 4. Automated Test Suite Results (`pytest -v`)

Executed via `python -m pytest -v` across all 10 test modules (`34` total tests):
- `tests/test_api.py` (3 tests: `/health`, `/trips`, `/predict` with strict ground-truth isolation, 404/503 error handling) — **`3 passed`**
- `tests/test_inference_and_evaluator.py` (3 tests: `CandidateModelScorer` artifact loading, schema validation, spy-verified route-free inference + isolated server evaluation) — **`3 passed`**
- `tests/test_viterbi_decoder.py` (10 test functions covering all 16 required Viterbi scenarios) — **`10 passed`**
- `tests/test_common_evaluator.py` (5 tests: Gokul's common evaluation framework) — **`5 passed`**
- `tests/test_scorer.py` & `tests/test_ui_client.py` (3 tests: Pranav's model scorer & Laptop 2 UI client) — **`3 passed`**
- `tests/test_candidate_generation.py`, `tests/test_data_loader.py`, `tests/test_graph.py`, `tests/test_label_generation.py`, `tests/test_route_alignment.py` — **`10 passed`**
- **Overall Result:** **`34 passed in 30.03s`**
