# PathSnap Phase 2 — Measured Inference, HMM/Viterbi & API Benchmark Results

**Owner:** Qamar Ahmed (Technical Lead)
**Evaluation Partition:** Held-out `student_test` partition (**10 trips**, **19,160 GPS observations**, **157,851 candidate rows**)
**Model Artifact Used:** `artifacts/phase1_pilot_tuned_hgb.joblib` (`HistGradientBoostingClassifier`)
**Source Data:** [`results/phase2/student_test_trip_metrics.csv`](../results/phase2/student_test_trip_metrics.csv), [`results/phase2/student_test_aggregate_summary.json`](../results/phase2/student_test_aggregate_summary.json)

---

## 1. Executive Summary (`student_test` — All 10 Trips)

Across all **10 held-out `student_test` trips (`19,160` observations, `19,103` confidently aligned observations)**:
- **Candidate Miss Rate:** Only **`29 / 19,103` (`0.1518%`)**, achieving **`99.8482%` candidate recall** at $R=100\text{ m}, K=10$.
- **HMM/Viterbi Improves Top-1 Accuracy on 10 / 10 Trips (`100%` of trips):**
  - **Mean Trip Observation Top-1 Accuracy:** Jumps from **`63.57%` (Classifier-Only)** to **`73.62%` (Classifier + HMM/Viterbi)** — an average improvement of **`+10.05%`** (and up to **`+21.65%`** on urban Trip `00000009`).
- **HMM/Viterbi Improves Route Continuity on 10 / 10 Trips (`100%` of trips):**
  - **Mean Trip Route Continuity:** Jumps from **`83.82%` (Classifier-Only)** to **`95.11%` (Classifier + HMM/Viterbi)** (`+11.29%`), with **`99.82%` 3-hop topological reachability**.
- **Sequence-Level Route Arc F1 Overlap:**
  - **Mean Route Arc F1:** Jumps from **`0.6395` (Classifier-Only)** to **`0.8207` (Classifier + HMM/Viterbi)** (`+18.12%`).

---

## 2. Trip-by-Trip `student_test` Comparison Table

| Trip ID | GPS Obs | Road Arcs | Candidates | Cand Miss Rate | Classifier-Only Top-1 Acc | **HMM/Viterbi Top-1 Acc** | **Top-1 Gain ($\Delta$)** | Classifier-Only Continuity | **HMM/Viterbi Continuity** | **Continuity Gain ($\Delta$)** | Route Arc F1 (Greedy $\to$ HMM) | Total Inference Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `00000000` | 2,503 | 18,196 | 13,602 | `0.16%` | `92.33%` | **`94.21%`** | **`+1.88%`** | `95.32%` | **`99.32%`** | **`+4.00%`** | `0.846` $\to$ **`0.927`** | `3,656.4 ms` |
| `00000007` | 1,354 | 4,369 | 7,792 | `0.00%` | `66.77%` | **`71.27%`** | **`+4.51%`** | `95.79%` | **`97.27%`** | **`+1.48%`** | `0.658` $\to$ **`0.794`** | `1,973.2 ms` |
| `00000009` | 1,049 | 13,106 | 10,374 | `0.78%` | `34.27%` | **`55.92%`** | **`+21.65%`** | `67.65%` | **`86.36%`** | **`+18.70%`** | `0.454` $\to$ **`0.762`** | `2,196.1 ms` |
| `00000040` | 2,995 | 27,026 | 18,485 | `0.00%` | `77.40%` | **`82.70%`** | **`+5.31%`** | `90.85%` | **`98.10%`** | **`+7.25%`** | `0.673` $\to$ **`0.882`** | `4,479.1 ms` |
| `00000052` | 262 | 3,577 | 2,487 | `0.00%` | `80.15%` | **`92.75%`** | **`+12.60%`** | `81.61%` | **`96.17%`** | **`+14.56%`** | `0.735` $\to$ **`0.888`** | `509.3 ms` |
| `00000056` | 3,958 | 27,631 | 36,612 | `0.08%` | `71.64%` | **`83.92%`** | **`+12.28%`** | `82.34%` | **`96.18%`** | **`+13.85%`** | `0.797` $\to$ **`0.916`** | `6,732.0 ms` |
| `00000065` | 768 | 13,184 | 7,611 | `0.00%` | `63.41%` | **`74.22%`** | **`+10.81%`** | `81.10%` | **`96.61%`** | **`+15.52%`** | `0.669` $\to$ **`0.903`** | `1,366.3 ms` |
| `00000076` | 1,486 | 56,027 | 14,776 | `0.61%` | `50.74%` | **`65.63%`** | **`+14.89%`** | `77.10%` | **`90.44%`** | **`+13.33%`** | `0.499` $\to$ **`0.724`** | `2,966.0 ms` |
| `00000077` | 3,911 | 9,622 | 37,658 | `0.13%` | `43.45%` | **`46.65%`** | **`+3.20%`** | `87.21%` | **`96.19%`** | **`+8.98%`** | `0.505` $\to$ **`0.635`** | `6,687.5 ms` |
| `00000091` | 874 | 12,303 | 8,454 | `0.00%` | `55.49%` | **`68.88%`** | **`+13.39%`** | `79.27%` | **`94.50%`** | **`+15.23%`** | `0.559` $\to$ **`0.776`** | `1,644.2 ms` |
| **Mean / Total** | **19,160** | **185,041** | **157,851** | **`0.15%`** | **`63.57%`** | **`73.62%`** | **`+10.05%`** | **`83.82%`** | **`95.11%`** | **`+11.29%`** | **`0.640` $\to$ `0.821`** | **`3,221.0 ms` / trip** |

---

## 3. Pipeline Latency Breakdown (`student_test` Mean per Trip)

- **Raw Data Loading & Validation (`.track`, `.nodes`, `.arcs`):** `86.7 ms`
- **Metric Projection, `STRtree` Construction & Candidate/Feature Generation:** `184.3 ms`
- **Trained Model Scoring (`HistGradientBoostingClassifier.predict_proba`):** `61.9 ms`
- **HMM + Viterbi Topological Decoding (`decode_trip_viterbi`):** `2,876.0 ms` (`~1.5 ms` per GPS observation)
- **Total Route-Free Inference Pipeline (`run_trip_inference`):** **`3,221.0 ms` (`3.22 s` per trip)**

---

## 4. Automated Test Suite Results (`pytest -v`)

Executed via `python -m pytest -v` across all 7 test modules (`26` total tests):
- `tests/test_api.py` (3 tests: `/health`, `/trips`, `/predict` with strict ground-truth isolation, 404/503 error handling) — **`3 passed`**
- `tests/test_inference_and_evaluator.py` (3 tests: `CandidateModelScorer` artifact loading, schema validation, spy-verified route-free inference + isolated server evaluation) — **`3 passed`**
- `tests/test_viterbi_decoder.py` (10 test functions covering all 16 required Viterbi scenarios) — **`10 passed`**
- `tests/test_candidate_generation.py`, `tests/test_data_loader.py`, `tests/test_graph.py`, `tests/test_label_generation.py`, `tests/test_route_alignment.py` — **`10 passed`**
- **Overall Result:** **`26 passed in 29.37s`**
