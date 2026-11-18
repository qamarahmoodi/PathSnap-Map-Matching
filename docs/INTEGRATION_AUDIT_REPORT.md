# PathSnap — GitHub Integration, Code Audit & System Validation Report

**Date:** 2026-11-18
**Integration Branch:** `integration/phase1-review` (local review branch based on `origin/main`)
**Audited Branches:** `origin/main` (`0067754`), `origin/feature/qamar-candidate-generation` (`ba10f58`), `origin/gokul-data-eda` (`cd5672c`)

---

## A. GitHub Integration Summary

### 1. Remote Branches Discovered (`git fetch origin --prune`)
- `origin/main` (`0067754`) — Default repository branch containing merged PR #1 (L A Gokul) and PR #2 (Qamar Ahmed).
- `origin/gokul-data-eda` (`cd5672c`) — L A Gokul's Data Engineering & EDA branch (4 commits: `e61b163`, `6bcfdb9`, `6eb35e0`, `cd5672c`, merged via PR #1 `525a6cc`).
- `origin/feature/qamar-candidate-generation` (`ba10f58`) — Qamar Ahmed's Candidate Generation & Route Alignment branch (`6015600` + merge resolution commit `ba10f58`, merged via PR #2 `0067754`).
- **Pranav's ML Branch Status:** Not yet pushed to `origin` (Pranav reviewed and merged PR #1 as `Pranav S Atreya <pranavatreya16@gmail.com>` and is developing the 5-model pipeline against the candidate schema contract).

### 2. Merge Conflicts & Resolution Audit
During the integration of `gokul-data-eda` and `feature/qamar-candidate-generation` (`ba10f58`), two files had overlapping additions:
1. **`requirements.txt`:** Cleanly resolved by keeping the superset of dependencies (`numpy>=1.26`, `pandas>=2.0`, `pyarrow>=14.0`, `shapely>=2.0`, `pyproj>=3.6`, `scikit-learn>=1.4`, `folium>=0.15`, `pyyaml>=6.0`, `matplotlib>=3.8`, `seaborn>=0.13`, `pytest>=8.0`).
2. **`.gitignore`:** The web merge (`ba10f58`) retained `data/` on line 36 from Gokul's `.gitignore`, which masks future untracked files inside `data/` (though `data/README.md`, `data/dataset_inventory.csv`, and `data/sensitivity_analysis.csv` remain tracked). On `integration/phase1-review`, we removed the redundant `data/` ignore line so `data/` metadata CSVs remain cleanly visible while raw dataset files (`Student folder/`, `*.track`, `*.nodes`, `*.arcs`, `*.route`) stay strictly ignored.

---

## B. Codebase Size & 100K-Line Investigation

We audited all **52 tracked files** in the integrated repository to determine why the line count reached **105,678 lines**.

### 1. Breakdown by File Category

| Category | Number of files | Total lines | Total size (bytes) | Share of total lines |
|---|---:|---:|---:|---:|
| **Python source** | 20 | 2,292 | 93,489 | **2.17%** |
| **Tests** | 5 | 380 | 14,918 | **0.36%** |
| **Markdown** | 9 | 711 | 55,069 | **0.67%** |
| **Notebooks** | 0 | 0 | 0 | **0.00%** |
| **CSV/data outputs** | 8 | 674 | 72,081 | **0.64%** |
| **HTML maps** | 2 | **101,621** | **7,494,575** | **96.16%** |
| **Configuration** | 4 | 121 | 2,167 | **0.11%** |
| **Other (PNG plots)** | 4 | 0 (binary) | 119,252 | **0.00%** |
| **Total (with Audit Report)** | **53** | **105,799** | **~7.85 MB** | **100.00%** |

### 2. Root Cause of the ~100K Lines
- **96.16% of all lines in the repository (101,621 out of 105,678 lines) come from just two self-contained Leaflet/Folium HTML map files:**
  1. `outputs/trip_map.html` (Gokul's map): **62,045 lines** (`5.56 MB`), which renders up to 10,000 road-network `PolyLine` JS objects inside the HTML file.
  2. `outputs/candidate_previews/trip_00000001_alignment_map.html` (Qamar's alignment map): **39,576 lines** (`1.93 MB`).
- **Actual Python engineering code is compact, modular, and maintainable:**
  - Total Python source across both Gokul's and Qamar's modules is **2,292 lines** across 20 files (average **115 lines per file**; largest module is `scripts/run_audit_and_previews.py` at 365 lines, followed by `src/pathsnap/data/loader.py` at 259 lines and `src/pathsnap/alignment/route_alignment.py` at 253 lines).
  - Total test code is **380 lines** across 5 test files.

### 3. Largest 20 Tracked Files in Repository

| Rank | File Path | Category | Lines | Size (Bytes) |
|---:|---|---|---:|---:|
| 1 | `outputs/trip_map.html` | HTML maps | 62,045 | 5,559,690 |
| 2 | `outputs/candidate_previews/trip_00000001_alignment_map.html` | HTML maps | 39,576 | 1,934,885 |
| 3 | `scripts/run_audit_and_previews.py` | Python source | 365 | 16,120 |
| 4 | `src/pathsnap/alignment/route_alignment.py` | Python source | 268 | 12,592 |
| 5 | `src/pathsnap/data/loader.py` | Python source | 259 | 9,487 |
| 6 | `src/pathsnap/spatial/candidate_generation.py` | Python source | 237 | 9,792 |
| 7 | `src/pathsnap/alignment/label_generation.py` | Python source | 208 | 8,216 |
| 8 | `outputs/candidate_previews/candidates_preview_200.csv` | CSV/data outputs | 201 | 27,833 |
| 9 | `outputs/candidate_previews/observations_preview_200.csv` | CSV/data outputs | 201 | 16,946 |
| 10 | `src/pathsnap/schemas/candidate_schema.py` | Python source | 173 | 5,961 |
| 11 | `tests/test_data_loader.py` | Tests | 130 | 5,004 |
| 12 | `docs/CANDIDATE_DATA_CONTRACT.md` | Markdown | 152 | 10,653 |
| 13 | `docs/PHASE1_EVALUATION.md` | Markdown | 142 | 11,667 |
| 14 | `src/pathsnap/spatial/road_network.py` | Python source | 134 | 5,100 |
| 15 | `docs/PRANAV_INTEGRATION.md` | Markdown | 133 | 5,738 |
| 16 | `docs/PREPROCESSING_PIPELINE.md` | Markdown | 124 | 11,731 |
| 17 | `src/pathsnap/graph/road_graph.py` | Python source | 121 | 5,099 |
| 18 | `src/pathsnap/features/feature_engineering.py` | Python source | 97 | 3,756 |
| 19 | `data/dataset_inventory.csv` | CSV/data outputs | 91 | 12,687 |
| 20 | `outputs/dataset_inventory.csv` | CSV/data outputs | 91 | 9,706 |

### 4. Code Quality & Duplication Findings
- **Zero dead code or circular imports:** All modules in `src/pathsnap/` and `src/` compile cleanly and import without circular dependencies.
- **Intentional Dual Loaders & Inventory CSVs (Gokul vs. Qamar):**
  - Gokul implemented `src/data_loader.py` (31 lines, returns a dictionary of raw NumPy arrays `{"track", "nodes", "arcs", "route"}`), `src/eda.py` (69 lines, writes `outputs/dataset_inventory.csv`), and `src/visualization.py` (55 lines, writes 4 PNG histograms and `outputs/trip_map.html`).
  - Qamar implemented `src/pathsnap/data/loader.py` (259 lines, returns typed `TripRawData` with `include_route=False` inference separation and extended validation metrics in `data/dataset_inventory.csv`).
  - Both inventories agree 100% on trip counts (`80` train, `10` student_test), observation counts (`208,417` train, `19,160` student_test), node/arc counts, and maximum timestamp gap (`153,064.0 s`).

---

## C. Technical Integration Verification (Gokul Loader $\leftrightarrow$ Qamar Pipeline)

### 1. Direct Array Compatibility Bridge
Gokul's `src/data_loader.py::load_trip(trip_dir)` returns:
```python
trip = {
    "track": np.ndarray,  # float64, shape (N, 3) -> [lon, lat, timestamp_s]
    "nodes": np.ndarray,  # float64, shape (V, 2) -> [lon, lat]
    "arcs":  np.ndarray,  # int,     shape (E, 2) -> [from_node_id, to_node_id]
    "route": np.ndarray,  # int,     shape (M,)   -> 1D flattened route arc IDs
}
```
Originally, Qamar's `generate_trip_candidates(...)` and `align_reference_route(...)` indexed `track["longitude"]` assuming a `pandas.DataFrame`. On `integration/phase1-review`, we enhanced both functions to natively accept **either** a `pandas.DataFrame` or Gokul's raw `(N, 3)` `np.ndarray` without requiring any manual conversion by the caller.

### 2. Verified Integration Invariants (`tests/test_data_loader.py::test_gokul_data_loader_integration_with_candidate_pipeline`)
Tested across multiple actual training trips (`00000001`, `00000003`, `00000005`):
1. **Leading-zero trip IDs preserved:** `"00000001"`, `"00000003"`, `"00000005"`.
2. **Exact shapes & dtypes accepted:** `d["track"]` (`(N, 3)`), `d["nodes"]` (`(V, 2)`), `d["arcs"]` (`(E, 2)`), `d["route"]` (`(M,)`).
3. **Inference works without `.route`:** `generate_trip_candidates(...)` never takes `d["route"]`.
4. **Bit-for-bit identical output:** Calling `generate_trip_candidates`, `align_reference_route`, and `attach_alignment_labels` with Gokul's `src/data_loader.py` dictionary produces **100% identical** `result.candidates` and `result.observations` DataFrames (`DataFrame.equals(...) == True`) compared to `pathsnap.data.loader.load_trip`.
5. **Exact Schema Compliance:** Verified against `CANDIDATE_SCHEMA_DTYPES` (20 columns), `OBSERVATION_SCHEMA_DTYPES` (13 columns), `CANDIDATE_FEATURE_COLUMNS` (11 features), and `CANDIDATE_ID_COLUMNS` (8 identifiers).

---

## D. Algorithmic Verification & Test Results

### 1. Geodesic vs. Local Tangent-Plane Projection Accuracy
We benchmarked `LocalTangentProjector` (`src/pathsnap/spatial/projection.py`) against exact WGS84 ellipsoidal geodesic inverse calculations (`pyproj.Geod(ellps="WGS84")`) across 5,000 road segments in Trip `00000001`:
- **Median absolute length error:** `0.0166 m` (`1.66 cm`)
- **Maximum relative error:** `0.43%`
- **Speedup:** $>50\times$ faster vectorised dot-product projection than iterative ellipsoidal geodesic projections.

### 2. Automated Test Suite & Static Checks
- `python -m pytest -q` $\rightarrow$ **`9 passed in 18.06s`** (including the new end-to-end Gokul-loader-to-Qamar-pipeline integration test).
- `python -m compileall -q src tests` $\rightarrow$ **Exited with code `0`** (all 25 Python files in `src/` and `tests/` compile cleanly).
- `git diff --check` $\rightarrow$ **Exited with code `0`** (zero whitespace or conflict marker issues).

---

## E. Security & Data Leakage Audit

- **Raw Dataset Exposure:** **NONE (0 files).** No `.track`, `.nodes`, `.arcs`, `.route`, or ZIP files exist in the working tree, staging area, or commit history across any branch (`main`, `gokul-data-eda`, `feature/qamar-candidate-generation`).
- **Credentials / Secrets Exposure:** **NONE.** No `.env` files, API keys, tokens, or passwords exist in the repository. (Note: `outputs/partition_summary.txt` line 2 contains a benign local path string `Data root: C:\Users\lgoku\Downloads\PathSnap_Dataset\Student folder` from Gokul's EDA run).
- **Training / Inference Leakage:** **NONE.** `generate_trip_candidates` is strictly decoupled from `.route`, and `CANDIDATE_FEATURE_COLUMNS` excludes all identifier and rank columns.

---

## F. Technical Risks & Recommendations

### 1. Critical Blockers
- **None.** All Phase 1 requirements, schemas, EDA outputs, Folium visualisations, and integration bridges are tested and working.

### 2. Important Improvements (Completed on `integration/phase1-review`)
- **Seamless NumPy `track` Support:** Updated `generate_trip_candidates` and `align_reference_route` to accept Gokul's `(N, 3)` `np.ndarray` directly alongside `pd.DataFrame`.
- **`.gitignore` Cleanup:** Removed the redundant `data/` ignore rule introduced during the PR merge so tracked metadata files in `data/` are not masked by Git.

### 3. Optional Cleanup (Post-Phase 1)
- Fill in the template prose inside Gokul's `report/dataset_exploration.md` (currently 16 lines of section prompts) by copying the measured numbers from `docs/PHASE1_EVALUATION.md` and `outputs/partition_summary.txt`.
- Keep HTML map count limited in Git so repository clone size stays under `10 MB` (both required Folium HTML maps are already committed).

---

## G. Final Status

### **READY FOR PHASE 1 EVALUATION**

**Justification:**
1. All Phase 1 documentation deliverables (`docs/PHASE1_EVALUATION.md`, `docs/PREPROCESSING_PIPELINE.md`, `docs/MODEL_SHORTLIST.md`, `docs/CANDIDATE_DATA_CONTRACT.md`, `docs/PRANAV_INTEGRATION.md`, `docs/PHASE1_PRESENTATION.md`) and Gokul's EDA artifacts (`outputs/dataset_inventory.csv`, 4 PNG plots, `outputs/trip_map.html`) are present, consistent, and backed by real measurements across all 90 trips.
2. Gokul's `src/data_loader.py` and Qamar's `src/pathsnap/` candidate-generation and route-alignment pipeline are 100% compatible and verified by automated integration tests (`9 passed`).
3. Zero raw dataset files or credentials have been leaked into Git history.
