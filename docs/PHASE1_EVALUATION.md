# PathSnap — Phase 1 Technical Evaluation Document

**Team Members:**
- **Qamar Ahmed (Technical Lead):** Candidate generation, spatial projection, feature engineering, monotonic DP reference alignment, HMM & Viterbi route reconstruction, core prediction pipeline.
- **Pranav Shashikiran Atreya (ML Engineering):** Five-model training & comparison pipeline, `GroupKFold` cross-validation, Bayesian hyperparameter optimisation, model persistence.
- **L A Gokul (Data Engineering & Documentation):** Dataset ingestion, integrity auditing, exploratory data analysis, Folium geographical visualisations, documentation.

---

## 1. Problem Understanding

Urban GPS telemetry suffers from multipath canyon reflections, signal attenuation, sampling dropouts, and stationary jitter. Naive nearest-road snapping assigns consecutive GPS points to disconnected or opposing carriageways. PathSnap solves this as a **hybrid Machine Learning + Probabilistic Graphical Model (HMM/Viterbi)** system:
1. **Spatial Indexing & Candidate Generation** reduces up to $307,392$ road segments per trip to a ranked shortlist of $K \le 10$ plausible directed road segments within $R = 100\text{ m}$ of each GPS observation.
2. **Supervised Candidate Probability Estimation** evaluates five distinct ML model families to estimate calibrated emission probabilities $P(y_{i,c}=1 \mid \mathbf{x}_{i,c})$ from 11 geometric and causal kinematic features.
3. **Topological Viterbi Decoding** combines ML emission probabilities with directed graph connectivity (`TripRoadGraph`) to reconstruct a physically connected route.

---

## 2. Dataset Directory Structure & Partition Inventory

Our verified dataset root resides at `../Student folder` (configurable via `PATHSNAP_DATA_ROOT`, excluded from Git via `.gitignore`):

```text
Student folder/
├── train/           # 80 trips (208,417 total GPS observations)
│   └── <TRIP_ID>/   # 8-digit zero-padded string IDs: "00000001" .. "00000099"
│       ├── <TRIP_ID>.track
│       ├── <TRIP_ID>.nodes
│       ├── <TRIP_ID>.arcs
│       └── <TRIP_ID>.route
└── student_test/    # 10 trips (19,160 total GPS observations)
    └── <TRIP_ID>/   # "00000000", "00000007", "00000009", "00000040", ...
```

*(Note: 10 reserved `evaluator_test` trips exist on the evaluation server for final two-laptop testing and are never accessed during development).*

---

## 3. Meaning of the Four Raw Files & Actual Audit Findings

Full statistics across all 90 trips are recorded in [`data/dataset_inventory.csv`](../data/dataset_inventory.csv):
1. **`.track` (3 columns, whitespace-separated, no header):** `longitude` (`float64`), `latitude` (`float64`), `timestamp_s` (`float64`).
   - **Audit:** $262$ to $19,313$ observations per trip (median $1,496.5$). Mean proportion of exact $1.0\text{ s}$ intervals is $94.32\%$ (median $97.6\%$), confirming nominal $1\text{ Hz}$ sampling. However, $47 / 90$ trips have time gaps $> 5\text{ s}$ and $30 / 90$ trips have gaps $> 30\text{ s}$ (max gap $153,064\text{ s}$).
2. **`.nodes` (2 columns, whitespace-separated, no header):** `longitude` (`float64`), `latitude` (`float64`). Row index ($0 \dots N_{\text{nodes}}-1$) is the trip-specific `node_id`.
   - **Audit:** $369$ to $163,991$ nodes per trip; zero isolated nodes across all 90 trips.
3. **`.arcs` (2 integer columns, whitespace-separated, no header):** `from_node_id`, `to_node_id` (`int32`). Row index ($0 \dots N_{\text{arcs}}-1$) is the trip-specific directed `arc_id`.
   - **Audit:** $649$ to $307,392$ directed arcs per trip (median $18,761.5$). Every node reference is valid ($0$ out-of-bounds references).
4. **`.route` (1 integer column, no header):** Chronological sequence of traversed `arc_id` values (`int32`).
   - **Audit:** $48$ to $1,558$ route segments per trip. Crucially, `len(.track) != len(.route)` in every trip, and $36 / 90$ trips contain repeated `arc_id` occurrences (loops/U-turns). Across all 90 trips, only 14 transitions in `.route` are non-adjacent in the node graph.

---

## 4. Preprocessing & Geographic Projection Pipeline

- **Why Degree-Space Euclidean Distance is Invalid:** Across our 90 trips, latitudes span $-34.64^\circ$ to $+61.70^\circ$ and longitudes span $-100.42^\circ$ to $+137.24^\circ$. At $\phi \approx 50^\circ\text{ N}$, $1^\circ$ of longitude is $35.5\%$ shorter than $1^\circ$ of latitude.
- **Trip-Centred WGS84 Ellipsoidal Tangent Projection (`LocalTangentProjector`):** Projects $(\text{lon}, \text{lat})$ into metric $(x_{\text{east, m}}, y_{\text{north, m}})$ using exact WGS84 prime-vertical ($N(\phi_0)$) and meridional ($M(\phi_0)$) radii of curvature around each trip's centroid. All lengths, orthogonal projections, and bearings are computed in metres and degrees clockwise from True North.

---

## 5. Feature Engineering Proposal (11 Features)

All 11 features in `CANDIDATE_FEATURE_COLUMNS` are strictly causal (no look-ahead):
1. `distance_to_segment_m` (`float32`): Perpendicular/clamped Euclidean distance from GPS point to road segment in metres.
2. `projection_fraction` (`float32` $\in [0, 1]$): Relative position of orthogonal projection along the directed segment.
3. `distance_to_start_m` (`float32`): Distance in metres to `from_node_id`.
4. `distance_to_end_m` (`float32`): Distance in metres to `to_node_id`.
5. `segment_length_m` (`float32`): Directed road segment length in metres.
6. `segment_bearing_deg` (`float32` $\in [0, 360)^\circ$): Road segment azimuth clockwise from North.
7. `gps_bearing_deg` (nullable `Float32` $\in [0, 360)^\circ$): Causal GPS trajectory azimuth ($i-1 \to i$). Set to `<NA>` at $i=0$, when $\Delta t_i \le 0$, $\Delta t_i > 30\text{ s}$, or when step displacement $< 2.5\text{ m}$ (stationary noise).
8. `bearing_difference_deg` (nullable `Float32` $\in [0, 180]^\circ$): Unsigned angular difference between GPS bearing and road bearing.
9. `gps_speed_mps` (nullable `Float32`): Causal speed $\Delta d_i / \Delta t_i$ in $\text{m/s}$ (`<NA>` when $\Delta t_i \le 0$, $\Delta t_i > 30\text{ s}$, or speed $> 65\text{ m/s}$).
10. `delta_time_s` (nullable `Float32`): Elapsed time $t_i - t_{i-1}$ in seconds (`<NA>` at $i=0$).
11. `is_fallback_candidate` (`bool`): `True` if retrieved via nearest-segment fallback when zero segments existed within `search_radius_m`.

---

## 6. Candidate-Generation Strategy & Sensitivity Analysis

- Indexed per trip with **`Shapely 2.x STRtree`** over metric road `LineString` geometries.
- Empirical sensitivity analysis across 10 training trips ($23,178$ observations, saved in [`data/sensitivity_analysis.csv`](../data/sensitivity_analysis.csv)) shows:
  - $R = 30\text{ m}, K = 5$: Miss rate $= 3.524\%$ (recall $96.476\%$).
  - **$R = 100\text{ m}, K = 10$ (Selected Default):** Miss rate $= \mathbf{0.270\%}$ (recall $\mathbf{99.730\%}$), average $8.48$ candidates/observation, positive class ratio $11.26\%$.

---

## 7. Training-Label Alignment Challenge & Monotonic DP Solution

- Implemented in `align_reference_route(...)` (`src/pathsnap/alignment/route_alignment.py`).
- Uses monotonic dynamic programming over `(observation_id i, route_position j)` where $j \in \{0, \dots, M_{\text{route}}-1\}$ preserves repeated arc traversals independently of `reference_arc_id`.
- Combines squared point-to-segment distance and directional agreement with arc-length progression penalties scaled by $\Delta t_i$.
- Classifies every observation into `"aligned"`, `"ambiguous"`, `"unmatched"`, or `"not_available"`.
- **Crucial Label Integrity Rule:** When `alignment_status == "aligned"` but `reference_arc_id` is absent from the top-$K$ candidates (`candidate_miss == True`), `label_eligible` is set to `False` and all candidate labels for that observation remain `<NA>`—never false negatives.

---

## 8. Five Proposed ML Models

Detailed in [`docs/MODEL_SHORTLIST.md`](MODEL_SHORTLIST.md):
1. **Logistic Regression** (Linear & Probabilistic baseline)
2. **Distance-Weighted KNN** (Instance-based local neighbourhood learning)
3. **Decision Tree Classifier** (Interpretable axis-aligned nonlinear partitioning)
4. **Random Forest Classifier** (Bootstrap-aggregated variance-reduced ensemble)
5. **HistGradientBoosting Classifier** (Sequential gradient boosting minimizing log-loss with native missing-value routing)

---

## 9. Evaluation Strategy & Metrics Hierarchy

- **Cross-Validation:** 5-fold `GroupKFold(n_splits=5)` grouped by `trip_id` across the 80 training trips.
- **Development Holdout:** 10 `student_test` trips evaluated for model comparison and HMM validation.
- **Metrics by Granularity:**
  - *Candidate-row level:* Binary Precision, Recall, Macro F1-score, Confusion Matrix, and **Log Loss** (crucial for HMM emission calibration).
  - *Observation level:* **Observation-Level Top-1 Accuracy** (fraction of aligned observations where the highest-probability candidate matches `reference_arc_id`) and **Candidate Miss Rate**.
  - *Sequence / Route level:* **Route Continuity** (fraction of consecutive predicted segments $(i-1, i)$ that are identical or connected via `graph.are_connected(a_{i-1}, a_i)`), Sequence Accuracy before and after Viterbi decoding, Training Time, and Inference Time.

---

## 10. Data Leakage Prevention

1. **Trip-Level Group Splitting:** All observations and candidates from a given `trip_id` stay strictly within the same CV fold.
2. **Strict Training/Inference Separation:** `generate_trip_candidates(...)` has no parameter for `.route`.
3. **Identifier Exclusion:** `trip_id`, `observation_id`, `candidate_arc_id`, `from_node_id`, `to_node_id`, and `candidate_rank` are excluded from `CANDIDATE_FEATURE_COLUMNS`.
4. **In-Fold Preprocessing:** Imputers and scalers are wrapped inside `sklearn.pipeline.Pipeline` and fitted only on training folds.
5. **Causal Kinematics:** Speeds and headings use backward differences ($i-1 \to i$) only.

---

## 11. Team Responsibility Allocation

| Team Member | Role | Phase 1 & Phase 2 Deliverables |
|---|---|---|
| **Qamar Ahmed** | Technical Lead | Spatial projection, `STRtree` candidate generation, causal feature engineering, monotonic DP alignment, `TripRoadGraph`, HMM + Viterbi decoder, FastAPI integration. |
| **Pranav Shashikiran Atreya** | ML Engineering | 5-model `sklearn` pipelines, 5-fold `GroupKFold` CV, student-test evaluation tables, Bayesian hyperparameter optimisation, model export. |
| **L A Gokul** | Data Engineering & Docs | Dataset loader validation, 90-trip `dataset_inventory.csv`, EDA distributions, interactive Folium trajectory/route maps, technical report & README. |

---

## 12. Open Technical Questions & Next Steps for Phase 2

1. **Class Weighting vs Probability Calibration:** Because positive candidates represent $11.26\%$ of training rows, using `class_weight='balanced'` improves raw binary F1 at threshold $0.5$ but shifts log-odds by a constant offset; we will evaluate whether uncalibrated balanced weights or natural priors yield superior Top-1 ranking and HMM log-emission scaling.
2. **Handling Unmatched Trajectory Tails (e.g., Trip `00000002`):** In Trip `00000002`, the vehicle continues driving for $868$ observations after the supplied `.route` ends. Our alignment classifier automatically flags these as `"unmatched"` (`label_eligible = False`) so they do not corrupt training labels.
3. **Next Steps (11:30 AM onwards):** Run full 80-trip candidate extraction, execute Pranav's 5-model comparison and Bayesian optimisation, and integrate Qamar's Viterbi decoder.
