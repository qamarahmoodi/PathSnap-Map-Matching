# PathSnap — Phase 1 Faculty Evaluation & Viva Preparation Notes

**Team Members:**
- **Pranav Shashikiran Atreya:** Five-model ML experimentation, `GroupKFold` cross-validation, Bayesian hyperparameter optimisation, HMM/Viterbi evaluation.
- **Qamar Ahmed:** Spatial projection (`LocalTangentProjector`), `STRtree` candidate generation, causal feature engineering, monotonic DP reference alignment, `TripRoadGraph`.
- **L A Gokul:** Raw dataset ingestion (`src/data_loader.py`), 90-trip integrity audit (`src/eda.py`), EDA plots & Folium maps (`src/visualization.py`).

---

## Topic 1: Dataset Structure and Understanding

### 1.1 The Four Raw Files per Trip
Each trip directory (`<trip_id>/`, 8-digit zero-padded string IDs `"00000000"`–`"00000099"`) contains four headerless whitespace-separated files:
1. **`<trip_id>.track` (3 float64 columns: `longitude`, `latitude`, `timestamp_s`):**
   - Chronological noisy GPS observations of the vehicle.
   - Nominally sampled at $1\text{ Hz}$ (`94.32%` mean exact $1.0\text{ s}$ intervals; median `97.6%`).
2. **`<trip_id>.nodes` (2 float64 columns: `longitude`, `latitude`):**
   - Road network vertices. The **0-based row index** ($0 \dots N_{\text{nodes}}-1$) is the **trip-specific `node_id`**.
3. **`<trip_id>.arcs` (2 int32 columns: `from_node_id`, `to_node_id`):**
   - Directed road segments connecting `from_node_id` $\to$ `to_node_id`. The **0-based row index** ($0 \dots N_{\text{arcs}}-1$) is the **trip-specific `arc_id`**.
   - **Critical Invariant:** `arc_id` and `node_id` are local to each trip (`arc_id = 10` in Trip `00000001` is a completely different street from `arc_id = 10` in Trip `00000002`).
4. **`<trip_id>.route` (1 int32 column: `arc_id`):**
   - Ordered chronological sequence of reference `arc_id` values traversed by the vehicle.
   - **Critical Invariant:** `len(.track) != len(.route)` in every trip (`208,417` GPS points vs `31,336` route segments in `train`). Row $i$ of `.track` **never** aligns 1-to-1 with row $i$ of `.route`.

### 1.2 Verified Dataset Partitions & Actual Statistics
Verified via [`data/dataset_inventory.csv`](data/dataset_inventory.csv), [`outputs/partition_summary.txt`](outputs/partition_summary.txt), and [`outputs/candidate_previews/dataset_audit_digest.json`](outputs/candidate_previews/dataset_audit_digest.json):

| Metric | Training Partition (`train`) | Student-Testing (`student_test`) | Combined Supplied (`90` Trips) |
| :--- | :---: | :---: | :---: |
| **Number of Trips** | **80** | **10** | **90** *(+10 reserved evaluator trips)* |
| **Total GPS Observations (`.track`)** | **208,417** | **19,160** | **227,577** |
| **Observations per Trip (Min / Median / Max)** | `361` / `1,496.5` / `19,313` | `262` / `1,568.5` / `3,958` | `262` / `1,496.5` / `19,313` |
| **Total Road Nodes (`.nodes`)** | **1,933,485** | **90,689** | **2,024,174** |
| **Total Directed Road Arcs (`.arcs`)** | **3,876,904** | **185,041** | **4,061,945** |
| **Road Arcs per Trip (Min / Median / Max)** | `649` / `19,117` / `307,392` | `3,577` / `15,436` / `56,027` | `649` / `18,761.5` / `307,392` |
| **Total Reference Route Segments (`.route`)** | **31,336** | **3,477** | **34,813** |
| **Non-Finite / Invalid Lon-Lat Rows** | `0` | `0` | `0` |
| **Out-of-Bounds Node or Route Arc IDs** | `0` | `0` | `0` |
| **Isolated Road Nodes** | `0` | `0` | `0` |
| **Trips with Timestamp Gaps $> 1.5\text{ s}$** | `60 / 80` | `6 / 10` | `66 / 90` |
| **Trips with Timestamp Gaps $> 5\text{ s}$ / $> 30\text{ s}$** | — | — | `47 / 90` (`> 5s`), `30 / 90` (`> 30s`) |
| **Maximum Timestamp Gap ($\Delta t_{\max}$)** | `153,064.0 s` (~42.5 hrs) | `210.0 s` | `153,064.0 s` |
| **Disconnected Transitions in `.route`** | `13` (`99.935%` connected) | `1` (`99.973%` connected) | `14` total across 34,813 segments |

---

## Topic 2: Preprocessing Methodology

Implemented in [`src/pathsnap/data/loader.py`](src/pathsnap/data/loader.py), [`src/pathsnap/spatial/projection.py`](src/pathsnap/spatial/projection.py), and [`src/pathsnap/features/feature_engineering.py`](src/pathsnap/features/feature_engineering.py):

1. **Coordinate & Topology Validation (`validate_trip_arrays`):**
   - Verifies finite floats, WGS84 bounds ($|\text{lon}| \le 180^\circ, |\text{lat}| \le 90^\circ$), monotonic timestamps ($\Delta t_i = t_i - t_{i-1}$), valid node indices in `.arcs` ($0 \le u, v < N_{\text{nodes}}$), and valid arc indices in `.route` ($0 \le a < N_{\text{arcs}}$).
2. **Local WGS84 Ellipsoidal Tangent-Plane Metric Projection (`LocalTangentProjector`):**
   - **Why needed:** Across our 90 trips, latitudes span $-34.64^\circ$ to $+61.70^\circ$ and longitudes span $-100.42^\circ$ to $+137.24^\circ$. At $\phi \approx 50^\circ\text{ N}$, $\cos(50^\circ) \approx 0.643$, so $1^\circ$ of longitude ($\approx 71.7\text{ km}$) is **$35.7\%$ shorter** than $1^\circ$ of latitude ($\approx 111.2\text{ km}$). Raw degree Euclidean distance severely distorts distances and orthogonal projections.
   - **Formula:** For each trip's centroid $(\lambda_0, \phi_0)$, we compute the exact WGS84 prime-vertical radius of curvature $N(\phi_0) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi_0}}$ and meridional radius of curvature $M(\phi_0) = \frac{a(1 - e^2)}{(1 - e^2 \sin^2\phi_0)^{3/2}}$, projecting $(\lambda, \phi)$ into local Cartesian metres:
     $$x_{\text{east, m}} = (\lambda - \lambda_0)\frac{\pi}{180} N(\phi_0)\cos\phi_0, \qquad y_{\text{north, m}} = (\phi - \phi_0)\frac{\pi}{180} M(\phi_0)$$
3. **Timestamp-Gap & Missing-Value Handling (`compute_gps_kinematics`):**
   - Kinematics are computed strictly **causally** ($i-1 \to i$).
   - **First observation ($i = 0$):** No predecessor exists $\implies$ `delta_time_s`, `gps_speed_mps`, `gps_bearing_deg`, and `bearing_difference_deg` are set to nullable `<NA>`.
   - **Stationary GPS jitter ($\Delta d_i < 2.5\text{ m}$):** When a vehicle is stopped at a traffic light, GPS noise wanders randomly in $360^\circ$. Computing bearing from $< 2.5\text{ m}$ displacement injects random angular noise, so `gps_bearing_deg` and `bearing_difference_deg` are masked to `<NA>`.
   - **Large GPS timestamp dropouts ($\Delta t_i > 30\text{ s}$ or $\Delta t_i \le 0$) or implausible speed spikes ($> 65\text{ m/s}$):** Masked to `<NA>` in kinematic features so finite-difference straight lines across signal dropouts do not corrupt road-heading comparisons.

---

## Topic 3: Feature Engineering (11 Candidate Features)

Defined in `CANDIDATE_FEATURE_COLUMNS` ([`src/pathsnap/schemas/candidate_schema.py`](src/pathsnap/schemas/candidate_schema.py)) and extracted in [`src/pathsnap/spatial/candidate_generation.py`](src/pathsnap/spatial/candidate_generation.py):

| # | Feature Column | Dtype | Nullable | Physical Meaning & Why It Is Used |
| :-: | :--- | :--- | :---: | :--- |
| 1 | `distance_to_segment_m` | `float32` | No | Shortest perpendicular/clamped Euclidean distance (m) from GPS point to the candidate road segment. Primary spatial proximity signal. |
| 2 | `projection_fraction` | `float32` | No | Orthogonal projection parameter $\alpha \in [0.0, 1.0]$ along the directed arc ($0 = \text{start node}$, $1 = \text{end node}$). Identifies whether the vehicle is mid-link or at an intersection endpoint. |
| 3 | `distance_to_start_m` | `float32` | No | Distance (m) from GPS point to `from_node_id`. Helps disambiguate entry into a new segment at intersections. |
| 4 | `distance_to_end_m` | `float32` | No | Distance (m) from GPS point to `to_node_id`. Indicates proximity to the outgoing junction. |
| 5 | `segment_length_m` | `float32` | No | Physical length (m) of the candidate road segment. Prevents tiny intersection connector arcs from dominating long arterial segments. |
| 6 | `segment_bearing_deg` | `float32` | No | Directed road azimuth $\theta_{\text{arc}} \in [0, 360)^\circ$ clockwise from True North. Distinguishes one-way vs opposing carriageways. |
| 7 | `gps_bearing_deg` | `Float32` | **Yes** | Causal vehicle movement azimuth $\theta_{\text{gps}} \in [0, 360)^\circ$ from $i-1 \to i$. |
| 8 | `bearing_difference_deg` | `Float32` | **Yes** | Smallest unsigned angle $|\theta_{\text{gps}} - \theta_{\text{arc}}| \in [0, 180]^\circ$. Crucial for rejecting opposing lanes ($\approx 180^\circ$) and perpendicular cross-streets ($\approx 90^\circ$). |
| 9 | `gps_speed_mps` | `Float32` | **Yes** | Causal speed $\Delta d_i / \Delta t_i$ (m/s). Tree/ensemble models use speed to weight heading agreement (high confidence at cruising speed, lower confidence near stops). |
| 10 | `delta_time_s` | `Float32` | **Yes** | Elapsed time $t_i - t_{i-1}$ (s). Signals sampling regularity vs GPS dropouts. |
| 11 | `is_fallback_candidate` | `bool` | No | `True` if zero arcs existed within `search_radius_m` and the candidate was retrieved via nearest-neighbour fallback. |

- **Strict Identifier Exclusion:** `trip_id`, `partition`, `observation_id`, `timestamp_s`, `candidate_rank`, `candidate_arc_id`, `from_node_id`, and `to_node_id` are stored in `CANDIDATE_ID_COLUMNS` and **never** passed to `model.fit()` or `model.predict_proba()`.

---

## Topic 4: Candidate-Generation Strategy & Sensitivity Analysis

Implemented in [`src/pathsnap/spatial/candidate_generation.py`](src/pathsnap/spatial/candidate_generation.py) and [`src/pathsnap/spatial/road_network.py`](src/pathsnap/spatial/road_network.py):

1. **Spatial Indexing via Shapely 2.x `STRtree`:**
   - Each trip has up to $307,392$ directed arcs. Brute-force $O(N_{\text{obs}} \times N_{\text{arcs}})$ distance calculation would require $> 6 \times 10^{10}$ checks.
   - We build a C-accelerated Sort-Tile-Recursive R-tree (`shapely.STRtree`) over the projected metric road segments once per trip, querying bounding boxes of radius $R = \text{search\_radius\_m}$ around all GPS points simultaneously in $< 0.2\text{ s}$ per trip.
2. **Exact Segment Filtering, Top-$K$ Ranking & Fallback Nearest Search:**
   - For all tree hits within the bounding box, we compute exact vectorised point-to-segment distances $d(p_i, s_a)$, filter to $d \le R$, sort ascending by distance, and retain up to $K = \text{max\_candidates\_k}$ unique `candidate_arc_id` rows (`candidate_rank = 1 .. K`).
   - **Fallback Nearest-Neighbour Search:** If a GPS point has $0$ arcs within $R = 100\text{ m}$ (e.g., severe urban canyon outlier), `STRtree.query_nearest` retrieves the nearest road segment up to `fallback_max_radius_m = 500 m` and flags `is_fallback_candidate = True` so no observation is left without candidates.
3. **Actual Executed Sensitivity Analysis ([`data/sensitivity_analysis.csv`](data/sensitivity_analysis.csv) — 10 Training Trips, `23,178` Observations, `22,193` Aligned):**

| Search Radius $R$ (m) | Max Candidates $K$ | Candidate Miss Count | **Candidate Miss Rate** | **Candidate Recall** | Avg Candidates / Obs | Positive Class Ratio | Query Time (10 Trips) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `30 m` | `5` | `782` | `3.5236%` | `96.4764%` | `3.844` | `24.03%` | `0.894s` |
| `50 m` | `5` | `592` | `2.6675%` | `97.3325%` | `4.316` | `21.59%` | `1.150s` |
| `50 m` | `10` | `60` | `0.2704%` | `99.7296%` | `7.079` | `13.49%` | `1.150s` |
| **`100 m` (Selected)** | **`10` (Selected)** | **`60`** | **`0.2704%`** | **`99.7296%`** | **`8.478`** | **`11.26%`** | **`1.921s`** |
| `100 m` | `15` | `9` | `0.0406%` | `99.9594%` | `11.678` | `8.20%` | `1.921s` |
| `150 m` | `20` | `2` | `0.0090%` | `99.9910%` | `16.211` | `5.91%` | `2.909s` |

- **Justification for Selected Operating Point ($R = 100\text{ m}, K = 10$):** Captures **`99.73%` recall** (`0.27%` miss rate) while keeping average candidates at `8.48` per observation and positive class balance at a healthy `11.26%` (avoiding the severe class imbalance and memory bloat of $K=20$).

---

## Topic 5: Five Proposed ML Models (Plus Executed 5-Trip Pilot)

Documented in [`docs/MODEL_SHORTLIST.md`](docs/MODEL_SHORTLIST.md) and implemented in [`src/models/model_pipeline.py`](src/models/model_pipeline.py):

1. **Logistic Regression (Linear & Probabilistic Family):**
   - *Suitability:* Models log-odds linearly over standardized distance and bearing mismatch; optimizes binary cross-entropy directly, producing well-calibrated probabilities for HMM emissions.
   - *Limitation:* Cannot capture conditional non-linear thresholds (e.g., bearing difference only matters when `gps_speed_mps > 3 m/s`).
2. **Distance-Weighted KNN (Instance-Based Non-Parametric Family):**
   - *Suitability:* Estimates probability from inverse-distance weighted $k=25$ nearest neighbours in standardized 11D feature space without parametric assumptions.
   - *Limitation:* $O(N_{\text{train}})$ query cost makes inference slow (`24.08s` across 5 folds vs `0.04s` for Logistic Regression).
3. **Decision Tree Classifier (Tree-Based Nonlinear Family):**
   - *Suitability:* Learns interpretable axis-aligned threshold splits (`criterion='log_loss'`, `max_depth=8`, `min_samples_leaf=30`).
   - *Limitation:* High variance across trips and piecewise-constant leaf probabilities (`0.6159` OOF Log Loss).
4. **Random Forest Classifier (Bagging Ensemble Family):**
   - *Suitability:* Averages $120$ decorrelated bootstrap trees (`max_depth=12`, `min_samples_leaf=15`), dramatically reducing single-tree variance and achieving the best **OOF Log Loss (`0.2102`)** on our 5-trip pilot.
   - *Limitation:* Higher memory footprint and slower inference than single boosted trees.
5. **HistGradientBoosting Classifier (Gradient Boosting Family):**
   - *Suitability:* Sequentially minimizes binary log-loss using histogram-binned additive trees; **natively routes `NaN` values** in `gps_bearing_deg` and `gps_speed_mps` without imputation distortion; achieves the highest **Mean Fold Macro F1 (`0.8013`)** and **OOF Macro F1 (`0.7445`)** on our 5-trip pilot.

> **Clear Distinction (Proposed Full 80-Trip Training vs Executed Phase 1 Pilot):**
> - **Executed in Phase 1:** Genuine 5-fold `GroupKFold` training and evaluation of all 5 models on the **5 representative training trips (`00000001`–`00000005`, `92,299` labeled rows)** saved in [`results/phase1/model_comparison.csv`](results/phase1/model_comparison.csv).
> - **Proposed for Phase 2:** Scaling the exact same pipeline to all **80 training trips** and evaluating on the **10 `student_test` trips**.

---

## Topic 6: Training-Label Alignment (`align_reference_route`)

Implemented in [`src/pathsnap/alignment/route_alignment.py`](src/pathsnap/alignment/route_alignment.py) and [`src/pathsnap/alignment/label_generation.py`](src/pathsnap/alignment/label_generation.py):

1. **Why Monotonic Dynamic Programming Over Route Positions $j \in \{0, \dots, M-1\}$?**
   - `.route` is a sequence of $M$ traversed arcs ($M \ll N_{\text{obs}}$), and $36 / 90$ trips traverse the same `arc_id` multiple times (loops/out-and-back trips). Indexing DP states by **route occurrence index $j$** (rather than unique `arc_id`) preserves exact chronological ordering and handles loops cleanly.
2. **Emission Cost, Transition Cost & Backtracking:**
   - **Emission Cost $E(i, j)$:** Normalized squared distance from GPS point $i$ to route segment $j$, plus directional bearing penalty when GPS heading is valid:
     $$E(i, j) = \left(\frac{d(p_i, r_j)}{\sigma}\right)^2 + w_{\text{brg}} \left(\frac{|\Delta \theta_{i,j}|}{180^\circ}\right)^2 \quad (\sigma = 15\text{ m},\; w_{\text{brg}} = 5.0)$$
   - **Monotonic Transition Cost $T(j_{\text{prev}} \to j_{\text{curr}})$ for $j_{\text{curr}} \ge j_{\text{prev}}$:**
     - If $j_{\text{curr}} = j_{\text{prev}}$ (staying on the same route segment): zero skip penalty, plus a small quadratic penalty only if projection moves backward by $> 5\text{ m}$.
     - If $j_{\text{curr}} > j_{\text{prev}}$ (advancing $\Delta j$ route segments):
       $$T(j_{\text{prev}} \to j_{\text{curr}}) = \frac{\lambda_{\text{skip}}}{\sqrt{\max(1, \Delta t_i)}}(\Delta j - 1) + \frac{\lambda_{\text{len}}}{\sigma}\left|\Delta s_{\text{route}}(j_{\text{prev}}, j_{\text{curr}}) - \Delta d_{\text{gps}, i}\right|$$
   - **Backtracking:** Recovers the globally optimal non-decreasing sequence $0 \le j_0 \le j_1 \le \dots \le j_{N-1} < M$ (`is_monotonic == True` verified across all preview trips).
3. **Four Alignment Statuses (`alignment_status`):**
   - `"aligned"`: Minimum DP alignment distance $d_i \le 50\text{ m}$ and unambiguous (`10,618 / 11,488` observations in the 5-trip preview).
   - `"ambiguous"`: $d_i > 20\text{ m}$ and a non-adjacent route segment with a different `arc_id` lies within `3.0 m` of $d_i$ (`2` observations in the 5-trip preview).
   - `"unmatched"`: $d_i > 50\text{ m}$—for example, in Trip `00000002`, the vehicle continues driving for **`868` observations** after the supplied `.route` ends (`alignment_distance_m` up to `242 m`). Flagging these `"unmatched"` prevents corrupting training labels!
   - `"not_available"`: Used during blind inference when `.route` is not supplied.
4. **Candidate Misses & `label_eligible` Rule:**
   - `label_eligible = True` **if and only if** `alignment_status == "aligned"` **AND** `candidate_miss == False` (`reference_arc_id` is present among the observation's generated candidates).
   - When `candidate_miss == True` (`21 / 10,618` aligned observations = `0.1978%`), `label_eligible` is set to `False` and **all candidate labels for that observation are set to `<NA>` (never `0`)** so the classifier is never taught to predict false negatives when the true road fell outside top-$K$.

---

## Topic 7: Evaluation Metrics Hierarchy

Implemented in [`src/models/model_pipeline.py`](src/models/model_pipeline.py) and [`src/sequence/viterbi_decoder.py`](src/sequence/viterbi_decoder.py):

1. **Candidate-Generation Stage Metric (Pre-ML):**
   - **Candidate Miss Rate:** Fraction of confidently `"aligned"` GPS observations whose `reference_arc_id` is absent from the generated candidate set:
     $$\text{Candidate Miss Rate} = \frac{\sum_{i \in \text{aligned}} \mathbb{I}(a^*_i \notin \mathcal{C}_i)}{|\text{aligned}|}$$
     Determines the theoretical upper bound ($1 - \text{Miss Rate} = 99.80\%$) for downstream pointwise classification without fallback recovery.
2. **Candidate-Row Classifier Metrics (Evaluated on `label_eligible` Candidate Rows):**
   - **Accuracy, Precision (Positive Class), Recall (Positive Class), Confusion Matrix `(TN, FP, FN, TP)`**.
   - **Macro F1-Score:** Unweighted mean of negative-class F1 and positive-class F1; essential because positive candidates are ~`11.5%` of rows (a trivial all-zero predictor gets `88.5%` accuracy but only `0.469` Macro F1).
   - **Binary Log Loss (Cross-Entropy):** Measures probability calibration quality. Because the HMM Viterbi decoder adds $\log P(y_{i,c}=1 \mid \mathbf{x}_{i,c})$ across time steps, poorly calibrated overconfident probabilities ($P \approx 0$ or $1$) cause log-likelihood underflow.
3. **Observation-Level & Sequence-Level Map-Matching Metrics:**
   - **Observation-Level Top-1 Accuracy:** For each `"aligned"` GPS observation $i$, pick the candidate $\hat{c}_i = \arg\max_{c \in \mathcal{C}_i} \hat{P}(y_{i,c}=1 \mid \mathbf{x}_{i,c})$ (or the Viterbi decoded state) and compute the fraction where $\text{arc}(\hat{c}_i) == \text{reference\_arc\_id}_i$.
   - **Route Continuity:** Fraction of consecutive observation steps $(i-1, i)$ within each trip where the predicted road segments satisfy directed topological connectivity `graph.are_connected(pred_arc[i-1], pred_arc[i])` (either staying on the same segment $u = v$ or direct node adjacency $\text{to\_node}(u) = \text{from\_node}(v)$).

---

## Topic 8: Data-Leakage Prevention

1. **Trip-Grouped Cross-Validation (`GroupKFold(n_splits=5)` by `trip_id`):**
   - Consecutive GPS observations 1 second apart in the same trip have nearly identical coordinates, speeds, and road candidates. Random row splitting leaks trajectory autocorrelation across train and validation sets. Grouping by `trip_id` guarantees that every validation fold tests generalisation to **completely unseen trips and unseen road networks**.
2. **Strict Partition Isolation (`train` vs `student_test` vs `evaluator_test`):**
   - `train` (80 trips): Used for candidate sensitivity analysis, model fitting, `GroupKFold` CV, and Bayesian hyperparameter tuning.
   - `student_test` (10 trips): Held-out development test partition; never used for fitting models or preprocessors.
   - `evaluator_test` (10 reserved trips): Strictly untouched and absent during development.
3. **In-Fold Preprocessing (`sklearn.pipeline.Pipeline`):**
   - `SimpleImputer(strategy="median", add_indicator=True)` and `StandardScaler()` are encapsulated inside each model's `Pipeline`, so medians and scaling parameters are fitted **only on the training trips of each CV fold**.
4. **Inference-Safe Candidate Generation & Feature Matrix:**
   - `generate_trip_candidates(...)` has no parameter for `.route`.
   - All identifier columns (`trip_id`, `observation_id`, `candidate_arc_id`, `from_node_id`, `to_node_id`, `candidate_rank`) and null-label rows (`label.isna()`) are excluded before fitting.

---

## Mandatory Status & Integration of Bayesian Optimisation and HMM/Viterbi

- **Bayesian Hyperparameter Optimisation:**
  - *Status:* Mandatory project component. Implemented in [`src/optimization/bayesian_tuning.py`](src/optimization/bayesian_tuning.py) using **Optuna TPE (`TPESampler`)** with `GroupKFold` by `trip_id`. Demonstrated via an executed 8-trial pilot on the 5 preview training trips ([`results/phase1/bayesian_trials.csv`](results/phase1/bayesian_trials.csv)); full 80-trip tuning will run in Phase 2.
- **HMM & Viterbi Route Reconstruction:**
  - *Status:* Mandatory project component. Implemented in [`src/sequence/viterbi_decoder.py`](src/sequence/viterbi_decoder.py) consuming ML candidate log-probabilities as emissions and `TripRoadGraph` hop connectivity as transitions. Verified via unit tests ([`tests/test_viterbi_decoder.py`](tests/test_viterbi_decoder.py)) and benchmarked on the 5 preview trips ([`results/phase1/hmm_test_results.csv`](results/phase1/hmm_test_results.csv), interactive map [`results/phase1/trip_00000001_viterbi_vs_greedy_map.html`](results/phase1/trip_00000001_viterbi_vs_greedy_map.html)).
