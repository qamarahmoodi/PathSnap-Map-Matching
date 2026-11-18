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
   - Nominally sampled at $1\text{ Hz}$ (`94.32%` mean exact $1.0\text{ s}$ intervals across all 90 trips; `93.83%` mean / `99.14%` median in `train`, `98.20%` mean / `99.93%` median in `student_test`).
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
| **Road Nodes per Trip (Min / Median / Max)** | `299` / `9,608.0` / `155,003` | `1,770` / `6,596.5` / `26,635` | `299` / `9,305.0` / `155,003` |
| **Total Directed Road Arcs (`.arcs`)** | **3,876,904** | **185,041** | **4,061,945** |
| **Road Arcs per Trip (Min / Median / Max)** | `649` / `19,117.0` / `307,392` | `3,577` / `15,436.0` / `56,027` | `649` / `18,761.5` / `307,392` |
| **Total Reference Route Segments (`.route`)** | **31,336** | **3,477** | **34,813** |
| **Route Segments per Trip (Min / Median / Max)** | `59` / `280.0` / `2,266` | `95` / `250.0` / `1,323` | `59` / `272.5` / `2,266` |
| **Non-Finite / Invalid Lon-Lat Rows** | `0` | `0` | `0` |
| **Out-of-Bounds Node or Route Arc IDs** | `0` | `0` | `0` |
| **Isolated Road Nodes** | `0` | `0` | `0` |
| **Trips with Timestamp Gaps $> 1.5\text{ s}$** | `60 / 80` | `6 / 10` | `66 / 90` |
| **Trips with Timestamp Gaps $> 5\text{ s}$ / $> 30\text{ s}$** | `45 / 80` (`> 5s`), `28 / 80` (`> 30s`) | `2 / 10` (`> 5s`), `2 / 10` (`> 30s`) | `47 / 90` (`> 5s`), `30 / 90` (`> 30s`) |
| **Maximum Timestamp Gap ($\Delta t_{\max}$)** | `153,064.0 s` (~42.5 hrs) | `210.0 s` | `153,064.0 s` |
| **Trips with Repeated `.route` Arcs (Loops)** | `9 / 80` (`67` repeated occurrences) | `1 / 10` (`255` repeated occurrences) | `10 / 90` (`322` total repeated occurrences) |
| **Disconnected Transitions in `.route`** | `13 / 31,256` (`99.935%` connected) | `1 / 3,467` (`99.973%` connected) | `14 / 34,723` total transitions |

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
   - **First observation ($i = 0$) or non-positive time step ($\Delta t_i \le 0$):** `delta_time_s`, `gps_speed_mps`, `gps_bearing_deg`, and `bearing_difference_deg` are set to nullable `<NA>`.
   - **Large positive GPS timestamp dropouts ($\Delta t_i > 30\text{ s}$):** **`delta_time_s` retains the positive elapsed gap duration** (e.g., `46.0 s`, `160.0 s`) so downstream models and sequence decoders can detect that a dropout occurred, whereas **`gps_speed_mps`, `gps_bearing_deg`, and `bearing_difference_deg` are masked to `<NA>`** so finite-difference straight lines across signal dropouts never fabricate velocities or headings.
   - **Implausible speed spikes ($\Delta d_i / \Delta t_i > 65\text{ m/s}$):** Masked to `<NA>` in `gps_speed_mps`, `gps_bearing_deg`, and `bearing_difference_deg`.
   - **Stationary GPS jitter ($\Delta d_i < 2.5\text{ m}$):** `gps_speed_mps` remains valid (near `0 m/s`), while `gps_bearing_deg` and `bearing_difference_deg` are masked to `<NA>` because the azimuth of $< 2.5\text{ m}$ stationary GPS noise is random.

---

## Topic 3: Feature Engineering (11 Candidate Features)

Defined in `CANDIDATE_FEATURE_COLUMNS` ([`src/pathsnap/schemas/candidate_schema.py`](src/pathsnap/schemas/candidate_schema.py)) and extracted in [`src/pathsnap/spatial/candidate_generation.py`](src/pathsnap/spatial/candidate_generation.py):

| # | Feature Column | Dtype | Nullable | Nulls in 5-Trip Preview (`98,636` rows) | Physical Meaning & Why It Is Used |
| :-: | :--- | :--- | :---: | :---: | :--- |
| 1 | `distance_to_segment_m` | `float32` | No | `0` | Shortest perpendicular/clamped Euclidean distance (m) from GPS point to the candidate road segment. Primary spatial proximity signal. |
| 2 | `projection_fraction` | `float32` | No | `0` | Orthogonal projection parameter $\alpha \in [0.0, 1.0]$ along the directed arc ($0 = \text{start node}$, $1 = \text{end node}$). Identifies whether the vehicle is mid-link or at an intersection endpoint. |
| 3 | `distance_to_start_m` | `float32` | No | `0` | Distance (m) from GPS point to `from_node_id`. Helps disambiguate entry into a new segment at intersections. |
| 4 | `distance_to_end_m` | `float32` | No | `0` | Distance (m) from GPS point to `to_node_id`. Indicates proximity to the outgoing junction. |
| 5 | `segment_length_m` | `float32` | No | `0` | Physical length (m) of the candidate road segment. Prevents tiny intersection connector arcs from dominating long arterial segments. |
| 6 | `segment_bearing_deg` | `float32` | No | `0` | Directed road azimuth $\theta_{\text{arc}} \in [0, 360)^\circ$ clockwise from True North. Distinguishes one-way vs opposing carriageways. |
| 7 | `gps_bearing_deg` | `Float32` | **Yes** | `6,257` | Causal vehicle movement azimuth $\theta_{\text{gps}} \in [0, 360)^\circ$ from $i-1 \to i$ (`<NA>` when $i=0$, $\Delta t_i \le 0$, $\Delta t_i > 30\text{ s}$, speed $> 65\text{ m/s}$, or $\Delta d_i < 2.5\text{ m}$). |
| 8 | `bearing_difference_deg` | `Float32` | **Yes** | `6,257` | Smallest unsigned angle $|\theta_{\text{gps}} - \theta_{\text{arc}}| \in [0, 180]^\circ$. Crucial for rejecting opposing lanes ($\approx 180^\circ$) and perpendicular cross-streets ($\approx 90^\circ$). |
| 9 | `gps_speed_mps` | `Float32` | **Yes** | `235` | Causal speed $\Delta d_i / \Delta t_i$ (m/s) (`<NA>` when $i=0$, $\Delta t_i \le 0$, $\Delta t_i > 30\text{ s}$, or speed $> 65\text{ m/s}$). |
| 10 | `delta_time_s` | `Float32` | **Yes** | `43` | Elapsed time $t_i - t_{i-1}$ (s) (`<NA>` only when $i=0$ or $\Delta t_i \le 0$; retains positive gap durations $> 30\text{ s}$). |
| 11 | `is_fallback_candidate` | `bool` | No | `0` (`271` `True`) | `True` if zero arcs existed within `search_radius_m` and the candidate was retrieved via nearest-neighbour fallback. |

- **Strict Identifier Exclusion:** `trip_id`, `partition`, `observation_id`, `timestamp_s`, `candidate_rank`, `candidate_arc_id`, `from_node_id`, and `to_node_id` are stored in `CANDIDATE_ID_COLUMNS` and **never** passed to `model.fit()` or `model.predict_proba()`.

---

## Topic 4: Candidate-Generation Strategy & Sensitivity Analysis

Implemented in [`src/pathsnap/spatial/candidate_generation.py`](src/pathsnap/spatial/candidate_generation.py) and [`src/pathsnap/spatial/road_network.py`](src/pathsnap/spatial/road_network.py):

1. **Spatial Indexing via Shapely 2.x `STRtree`:**
   - Each trip has up to $307,392$ directed arcs. Brute-force $O(N_{\text{obs}} \times N_{\text{arcs}})$ distance calculation would require $> 6 \times 10^{10}$ checks.
   - We build a C-accelerated Sort-Tile-Recursive R-tree (`shapely.STRtree`) over the projected metric road segments once per trip, querying `predicate="dwithin"` of radius $R = \text{search\_radius\_m}$ around all GPS points simultaneously in $< 0.2\text{ s}$ per trip.
2. **Exact Segment Filtering, Top-$K$ Ranking & Fallback Nearest Search:**
   - For all tree hits within radius $R$, we compute exact vectorised point-to-segment distances $d(p_i, s_a)$, sort deterministically by `(observation_id ASC, distance_to_segment_m ASC, candidate_arc_id ASC)`, and retain up to $K = \text{max\_candidates\_k}$ unique `candidate_arc_id` rows (`candidate_rank = 1 .. K`).
   - **Fallback Nearest-Neighbour Search:** If a GPS point has $0$ arcs within $R = 100\text{ m}$ (e.g., off-route detour), `STRtree.query_nearest` retrieves the single nearest road segment up to `fallback_max_radius_m = 500 m` and flags `is_fallback_candidate = True`.
3. **Actual Executed Sensitivity Analysis ([`data/sensitivity_analysis.csv`](data/sensitivity_analysis.csv) — 10 Training Trips, `23,178` Observations, `22,193` Aligned):**

| Search Radius $R$ (m) | Max Candidates $K$ | Candidate Miss Count | **Candidate Miss Rate** | **Candidate Recall** | Avg Candidates / Obs | Positive Class Ratio | Query Time (10 Trips) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `30 m` | `5` | `782` | `3.5236%` | `96.4764%` | `3.844` | `24.03%` | `0.894s` |
| `30 m` | `10` | `288` | `1.2977%` | `98.7023%` | `5.440` | `17.37%` | `0.894s` |
| `50 m` | `5` | `592` | `2.6675%` | `97.3325%` | `4.316` | `21.59%` | `1.150s` |
| `50 m` | `10` | `60` | `0.2704%` | `99.7296%` | `7.079` | `13.49%` | `1.150s` |
| **`100 m` (Selected)** | **`10` (Selected)** | **`60`** | **`0.2704%`** | **`99.7296%`** | **`8.478`** | **`11.26%`** | **`1.921s`** |
| `100 m` | `15` | `9` | `0.0406%` | `99.9594%` | `11.678` | `8.20%` | `1.921s` |
| `150 m` | `20` | `2` | `0.0090%` | `99.9910%` | `16.211` | `5.91%` | `2.909s` |

- **Why $R = 50\text{ m}$ and $R = 100\text{ m}$ Have Identical Miss Counts (`60`) at $K = 10$, and Why $R = 100\text{ m}$ Is Defensible:**
  - By definition of `align_reference_route`, an observation is classified as `"aligned"` **only if** its distance to the aligned reference route segment is $\le \text{alignment\_max\_confident\_distance\_m} = 50.0\text{ m}$ (observed max across the 5 preview trips is `49.80 m`). Therefore, any radius $R \ge 50.0\text{ m}$ already contains $100\%$ of `"aligned"` reference arcs! The remaining `60` misses at $K = 10$ (`9` at $K=15$, `2` at $K=20$) are purely **rank-cap misses** at dense urban intersections where more than $K$ short road segments lie closer to the GPS point than the reference arc.
  - Why keep $R = 100\text{ m}$ instead of $50\text{ m}$? During blind inference and sequential Viterbi decoding, GPS points occasionally drift $50\text{--}100\text{ m}$ from the road network (in our 5-trip preview, `510` observations have their closest road $> 50\text{ m}$ away, whereas only `271` exceed $100\text{ m}$). Under $R = 50\text{ m}$, all `510` observations would collapse to a single `query_nearest` fallback arc (`K=1`), depriving the Viterbi decoder of alternative connected road segments. Setting $R = 100\text{ m}, K = 10$ preserves multi-candidate Viterbi trellis branching in the $50\text{--}100\text{ m}$ drift zone at negligible extra query cost (`0.19 s/trip` vs `0.12 s/trip`), while $R = 50\text{ m}, K = 10$ remains an equally valid faster alternative for pure training-row extraction.

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
   - *Limitation:* High variance across trips and coarse piecewise-constant leaf probabilities (`0.6159` OOF Log Loss).
4. **Random Forest Classifier (Bagging Ensemble Family):**
   - *Suitability:* Averages $120$ decorrelated bootstrap trees (`max_depth=12`, `min_samples_leaf=15`), reducing single-tree variance and achieving the lowest **OOF Log Loss (`0.2102`)** on our 5-trip pilot.
   - *Limitation:* Higher memory footprint and slower inference (`0.42s`) than single boosted trees or linear models.
5. **HistGradientBoosting Classifier (Gradient Boosting Family):**
   - *Suitability:* Sequentially minimizes binary log-loss using histogram-binned additive trees; **natively routes `NaN` values** in `gps_bearing_deg` and `gps_speed_mps` without imputation distortion; achieves the highest **Mean Fold Macro F1 (`0.8013`)** and **OOF Macro F1 (`0.7445`)** on our 5-trip pilot.
   - *Limitation:* On small unbalanced trip folds (such as Fold 4 where Trip `00000002` is held out), uncalibrated boosted log-odds can be overconfident (`0.2759` OOF Log Loss vs `0.2378` for Logistic Regression).

> **Critical Scope & Imbalance Note (5-Trip Pilot vs Full 80-Trip Training):**
> - **Executed in Phase 1:** 5-fold `GroupKFold` cross-validation on the **5 representative training trips (`00000001`–`00000005`, `92,299` labeled rows)** saved in [`results/phase1/model_comparison.csv`](results/phase1/model_comparison.csv).
> - **Effect of Trip `00000002` Size Asymmetry:** Trip `00000002` contains `59,156` of the `92,299` labeled rows (`64.1%` of the 5-trip pilot). In Fold 4, when Trip `00000002` is the validation fold, models train on only `33,143` rows (`35.9%`) from the 4 smaller trips (`00000001`, `00000003`, `00000004`, `00000005`), causing lower recall and lower pooled Out-of-Fold (`oof_*`) scores compared to the unweighted mean across folds (`cv_*_mean`). Furthermore, every ML model substantially beats the **Nearest-Segment (`candidate_rank == 1`) Baseline (`52.89%` aligned Top-1 accuracy)** on the 5 trips (Logistic Regression reaches `65.00%` greedy and `84.99%` with Viterbi).
> - **Scheduled for Phase 2:** Running the exact same pipeline across all **80 training trips** (~`1.5M` candidates, where each 16-trip validation fold is balanced at ~`20%` of the data) and evaluating on the **10 held-out `student_test` trips**.

---

## Topic 6: Training-Label Alignment (`align_reference_route`)

Implemented in [`src/pathsnap/alignment/route_alignment.py`](src/pathsnap/alignment/route_alignment.py) and [`src/pathsnap/alignment/label_generation.py`](src/pathsnap/alignment/label_generation.py):

1. **Why Monotonic Dynamic Programming Over Route Positions $j \in \{0, \dots, M-1\}$?**
   - `.route` is a sequence of $M$ traversed arcs ($M \ll N_{\text{obs}}$), and $10 / 90$ trips ($322$ repeated arc occurrences) traverse the same `arc_id` multiple times (loops/out-and-back trajectories). Indexing DP states by **route occurrence index $j$** (rather than unique `arc_id`) preserves exact chronological ordering and handles loops cleanly.
2. **Emission Cost, Transition Cost & Backtracking:**
   - **Emission Cost $E(i, j)$:** Normalized squared distance from GPS point $i$ to route segment $j$, plus directional bearing penalty when GPS heading is valid:
     $$E(i, j) = \left(\frac{d(p_i, r_j)}{\sigma}\right)^2 + w_{\text{brg}} \left(\frac{|\Delta \theta_{i,j}|}{180^\circ}\right)^2 \quad (\sigma = 15\text{ m},\; w_{\text{brg}} = 5.0)$$
   - **Monotonic Transition Cost $T(j_{\text{prev}} \to j_{\text{curr}})$ for $j_{\text{curr}} \ge j_{\text{prev}}$:**
     - If $j_{\text{curr}} = j_{\text{prev}}$ (staying on the same route segment): zero skip penalty, plus a quadratic penalty $((-\Delta s - 5.0)/\sigma)^2$ only if orthogonal projection moves backward along the arc by $> 5.0\text{ m}$.
     - If $j_{\text{curr}} > j_{\text{prev}}$ (advancing $\Delta j$ route segments):
       $$T(j_{\text{prev}} \to j_{\text{curr}}) = \frac{\lambda_{\text{skip}}}{\sqrt{\max(1, \Delta t_i)}}(\Delta j - 1) + \frac{\lambda_{\text{len}}}{\sigma}\left|\Delta s_{\text{route}}(j_{\text{prev}}, j_{\text{curr}}) - \Delta d_{\text{gps}, i}\right|$$
   - **Backtracking:** Recovers the globally optimal non-decreasing sequence $0 \le j_0 \le j_1 \le \dots \le j_{N-1} < M$ (`is_monotonic == True` verified across all preview trips).
3. **Four Alignment Statuses (`alignment_status`):**
   - `"aligned"`: Minimum DP alignment distance $d_i \le 50\text{ m}$ and unambiguous (`10,618 / 11,488` observations in the 5-trip preview).
   - `"ambiguous"`: $d_i \le 50\text{ m}$, $d_i > 20\text{ m}$, and a non-adjacent route segment ($|j - j^*| > 2$) with a different `arc_id` lies within `3.0 m` of $d_i$ (`2` observations in the 5-trip preview).
   - `"unmatched"`: $d_i > 50\text{ m}$—for example, in Trip `00000002`, the vehicle continues driving for **`868` observations** after the supplied `.route` ends (`alignment_distance_m` up to `569.8 m`, p95 across trip `242.4 m`). Flagging these `"unmatched"` prevents corrupting training labels!
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
     Determines the theoretical upper bound ($1 - \text{Miss Rate} = 99.8022\%$ on the 5 preview trips) for downstream pointwise classification.
2. **Candidate-Row Classifier Metrics (Evaluated on `label_eligible` Candidate Rows):**
   - **Accuracy, Precision (Positive Class), Recall (Positive Class), Confusion Matrix `(TN, FP, FN, TP)`**.
   - **Macro F1-Score:** Unweighted mean of negative-class F1 and positive-class F1; essential because positive candidates are `11.48%` of labeled rows (a trivial all-zero predictor gets `88.52%` accuracy but only `0.4695` Macro F1).
   - **Binary Log Loss (Cross-Entropy):** Measures probability calibration quality. Because the HMM Viterbi decoder sums $\log P(y_{i,c}=1 \mid \mathbf{x}_{i,c})$ across time steps, poorly calibrated overconfident probabilities ($P \approx 0$ or $1$) degrade Viterbi path scoring.
3. **Observation-Level & Sequence-Level Map-Matching Metrics:**
   - **Observation-Level Top-1 Accuracy:** We report both:
     - `obs_top1_accuracy` (over all `10,618` `"aligned"` observations, penalized by candidate misses), and
     - `obs_top1_accuracy_eligible` (over the `10,597` `label_eligible` observations where the true arc is present in the candidate set).
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

## Mandatory Status, Limitations & Trade-Offs of Bayesian Optimisation and HMM/Viterbi

1. **Bayesian Hyperparameter Optimisation (`src/optimization/bayesian_tuning.py`):**
   - *Methodology:* Uses **`optuna.samplers.TPESampler(seed=42, n_startup_trials=3)`** to tune `HistGradientBoostingClassifier` over 6 hyperparameters using 5-fold `GroupKFold`.
   - *Important Methodological Limitation:* In this Phase 1 pilot, the 8 Optuna trials were evaluated on the same 5-trip `GroupKFold` split (`oof_macro_f1` improved from `0.744416` in Baseline Trial #1 to `0.745228` in Best Trial #7). Selecting the best trial on CV folds introduces slight selection optimism; unbiased generalisation of the tuned model vs. the untuned model will be measured on the **10 held-out `student_test` trips** after full 80-trip training in Phase 2.
2. **HMM & Viterbi Route Reconstruction (`src/sequence/viterbi_decoder.py`):**
   - *Methodology:* Combines log-emission probabilities $\log \hat{P}(y_{i,c}=1\mid\mathbf{x}_{i,c})$ with `TripRoadGraph` topological transition scores (`0`-hop same arc: `-0.15`, `1`-hop direct successor: `-0.60`, U-turn penalty: `-4.00`, `2`-hop: `-2.20`, `3`-hop: `-3.80`, disconnected: `-10.50`, with linear gap relaxation over `10..60s` and uniform reset $\ge 60\text{s}$).
   - *Honest Trade-Off Analysis (Why Viterbi Improves All 5 Trips with Logistic Regression, but Drops Pointwise Accuracy on Trip `00000002` with Uncalibrated HGB):*
     - With well-calibrated **Logistic Regression (OOF)** probabilities (`0.2378` Log Loss), Viterbi improves Observation Top-1 Accuracy on **every single trip**, boosting the 5-trip aggregate from **`65.00%` to `84.99%` (`+19.98%`)** and Route Continuity from **`86.84%` to `97.09%` (`+10.25%`)** (including Trip `00000002`: `52.24%` $\to$ `79.73%`).
     - With **Tuned HistGradientBoosting (OOF)** (`0.2728` Log Loss), Viterbi strongly boosts accuracy on Trips `00000001` (`75.36%` $\to$ `93.31%`), `00000003` (`76.49%` $\to$ `81.73%`), `00000004` (`56.39%` $\to$ `63.63%`), and `00000005` (`98.18%` $\to$ `99.33%`), and boosts Trip `00000002` Route Continuity from `83.87%` to `96.09%`. However, on Trip `00000002`, HGB's pointwise Top-1 accuracy drops slightly from **`47.45%` to `43.75%` (`-3.70%`)**. This occurs because in Fold 4 (where Trip `00000002` is held out), HGB is trained on only `33,143` rows from the 4 small trips and produces sharper, less-calibrated log-odds (`0.3177` fold Log Loss vs `0.2675` for Logistic Regression); when HGB emits an overconfident score on an alternative connected branch at an intersection, Viterbi locks onto that connected parallel path for several steps to preserve topological continuity (`96.09%`). This directly motivates **full 80-trip training + post-hoc probability/temperature calibration** in Phase 2.
