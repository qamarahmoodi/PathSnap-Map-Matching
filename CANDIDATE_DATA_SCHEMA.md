# Candidate Data Schema & Interface Specification (`PROVISIONAL`)

> [!IMPORTANT]
> **STATUS: PROVISIONAL — AWAITING APPROVAL FROM QAMAR AHMED**  
> This document defines the exact data contract between **Candidate Generation & Label Assignment** (owned by Qamar Ahmed, coordinated with L A Gokul and Pranav S Atreya) and **ML Model Experimentation, Bayesian Optimisation, and HMM Emission Scoring** (owned by Pranav S Atreya).  
> **No supervised model training will execute until Qamar reviews and confirms this schema and the label-generation methodology.**

---

## 1. Fundamental Scientific Constraints

1. **No Direct Index Alignment:** In raw trip data, `<trip_id>.track` has $T$ GPS observations (e.g., 1,315 rows in trip `00000001`), whereas `<trip_id>.route` has $R$ chronological traversed arc IDs (e.g., 525 rows in trip `00000001`). Row $i$ of `.track` **never** corresponds directly to row $i$ of `.route`.
2. **Trip-Local Arc IDs:** In `<trip_id>.arcs`, the 0-based row index is the `arc_id` for **that specific trip only** (e.g., `arc_id = 211` in trip `00000000` is a completely different physical road segment from `arc_id = 211` in trip `00000001`). Therefore:
   - `arc_id` must **never** be used as a raw numerical feature or a global multi-class target label.
   - Instead, map-matching must be formulated as **Candidate Ranking / Binary Candidate Classification** over $(t, k)$ pairs (GPS observation $t$, candidate road segment $k$ within search radius $R_{\text{search}}$), where each candidate is represented by **trip-invariant geometric, kinematic, and topological features**.
3. **Group-Aware Partitioning:** All candidate rows belonging to the same `trip_id` must stay together in the same split (`train` vs. `student_test`, and within `GroupKFold` cross-validation folds).

---

## 2. Candidate Table Granularity

Each row in the candidate dataset represents **one candidate road segment $k$ considered for one GPS observation $t$ in one trip (`trip_id`)**.

- Let trip $m$ have $T_m$ GPS observations ($t \in \{0, 1, \dots, T_m - 1\}$).
- For each GPS observation $t$, the candidate generator selects up to $K_{\max}$ candidate directed arcs within spatial radius $R_{\text{search}}$ (e.g., $R_{\text{search}} = 50\text{ m}$ to $100\text{ m}$, $K_{\max} = 5$ to $10$).
- If a GPS observation has **zero** arcs within $R_{\text{search}}$, either:
  - Fall back to the top-$K_{\text{fallback}}$ nearest arcs regardless of radius (flagging `is_fallback_candidate = 1`), OR
  - Record the observation in the trip metadata so **Candidate Miss Rate** is tracked accurately.

---

## 3. Required Identifier & Metadata Columns

These columns identify the observation and map candidate predictions back to the trip's road network for evaluation and HMM/Viterbi decoding. **They are NEVER passed as input features to the ML classifier.**

| Column Name | Data Type | Description | Required? |
| :--- | :--- | :--- | :--- |
| `partition` | `str` | Dataset split: `"train"`, `"student_test"`, or `"evaluator_test"` | Yes |
| `trip_id` | `str` | 8-character zero-padded trip identifier (e.g., `"00000001"`) | Yes |
| `obs_idx` | `int32` | 0-based chronological index $t \in [0, T_m - 1]$ of the GPS point in `<trip_id>.track` | Yes |
| `timestamp` | `float64` | Raw timestamp (seconds) from column 3 of `<trip_id>.track` | Yes |
| `gps_lon` | `float64` | Raw GPS longitude from column 1 of `<trip_id>.track` | Yes |
| `gps_lat` | `float64` | Raw GPS latitude from column 2 of `<trip_id>.track` | Yes |
| `candidate_rank` | `int16` | 0-based spatial distance rank $k \in [0, K_t - 1]$ among candidates at `obs_idx` | Yes |
| `arc_id` | `int32` | 0-based row index of the candidate road segment in `<trip_id>.arcs` | Yes |
| `from_node_id` | `int32` | Start node ID (`col 0` of `<trip_id>.arcs`) for topological transition checks | Yes |
| `to_node_id` | `int32` | End node ID (`col 1` of `<trip_id>.arcs`) for topological transition checks | Yes |

---

## 4. Target / Label Column (`train` and `student_test` Only)

| Column Name | Data Type | Description | Availability |
| :--- | :--- | :--- | :--- |
| `label` | `int8` | `1` if `arc_id` is the aligned reference road segment for GPS observation `obs_idx`, else `0` | `train`, `student_test` only (`None`/`NaN` in live API inference) |
| `ref_arc_id` | `int32` | The aligned reference `arc_id` for observation `obs_idx` (or `-1` if unaligned / gap). Used to compute **Candidate Miss Rate** even when the true arc is not in the candidate set. | `train`, `student_test` only |

> [!WARNING]
> **Critical Alignment Requirement for Qamar:**  
> Because `.route` is a sequence of unique/chronological traversed arcs (length $R_m \ll T_m$) rather than a per-timestamp label vector of length $T_m$, **how each GPS observation `obs_idx` is assigned its reference `ref_arc_id` from `<trip_id>.route`** must be explicitly agreed upon before training!  
> Common valid approaches include:
> 1. **Monotonic Sequence Alignment (Dynamic Time Warping / Subsequence Viterbi on `.route`):** Project each GPS observation onto the ordered subgraph restricted to arcs in `<trip_id>.route` enforcing monotonic progression along the `.route` sequence.
> 2. **Spatial Projection onto the Reference Route Polyline:** Project each GPS point $(x_t, y_t)$ onto the geometric polyline formed by the ordered sequence of arcs in `.route` (with topological continuity constraint so back-and-forth loops aren't misassigned).
>
> **Anti-Fabrication Rule:** `label` must ALWAYS come from `<trip_id>.route` alignment—never from heuristics on candidate features (e.g., never label the closest candidate `candidate_rank == 0` as `1` without `.route`).

---

## 5. Candidate Feature Columns (`X` Matrix Input to ML Models)

All features must be **trip-invariant**, numeric (`float32`/`float64`), and computed solely from `.track`, `.nodes`, and `.arcs` (never using `.route`).

### A. Spatial & Geometric Features (Point-to-Segment)
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `dist_point_to_arc_m` | `float32` | Orthogonal / geodesic distance (in meters) from GPS observation $(lon_t, lat_t)$ to the candidate road segment `arc_id` |
| `dist_to_from_node_m` | `float32` | Distance (meters) from GPS point to the candidate arc's start node (`from_node_id`) |
| `dist_to_to_node_m` | `float32` | Distance (meters) from GPS point to the candidate arc's end node (`to_node_id`) |
| `arc_length_m` | `float32` | Physical length (meters) of the candidate road segment |
| `proj_fraction` | `float32` | Normalized projection position along the directed arc $\alpha \in [0.0, 1.0]$ ($0 = \text{from\_node}$, $1 = \text{to\_node}$) |
| `is_endpoint_projection` | `int8` | `1` if the closest point on the segment is clamped to either endpoint ($\alpha = 0$ or $\alpha = 1$), else `0` |
| `dist_ratio_to_best` | `float32` | `dist_point_to_arc_m / (min_dist_at_obs + 1e-3)` — relative distance compared to the closest candidate at timestamp $t$ |
| `dist_diff_from_best_m` | `float32` | `dist_point_to_arc_m - min_dist_at_obs` — excess distance (meters) over the closest candidate at timestamp $t$ |
| `dist_zscore_at_obs` | `float32` | Standardized distance within the candidate set at timestamp $t$ |
| `num_candidates_at_obs` | `int16` | Total number of candidates $K_t$ found within search radius at timestamp $t$ |

### B. Kinematic & Heading Alignment Features
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `gps_speed_mps` | `float32` | Estimated instantaneous vehicle speed (m/s) from finite difference between $t-1$ and $t$ (or $t$ and $t+1$) |
| `gps_bearing_deg` | `float32` | Estimated movement bearing $\theta_{\text{gps}} \in [0^\circ, 360^\circ)$ between consecutive GPS observations |
| `arc_bearing_deg` | `float32` | Directed bearing $\theta_{\text{arc}} \in [0^\circ, 360^\circ)$ from `from_node_id` to `to_node_id` |
| `abs_heading_diff_deg` | `float32` | Minimal angular difference $|\Delta \theta| \in [0^\circ, 180^\circ]$ between `gps_bearing_deg` and `arc_bearing_deg` |
| `cos_heading_diff` | `float32` | $\cos(\Delta \theta) \in [-1.0, 1.0]$ (`+1` = moving parallel in same direction, `-1` = moving opposite to directed arc) |
| `sin_heading_diff` | `float32` | $|\sin(\Delta \theta)| \in [0.0, 1.0]$ (perpendicularity between GPS movement vector and road segment) |
| `dt_prev_s` | `float32` | Time elapsed $\Delta t = t_i - t_{i-1}$ (seconds) since previous GPS observation (`0.0` at $t=0$) |
| `dt_next_s` | `float32` | Time to next GPS observation $t_{i+1} - t_i$ (seconds) |

### C. Local Road-Network & Contextual Continuity Features
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `out_degree_to_node` | `int16` | Number of outgoing arcs from `to_node_id` in `<trip_id>.arcs` |
| `in_degree_from_node` | `int16` | Number of incoming arcs into `from_node_id` in `<trip_id>.arcs` |
| `has_reverse_arc` | `int8` | `1` if `(to_node_id, from_node_id)` also exists in `<trip_id>.arcs` (bidirectional road), else `0` |
| `connected_to_prev_top1` | `int8` | `1` if any candidate at $t-1$ shares `arc_id` or connects into `from_node_id` of this candidate, else `0` |
| `min_topological_hops_from_prev` | `int16` | Minimum directed graph hops from any candidate at $t-1$ to this candidate at $t$ (`0` = same arc, `1` = adjacent arc, `2` = 2-hop, `99` = disconnected/exceeds bound) |

---

## 6. Model Output & HMM Decoder Interface

Once trained, Pranav's ML pipeline and inference wrapper will accept a DataFrame (or dictionary of arrays) conforming to Sections 3 & 5 and output:

1. **Emission / Candidate Scoring Output (`predict_candidates(df_candidates)`):**
   Returns the input identifier columns plus:
   - `pred_prob`: Calibrated probability $P(y_{t,k} = 1 \mid \mathbf{x}_{t,k}) \in (0, 1)$ for each candidate row.
   - `pred_log_prob`: Log emission probability $\log P(y_{t,k} = 1 \mid \mathbf{x}_{t,k})$ for direct consumption by the log-domain Viterbi decoder.
   - `pred_label`: Binary thresholded prediction (`0` or `1`) for candidate-level classification metrics.
   - `is_obs_top1`: `1` for the highest-scoring candidate at each `(trip_id, obs_idx)`, else `0` (Classifier-Only greedy baseline).

2. **Sequence Decoding Interface (`decode_route(df_candidates, trip_arcs, trip_nodes, hmm_config)`):**
   Returns:
   - `obs_predictions`: DataFrame with one row per `(trip_id, obs_idx)` containing `pred_arc_id_greedy` (Classifier-Only) and `pred_arc_id_viterbi` (Classifier + HMM/Viterbi).
   - `reconstructed_route`: Deduplicated / connected chronological sequence of `arc_id` values representing the reconstructed vehicle trajectory.
