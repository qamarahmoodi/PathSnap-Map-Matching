# PathSnap Phase 2 — HMM & Viterbi Route Reconstruction Handoff

**Owner:** Qamar Ahmed (Technical Lead)
**Target Consumers:** L A Gokul (Common Evaluation & Reporting), Pranav Shashikiran Atreya (ML Integration & UI)
**Implementation:** [`src/sequence/viterbi_decoder.py`](../src/sequence/viterbi_decoder.py), [`src/pathsnap/graph/road_graph.py`](../src/pathsnap/graph/road_graph.py)

---

## 1. Hidden-State Representation

For a trip with $N_{\text{obs}}$ chronological GPS observations ($t \in \{0, 1, \dots, N_{\text{obs}}-1\}$):
- At observation $t$, the hidden state space $\mathcal{S}_t = \{c_{t,1}, c_{t,2}, \dots, c_{t,K_t}\}$ consists of the $K_t \le K$ candidate road segments retrieved around GPS point $t$.
- Every hidden state $c_{t,k}$ explicitly retains its reversible mapping to:
  - `trip_id` (`string`, 8-digit zero-padded)
  - `observation_id` (`int32`, $0 \dots N_{\text{obs}}-1$)
  - `candidate_arc_id` (`int32`, exact row index in `<trip_id>.arcs`)
  - `candidate_rank` (`int16`, $1 \dots K_t$)
  - `predicted_probability` (`float64` $\in [0, 1]$, from the trained classifier)

---

## 2. Emission Probability Model

1. **Classifier Probability Input:** The decoder consumes the positive-class posterior probability $p_{t,k} = P(y_{t,k}=1 \mid \mathbf{x}_{t,k})$ produced by `CandidateModelScorer.score_candidates(...)`.
2. **Validation & Numerical Floor:**
   - Rejects `NaN`, `Inf`, or out-of-range ($< 0$ or $> 1$) scores with a `ValueError`.
   - Clips probabilities to $[\epsilon_{\text{floor}}, 1 - \epsilon_{\text{floor}}]$ where `prob_floor` $= 10^{-6}$ so $\log(p_{t,k})$ never underflows to $-\infty$.
   - Supports optional within-observation normalization (`normalize_emissions_within_obs = False` by default, preserving raw calibrated model likelihoods).
3. **Log-Emission Formula:**
   $$\log B(c_{t,k}) = w_{\text{emit}} \cdot \log\big(\text{clip}(p_{t,k}, 10^{-6}, 1 - 10^{-6})\big) - 10^{-9} \cdot d_{\text{seg}}(t, k)$$
   where the $10^{-9} \cdot d_{\text{seg}}(t, k)$ term provides a deterministic physical tie-breaker favoring the closer segment when two candidates have identical probabilities.

---

## 3. Topological & Physical Transition Model (`compute_log_transition_matrix`)

Using `TripRoadGraph` (`arc_endpoints`, `arc_lengths_m`, `outgoing_arcs_by_node`), the log-transition score $\log A(u \to v \mid \Delta t_t)$ between candidate arc $u$ at step $t-1$ and candidate arc $v$ at step $t$ is defined by directed road-network reachability:

| Transition Case | Topological Condition in `TripRoadGraph` | Base Log-Transition Score | Rationale |
|---|---|---|---|
| **Same Road Segment (`hop = 0`)** | $u == v$ | `same_arc_log_prob = -0.15` | At $1\text{ Hz}$ sampling, vehicles remain on the same road segment across multiple consecutive seconds. |
| **Direct Outgoing Successor (`hop = 1`)** | $\text{to\_node}(u) == \text{from\_node}(v)$ and $\text{to\_node}(v) \ne \text{from\_node}(u)$ | `direct_successor_log_prob = -0.60` | Smooth forward progression across an intersection node. |
| **Immediate U-Turn (`hop = 1`, reverse)** | $\text{to\_node}(u) == \text{from\_node}(v)$ and $\text{to\_node}(v) == \text{from\_node}(u)$ | `-0.60 + (-4.00) = -4.60` | Penalizes oscillating back-and-forth onto the opposing carriageway of a bidirectional road. |
| **2-Hop Connected (`hop = 2`)** | $\exists w \in \text{succ}(u)$ s.t. $v \in \text{succ}(w)$ | `two_hop_log_prob = -2.20` | Permits skipping a short intermediate intersection connector arc across a $1\text{ s}$ step. |
| **3-Hop Connected (`hop = 3`)** | Shortest directed path is 3 arc transitions | `three_hop_log_prob = -3.80` | Handles multi-segment progression across higher speeds or minor $2\text{--}5\text{ s}$ sampling intervals. |
| **Disconnected (`hop = -1`)** | No directed path within `max_hops = 3` | `disconnected_log_prob = -10.50` | Strongly suppresses jumping across buildings or onto parallel disconnected streets, while acting as a controlled backoff if all candidate pairs are disconnected. |

---

## 4. Viterbi Dynamic Programming Recurrence

In the log domain, for each observation $t$ with $K_t > 0$ candidates:
1. **Initialization (first valid step $t_0$):**
   $$V_{t_0}(k) = \log B(c_{t_0, k}), \qquad \text{bp}_{t_0}(k) = -1$$
2. **Forward Recursion ($t > t_0$, previous valid step $t_{\text{prev}}$):**
   $$V_t(k) = \log B(c_{t, k}) + \max_{i \in \{1, \dots, K_{t_{\text{prev}}}\}} \Big[ V_{t_{\text{prev}}}(i) + \log A\big(c_{t_{\text{prev}}, i} \to c_{t, k} \mid \Delta t\big) \Big]$$
   $$\text{bp}_t(k) = \arg\max_{i \in \{1, \dots, K_{t_{\text{prev}}}\}} \Big[ V_{t_{\text{prev}}}(i) + \log A\big(c_{t_{\text{prev}}, i} \to c_{t, k} \mid \Delta t\big) \Big]$$
3. **Termination & Backtracking:**
   $$k^*_{t_{\text{last}}} = \arg\max_k V_{t_{\text{last}}}(k), \qquad k^*_{t_{\text{prev}}} = \text{bp}_t(k^*_t)$$

---

## 5. Explicit Handling of Missing Candidates, Fallback Candidates & GPS Gaps

Every observation in `observation_predictions` and `hmm_viterbi_predictions` is assigned an explicit `decoder_status` and `gap_status`:

| Edge Case | How the Decoder Handles It | `decoder_status` | `gap_status` |
|---|---|---|---|
| **Normal Connected Decoding** | Standard Viterbi transition (`hop` $\in \{0, 1, 2, 3\}$) on non-fallback candidate | `"normal"` | `"normal"` or `"moderate_gap"` |
| **Fallback Candidate Selected** | Candidate retrieved via `STRtree.query_nearest` ($> 100\text{ m}$ from road) is selected | `"fallback_decoded"` | `"normal"` / `"moderate_gap"` |
| **Moderate GPS Time Gap ($10\text{ s} < \Delta t < 60\text{ s}$)** | Transition penalties are linearly relaxed by factor $\max(0.25, 1 - (\Delta t - 10)/60)$ | `"normal"` / `"disconnected_bridge"` | `"moderate_gap"` |
| **Large GPS Dropout ($\Delta t \ge 60\text{ s}$)** | Transition matrix resets to uniform $-\log(K_t)$ so the path starts cleanly at the new location | `"gap_reset"` | `"reset_gap"` |
| **Empty Candidate Set ($K_t = 0$)** | Carries forward the previously decoded `arc_id` (or `-1` if at start of trip) | `"carried_forward"` or `"unmatched"` | `"normal"` / `"moderate_gap"` |
| **Disconnected Transition Forced (`hop = -1`)** | No connected candidate pair exists between $t-1$ and $t$; uses backoff `-10.50` | `"disconnected_bridge"` | `"normal"` / `"moderate_gap"` |

---

## 6. Output Schemas for Gokul's Evaluation Framework

`run_trip_inference(...)` produces two standardized DataFrames (plus sample exports in `results/phase2/`):

### A. `classifier_only_predictions` DataFrame
| Column | Dtype | Description |
|---|---|---|
| `trip_id` | `string` | 8-digit trip ID |
| `observation_id` | `int32` | 0-based observation index |
| `timestamp_s` | `float64` | GPS timestamp in seconds |
| `predicted_arc_id` | `int32` | Highest-scoring candidate `arc_id` at observation $t$ |
| `model_score` | `float64` | Raw positive-class probability $P(y=1\mid\mathbf{x})$ |
| `prediction_status` | `string` | `"normal"`, `"fallback_candidate"`, or `"unmatched"` |
| `is_connected_step` | `bool` | Whether `predicted_arc_id[t-1]` connects to `predicted_arc_id[t]` |

### B. `hmm_viterbi_predictions` DataFrame
| Column | Dtype | Description |
|---|---|---|
| `trip_id` | `string` | 8-digit trip ID |
| `observation_id` | `int32` | 0-based observation index |
| `timestamp_s` | `float64` | GPS timestamp in seconds |
| `predicted_arc_id` | `int32` | Viterbi decoded `arc_id` at observation $t$ |
| `decoded_score` | `float64` | Model emission probability of the Viterbi-selected state |
| `decoder_status` | `string` | `"normal"`, `"fallback_decoded"`, `"gap_reset"`, `"carried_forward"`, `"disconnected_bridge"`, `"unmatched"` |
| `gap_status` | `string` | `"initial"`, `"normal"`, `"moderate_gap"`, `"reset_gap"` |
| `transition_hops` | `int32` | Directed graph hops from step $t-1$ (`0`, `1`, `2`, `3`, or `-1` if disconnected) |
| `is_connected_step` | `bool` | Whether `predicted_arc_id[t-1]` connects to `predicted_arc_id[t]` |
