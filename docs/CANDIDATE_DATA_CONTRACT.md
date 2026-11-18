# PathSnap — Candidate Data Contract & Interface Specification

**Owner:** Qamar Ahmed (Technical Lead)
**Consumers:** Pranav Shashikiran Atreya (ML Engineering), L A Gokul (Data Engineering & Visualisation)

---

## 1. Overview & Design Principles

PathSnap reformulates GPS map-matching as a two-stage structured prediction problem:
1. **Stage 1 (Pointwise Candidate Scoring):** For each GPS observation $i$ in trip $T$, spatial indexing (`Shapely 2.x STRtree` over a local WGS84 tangent-plane metric projection) retrieves up to $K$ nearby directed road segments (`candidate_arc_id`). A supervised classifier predicts the probability $P(y_{i,c} = 1 \mid \mathbf{x}_{i,c})$ that candidate arc $c$ is the true traversed road segment.
2. **Stage 2 (Sequential HMM + Viterbi Decoding):** Classifier emission probabilities are combined with directed road-network topology (`TripRoadGraph`) to decode a globally connected sequence of road segments via the Viterbi algorithm.

### Strict Training / Inference Separation
- `generate_trip_candidates(...)` **never accesses `.route`**. It works identically during training, validation, and blind evaluation.
- `align_reference_route(...)` and `attach_alignment_labels(...)` are **separate, optional operations** used only when reference routes are permitted (training and development-stage evaluation).

---

## 2. Primary Output 1: `candidates.parquet`

One row represents **one `(GPS observation, candidate road arc)` pair**.

- **Unique Composite Key:** `(trip_id, observation_id, candidate_arc_id)`
- **Ordering:** Sorted deterministically by `(trip_id, observation_id, candidate_rank)` where `candidate_rank = 1` is the closest segment in metres.
- **Deduplication:** Repeated `candidate_arc_id` values within the same `(trip_id, observation_id)` are strictly prohibited and automatically deduplicated.

### Exact Column Schema (`CANDIDATE_SCHEMA_DTYPES`)

| Column | Pandas / Arrow Dtype | Nullable | Role | Definition & Units |
|---|---|---|---|---|
| `trip_id` | `string` | No | Identifier / Group Key | 8-character zero-padded trip ID (e.g., `"00000001"`). Must be used for `GroupKFold`. |
| `partition` | `string` | No | Metadata | Dataset split (`"train"`, `"student_test"`, `"eval"`). |
| `observation_id` | `int32` | No | Identifier | 0-based sequential GPS index within the trip (`0 .. N_obs-1`). |
| `timestamp_s` | `float64` | No | Metadata | Raw GPS observation timestamp in seconds. |
| `candidate_rank` | `int16` | No | Metadata | 1-based proximity rank within the observation (`1 .. K`), ordered by `distance_to_segment_m` ascending. **Do not use as an ML feature** (to avoid rank-shortcut bias; let the model learn from continuous geometric features). |
| `candidate_arc_id` | `int32` | No | Identifier | Trip-specific 0-based row index in `<trip_id>.arcs`. **Never use as an ML feature** (arc IDs have no cross-trip meaning). |
| `from_node_id` | `int32` | No | Graph Metadata | Trip-specific start node ID (`arcs[candidate_arc_id, 0]`). |
| `to_node_id` | `int32` | No | Graph Metadata | Trip-specific end node ID (`arcs[candidate_arc_id, 1]`). |
| `distance_to_segment_m` | `float32` | No | **ML Feature** | Shortest Euclidean distance in metres from projected GPS point to the directed line segment. |
| `projection_fraction` | `float32` | No | **ML Feature** | Normalized orthogonal projection position along the directed segment, clamped to $[0.0, 1.0]$ ($0.0 =$ start node, $1.0 =$ end node). |
| `distance_to_start_m` | `float32` | No | **ML Feature** | Euclidean distance in metres from GPS observation to `from_node_id`. |
| `distance_to_end_m` | `float32` | No | **ML Feature** | Euclidean distance in metres from GPS observation to `to_node_id`. |
| `segment_length_m` | `float32` | No | **ML Feature** | Physical length of the directed road segment in metres. |
| `segment_bearing_deg` | `float32` | No | **ML Feature** | Directed road azimuth in $[0, 360)^\circ$ clockwise from True North ($0^\circ=\text{N}, 90^\circ=\text{E}$). |
| `gps_bearing_deg` | `Float32` (nullable) | **Yes** | **ML Feature** | Causal GPS movement azimuth from observation $i-1 \to i$ in $[0, 360)^\circ$. Null (`<NA>`) at `observation_id == 0`, when $\Delta t \le 0$, $\Delta t > 30\text{ s}$, or step displacement $< 2.5\text{ m}$. |
| `bearing_difference_deg` | `Float32` (nullable) | **Yes** | **ML Feature** | Smallest unsigned angle $|\text{gps\_bearing} - \text{segment\_bearing}| \in [0, 180]^\circ$. Null whenever `gps_bearing_deg` is null. |
| `gps_speed_mps` | `Float32` (nullable) | **Yes** | **ML Feature** | Causal point-to-point speed $\Delta d / \Delta t$ in $\text{m/s}$. Null at `observation_id == 0`, when $\Delta t \le 0$, $\Delta t > 30\text{ s}$, or when $\Delta d / \Delta t > 65\text{ m/s}$. |
| `delta_time_s` | `Float32` (nullable) | **Yes** | **ML Feature** | Backward time step $t_i - t_{i-1}$ in seconds. Null at `observation_id == 0` or when $\Delta t \le 0$ (retains positive elapsed gap values across dropouts $> 30\text{ s}$). |
| `is_fallback_candidate` | `bool` | No | **ML Feature** | `True` if no road existed within `search_radius_m` and this candidate was retrieved via fallback nearest-neighbour search. |
| `label` | `Int8` (nullable) | **Yes** | **Target ($y$)** | `1` if candidate matches confidently aligned reference arc; `0` for non-matching candidates of a `label_eligible` observation; `<NA>` during inference or when `label_eligible == False`. |

---

## 3. Primary Output 2: `observations.parquet`

Every GPS observation in `.track` appears as **exactly one row** in `observations.parquet`, even when zero candidates are generated or when alignment fails.

- **Unique Key:** `(trip_id, observation_id)`

### Exact Column Schema (`OBSERVATION_SCHEMA_DTYPES`)

| Column | Pandas / Arrow Dtype | Nullable | Definition |
|---|---|---|---|
| `trip_id` | `string` | No | 8-character zero-padded trip ID. |
| `partition` | `string` | No | Dataset split (`"train"`, `"student_test"`, `"eval"`). |
| `observation_id` | `int32` | No | 0-based observation index in `.track`. |
| `timestamp_s` | `float64` | No | GPS timestamp in seconds. |
| `gps_longitude` | `float64` | No | Raw WGS84 longitude in degrees. |
| `gps_latitude` | `float64` | No | Raw WGS84 latitude in degrees. |
| `reference_arc_id` | `Int32` (nullable) | **Yes** | Aligned arc ID from `.route` (`<NA>` when `alignment_status` is `unmatched` or `not_available`). |
| `reference_route_position` | `Int32` (nullable) | **Yes** | 0-based index into `.route` sequence (`<NA>` when `unmatched` or `not_available`). Distinguishes repeated traversals of the same arc ID. |
| `alignment_distance_m` | `Float32` (nullable) | **Yes** | Distance in metres from GPS point to the aligned reference route segment (`NaN` when `not_available`). |
| `alignment_status` | `string` | No | Exactly one of: `"aligned"`, `"ambiguous"`, `"unmatched"`, `"not_available"`. |
| `candidate_count` | `int16` | No | Number of candidate rows generated for this observation (`0 .. K`). |
| `candidate_miss` | `boolean` (nullable) | **Yes** | For `"aligned"` observations: `False` if `reference_arc_id` is in the candidate set, `True` if absent. For all other alignment statuses: `<NA>`. |
| `label_eligible` | `bool` | No | `True` **if and only if** `alignment_status == "aligned"` AND `candidate_miss == False`. |

---

## 4. Candidate Label & Miss Rate Policy

1. **When `label_eligible == True`:**
   - The candidate with `candidate_arc_id == reference_arc_id` receives `label = 1`.
   - All other candidates for that observation receive `label = 0`.
2. **When `candidate_miss == True` (`alignment_status == "aligned"`, but `reference_arc_id` is absent from candidates):**
   - `label_eligible = False`.
   - **All candidates for that observation keep `label = <NA>`**. We never turn candidate misses into all-zero false negative training examples, because doing so would teach the classifier to suppress every nearby road when the vehicle is slightly outside the search radius or cap $K$.
3. **When `alignment_status` is `"ambiguous"`, `"unmatched"`, or `"not_available"`:**
   - `candidate_miss = <NA>`, `label_eligible = False`, and all candidate `label` values remain `<NA>`.
4. **Candidate Miss Rate Formula:**
   $$\text{Candidate Miss Rate} = \frac{\text{Count}(\text{alignment\_status} == \text{'aligned'} \land \text{candidate\_miss} == \text{True})}{\text{Count}(\text{alignment\_status} == \text{'aligned'})}$$
   We always report both the numerator, the denominator (`aligned` count), and the counts of excluded `ambiguous`/`unmatched` observations.

---

## 5. HMM Road Graph Contract (`TripRoadGraph`)

```python
@dataclass
class TripRoadGraph:
    trip_id: str
    arc_endpoints: np.ndarray          # int32, shape (num_arcs, 2) -> [from_node_id, to_node_id]
    arc_lengths_m: np.ndarray          # float32, shape (num_arcs,)
    arc_bearings_deg: np.ndarray       # float32, shape (num_arcs,)
    outgoing_arcs_by_node: dict[int, np.ndarray]
```

- `graph.successor_arcs(arc_id: int) -> np.ndarray`: Returns a 1D `int32` array of all directed arcs $a'$ such that $\text{from\_node}(a') == \text{to\_node}(\text{arc\_id})$.
- `graph.are_connected(from_arc_id: int, to_arc_id: int) -> bool`: Returns `True` if `from_arc_id == to_arc_id` (staying on the same road segment) or `to_arc_id` is a direct outgoing successor of `from_arc_id`.
- `graph.shortest_hop_transition_type(from_arc_id: int, to_arc_id: int, max_hops: int = 3) -> int`: Returns `0` (same segment), `1` (direct successor), `2..max_hops` (skipped intermediate segments across a 1 Hz step or GPS gap), or `-1` (disconnected).

---

## 6. Usage Example

```python
from pathsnap import (
    CandidateConfig,
    align_reference_route,
    attach_alignment_labels,
    generate_trip_candidates,
    load_trip,
    summarize_trip_labels,
)

# 1. Inference-safe candidate generation (does NOT use .route)
raw = load_trip("00000001", partition="train", include_route=True)
config = CandidateConfig(search_radius_m=100.0, max_candidates_k=10)

result = generate_trip_candidates(
    trip_id=raw.trip_id,
    partition=raw.partition,
    track=raw.track,
    nodes=raw.nodes,
    arcs=raw.arcs,
    config=config,
)

# 2. Training / Development only: align reference route and attach labels
alignment = align_reference_route(
    track=raw.track,
    nodes=raw.nodes,
    arcs=raw.arcs,
    route=raw.route,
    config=config,
    trip_id=raw.trip_id,
)
labeled_result = attach_alignment_labels(result, alignment)
summary = summarize_trip_labels(labeled_result)
print(summary.to_dict())
```
