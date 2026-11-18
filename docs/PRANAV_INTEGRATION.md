# PathSnap — ML Pipeline Integration Guide for Pranav

**Author:** Qamar Ahmed
**Target Audience:** Pranav Shashikiran Atreya (ML Engineering — 5 Models, GroupKFold CV, Bayesian Optimisation)

---

## 1. Quick-Start Import & Data Generation

Install the repository package in editable mode from the repo root:

```powershell
python -m pip install -e .
```

Import the official contract constants and pipeline functions directly from `pathsnap`:

```python
import pandas as pd
from pathsnap import (
    CANDIDATE_FEATURE_COLUMNS,
    CANDIDATE_ID_COLUMNS,
    CANDIDATE_TARGET_COLUMN,
    CandidateConfig,
    align_reference_route,
    attach_alignment_labels,
    generate_trip_candidates,
    list_trips,
    load_trip,
)
```

---

## 2. Exact Column Separation for Model Training

Never pass `trip_id`, `observation_id`, `candidate_arc_id`, `from_node_id`, `to_node_id`, or `candidate_rank` as predictive features into any ML classifier.

Use the exported constant lists from `pathsnap.schemas.candidate_schema`:

### A. Predictive Feature Columns (`CANDIDATE_FEATURE_COLUMNS` — 11 columns)
```python
CANDIDATE_FEATURE_COLUMNS = [
    "distance_to_segment_m",   # float32 (non-null)
    "projection_fraction",     # float32 in [0, 1] (non-null)
    "distance_to_start_m",     # float32 (non-null)
    "distance_to_end_m",       # float32 (non-null)
    "segment_length_m",        # float32 (non-null)
    "segment_bearing_deg",     # float32 in [0, 360) (non-null)
    "gps_bearing_deg",         # nullable Float32 in [0, 360)
    "bearing_difference_deg",  # nullable Float32 in [0, 180]
    "gps_speed_mps",           # nullable Float32 in m/s
    "delta_time_s",            # nullable Float32 in seconds
    "is_fallback_candidate",   # bool (non-null)
]
```

### B. Identifier & Grouping Columns (`CANDIDATE_ID_COLUMNS`)
```python
CANDIDATE_ID_COLUMNS = [
    "trip_id",           # Use as `groups` in GroupKFold(n_splits=5)
    "partition",         # 'train' or 'student_test'
    "observation_id",    # Group by (trip_id, observation_id) to compute Observation-level Top-1 Accuracy
    "timestamp_s",
    "candidate_rank",    # Proximity baseline rank (rank == 1 is the Nearest-Road baseline!)
    "candidate_arc_id",  # Predicted arc ID to pass to HMM / compare with reference_arc_id
    "from_node_id",
    "to_node_id",
]
```

### C. Target Column (`CANDIDATE_TARGET_COLUMN`)
```python
CANDIDATE_TARGET_COLUMN = "label"  # nullable Int8: 1 (positive), 0 (negative), <NA> (unlabeled/ineligible)
```

---

## 3. Filtering Training Rows vs Evaluating Top-1 Accuracy

### Step 1: Filtering for Supervised Model Fitting (`train` partition)
When fitting your 5 classifiers (`LogisticRegression`, `KNeighborsClassifier(weights='distance')`, `DecisionTreeClassifier`, `RandomForestClassifier`, `HistGradientBoostingClassifier`), filter `candidates_df` to rows where `label.notna()`:

```python
train_cands = candidates_df[
    (candidates_df["partition"] == "train") & (candidates_df["label"].notna())
].copy()

X_train = train_cands[CANDIDATE_FEATURE_COLUMNS].astype("float32")
y_train = train_cands[CANDIDATE_TARGET_COLUMN].astype("int8").to_numpy()
groups_train = train_cands["trip_id"].to_numpy()
```

### Step 2: Handling Nullable Features (`gps_bearing_deg`, `bearing_difference_deg`, `gps_speed_mps`, `delta_time_s`)
- Observation `0` of every trip and observations with stationary GPS jitter ($< 2.5\text{ m}$) or long dropouts ($> 30\text{ s}$) have `NaN` in these 4 columns.
- `HistGradientBoostingClassifier` handles `NaN` natively.
- For `LogisticRegression`, `KNN`, `DecisionTree`, and `RandomForest`, wrap your pipeline with `sklearn.impute.SimpleImputer(strategy="median", add_indicator=True)` **inside** your `sklearn.pipeline.Pipeline` so imputation statistics and `StandardScaler` are fitted strictly on each training fold (zero data leakage).

### Step 3: Computing Observation-Level Top-1 Accuracy & Route Continuity
For evaluation on validation folds or `student_test`:
1. Predict positive class probability `prob = model.predict_proba(X_eval)[:, 1]`.
2. Attach `prob` to the candidate rows and select the highest-scoring candidate per `(trip_id, observation_id)`:
   ```python
   eval_cands["pred_prob"] = model.predict_proba(X_eval)[:, 1]
   top1_preds = (
       eval_cands.sort_values(
           ["trip_id", "observation_id", "pred_prob", "distance_to_segment_m"],
           ascending=[True, True, False, True],
       )
       .groupby(["trip_id", "observation_id"], as_index=False)
       .first()
   )
   ```
3. Join `top1_preds` with `observations_df` on `["trip_id", "observation_id"]`:
   - **Observation-Level Top-1 Accuracy (on aligned observations):**
     ```python
     aligned_mask = merged["alignment_status"] == "aligned"
     top1_acc = (
         merged.loc[aligned_mask, "candidate_arc_id"]
         == merged.loc[aligned_mask, "reference_arc_id"]
     ).mean()
     ```
   - **Route Continuity:** For consecutive observations $(i-1, i)$ in a trip, check `graph.are_connected(pred_arc[i-1], pred_arc[i])` using `result.graph`.

---

## 4. Pre-Generated Preview Files Ready in Repository

You can immediately inspect and test your code against real generated files in `outputs/candidate_previews/`:
- `outputs/candidate_previews/candidates.parquet` (first 5 training trips, full schema)
- `outputs/candidate_previews/observations.parquet` (first 5 training trips, full schema)
- `outputs/candidate_previews/candidates_preview_200.csv` (human-readable CSV sample)
- `outputs/candidate_previews/observations_preview_200.csv` (human-readable CSV sample)
