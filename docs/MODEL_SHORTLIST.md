# PathSnap — Shortlisted Five Machine Learning Models

**Owner:** Pranav Shashikiran Atreya (ML Engineering), Qamar Ahmed (Technical Lead)

---

## 1. Candidate-Classification Problem Formulation

Before sequential HMM decoding, we formulate road-segment scoring as a **supervised binary probability estimation task over `(GPS observation, candidate road arc)` pairs**:

- **One Sample:** A single row from `candidates.parquet` where `label_eligible == True` (identified by `(trip_id, observation_id, candidate_arc_id)`), represented by the 11-dimensional continuous/boolean feature vector `CANDIDATE_FEATURE_COLUMNS`.
- **Positive Label (`label = 1`):** The candidate road segment matches the vehicle's confidently aligned reference route arc at that observation ($11.26\%$ of labeled training rows at $R=100\text{ m}, K=10$).
- **Negative Label (`label = 0`):** A nearby candidate road segment within the spatial search radius that was *not* the traversed segment ($88.74\%$ of labeled training rows).
- **Required Model Output:** Calibrated posterior probability $P(y_{i,c} = 1 \mid \mathbf{x}_{i,c}) \in [0, 1]$ via `predict_proba(X)[:, 1]`, which serves both for pointwise **Observation-Level Top-1 ranking** and as the **emission likelihood** in our downstream Viterbi decoder.

---

## 2. Consolidated Five-Model Shortlist Comparison

| Model | Algorithm family | Reason for selection | Important hyperparameters | Probability output |
|---|---|---|---|---|
| **1. Logistic Regression** | Linear & Probabilistic Classification | Fast, convex, highly interpretable baseline that directly models log-odds as a linear combination of perpendicular distance, endpoint proximity, and bearing difference; naturally outputs calibrated probabilities under cross-entropy loss. | `penalty` (`'l2'`), `C` (inverse regularisation strength), `class_weight` (`None` vs `'balanced'`), `solver` (`'lbfgs'`), `StandardScaler` + `SimpleImputer(add_indicator=True)` | Native sigmoid probability $P(y=1\mid\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x} + b)$ trained via log-loss minimisation. |
| **2. Distance-Weighted KNN** | Instance-Based (Non-Parametric) Learning | Tests whether local neighbourhoods in standardized geometric/kinematic feature space can classify road candidates without global parametric assumptions. | `n_neighbors` ($k \in \{7, 15, 31, 51\}$), `weights` (`'distance'`), `metric` (`'manhattan'`, `'euclidean'`, `'minkowski'`), `StandardScaler` | Inverse-distance weighted fraction of positive neighbours: $P(y=1\mid\mathbf{x}) = \frac{\sum_{j \in \mathcal{N}_k(\mathbf{x})} w_j y_j}{\sum_{j \in \mathcal{N}_k(\mathbf{x})} w_j}$ where $w_j = 1 / d(\mathbf{x}, \mathbf{x}_j)$. |
| **3. Decision Tree Classifier** | Tree-Based Nonlinear Learning | Captures axis-aligned nonlinear thresholds and conditional interactions (e.g., *bearing difference matters only when `gps_speed_mps` $> 3\text{ m/s}$ and `projection_fraction` $\in (0.1, 0.9)$*) with transparent rule inspection. | `criterion` (`'gini'`, `'log_loss'`), `max_depth` ($\{4, 6, 8, 12\}$), `min_samples_split`, `min_samples_leaf` ($\ge 20$ for smooth leaf probabilities), `class_weight` | Empirical positive class frequency in the terminal leaf node (smoothed by enforcing `min_samples_leaf >= 20`). |
| **4. Random Forest Classifier** | Bagging Ensemble Learning | Reduces the high variance of a single decision tree by averaging bootstrap-aggregated decorrelated trees over random feature subsets, yielding robust nonlinear boundaries and smooth probability estimates. | `n_estimators` ($100\text{--}300$), `max_depth` ($\{8, 12, 16\}$), `max_features` (`'sqrt'`, `0.5`), `min_samples_leaf` ($\{10, 25\}$), `class_weight` (`'balanced_subsample'`, `None`) | Mean of predicted leaf-node class probabilities across all $B$ bootstrap trees in the forest. |
| **5. HistGradientBoosting Classifier** | Gradient Boosting (Sequential Additive Trees) | Sequentially minimizes binary log-loss using histogram-binned gradient boosted decision trees; natively handles `NaN` values in `gps_bearing_deg` and `gps_speed_mps` (stationary points / 1st observations / GPS gaps) and scales rapidly to $> 1.5\text{M}$ candidate rows. | `learning_rate` ($\in [0.01, 0.2]$), `max_iter` ($100\text{--}400$), `max_leaf_nodes` ($\{15, 31, 63\}$), `max_depth`, `min_samples_leaf`, `l2_regularization` | Native logistic sigmoid transformation of the additive boosted log-odds ensemble score, directly minimizing binary log-loss. |

---

## 3. Why These Five Models Provide Meaningful Algorithmic Diversity

1. **Linear vs Nonlinear Decision Boundaries:**
   - In clean open-road segments, candidate probability decreases monotonically with `distance_to_segment_m` and `bearing_difference_deg`, which **Logistic Regression** captures cleanly.
   - At complex urban intersections, parallel service roads, or turn lanes, feature interactions are strongly conditional: a candidate segment with a $12\text{ m}$ perpendicular distance may still be the true arc if `projection_fraction` is near $0.0$ (just entering a turn), `gps_speed_mps` is low ($< 4\text{ m/s}$), and `bearing_difference_deg` is transitioning. **Decision Trees, Random Forests, and HistGradientBoosting** capture these multi-way nonlinear interactions without manual polynomial feature expansion.
2. **Parametric vs Non-Parametric Local Geometry:**
   - **Distance-Weighted KNN** evaluates whether local similarity in standardized 11D feature space suffices, providing an empirical contrast against global hyperplane (Logistic Regression) and hierarchical partitioning (Tree/Ensemble) methods.
3. **Variance Reduction (Bagging) vs Bias/Loss Reduction (Boosting):**
   - Comparing **Decision Tree** $\to$ **Random Forest** isolates the effect of variance reduction via bootstrap aggregation.
   - Comparing **Random Forest** $\to$ **HistGradientBoosting** contrasts parallel bagging against sequential residual correction under direct log-loss optimisation, while testing native missing-value routing (`NaN` splits for stationary/first-observation GPS points) versus median imputation.
4. **Compatibility with Sequential HMM Decoding:**
   - The downstream Viterbi decoder sums log-emission probabilities $\log P(y_{i,c}=1 \mid \mathbf{x}_{i,c})$ across hundreds of time steps. Overconfident or zero probabilities ($P = 0$ or $P = 1$, common in unpruned Decision Trees or small-$k$ KNN) cause log-probability underflow and prevent the HMM from overriding a locally noisy GPS point using road-network connectivity. Evaluating **Log Loss** alongside **Observation-Level Top-1 Accuracy** across these five families directly measures both ranking power and probability calibration for Viterbi integration.
