# PathSnap — ML Component Requirements & Audit Checklist

**Owner:** Pranav Shashikiran Atreya  
**Branch:** `feature/pranav-ml`  
**Status:** Phase 1 Audit Complete (Pre-Training)

---

## 1. Authoritative Guidelines Checklist (All 5 PDFs Audited)

### A. Main Guidelines (`Main Guidelines.pdf`)
- [x] Understand two-laptop deployment architecture:
  - **Laptop 1 (Server / Model & API):** Hosts the trained classifier, learned preprocessing pipeline, HMM/Viterbi route reconstruction engine, road-network graph loader, and server-side evaluation against reference `.route` data.
  - **Laptop 2 (Client / Evaluator UI):** Displays the 10 reserved evaluator-testing trips, sends the selected trip ID to Laptop 1's FastAPI endpoint, and renders the predicted route and evaluation metrics.
- [ ] Follow the 12-stage workflow: problem understanding -> raw dataset inspection -> independent candidate & feature engineering -> 5+ model training & comparison -> evidence-based model selection -> Bayesian hyperparameter optimisation -> HMM/Viterbi route reconstruction -> artifact export -> FastAPI backend -> UI -> integration -> technical report.

### B. Dataset Exploration (`Dataset Exploration.pdf`)
- [x] Verify dataset partitions and trip counts:
  - `train`: **80 trips** (`00000001` to `00000099` minus the 10 student-test and 10 evaluator-test IDs; 208,417 total GPS points, 1,933,485 nodes, 3,876,904 arcs, 31,336 reference route segments).
  - `student_test`: **10 trips** (`00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`; 19,160 total GPS points).
  - `evaluator_test`: **10 trips** reserved for final live evaluation (missing from student ZIP by design: `00000015`, `00000017`, `00000023`, `00000030`, `00000046`, `00000050`, `00000061`, `00000074`, `00000082`, `00000085`). **Never fabricate or use evaluator-testing trips during development.**
- [x] Verify raw file schemas (tab/whitespace-separated, no headers):
  - `<trip_id>.track`: 3 float columns (`longitude`, `latitude`, `timestamp` in seconds, ~1 Hz nominal).
  - `<trip_id>.nodes`: 2 float columns (`longitude`, `latitude`), where 0-based row index = `node_id`.
  - `<trip_id>.arcs`: 2 integer columns (`from_node_id`, `to_node_id`), where 0-based row index = trip-local `arc_id`.
  - `<trip_id>.route`: 1 integer column (`arc_id`), representing the chronological sequence of traversed reference road segments (NOT 1-to-1 aligned with `.track` rows).
- [ ] Account for geographical coordinates properly (angular lon/lat vs. geodesic/projected metric distances in meters; verify coordinate order `[lon, lat]`).
- [ ] Document limitations of `.route` reference paths (do not silently assume error-free physical ground truth).

### C. Models and Training Guidelines (`Models and Training Guidelines.pdf`)
- [ ] **Minimum Five Distinct Models:** Train and evaluate at least **5 distinct machine learning models** spanning multiple algorithm families:
  1. *Linear / Probabilistic:* e.g., Logistic Regression / SGDClassifier / Gaussian Naive Bayes
  2. *Instance-based / Kernel:* e.g., Distance-Weighted KNN / Calibrated Linear or RBF SVM
  3. *Tree / Bagging Ensemble:* e.g., Random Forest Classifier / Extra Trees Classifier
  4. *Gradient Boosting:* e.g., LightGBM / XGBoost / CatBoost / HistGradientBoostingClassifier
  5. *Neural Network:* e.g., Multi-Layer Perceptron (`MLPClassifier` with epoch-wise loss curve tracking)
- [ ] **Group-Aware Cross-Validation:** Use `GroupKFold` (grouped by `trip_id`) on the **80 training trips**. Never randomly split candidate rows from the same trip across train and validation folds.
- [ ] **Leakage-Free Preprocessing:** Fit all scalers (`StandardScaler`/`RobustScaler`), imputers, and feature selectors strictly on training folds only.
- [ ] **Development Evaluation:** Evaluate all trained models on the **10 student-testing trips** (`student_test`).
- [ ] **Mandatory Evaluation Metrics:**
  - Candidate classification metrics: **Accuracy**, **Precision**, **Recall**, **Macro F1-score**, **Positive-class Binary F1-score**, **Confusion Matrix**, and **Log Loss** (where probabilities are produced).
  - Map-matching / observation-level metrics: **Observation-level Top-1 Accuracy** (selecting the candidate with highest predicted score per GPS observation `t` and comparing against reference segment), **Candidate Miss Rate** (fraction of GPS observations where the true reference arc is absent from the candidate set), **Route Continuity**, and **Sequence-level Performance**.
  - Operational metrics: **Training Time (s)** and **Inference Time (s / ms per trip & per observation)**.
- [ ] **Evidence-Based Model Selection:** Select the final model using validation stability, student-test generalisation, probability calibration/quality (Log Loss for HMM emissions), Observation Top-1 Accuracy, and inference efficiency—never training accuracy alone.

### D. Bayesian Hyperparameter Optimisation (`Models and Training Guidelines.pdf` §7 & `Deliverables.pdf` §4)
- [ ] Select the most promising model from the 5-model comparison for Bayesian tuning.
- [ ] Define and justify a structured hyperparameter search space (continuous log-uniform/uniform ranges, integer ranges, categorical choices).
- [ ] Use a genuine Bayesian optimisation algorithm (e.g., **Optuna TPE — Tree-structured Parzen Estimator** or Gaussian Process via `scikit-optimize`) optimizing a group-aware CV objective on the 80 training trips.
- [ ] Record every optimisation trial, hyperparameter configuration, objective score, and elapsed timestamp in `results/bayesian_trials.csv` and `artifacts/`.
- [ ] Refit the selected model with the best hyperparameters on the full 80 training trips.
- [ ] Evaluate and compare **Untuned vs. Tuned** performance on the 10 student-testing trips across:
  - CV Macro F1-score
  - Student-testing Top-1 Accuracy
  - Student-testing Macro F1-score
  - Log Loss
  - Inference Time
  - Total tuning computation cost

### E. HMM and Viterbi Route Reconstruction (`Models and Training Guidelines.pdf` §8–9 & `Deliverables.pdf` §5)
- [ ] Coordinate with Qamar Ahmed on the shared sequence decoding architecture so there is no duplicate/incompatible implementation.
- [ ] **Hidden States:** Candidate road segments `(arc_id)` at each GPS observation step `t`. Preserve exact candidate-to-road-segment (`arc_id`, `from_node_id`, `to_node_id`) correspondence.
- [ ] **Emission Probabilities:** Derived from the tuned classifier's predicted probabilities $P(y_{t,k}=1 \mid \mathbf{x}_{t,k})$ (or log-probabilities $\log P$, with temperature scaling / probability floor $\epsilon$ to prevent zero underflow).
- [ ] **Transition Probabilities:** Constructed using road-network topology from `.arcs` and `.nodes`:
  - Same arc ($a_{t-1} = a_t$): high self-transition probability (vehicle continues on same road segment).
  - Directly connected arcs ($\text{to\_node}(a_{t-1}) = \text{from\_node}(a_t)$): high transition probability weighted by topological/shortest-path consistency and vehicle displacement.
  - Multi-hop connected or disconnected arcs: exponential decay penalty $\exp(-\beta \cdot d_{\text{net}})$ or floor penalty $\log \epsilon_{\text{disc}}$ for invalid transitions.
- [ ] **Edge-Case Handling:**
  - *Missing candidate sets:* Fallback state or nearest-arc emission recovery so decoding never crashes.
  - *Invalid / disconnected transitions:* Backoff transition penalty when all candidate pairs between $t-1$ and $t$ are disconnected.
  - *Large GPS time gaps ($\Delta t \gg 1\text{ s}$):* Relax transition distance penalties proportionally to $\Delta t \cdot v_{\max}$ or reset transition prior across large dropouts.
- [ ] **Required Comparison:** Evaluate **Classifier-Only (Greedy Pointwise Top-1)** vs. **Classifier + HMM/Viterbi** on the 10 student-testing trips using Observation-level Top-1 Accuracy, Sequence-level Route Similarity (e.g., sequence edit/Jaccard/F1 match against `.route`), and Route Continuity (fraction of consecutive predicted segments that are topologically connected in `.arcs`).

### F. Deliverables, Visualisations & Academic Integrity (`Deliverables.pdf` & `Assessment and Plagiarism Policy.pdf`)
- [ ] Generate the mandatory consolidated comparison table (§3.1 of `Deliverables.pdf`):
  - Columns: `Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Inference Time`
- [ ] Answer compulsory questions A (1–2), B (5–6), and C (7–9) for **every** trained model grounded in actual executed results.
- [ ] Produce mandatory plots from real experimental logs:
  - Cross-validation Macro F1-score comparison across all 5+ models.
  - Student-testing Observation Top-1 Accuracy and Macro F1-score comparison.
  - Log Loss comparison across models.
  - Confusion matrices for models.
  - Epoch-wise training & validation loss curves **only** for models that naturally produce epoch histories (e.g., `MLPClassifier` / boosting staged predictions)—never fabricate training curves for non-iterative models.
  - Interactive **Folium** geographical maps showing raw GPS observations, road network geometry, reference route, and predicted route (classifier-only vs. HMM/Viterbi).
- [ ] Maintain strict reproducibility: log all random seeds, hyperparameters, feature lists, execution times, and saved model weights.
