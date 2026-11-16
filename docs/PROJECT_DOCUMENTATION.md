# Reproducibility and validation notes — Hotel Booking Cancellation Prediction

## Scope of this report

This document records the saved Kaggle execution contained in the published notebook. It is a consistency report based on the notebook's saved code, outputs, Markdown explanations, and figure outputs. It does **not** claim that the notebook was independently rerun as part of preparing this document.

The saved notebook contains 20 executed code cells, with execution counts 1–20 and no recorded execution exception. Its kernel metadata records Python 3.10.9. The Kaggle runtime versions reported for this run are pandas 1.5.3, NumPy 1.23.5, and scikit-learn 1.2.1. The notebook metadata does not itself embed the versions of those three packages; the package versions are recorded here from the Kaggle runtime information supplied for this run.

## Dataset and data-quality checks

- Source: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand), file `hotel_bookings.csv`.
- Raw shape: **119,390 rows × 32 columns**.
- Target: `is_canceled`, where 0 means Not Cancelled and 1 means Cancelled.
- Cancellations: **44,224 (37.0%)**; not cancelled: **75,166 (63.0%)**.
- Exact duplicate raw rows: **31,994 (26.8%)**. The dataset has no unique reservation identifier, so identical rows are not automatically treated as data-entry errors.
- Missing values: `company` 112,593 (94.31%); `agent` 16,340 (13.69%); `country` 488 (0.41%); `children` 4.

The analysis removes `reservation_status` and `reservation_status_date` because they reveal the final outcome, `assigned_room_type` because its timing may change, and `company` because it is highly missing and identifier-like. Missing `agent` values are represented as `Unknown`; `agent` is treated as categorical. Two derived features are added: `total_nights` and `total_guests`.

## Repeated predictor patterns and split

Grouping is based on retained predictor columns only; the target is excluded from the group key. The analysis retains every row and uses the groups only to prevent identical predictor patterns from crossing the train/test split and grouped cross-validation folds.

- Distinct retained-predictor patterns: **85,856**.
- Repeated retained-predictor rows: **33,534 (28.1%)**.
- Predictor patterns with conflicting cancellation labels: **282**, covering **2,084 rows**.
- Split: first held-out fold from `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
- Training rows: **95,648**; held-out test rows: **23,742**.
- Cancellation rate: **37.4%** in training and **35.6%** in the held-out fold.
- Unique predictor groups: **68,652** in training and **17,204** in test.
- Shared predictor groups across the split: **0**.
- Majority-class reference accuracy on the test fold: **0.6440**; cancellation F1: **0.0000**.

The split is group-aware, not guest-aware or reservation-aware: the source data does not provide a unique booking ID. Grouping therefore controls overlap of identical recorded predictor patterns, not all possible relationships between records.

## Preprocessing and model selection

Preprocessing is kept inside a scikit-learn pipeline. Numeric features are imputed with the training median. Categorical features are imputed with the most frequent training category and one-hot encoded; unseen categories are ignored at prediction time.

A five-fold `StratifiedGroupKFold` search is run on the training partition only. The test fold is not used to select hyperparameters.

- Candidate configurations: **32**.
- Cross-validation folds: **5**.
- Total fits: **160**.
- Selection metric: F1 for the Cancelled class.
- Best mean grouped-CV F1: **0.794842** (fold SD **0.005376**; rounded in the notebook to 0.7948 and 0.0054).
- Spread across the top five mean CV F1 scores: **0.0007**.

Selected settings:

| Hyperparameter | Selected value |
|---|---|
| `criterion` | `gini` |
| `max_depth` | `12` |
| `min_samples_split` | `10` |
| `min_samples_leaf` | `10` |
| `class_weight` | `balanced` |
| `ccp_alpha` | `0.0` (fixed, not tuned) |

## Held-out results

Both trees are evaluated on the same held-out test fold of 23,742 rows. The confusion-matrix class order is `[Not Cancelled, Cancelled]`.

| Metric | Unrestricted tree | CV-tuned tree |
|---|---:|---:|
| Accuracy | 0.8342 | 0.8465 |
| Precision — Cancelled | 0.7715 | 0.7708 |
| Recall — Cancelled | 0.7590 | 0.8096 |
| F1 — Cancelled | 0.7652 | **0.7897** |
| Training F1 | 0.9936 | 0.8152 |
| Train–test F1 gap | 0.2284 | 0.0254 |
| Depth | 49 | 12 |
| Leaves | 9,823 | 373 |

Confusion matrix counts:

| Actual class / Predicted class | Unrestricted: Not Cancelled | Unrestricted: Cancelled | Tuned: Not Cancelled | Tuned: Cancelled |
|---|---:|---:|---:|---:|
| Not Cancelled | 13,391 | 1,900 | 13,256 | 2,035 |
| Cancelled | 2,037 | 6,414 | 1,609 | 6,842 |

The tuned tree's held-out F1 increased by **0.0245**. Precision changed by **−0.0007** (0.7715 to 0.7708), while recall increased by **0.0506** (0.7590 to 0.8096). It missed **428 fewer cancellations** and generated **135 additional false alerts**. This is a trade-off, not an unqualified improvement: the preferred balance depends on the operational cost of each error type.

A paired predictor-group bootstrap with **2,000 resamples** produced a 95% percentile interval of **[+0.0141, +0.0346]** for the tuned-minus-unrestricted F1 difference. The interval is conditional on this held-out partition and the two fitted models. It does not include uncertainty from retraining, later time periods, or other hotels.

## Feature and correlation interpretation

The five highest impurity-based feature importances in the fitted tuned tree are:

| Feature | Importance |
|---|---:|
| `deposit_type_Non Refund` | 0.3298 |
| `market_segment_Online TA` | 0.1295 |
| `lead_time` | 0.0922 |
| `total_of_special_requests` | 0.0799 |
| `country_PRT` | 0.0684 |

`deposit_type_Non Refund` is also the first split. This describes how the fitted tree uses the data; it does not establish that deposit type causes cancellation. The pattern may reflect booking policies or recording practices. Because `class_weight="balanced"` is selected, the `value` fields in the displayed tree are weighted class totals rather than raw booking counts.

The Pearson correlation heatmap is calculated on the training partition only. The strongest linear association with cancellation is `lead_time` (+0.30). Other visible associations include `total_of_special_requests` (−0.24), `required_car_parking_spaces` (−0.20), `booking_changes` (−0.15), and `previous_cancellations` (+0.12). Correlation is descriptive and captures linear association only; the tree can use nonlinear patterns and interactions.

## Figures

The `images/` directory contains nine figures extracted from the saved notebook's executed chart outputs:

1. `cancellation_distribution.png`
2. `missing_values.png`
3. `correlation_matrix.png`
4. `baseline_confusion_matrix.png`
5. `tuned_confusion_matrix.png`
6. `train_test_f1_comparison.png`
7. `error_tradeoff.png`
8. `tuned_tree_depth_0_to_3.png`
9. `feature_importances.png`

The notebook saves these figures under the same filenames when rerun. The repository's `run_metrics.json` is a static snapshot and is not automatically rewritten by executing the notebook.

## Limitations and interpretation boundaries

- No chronological hold-out or external-hotel validation was performed. The data covers 2015–2017, and calendar or price-related fields may capture period-specific patterns.
- The intended prediction point needs a source-system audit. Some retained fields, including `booking_changes`, `adr`, special requests, and parking spaces, may change during the reservation lifecycle.
- Identical predictor patterns do not prove that rows represent the same guest or reservation. The source data has no unique booking identifier.
- A total of 282 retained-predictor patterns (2,084 rows) have conflicting target labels.
- High-cardinality `agent` and `country` categories may encode dataset-specific patterns.
- Missing `country` values (488 rows) are imputed with the most frequent category (`PRT`); a separate `Unknown` category would be a useful sensitivity check.
- Impurity-based feature importance is not causal evidence. The strong `deposit_type_Non Refund` signal may reflect booking policies or recording practices.
- The split, selected settings, and error trade-off may change with the dataset, random seed, or library version. The reported metrics describe this saved run only.
- An operational model would need a prediction-time definition, a cost-sensitive decision threshold, future-period validation, and monitoring.

## Reproduction notes

The saved Kaggle notebook metadata records Python **3.10.9**. The Kaggle runtime versions reported for this saved run are pandas **1.5.3**, NumPy **1.23.5**, and scikit-learn **1.2.1**. Exact versions for Matplotlib, Seaborn, IPython, and Jupyter were not captured, so they are not pinned in `requirements.txt`.

To reproduce the analysis, download the source CSV from the dataset page, attach it as Kaggle input or place it at `data/hotel_bookings.csv`, install `requirements.txt`, and run all notebook cells. Because the split is generated by `StratifiedGroupKFold`, library versions, data changes, or random-state changes can alter the fold and downstream results. If a new run is published, update the notebook outputs, Markdown explanations, figures, README, this report, and `run_metrics.json` together.
