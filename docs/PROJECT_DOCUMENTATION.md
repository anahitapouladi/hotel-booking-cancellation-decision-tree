# Validation notes

These notes document how the numbers in the README and `docs/run_metrics.json` relate to the notebook outputs, and what has not been verified.

## Source of the reported results

All results come from the saved outputs of the last completed run of `notebooks/Hotel_Booking_Cancellation_Prediction_Decision_Tree.ipynb`.

- `docs/run_metrics.json` is a **static snapshot** copied from those saved outputs. The notebook does not create or update it.
- The nine figures in `images/` are the images displayed in the notebook's saved outputs. They may have a lower resolution than the files the notebook writes when it is rerun (`save_figure` uses 170 DPI).
- Package versions and the SHA-256 of the raw CSV were not recorded in the saved outputs, so they are not included in the snapshot.

If you rerun the notebook and publish new results, update the README, the figures, and the snapshot together. Different package versions can also change the grouped split and the selected model.

## Reported run

- Input: **119,390 rows × 32 columns**; **31,994 exact duplicate rows**.
- Repeated retained-predictor rows: **33,534**.
- Group-aware split: **95,648 training rows** and **23,742 held-out test rows**, with **0 shared predictor groups**.
- Cancellation rate: about **37.4%** in training and **35.6%** in test. Stratification across groups is approximate.
- Grouped CV: **32 candidates × 5 folds = 160 fits**; best mean F1 **0.7948** (fold SD **0.0054**).
- Selected parameters: `class_weight="balanced"`, `criterion="gini"`, `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=10`, `ccp_alpha=0.0` (fixed).

## Held-out results

| Metric | Unrestricted tree | CV-tuned tree |
|---|---:|---:|
| Accuracy | 0.8342 | 0.8465 |
| Precision (Cancelled) | 0.7715 | 0.7708 |
| Recall (Cancelled) | 0.7590 | 0.8096 |
| F1 (Cancelled) | 0.7652 | 0.7897 |
| Confusion matrix (TN, FP, FN, TP) | 13,391; 1,900; 2,037; 6,414 | 13,256; 2,035; 1,609; 6,842 |
| Depth / leaves | 49 / 9,823 | 12 / 373 |

- F1 difference: **+0.0245**; paired group-bootstrap 95% interval (2,000 resamples): **[+0.0141, +0.0346]**.
- Error trade-off: **135 more false alerts** and **428 fewer missed cancellations**.

## Consistency checks

- The confusion-matrix counts reproduce the reported precision, recall, accuracy, and F1 for both trees.
- The README, this report, and the JSON snapshot all use the same 23,742-row held-out run.
- The notebook's Python code parses without syntax errors. The raw CSV is not included in this repository, so these notes do not claim a fresh end-to-end rerun.

## Interpretation boundaries

- The held-out fold was not used for tuning.
- Grouping avoids exact predictor-pattern overlap, but it does not show that guests or bookings are independent.
- The bootstrap interval is conditional on this held-out split and the fitted models. It does not cover future periods or other hotels.
- Features that can change during the booking lifecycle need a defined prediction time before any operational use.

## Figure files

`baseline_confusion_matrix.png`, `cancellation_distribution.png`, `correlation_matrix.png`, `error_tradeoff.png`, `feature_importances.png`, `missing_values.png`, `train_test_f1_comparison.png`, `tuned_confusion_matrix.png`, `tuned_tree_depth_0_to_3.png` (all in `images/`).
