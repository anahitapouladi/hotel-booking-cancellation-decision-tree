# Reproducibility notes — Hotel Booking Cancellation Prediction

## What I checked

I reran the whole notebook from `hotel_bookings.csv` (Python 3.13, scikit-learn 1.9.1). All 20 code cells ran without errors and exported the nine figures. The README, notebook text and figures all describe this same run.

## Data checks

- Raw shape: **119,390 rows × 32 columns**; **44,224** cancellations (37.0%).
- Exact duplicate rows: **31,994**; repeated retained-predictor rows: **33,534** across **85,856** unique predictor patterns.
- Conflicting labels: **282** retained-predictor patterns (covering **2,084** rows) contain both outcomes.
- Held-out split: **95,515** training and **23,875** test rows, with **zero** predictor groups shared between them.
- Preprocessing sits inside the fitted pipeline, and the five-fold hyperparameter search uses training groups only.

## Main results

| Metric | Unrestricted tree | CV-tuned tree |
|---|---:|---:|
| Test accuracy | 0.8294 | 0.8469 |
| Cancellation precision | 0.7786 | 0.8159 |
| Cancellation recall | 0.7537 | 0.7576 |
| Cancellation F1 | 0.7659 | 0.7856 |
| False alerts | 1,896 | 1,512 |
| Missed cancellations | 2,178 | 2,144 |
| Depth | 55 | 12 |
| Leaves | 9,861 | 508 |

The held-out F1 gain is **+0.0197**, with a paired group-bootstrap 95% interval of **[+0.0106, +0.0292]** (2,000 resamples). The interval describes this held-out sample only; it says nothing about future periods or retraining.

Selected settings: `gini`, `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=5`, `class_weight=None`, `ccp_alpha=0.0`. Mean grouped-CV F1 is **0.7959** (fold SD **0.0056**).

## Limitations

- No chronological hold-out or separate-hotel test was run, and some retained fields may be updated after a booking is created.
- Identical predictor patterns are not necessarily the same reservation or guest, and feature importance does not show cause and effect.
- Results can shift slightly with the library version or platform, so check the figures and text together if you rerun.
- The upstream dataset license and the public GitHub/Kaggle links should be confirmed before publishing.
