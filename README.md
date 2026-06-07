### Viettel-package-stickiness 

Predict, at the moment of sale, whether a newly registered mobile data plan
will be "sticky" (survive ≥ 6 months) — to flag at-risk revenue early and score
sales quality, not just volume.

> Built on ~162K rows across 3 months (Mar–May 2026), enabling out-of-time validation, spanning 327 packages, 1561 sellers, and 73 sales clusters.


## 🎯 Problem
Telco sales teams are measured on volume (packages sold), which creates an incentive to register plans that churn almost immediately — inflating short-term numbers while destroying real revenue. The target `TUOI_THO_6THANG` marks whether a plan stayed alive ≥ 6 months (**58% positive / 42% churn**, a healthy class balance).

We frame this as a point-of-sale binary classification: using only signals known when the plan is sold, estimate `P(plan survives ≥ 6 months)`.

## Status
Phases 1–3 complete: data pipeline (3 months merged), EDA, leakage-audited feature set, baseline + LightGBM model.

## Results (out-of-time validation: train Mar–Apr, test May)
| Model | ROC-AUC | PR-AUC|
|------ |---------|-------| 
| Logistic Regression (baseline, numeric only) | 0.651 | 0.732 |
| LightGBM (full feature set) |0.891| 0.897|

## Leakage hunting (the core of this project)
A naive baseline scored AUC 0.9995. Rather than trust it, I traced the cause with a per-feature AUC scan and removed three distinct leakage sources, each with a different mechanism:
 - tenure_days (solo AUC 0.9995) — derived from NGAY_KICH_HOAT, which encodes package activation, not subscriber tenure → re-encodes the label.
- NUM_TT_BAT_MAY, GROUP_CHANNEL_TYPE_ID (solo AUC ~0.88) — presence-of-value leakage: only surviving plans populate these fields.
I also diagnosed GOI_DATA as high-cardinality memorization (not leakage — 98.6% of test packages seen in train) and confirmed DT_GOI_FULL / SYSTEM_CLIENT were legitimate despite high single-feature AUC. Key lesson: distinguish real leakage (remove) from valid-but-strong features (keep), using single-feature AUC + column semantics rather than intuition.


