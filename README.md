### Viettel-package-stickiness 

Predict, at the moment of sale, whether a newly registered mobile data plan
will be "sticky" (survive ≥ 6 months) — to flag at-risk revenue early and score
sales quality, not just volume.

> Built on 59,603 real Viettel data-plan registrations (Thanh Hoa province, May 2026), spanning 327 packages, 1561 sellers, and 73 sales clusters.


## 🎯 Problem

Telco sales teams are measured on volume (packages sold), which creates an incentive to register plans that churn almost immediately — inflating short-term numbers while destroying real revenue. The target `TUOI_THO_6THANG` marks whether a plan stayed alive ≥ 6 months (**58% positive / 42% churn**, a healthy class balance).

We frame this as a point-of-sale binary classification: using only signals known when the plan is sold, estimate `P(plan survives ≥ 6 months)`.