
# Viettel Package-Stickiness Prediction 📶

**Predict, at the moment of sale, whether a newly registered mobile data plan will survive ≥6 months** — to flag at-risk revenue early and score sales quality, not just volume.

**🔗 Live demo:** https://viettel-package-stickiness.streamlit.app

---

## Overview

A leakage-audited retention model on **162K real Viettel data-plan registrations** (Thanh Hoa province, Mar–May 2026; 51 raw fields, 327 packages, 1,561 sellers). The target `TUOI_THO_6THANG` marks whether a plan survived ≥6 months (58% / 42% balance).

Framed as **point-of-sale binary classification**: using only signals known *when the plan is sold*, estimate `P(survives ≥ 6 months)`.

> Proof-of-concept: single province, 3 months. Not a national production model.

## Results (out-of-time validation: train Mar–Apr, test May)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression (baseline, numeric only) | 0.651 | 0.732 |
| **LightGBM (full feature set)** | **0.867** | **0.880** |

## ⚠️ Leakage hunting — the core of this project

A naive baseline scored **0.9995 AUC**. Instead of trusting it, I traced the cause with **per-feature AUC scans + SHAP** and removed **4 leakage sources across 3 distinct mechanisms**:

| Source | Mechanism | Evidence |
|---|---|---|
| `tenure_days` (from `NGAY_KICH_HOAT`) | semantic — encodes *package activation*, not subscriber tenure | solo AUC 0.9995; tenure=0 (34% of rows) survives 0.1% |
| `NUM_TT_BAT_MAY`, `GROUP_CHANNEL_TYPE_ID` | presence-of-value — only active plans populate them | solo AUC ~0.88, ~70% missing |
| `SYSTEM_CLIENT` | per-value leak hidden in a valid column | several client codes survive at exactly 0% over 100s of rows |

I also **kept** features that were strong but legitimate (`GOI_DATA` — diagnosed as high-cardinality memorization, not leakage; `DT_GOI_FULL` — monotonic price–survival relation). 

**Key lesson:** distinguish real leakage (remove) from valid-but-strong features (keep), using single-feature AUC + column semantics — and know *when to stop suspecting*. Result: 0.9995 → an honest **0.867**.

## Key findings (EDA)

- **Channel** strongly separates retention: CHUOI 0.12 → HKD 0.64 (baseline 0.58).
- **Out-of-province** subscribers look stickier (0.79 vs 0.49) — but it's **confounded by channel** (96% buy via the high-retention HKD channel), shown via cross-tab analysis.
- **Label rate is stable across months** (0.55–0.60) → safe to merge for out-of-time validation.

## Dashboard

Interactive Streamlit app with 3 tabs:
1. **Overview** — EDA findings + SHAP feature importance
2. **Score a plan** — enter point-of-sale features → survival probability
3. **Seller scorecard** — *actual vs model-expected* survival per seller, isolating individual skill from product-mix advantage

## Scorecard validation
To check the residual measures real seller skill rather than noise, I computed residuals independently in two periods (train Mar→test Apr, and train Mar–Apr→test May) and correlated them per seller. Across 690 sellers present in both, residuals correlate at r = 0.31 — a clear positive signal that flagged sellers stay flagged, indicating the score captures something stable over time. It remains a decision-support tool (review candidates), not a definitive performance verdict; per-seller confidence intervals are listed as future work.

## Tech stack

Python · pandas · LightGBM · scikit-learn · SHAP · Streamlit · pytest

## Quickstart

```bash
pip install -r requirements.txt
python scripts/make_parquet.py        # build data/processed/tbm.parquet
python -m src.model                   # baseline + LightGBM
python -m src.scorecard               # seller scorecard
streamlit run app/streamlit_app.py    # dashboard
```
