"""Viettel Package-Stickiness Dashboard."""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))   # cho thấy thư mục gốc -> import src

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Viettel Package Stickiness", layout="wide")
st.markdown("""
<div style="padding:1.5rem 1.8rem; border-radius:16px;
            background:linear-gradient(120deg,#1a1a2e 0%,#16213e 45%,#e94560 110%);
            margin-bottom:1.2rem;">
  <div style="display:flex; align-items:center; gap:.6rem;">
    <span style="font-size:1.8rem;">📶</span>
    <h1 style="margin:0; color:#fff; font-size:1.7rem; font-weight:700;">
      Package-Stickiness Intelligence</h1>
  </div>
  <p style="margin:.5rem 0 0; color:#cfd3dc; font-size:.95rem;">
    Predicting whether a telco data plan survives ≥6 months — at the moment of sale.
  </p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Registrations", "162K")
m2.metric("Model AUC", "0.867", help="Out-of-time validation (train Mar–Apr, test May)")
m3.metric("Leaks removed", "4", help="Found via per-feature AUC scan + SHAP")
m4.metric("Sellers scored", "1,561")


st.caption("Predict at point-of-sale whether a data plan survives ≥6 months. "
           "Proof-of-concept: Thanh Hoa, Mar–May 2026.")
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔮 Score a plan", "🏆 Seller scorecard"])

#  TAB 1: OVERVIEW 
with tab1:
    st.header("Project overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Registrations", "162K")
    col2.metric("Model AUC (out-of-time)", "0.867")
    col3.metric("Leakage sources removed", "4")

    st.subheader("Key EDA findings")
    st.markdown(
        "- **Channel** strongly separates retention: CHUOI 0.12 → HKD 0.64 (baseline 0.58).\n"
        "- **Out-of-province** subscribers look stickier (0.79 vs 0.49) — but it's "
        "**confounded by channel** (96% buy via HKD).\n"
        "- **Label rate stable across months** (0.55–0.60) → safe to merge for out-of-time validation."
    )

    st.subheader("What the model relies on (SHAP)")
    shap_img = ROOT / "reports" / "figures" / "shap_importance.png"
    if shap_img.exists():
        st.image(str(shap_img), use_column_width=True)
    else:
        st.info("Run the SHAP notebook to generate reports/figures/shap_importance.png")

    st.subheader("The leakage story")
    st.markdown(
        "A naive baseline scored **0.9995 AUC**. Tracing it with per-feature AUC scans + SHAP "
        "revealed 4 leaks across 3 mechanisms (activation-date proxy, presence-of-value, "
        "per-value channel leak). Removing them brought AUC to an honest **0.867**."
    )

# TAB 2: SCORE A PLAN 

@st.cache_resource
def load_model():
    """Train 1 lần khi app khởi động."""
    import lightgbm as lgb
    from src.data import get_model_data
    X, y, thang = get_model_data()
    Xc = X.copy()
    cat = list(Xc.select_dtypes(["object", "category"]).columns)
    for c in cat:
        Xc[c] = Xc[c].astype("category")
    tr = thang.isin([3, 4])
    model = lgb.LGBMClassifier(n_estimators=300, min_child_samples=100,
                               random_state=42, verbose=-1)
    model.fit(Xc[tr], y[tr], categorical_feature=cat)
    return model


@st.cache_data
def load_template():
    from src.data import get_model_data
    X, _, thang = get_model_data()
    Xc = X.copy()
    for c in Xc.select_dtypes(["object", "category"]).columns:
        Xc[c] = Xc[c].astype("category")
    return Xc, Xc[thang == 5].iloc[[0]].copy()

with tab2:
    st.header("Score a single plan")
    st.caption("Enter point-of-sale features → probability the plan survives ≥6 months.")

    try:
        model = load_model()
        Xfull, row = load_template()

        c1, c2 = st.columns(2)
        gia   = c1.number_input("Giá gói (GIA_GOI)", value=120000, step=10000)
        chuky = c2.number_input("Chu kỳ gói (CHUKY_GOI_N)", value=30, step=1)
        kenh  = c1.selectbox("Kênh (KENH_DANG_KY)",
                             sorted(Xfull["KENH_DANG_KY"].dropna().unique()))
        oop   = c2.selectbox("Ngoại tỉnh? (is_out_of_province)", [0, 1])
        band  = c1.selectbox("Price band", ["low", "mid", "high"])

        row = row.copy()
        row["GIA_GOI"] = gia
        row["CHUKY_GOI_N"] = chuky
        row["KENH_DANG_KY"] = pd.Categorical([kenh], categories=Xfull["KENH_DANG_KY"].cat.categories)
        row["is_out_of_province"] = oop
        row["price_band"] = pd.Categorical([band], categories=Xfull["price_band"].cat.categories)

        if st.button("Score", type="primary"):
            p = model.predict_proba(row)[0, 1]
            st.metric("Survival probability (≥6 months)", f"{p:.1%}")
            if p >= 0.58:
                st.success("Above baseline (0.58) — likely sticky.")
            else:
                st.warning("Below baseline (0.58) — at-risk plan.")
    except FileNotFoundError:
        st.warning("Run `python3 -m src.scorecard` first to save models/lgbm.pkl")


# TAB 3: SELLER SCORECARD (khung, làm sau) 
with tab3:
    st.header("Seller scorecard — actual vs model-expected survival")
    st.caption("Residual = actual − expected. Negative = sells worse than the "
               "plan mix predicts (review candidate). Removes product-mix advantage.")
    st.caption("⚠️ Seller IDs anonymized (Seller_NNN). Internal-facing tool by design — "
               "shown to demonstrate methodology, not to expose individuals.Validated for temporal stability (residuals correlate r=0.31 across two "
           "independent periods, n=690). Decision-support tool, not a performance verdict. "
           "Seller IDs anonymized.") 

    sc_path = ROOT / "reports" / "scorecard.csv"
    if not sc_path.exists():
        st.warning("Run `python3 -m src.scorecard` to generate reports/scorecard.csv")
    else:
        sc = pd.read_csv(sc_path)

        min_n = st.slider("Min sales per seller", 30, 200, 30, step=10)
        view = sc[sc["n_sales"] >= min_n].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Sellers shown", len(view))
        c2.metric("Worst residual", f"{view['residual'].min():.3f}")
        c3.metric("Best residual", f"{view['residual'].max():.3f}")

        st.subheader("🚩 Red flags — selling below expectation")
        red = view.nsmallest(15, "residual")
        st.dataframe(
            red.style.background_gradient(subset=["residual"], cmap="Reds_r"),
            use_container_width=True,
        )

        st.subheader("⭐ Top performers — beating expectation")
        top = view.nlargest(15, "residual")
        st.dataframe(
            top.style.background_gradient(subset=["residual"], cmap="Greens"),
            use_container_width=True,
        )
