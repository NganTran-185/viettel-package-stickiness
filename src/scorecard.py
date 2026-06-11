
import pandas as pd
import lightgbm as lgb
from src.data import load, get_model_data


def build_scorecard(sellers, y_true, expected, min_sales=30):
    
    t = pd.DataFrame({
        "seller":   sellers.values,
        "actual":   pd.Series(y_true).values,
        "expected": expected,
    })
    g = t.groupby("seller").agg(
        n_sales=("actual", "size"),
        actual_rate=("actual", "mean"),
        expected_rate=("expected", "mean"),
    )
    g = g[g["n_sales"] >= min_sales]              
    g["residual"] = g["actual_rate"] - g["expected_rate"]
    g = g.sort_values("residual")
    g = g.reset_index(drop=True)
    g.insert(0, "seller", [f"Seller_{i:03d}" for i in range(1, len(g) + 1)])
    return g.sort_values("residual") 

def _residual_for(train_months, test_month, df, Xc, y, thang, cat, min_sales=20):
    """Tính residual cho từng seller ở một kỳ (train -> test) độc lập."""
    import lightgbm as lgb
    tr = thang.isin(train_months)
    te = thang == test_month
    m = lgb.LGBMClassifier(n_estimators=300, min_child_samples=100,
                           random_state=42, verbose=-1)
    m.fit(Xc[tr], y[tr], categorical_feature=cat)
    expected = m.predict_proba(Xc[te])[:, 1]
    t = pd.DataFrame({
        "seller": df.loc[te, "USER_DANG_KY"].values,
        "actual": y[te].values,
        "expected": expected,
    })
    g = t.groupby("seller").agg(n=("actual","size"),
                                actual=("actual","mean"),
                                expected=("expected","mean"))
    g = g[g["n"] >= min_sales]
    return (g["actual"] - g["expected"]).rename(f"resid_{test_month}")


def validate_stability(df, Xc, y, thang, cat):
    """So residual kỳ A (->thg4) vs kỳ B (->thg5) trên cùng seller."""
    rA = _residual_for([3], 4, df, Xc, y, thang, cat)        # train 3 -> test 4
    rB = _residual_for([3,4], 5, df, Xc, y, thang, cat)      # train 3+4 -> test 5
    both = pd.concat([rA, rB], axis=1).dropna()              # seller có ở cả 2 kỳ
    corr = both.iloc[:,0].corr(both.iloc[:,1])
    print(f"Sellers in both periods: {len(both)}")
    print(f"Residual correlation (period A vs B): {corr:.3f}")
    return both, corr


if __name__ == "__main__":
    df = load()
    X, y, thang = get_model_data(df)
    tr, te = thang.isin([3, 4]), thang == 5

    cat = list(X.select_dtypes(["object", "category"]).columns)
    Xc = X.copy()
    for c in cat:
        Xc[c] = Xc[c].astype("category")

    model = lgb.LGBMClassifier(n_estimators=300, min_child_samples=100,
                               random_state=42, verbose=-1)
    model.fit(Xc[tr], y[tr], categorical_feature=cat)

    expected = model.predict_proba(Xc[te])[:, 1]

    sellers = df.loc[te, "USER_DANG_KY"]
    y_test  = y[te]

    sc = build_scorecard(sellers, y_test, expected, min_sales=30)
    both, corr = validate_stability(df, Xc, y, thang, cat)

    print("=== 15 nhân viên CỜ ĐỎ (bán tệ hơn kỳ vọng nhất) ===")
    print(sc.head(15).round(3).to_string())
    print("\n=== 15 nhân viên TỐT NHẤT (vượt kỳ vọng) ===")
    print(sc.tail(15).round(3).to_string())
    print(f"\nTổng nhân viên đủ mẫu (>={30} gói):", len(sc))

    import joblib, os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/lgbm.pkl")
    sc.to_csv("reports/scorecard.csv")
    print("Saved models/lgbm.pkl + reports/scorecard.csv")