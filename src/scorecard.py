"""Seller quality scorecard: actual vs model-expected survival.

Ý tưởng: model cho 'expected' (xác suất sống theo ĐẶC ĐIỂM gói, không nhìn
kết cục). Gom theo nhân viên rồi so với 'actual' (kết cục thật). Residual âm
= bán tệ hơn mức đáng lẽ -> cờ đỏ. Cách này loại bỏ lợi thế/bất lợi do mix
sản phẩm, chỉ còn lại 'kỹ năng' của người bán.
"""
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
    return g.sort_values("residual")               # sort the worst lên đầu


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