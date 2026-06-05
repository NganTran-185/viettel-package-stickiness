from pathlib import Path
import yaml
import pandas as pd
from src.features import build_features

ROOT = Path(__file__).resolve().parents[1]
CONFIG  = yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8"))
COLUMNS = yaml.safe_load(open(ROOT / CONFIG["paths"]["columns"], encoding="utf-8"))

TARGET       = CONFIG["target"]
FEATURE_COLS = COLUMNS["feature"]
LEAKAGE_COLS = COLUMNS["leakage"]
ENGINEERED = ["tenure_days", "usage_trend", "is_out_of_province", "has_usage_history", "price_band"] 

def load():
    return pd.read_parquet(ROOT / CONFIG["paths"]["processed"])
    

def get_Xy(df = None):
    if df is None:
        df = load()
    cols = (c for c in FEATURE_COLS if c in df.columns)
    X = df[cols].copy()
    y = df[TARGET].astype(int)
    return X, y


def validate(df:pd.DataFrame) -> dict: 
   report = {
       "n_rows": len(df),
       "n_cols": df.shape[1],
       "target_rate": float(df[TARGET].mean()),
       "duplicate_rows" : int(df.duplicated().sum()),
       "missing_target": int(df[TARGET].isna().sum()),
   }
   assert report["missing_target"] == 0, "Target thiếu gía trị"
   assert set(FEATURE_COLS).issubset(df.columns), "Thiếu cột feature"
   return report

def get_model_data (df = None):
    if df is None:
        df = load()
    df = build_features(df)

    base = [c for c in FEATURE_COLS if c in df.columns and c != "NGAY_KICH_HOAT"]
    feat_cols = base + ENGINEERED 

    X = df[feat_cols].copy()
    y = df[TARGET].astype(int)
    thang = df["thang"].copy()  
    return X, y, thang
    

if __name__ == "__main__":
    df = load()
    print(validate(df))
    X, y = get_Xy(df)
    print("X:", X.shape, "| y rate:", round(y.mean(), 3))

    Xm, ym, thang = get_model_data(df)
    print("X (model):", Xm.shape)
    print("Phân bố tháng:\n", thang.value_counts().sort_index())
    print("Còn cột leakage trong X?", set(Xm.columns) & set(LEAKAGE_COLS))
    print("Các cột X:", list(Xm.columns))