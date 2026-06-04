from pathlib import Path
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG  = yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8"))
COLUMNS = yaml.safe_load(open(ROOT / CONFIG["paths"]["columns"], encoding="utf-8"))

TARGET       = CONFIG["target"]
FEATURE_COLS = COLUMNS["feature"]
LEAKAGE_COLS = COLUMNS["leakage"]


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


if __name__ == "__main__":
    df = load()
    print(validate(df))
    X, y = get_Xy(df)
    print("X:", X.shape, "| y rate:", round(y.mean(), 3))