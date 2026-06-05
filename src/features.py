import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. tenure_days
    sold = pd.to_datetime(df["NGAY_BAN_GOI"], format="%Y%m%d", errors="coerce")
    act  = pd.to_datetime(df["NGAY_KICH_HOAT"], format="%Y%m%d", errors="coerce")
    raw_tenure = (sold - act).dt.days
    print("Số dòng tenure âm:", (raw_tenure < 0).sum())
    df["tenure_days"] = raw_tenure.clip(lower=0)

    # 2. usage_trend
    df["usage_trend"] = (pd.to_numeric(df["TONG_TIEU_DUNG_N_1"], errors="coerce")
                         - pd.to_numeric(df["TONG_TIEU_DUNG_N_3"], errors="coerce"))

    # 3. is_out_of_province
    df["is_out_of_province"] = (df["HOME_TINH"] != "THA").astype(int)

    # 4. has_usage_history   => thue bao moi khong lich su tieu dung khac voi nguoi cu
    df["has_usage_history"] = pd.to_numeric(
        df["TONG_TIEU_DUNG_N_3"], errors="coerce").notna().astype(int)
    
    # 5. price_band => nhom gia goi (goi re va dat co do ben khac nhau) 
    gia = pd.to_numeric(df["GIA_GOI"], errors="coerce")
    df["price_band"] = pd.cut(gia,
        bins=[0, 50_000, 120_000, 1e9],
        labels=["low", "mid", "high"])
    
    return df                       


if __name__ == "__main__":
    from src.data import load
    df = build_features(load())
    cols = ["tenure_days", "usage_trend", "is_out_of_province", "has_usage_history", "price_band"]
    print(df[cols].describe())
    print(df[cols].isna().sum())  
    print(df["price_band"].value_counts())        # kiểm feature mới có bị thiếu nhiều không