import pandas as pd

sources = {
    3: ("data/raw/04_032026.xlsx", "T3"),
    4: ("data/raw/04_032026.xlsx", "Sheet2"),
    5: ("data/raw/viettel_thanhhoa_202605.xlsx", "Sheet2"),
}

frames = {}
for thang, (path, sheet) in sources.items():
    frames[thang] = pd.read_excel(path, sheet_name=sheet)

common = [c for c in frames[5].columns
          if all(c in frames[t].columns for t in (3, 4))]

parts = []
for thang, d in frames.items():
    d = d[common].copy()
    d["thang"] = thang                    
    parts.append(d)

df = pd.concat(parts, ignore_index=True)

for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip())

df.to_parquet("data/processed/viettel.parquet", index=False)
print(df.shape)
print(df["thang"].value_counts().sort_index())   