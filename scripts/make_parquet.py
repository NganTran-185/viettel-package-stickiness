import pandas as pd 
df = pd.read_excel('data/raw/viettel_thanhhoa_202605.xlsx', sheet_name='Sheet2')

for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = df[c].where(df[c].isna(), df[c].astype(str))
df.to_parquet('data/processed/viettel.parquet', index=False)
print(df.shape)