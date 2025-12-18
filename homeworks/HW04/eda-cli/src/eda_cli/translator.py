import pandas as pd

df = pd.read_csv("data/example.csv")
df.to_parquet("data/example.parquet", engine="pyarrow")
