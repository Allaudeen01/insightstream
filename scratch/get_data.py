import polars as pl
import json
import os

path = r"C:\Users\ALI\AppData\Local\Temp\insightstream_sessions\70e5db51-821d-46fb-8082-8f0217d4e1c6\data.parquet"
df = pl.read_parquet(path)
data = {
    "columns": df.columns,
    "head": df.head(10).to_dicts()
}
with open("scratch/output.json", "w") as f:
    json.dump(data, f)
print("Done")
