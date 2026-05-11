import polars as pl
from pathlib import Path
import json

session_id = "027f9e7e-5686-4388-8d4d-f85a00c73d16"
session_path = Path(r"C:\Users\ALI\AppData\Local\Temp\insightstream_sessions") / session_id

print(f"Loading session: {session_id}")
print(f"Session path: {session_path}")
print(f"Exists: {session_path.exists()}")

# Load metadata
metadata_file = session_path / "metadata.json"
with open(metadata_file, "r") as f:
    metadata = json.load(f)
print(f"Metadata: {metadata}")

# Load DataFrame
data_file = session_path / "data.parquet"
try:
    df = pl.read_parquet(data_file)
    print(f"DataFrame loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"Dtypes: {df.dtypes}")
except Exception as e:
    print(f"Error loading DataFrame: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
