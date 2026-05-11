import polars as pl
from pathlib import Path
import json
from insight_engine import run_insight_engine

session_id = "027f9e7e-5686-4388-8d4d-f85a00c73d16"
session_path = Path(r"C:\Users\ALI\AppData\Local\Temp\insightstream_sessions") / session_id

print(f"Loading session: {session_id}")

# Load DataFrame
data_file = session_path / "data.parquet"
df = pl.read_parquet(data_file)
print(f"DataFrame loaded: {df.shape}")

# Run insight engine
print("\nRunning insight engine...")
try:
    result = run_insight_engine(df, max_insights=10, max_charts=0)
    print(f"Success! Result keys: {result.keys()}")
    print(f"Strategic brief items: {len(result.get('strategic_brief', []))}")
    print(f"Executive summary length: {len(result.get('executive_summary', ''))}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
