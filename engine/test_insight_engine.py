import polars as pl
import pandas as pd
from insight_engine import run_insight_engine
import re

def strip_emojis(text):
    if not isinstance(text, str): return str(text)
    return re.sub(r'[^\x00-\x7F]+', '', text)

def test_engine_v2():
    # Mock data with an anomaly: Delivery Days and Returns are NOT correlated (Surprise!)
    df = pl.DataFrame({
        "order_id": [f"ORD_{i}" for i in range(100)],
        "category": ["Electronics", "Fashion", "Home"] * 33 + ["Electronics"],
        "price": [100, 20, 50] * 33 + [100],
        "quantity": [1] * 100,
        "returned": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0] * 10,
        "delivery_days": [2, 1, 15, 12, 3, 2, 14, 11, 4, 3] * 10,
        "marketing_spend": [10, 5, 15] * 33 + [10]
    })
    
    print("\nStarting E2E Decision Intelligence Test (V2)...")
    
    results = run_insight_engine(df)
    
    print("\n--- EXECUTIVE SUMMARY ---")
    print(strip_emojis(results["executive_summary"]))
    
    print("\n--- STRATEGIC INSIGHTS (SYNTHESIZED) ---")
    for ins in results["insights"]:
        impact = strip_emojis(ins['impact']).strip()
        print(f"\n[{impact}] {ins['title']}")
        if ins.get("is_unexpected"):
            print("WARNING: UNEXPECTED INSIGHT TRIGGERED")
        print(strip_emojis(ins["description"]))
        
    print("\n--- CORE DRIVERS ---")
    for d in results["key_drivers"]:
        col = strip_emojis(d['column'])
        type_str = strip_emojis(d.get('type', d.get('impact', 'unknown')))
        print(f"Driver: {col} | Impact: {type_str} | r={d.get('r')}")

if __name__ == "__main__":
    test_engine_v2()
