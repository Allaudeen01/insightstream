import polars as pl
import pandas as pd
from insight_engine import BusinessRuleEngine, DataProfile, ComputedMetric, ColumnProfile

def test_engine():
    engine = BusinessRuleEngine()
    
    # Mock data for return rate 0% vs high%
    df = pl.DataFrame({
        "Category": ["A", "A", "B", "B", "C", "C"] * 4, # 24 rows
        "Returned": [0, 0, 1, 1, 0, 1] * 4, 
        "Delivery_Days": [1, 2, 8, 9, 3, 4] * 4, 
        "Payment_Method": ["Card", "Card", "Card", "UPI", "UPI", "Cash"] * 4,
        "Price": [20, 20, 50, 50, 50, 50] * 4, # A has very low price (revenue), meaning worst revenue performer
        "Quantity": [1] * 24
    })
    
    profile = DataProfile(row_count=24, col_count=6)
    profile.category_col = "Category"
    profile.return_col = "Returned"
    profile.delivery_days_col = "Delivery_Days"
    profile.categoricals = ["Category", "Payment_Method"]
    profile.profiles["Category"] = ColumnProfile("Category", "categorical")
    profile.profiles["Payment_Method"] = ColumnProfile("Payment_Method", "categorical")
    profile.profiles["Returned"] = ColumnProfile("Returned", "binary")
    profile.profiles["Delivery_Days"] = ColumnProfile("Delivery_Days", "numerical")
    profile.profiles["Price"] = ColumnProfile("Price", "numerical")
    profile.profiles["Quantity"] = ColumnProfile("Quantity", "numerical")
    profile.price_col = "Price"
    profile.qty_col = "Quantity"
    profile.numericals = ["Delivery_Days", "Price", "Quantity"]

    # Add the series
    profile._return_count_series = df["Returned"]
    profile._revenue_series = df["Price"] * df["Quantity"]

    metrics = {
        "return_rate": ComputedMetric("Return Rate", 0.5, "50%", "Test")
    }

    # Evaluate
    insights, warnings = engine.evaluate(df, profile, metrics)
    
    print("\n=== GENERATED INSIGHTS ===")
    for ins in insights:
        print(f"[{ins.confidence.upper()}] Pts: {ins.score:.1f} | {ins.title}")
        print(f"   Desc: {ins.description}")
        print(f"   Rec: {ins.recommendation}\n")

if __name__ == "__main__":
    test_engine()
