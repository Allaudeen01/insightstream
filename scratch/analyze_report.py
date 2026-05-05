import polars as pl
import json
import os

def analyze():
    path = r"C:\Users\ALI\AppData\Local\Temp\insightstream_sessions\70e5db51-821d-46fb-8082-8f0217d4e1c6\data.parquet"
    df = pl.read_parquet(path)
    
    target = "Happiness Score"
    numeric_cols = ["Happiness Score", "GDP per Capita", "Family Support", "Life Expectancy", "Freedom", "Corruption Index", "Generosity", "Population (Millions)"]
    
    # Correlations
    corrs = []
    for c in numeric_cols:
        if c == target: continue
        pearson = df.select(pl.corr(target, c)).to_series()[0]
        spearman = df.select([
            pl.col(target).rank().alias("r1"),
            pl.col(c).rank().alias("r2")
        ]).select(pl.corr("r1", "r2")).to_series()[0]
        
        corrs.append({
            "column": c,
            "pearson": float(pearson) if pearson is not None else 0,
            "spearman": float(spearman) if spearman is not None else 0
        })
    
    # Regional Breakdown
    regional = df.group_by("Region").agg([
        pl.col("Happiness Score").median().alias("Happiness_median"),
        pl.col("GDP per Capita").median().alias("GDP_median"),
        pl.col("Life Expectancy").median().alias("LifeExpectancy_median"),
        pl.count().alias("count")
    ]).sort("Happiness_median", descending=True).to_dicts()
    
    # Outliers
    outliers = []
    for c in numeric_cols:
        m = df[c].median()
        mad = (df[c] - m).abs().median()
        if mad > 0:
            count = ((df[c] - m).abs() > 3 * mad).sum()
            if count > 0:
                outliers.append({"column": c, "count": int(count)})

    results = {
        "domain": "World Happiness / Socio-economic",
        "target": target,
        "correlations": corrs,
        "regional": regional,
        "outliers": outliers,
        "row_count": len(df)
    }
    
    with open("scratch/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done")

if __name__ == "__main__":
    analyze()
