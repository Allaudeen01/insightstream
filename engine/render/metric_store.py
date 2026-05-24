"""
engine/render/metric_store.py
──────────────────────────────
Canonical store for pre-computed metrics.
The LLM references metrics via {{metric:COLUMN.STATISTIC}} placeholders;
the renderer resolves them from this store — preventing hallucinated numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class MetricKey:
    column: str
    statistic: str            # 'mean' | 'median' | 'min' | 'max' | 'std' | 'count' | 'pct_missing' | 'nunique'
    scope: Optional[str] = None  # e.g., 'has_chip=YES' or None for global


class MetricStore:
    def __init__(self):
        self._store: dict[MetricKey, float] = {}

    def put(self, key: MetricKey, value: float) -> None:
        """Store a metric, rounded to consistent precision."""
        self._store[key] = round(float(value), 4)

    def get(self, key: MetricKey) -> float:
        return self._store[key]

    def format(self, key: MetricKey, fmt: str = "auto") -> str:
        v = self.get(key)
        if fmt == "currency":
            return f"${v:,.0f}"
        if fmt == "percent":
            return f"{v:.1%}"
        if fmt == "integer":
            return f"{int(round(v)):,}"
        # auto
        return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4f}"

    def __contains__(self, key: MetricKey) -> bool:
        return key in self._store

    def keys(self):
        return self._store.keys()


def build_metric_store(df: pd.DataFrame, semantics: dict) -> MetricStore:
    """
    Compute global and scoped metrics for all meaningful columns.
    Scoped metrics cover every (categorical_meaningful × numeric/monetary) pair.
    """
    store = MetricStore()

    for col, sem in semantics.items():
        if sem.tag in ("identifier", "random_token", "free_text", "categorical_degenerate"):
            continue

        # Global stats
        if pd.api.types.is_numeric_dtype(df[col]):
            store.put(MetricKey(col, "mean"),   df[col].mean())
            store.put(MetricKey(col, "median"), df[col].median())
            store.put(MetricKey(col, "min"),    df[col].min())
            store.put(MetricKey(col, "max"),    df[col].max())
            store.put(MetricKey(col, "std"),    df[col].std())

        store.put(MetricKey(col, "pct_missing"), df[col].isna().mean())
        store.put(MetricKey(col, "nunique"),     float(df[col].nunique()))
        store.put(MetricKey(col, "count"),       float(df[col].notna().sum()))

    # Scoped means: every categorical_meaningful × numeric/monetary pair
    cat_cols = [c for c, s in semantics.items() if s.tag == "categorical_meaningful"]
    num_cols = [c for c, s in semantics.items() if s.tag in ("numeric_meaningful", "monetary")]

    for group_col in cat_cols:
        for target_col in num_cols:
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                continue
            try:
                grp = df.groupby(group_col, dropna=False)[target_col].mean()
                for val, mean_val in grp.items():
                    if pd.notna(mean_val):
                        store.put(
                            MetricKey(target_col, "mean", f"{group_col}={val}"),
                            mean_val,
                        )
            except Exception:
                pass

    return store
