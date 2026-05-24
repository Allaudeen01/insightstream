"""
engine/render/template_guards.py
──────────────────────────────────
Guards against common template-generation bugs:
  - "top 5 categories" when there are only 4
  - "relatively balanced" when one category is 52% and another is 3%
"""
from __future__ import annotations

import pandas as pd


def safe_top_n(series: pd.Series, n: int) -> tuple[pd.Series, int]:
    """
    Return (top_n_counts, actual_n_used).
    Never claims more categories than actually exist.
    """
    actual = min(n, series.nunique(dropna=False))
    return series.value_counts(dropna=False).head(actual), actual


def balance_descriptor(counts: pd.Series) -> str:
    """
    Return one of: 'dominated', 'uneven', 'balanced', 'single-valued'.

    Rules:
      dominated  — top category ≥ 50% of total, OR max/min ratio ≥ 10
      balanced   — max/min ratio ≤ 1.5
      uneven     — everything else
    """
    if len(counts) < 2:
        return "single-valued"
    nz = counts[counts > 0]
    if len(nz) < 2:
        return "single-valued"
    ratio = nz.max() / nz.min()
    top_share = nz.max() / nz.sum()
    if top_share >= 0.5 or ratio >= 10:
        return "dominated"
    if ratio <= 1.5:
        return "balanced"
    return "uneven"


def describe_distribution(col_name: str, counts: pd.Series) -> str:
    """
    Generate a one-sentence description of a categorical distribution.
    Uses balance_descriptor to pick the right framing.
    """
    desc = balance_descriptor(counts)
    top = counts.idxmax()
    top_n = int(counts.max())
    top_pct = top_n / counts.sum()
    n_cats = int((counts > 0).sum())

    if desc == "single-valued":
        return (
            f"{col_name} has only one distinct value ('{top}') — "
            f"it carries no analytical signal."
        )
    if desc == "dominated":
        return (
            f"{col_name} is dominated by '{top}' "
            f"({top_n:,} of {int(counts.sum()):,}, {top_pct:.0%}). "
            f"Conclusions about {col_name} will largely reflect this segment."
        )
    if desc == "balanced":
        return (
            f"{col_name} is roughly balanced across {n_cats} categories, "
            f"with '{top}' slightly ahead at {top_pct:.0%}."
        )
    # uneven
    return (
        f"{col_name} is uneven across {n_cats} categories, "
        f"with '{top}' the largest at {top_pct:.0%}."
    )
