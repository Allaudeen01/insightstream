"""
engine/analysis/segmentation.py
─────────────────────────────────
Auto multivariate segmentation: for every (categorical_meaningful ×
numeric/monetary) pair, check whether group means differ enough to be
a finding, gated by minimum sample size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Segmentation:
    group_col: str
    target_col: str
    group_stats: pd.DataFrame   # index=group, columns=[count, mean, median, std]
    spread_ratio: float          # max(mean) / min(mean) for non-zero means; inf if any zero
    min_group_n: int
    is_significant: bool         # passed both small-n and spread thresholds
    headline: str                # human-readable summary, with metric placeholders


def segment(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    semantics: Optional[dict] = None,
    min_n: int = 30,
    spread_threshold: float = 2.0,
) -> Segmentation:
    g = (
        df.groupby(group_col, dropna=False)[target_col]
        .agg(["count", "mean", "median", "std"])
        .sort_values("mean", ascending=False)
    )
    means = g["mean"].replace(0, float("nan")).dropna()
    spread = (means.max() / means.min()) if len(means) >= 2 else 1.0
    min_n_obs = int(g["count"].min())
    sig = (min_n_obs >= min_n) and (spread >= spread_threshold) and (len(g) >= 2)

    top = g.index[0]
    bot = g.index[-1]

    # Use currency formatting for monetary targets (D2 fix)
    is_monetary = (
        semantics is not None
        and target_col in semantics
        and semantics[target_col].tag == "monetary"
    )
    fmt = "fmt:currency:" if is_monetary else ""

    headline = (
        f"{group_col}={top} has the highest mean {target_col} "
        f"({{{{{fmt}metric:{target_col}.mean|scope={group_col}={top}}}}}), "
        f"vs {group_col}={bot} at "
        f"({{{{{fmt}metric:{target_col}.mean|scope={group_col}={bot}}}}}) — "
        f"a {spread:.1f}x spread."
    )
    return Segmentation(
        group_col=group_col,
        target_col=target_col,
        group_stats=g,
        spread_ratio=float(spread),
        min_group_n=min_n_obs,
        is_significant=sig,
        headline=headline,
    )


def auto_segment_all(
    df: pd.DataFrame,
    semantics: dict,
    min_n: int = 30,
    spread_threshold: float = 2.0,
) -> list[Segmentation]:
    """
    Run segmentation for all (categorical_meaningful × numeric/monetary) pairs.
    Returns significant findings sorted by spread_ratio descending.

    Promotion rules:
      spread_ratio >= 5  → CRITICAL
      spread_ratio >= 2  → IMPORTANT
    """
    findings: list[Segmentation] = []
    cat_cols = [c for c, s in semantics.items() if s.tag == "categorical_meaningful"]
    num_cols = [c for c, s in semantics.items() if s.tag in ("numeric_meaningful", "monetary")]

    for g in cat_cols:
        for t in num_cols:
            if not pd.api.types.is_numeric_dtype(df[t]):
                continue
            # Pass semantics so monetary targets get currency formatting
            seg = segment(df, g, t, semantics=semantics,
                          min_n=min_n, spread_threshold=spread_threshold)
            if seg.is_significant:
                findings.append(seg)

    # Rank by spread descending — highest spread is most newsworthy
    findings.sort(key=lambda s: s.spread_ratio, reverse=True)
    return findings


def impact_for(seg: Segmentation) -> str:
    """Map spread_ratio to impact label."""
    if seg.spread_ratio >= 5:
        return "CRITICAL"
    return "IMPORTANT"
