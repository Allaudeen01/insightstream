"""
engine/analysis/outlier_profile.py
────────────────────────────────────
Characterises the categorical profile of an outlier set detected by
IsolationForest (or any method that returns a list of row indices).

Instead of just counting outliers, this module answers:
"Who are the outliers?" — e.g., "96% of outliers have card_type=Credit".

The narrative is injected into the LLM prompt so findings about anomalies
describe the outlier population rather than just reporting a count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class OutlierProfile:
    n_outliers: int
    pct_of_total: float                  # n_outliers / total_rows (exact)
    modal_profile: dict[str, str]        # col → most common value among outliers
    modal_pct: dict[str, float]          # col → fraction of outliers with modal value
    narrative: str                       # human-readable summary sentence


def profile_outliers(
    df: pd.DataFrame,
    semantics: dict,
    anomaly_indices,                     # list[int] from df.index[preds == -1]
) -> Optional[OutlierProfile]:
    """
    Compute the categorical modal profile of the outlier rows.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset (not mutated).
    semantics : dict
        {col_name: ColumnSemantics} — used to identify categorical columns.
    anomaly_indices : list[int] | None
        Row positions returned by _detect_anomalies (df.index[preds == -1]).
        Pass None or [] to get None back.

    Returns
    -------
    OutlierProfile | None
        None when anomaly_indices is None or empty.
        OutlierProfile.pct_of_total == len(anomaly_indices) / len(df) exactly.
    """
    if not anomaly_indices:
        return None

    # Filter to valid indices (guard against stale index values)
    valid_idx = [i for i in anomaly_indices if i in df.index]
    n_invalid = len(anomaly_indices) - len(valid_idx)
    if n_invalid > 0:
        print(f"[outlier_profile] Filtered {n_invalid} invalid indices")
    if not valid_idx:
        return None

    outlier_df = df.loc[valid_idx]
    n = len(outlier_df)
    pct = n / len(df)   # exact fraction — not rounded here

    # Categorical columns with ≤ 20 unique values
    cat_cols = [
        c for c, s in semantics.items()
        if s.tag == "categorical_meaningful"
        and c in df.columns
        and df[c].nunique() <= 20
    ]

    modal_profile: dict[str, str] = {}
    modal_pct: dict[str, float] = {}

    for col in cat_cols:
        counts = outlier_df[col].value_counts(dropna=True)
        if len(counts) == 0:
            continue
        top_val = str(counts.index[0])
        top_pct = float(counts.iloc[0]) / n
        modal_profile[col] = top_val
        modal_pct[col] = round(top_pct, 3)

    # Build narrative: only mention columns where ≥ 70% of outliers share the modal value
    parts = [
        f"{modal_pct[col]:.0%} of outliers have {col}={modal_profile[col]}"
        for col in modal_profile
        if modal_pct[col] >= 0.70
    ]
    narrative = "; ".join(parts) if parts else f"{n} outliers detected"

    return OutlierProfile(
        n_outliers=n,
        pct_of_total=pct,          # exact, not rounded
        modal_profile=modal_profile,
        modal_pct=modal_pct,
        narrative=narrative,
    )


def build_outlier_profile_block(profile: Optional[OutlierProfile]) -> str:
    """
    Return a prompt injection block describing the outlier population.
    Returns empty string when profile is None.
    """
    if profile is None:
        return ""

    lines = [
        "\n=== OUTLIER CHARACTERIZATION ===",
        f"{profile.n_outliers} outliers ({profile.pct_of_total:.1%} of data).",
        f"Modal profile: {profile.narrative}",
        "INSTRUCTION: Use this profile to characterize outliers in findings. "
        "Do NOT just report the count — describe who the outliers are.",
        "=== END OUTLIER CHARACTERIZATION ===\n",
    ]
    return "\n".join(lines)
