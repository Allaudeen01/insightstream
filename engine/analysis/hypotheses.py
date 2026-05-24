"""
engine/analysis/hypotheses.py
──────────────────────────────
Detects ambiguous evidence and emits structured hypotheses rather than
confident assertions. Rendered in a dedicated "Open Questions" section.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Hypothesis:
    observation: str
    candidates: list[str]       # 2–4 plausible explanations
    disambiguating_info: str    # what data would resolve this


def detect_ambiguities(
    df: pd.DataFrame,
    semantics: dict,
    coerce_reports: dict,
) -> list[Hypothesis]:
    """
    Scan for three classes of ambiguity:
      1. Partial parse failure (coerce success_rate 50–95%)
      2. Zero values clustered at the bottom of a monetary column (>5%)
      3. Duplicate values without a unique identifier
    """
    out: list[Hypothesis] = []

    # 1) Partial parse failure
    for col, rpt in coerce_reports.items():
        sr = rpt.get("success_rate", 1.0)
        if 0.50 <= sr < 0.95:
            out.append(Hypothesis(
                observation=(
                    f"Column '{col}' parsed successfully for "
                    f"{sr:.0%} of values; "
                    f"sample failures: {rpt.get('sample_failures', [])!r}."
                ),
                candidates=[
                    "Mixed formatting (some rows use a different unit or symbol)",
                    "Genuine sentinel values for missing data (e.g., '-', 'N/A')",
                    "A second logical column merged into this one",
                ],
                disambiguating_info=(
                    "Inspect the failing rows alongside related columns "
                    "to determine whether the variation is structural or noise."
                ),
            ))

    # 2) Zero values clustered at the bottom of a monetary column
    for col, sem in semantics.items():
        if sem.tag == "monetary" and pd.api.types.is_numeric_dtype(df[col]):
            zero_count = int((df[col] == 0).sum())
            zero_share = float((df[col] == 0).mean())
            if zero_count > 0:  # any zeros in a monetary column warrant a hypothesis
                out.append(Hypothesis(
                    observation=(
                        f"{zero_count:,} of '{col}' values are exactly zero "
                        f"({zero_share:.1%} of rows)."
                    ),
                    candidates=[
                        f"A subset of accounts genuinely has a {col} of 0 "
                        f"(e.g., prepaid, closed)",
                        "Zero is being used as a sentinel for missing/unknown",
                        "A coding convention for unlimited or uncapped",
                    ],
                    disambiguating_info=(
                        f"Cross-tabulate {col}==0 against account-status or "
                        "product-type columns if available."
                    ),
                ))

    # 3) Duplicates without a unique identifier
    id_cols = [c for c, s in semantics.items() if s.tag == "identifier"]
    cat_meaningful = [c for c, s in semantics.items() if s.tag == "categorical_meaningful"]
    for col in cat_meaningful:
        dup_share = 1 - (df[col].nunique() / len(df))
        if dup_share > 0.3 and not id_cols:
            out.append(Hypothesis(
                observation=(
                    f"'{col}' has {dup_share:.0%} repeated values, "
                    f"and the dataset has no unique row identifier."
                ),
                candidates=[
                    "Different entities sharing the same label (e.g., homonyms)",
                    "The same entity appearing in multiple rows (multiple records per entity)",
                    "A merge from multiple sources without deduplication",
                ],
                disambiguating_info=(
                    "Add or join a primary key, or compare other columns "
                    "within duplicate groups to estimate the rate of each case."
                ),
            ))

    return out
