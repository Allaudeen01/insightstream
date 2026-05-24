"""
engine/classifiers/semantics.py
────────────────────────────────
Tags each DataFrame column with a semantic role so downstream code
can apply the right analysis, formatting, and exclusion rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

SemanticTag = Literal[
    "identifier",
    "random_token",
    "monetary",
    "temporal",
    "numeric_meaningful",
    "categorical_meaningful",
    "categorical_degenerate",
    "free_text",
]


@dataclass
class ColumnSemantics:
    name: str
    tag: SemanticTag
    confidence: float           # 0..1
    reasons: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)  # e.g., {'currency': 'USD'} for monetary


# ── Name-based hint patterns (primary signal) ────────────────────────────────
_IDENTIFIER_PATTERNS = [
    r"\bid\b", r"_id$", r"^id_", r"\buuid\b", r"\bguid\b",
]
_RANDOM_TOKEN_PATTERNS = [
    r"\bcvv\b", r"\bpin\b(?!_)", r"\bcard_number\b",
    r"\bcvc\b", r"\btoken\b", r"\bhash\b", r"\bssn\b",
]
_MONETARY_PATTERNS = [
    r"\bprice\b", r"\bcost\b", r"\bamount\b", r"\brevenue\b",
    r"limit", r"\bsalary\b", r"\bbalance\b", r"\bfee\b",
]
_TEMPORAL_PATTERNS = [
    r"date", r"_at$", r"time", r"(?:^|_)year(?:_|$)",
    r"expires?", r"\bcreated\b", r"\bupdated\b",
]


def _name_hit(name: str, patterns: list[str]) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in patterns)


def classify_column(col_name: str, series: pd.Series) -> ColumnSemantics:
    n = len(series)
    nunique = series.nunique(dropna=True)
    reasons: list[str] = []

    # 1) Degenerate — zero or one distinct non-null value
    if nunique <= 1:
        return ColumnSemantics(
            col_name, "categorical_degenerate", 1.0,
            reasons=[f"only {nunique} unique value(s)"],
        )

    # 2) Identifier — name match OR uniqueness ratio near 1.0
    uniq_ratio = nunique / n if n else 0
    if _name_hit(col_name, _IDENTIFIER_PATTERNS) or uniq_ratio >= 0.98:
        reason = (
            "name matches id pattern"
            if _name_hit(col_name, _IDENTIFIER_PATTERNS)
            else f"uniqueness ratio {uniq_ratio:.2f}"
        )
        return ColumnSemantics(col_name, "identifier", 0.95, reasons=[reason])

    # 3) Random token — name-based
    if _name_hit(col_name, _RANDOM_TOKEN_PATTERNS):
        return ColumnSemantics(
            col_name, "random_token", 0.95,
            reasons=["name matches random-token pattern"],
        )

    # 4) Temporal — name or dtype
    if _name_hit(col_name, _TEMPORAL_PATTERNS) or pd.api.types.is_datetime64_any_dtype(series):
        return ColumnSemantics(
            col_name, "temporal", 0.9,
            reasons=["name or dtype indicates temporal"],
        )

    # 5) Monetary — name-based, but only when column is numeric (or will be after coercion)
    #    We check name first; if the series is still object dtype here, the caller
    #    should have applied coerce_numeric before classify_column.
    if _name_hit(col_name, _MONETARY_PATTERNS) and pd.api.types.is_numeric_dtype(series):
        return ColumnSemantics(
            col_name, "monetary", 0.9,
            reasons=["name matches monetary pattern"],
            extras={"currency": "USD"},
        )

    # 5b) Monetary by name alone (string column not yet coerced — still tag it)
    if _name_hit(col_name, _MONETARY_PATTERNS):
        return ColumnSemantics(
            col_name, "monetary", 0.75,
            reasons=["name matches monetary pattern (not yet numeric)"],
            extras={"currency": "USD"},
        )

    # 6) Numeric meaningful — numeric dtype, not flagged above
    if pd.api.types.is_numeric_dtype(series):
        return ColumnSemantics(
            col_name, "numeric_meaningful", 0.85,
            reasons=["numeric dtype, no special-name match"],
        )

    # 7) Free text — object dtype, very high cardinality, mostly long strings
    if series.dtype == object:
        avg_len = series.astype(str).str.len().mean()
        if uniq_ratio > 0.7 and avg_len > 30:
            return ColumnSemantics(
                col_name, "free_text", 0.8,
                reasons=[f"high cardinality + long strings (avg_len={avg_len:.0f})"],
            )

    # 8) Default: categorical meaningful
    return ColumnSemantics(
        col_name, "categorical_meaningful", 0.8,
        reasons=[f"{nunique} distinct values, object dtype"],
    )


def classify_dataframe(df: pd.DataFrame) -> dict[str, ColumnSemantics]:
    """Return {col_name: ColumnSemantics} for every column in df."""
    return {c: classify_column(c, df[c]) for c in df.columns}
