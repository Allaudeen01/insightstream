"""
engine/utils/coerce.py
──────────────────────
Robust numeric coercion that handles currency strings ($24,295),
thousands separators, parenthesised negatives, percent values,
and scale suffixes (K/M/B).
"""
from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

_CURRENCY_PATTERN = re.compile(r"[\$€£¥₹]")
_SCALE_SUFFIXES = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def coerce_numeric(series: pd.Series) -> Tuple[pd.Series, dict]:
    """
    Attempt to coerce a string Series to numeric, handling common formats.

    Returns (coerced_series, report) where report = {
        'success_rate': float,         # 0..1, share of non-null inputs that parsed
        'sample_failures': list[str],  # up to 5 raw values that failed
        'detected_format': str,        # 'currency' | 'percent' | 'scale_suffix' | 'plain' | 'mixed'
    }
    """
    if pd.api.types.is_numeric_dtype(series):
        return series, {
            "success_rate": 1.0,
            "sample_failures": [],
            "detected_format": "plain",
        }

    s = series.astype(str).str.strip()
    non_null_mask = s.notna() & (s != "") & (s.str.lower() != "nan")

    detected = "plain"
    if s[non_null_mask].str.contains(_CURRENCY_PATTERN, regex=True).any():
        detected = "currency"
    elif s[non_null_mask].str.endswith("%").any():
        detected = "percent"
    elif s[non_null_mask].str.upper().str[-1:].isin(list(_SCALE_SUFFIXES)).any():
        detected = "scale_suffix"

    cleaned = (
        s.str.replace(_CURRENCY_PATTERN, "", regex=True)
         .str.replace(",", "", regex=False)
         .str.replace(" ", "", regex=False)
    )

    # Parentheses negatives: (100) -> -100
    paren_mask = cleaned.str.match(r"^\(.*\)$")
    cleaned = cleaned.where(~paren_mask, "-" + cleaned.str.strip("()"))

    # Percent
    pct_mask = cleaned.str.endswith("%")
    cleaned = cleaned.where(~pct_mask, cleaned.str.rstrip("%"))

    # Scale suffix
    def _apply_scale(val: str) -> str:
        if not isinstance(val, str):
            return val  # guard: skip floats/NaN that slipped through
        if not val:
            return val
        last = val[-1].upper()
        if last in _SCALE_SUFFIXES and val[:-1].replace(".", "").replace("-", "").isdigit():
            return str(float(val[:-1]) * _SCALE_SUFFIXES[last])
        return val

    cleaned = cleaned.apply(_apply_scale)

    coerced = pd.to_numeric(cleaned, errors="coerce")
    if pct_mask.any():
        coerced = coerced.where(~pct_mask, coerced / 100.0)

    n_in = int(non_null_mask.sum())
    n_out = int(coerced[non_null_mask].notna().sum())
    success_rate = (n_out / n_in) if n_in else 1.0
    failures = s[non_null_mask & coerced.isna()].head(5).tolist()

    return coerced, {
        "success_rate": float(success_rate),
        "sample_failures": failures,
        "detected_format": detected,
    }
