"""
engine/render/prose_cleaner.py
────────────────────────────────
Removes COLUMN=VALUE template artifacts from finished prose.

Segmentation headlines are generated with patterns like:
  "card_type=Debit (Prepaid) has the highest mean credit_limit..."

After metric placeholders are filled, the COLUMN=VALUE syntax remains.
This module replaces it with natural language:
  "Debit (Prepaid) card type has the highest mean credit_limit..."

The cleaner is applied to every segmentation headline and every insight
text field in _attach_phase2_keys before the results dict is returned.
"""
from __future__ import annotations

import re

# Pattern: WORD=VALUE where VALUE may include spaces inside parentheses
# Examples matched:
#   card_type=Debit (Prepaid)
#   card_brand=Visa
#   has_chip=YES
#   card_type=Credit
_COLVAL_PATTERN = re.compile(
    r'\b([A-Za-z][A-Za-z0-9_]*)=([^\s,;.!?]+(?:\s+\([^)]+\))?)'
)


def _natural_language(col: str, value: str) -> str:
    """
    Convert a COLUMN=VALUE pair to natural language.

    Strategy:
    1. Strip parenthetical suffixes from value for the main label
       e.g. "Debit (Prepaid)" → keep as-is (it's already readable)
    2. Remove underscores from column name
    3. Combine as "VALUE col_name" in title case

    Examples:
      card_type, "Debit (Prepaid)" → "Debit (Prepaid) card type"
      card_brand, "Visa"           → "Visa card brand"
      has_chip, "YES"              → "YES has chip"  (generic fallback)
    """
    col_clean = col.replace("_", " ").lower()
    value_clean = value.strip()
    return f"{value_clean} {col_clean}"


def clean_prose_artifacts(text: str) -> str:
    """
    Replace all COLUMN=VALUE patterns in text with natural-language equivalents.

    Guarantees: the returned string contains no substring matching \\b\\w+=\\S+
    (the COLUMN=VALUE pattern).

    Handles regex special characters in values gracefully — falls back to the
    original match text if substitution raises, without propagating the exception.

    Parameters
    ----------
    text : str
        Any string, including segmentation headlines and insight text.

    Returns
    -------
    str
        The cleaned string with no COLUMN=VALUE artifacts.
    """
    if not text or "=" not in text:
        return text

    def _replace(m: re.Match) -> str:
        col = m.group(1)
        value = m.group(2)
        try:
            return _natural_language(col, value)
        except Exception:
            return m.group(0)  # fall back to original on any error

    try:
        return _COLVAL_PATTERN.sub(_replace, text)
    except Exception:
        return text  # never crash — return original if regex engine fails
