"""
engine/analysis/extremum_validator.py
──────────────────────────────────────
Post-LLM validator that drops insights whose extremum claims (highest,
lowest, most, least, etc.) are contradicted by the ground-truth data.

Prevents the "two brands both claimed highest" class of bug where the LLM
emits bare numbers instead of {{metric:...}} placeholders and gets the
ranking wrong.

Key design decisions:
- Only validates a (cat, num) pair when the insight explicitly mentions
  the categorical column name (not just a value), to avoid cross-pair
  false positives (e.g. "credit" in "credit limit" triggering card_type
  validation).
- Uses character-boundary checks to avoid matching column-name substrings
  (e.g. "credit" inside "credit_limit").
- Checks longer values before shorter ones so "Debit (Prepaid)" is matched
  before "Debit".
"""
from __future__ import annotations

import re as _re
import pandas as pd

EXTREMUM_WORDS = (
    "highest", "lowest", "most", "least",
    "largest", "smallest", "greatest", "biggest",
    "minimum", "maximum",
)

_TOP_WORDS = {"highest", "most", "largest", "greatest", "biggest", "maximum"}
_BOT_WORDS = {"lowest", "least", "smallest", "minimum"}


def _value_in_text(val_lc: str, text: str, all_values: list[str]) -> bool:
    """
    Return True if val_lc appears in text as a standalone token (not embedded
    in a column name like credit_limit) and no longer value from the same
    column also appears in the text (to avoid "debit" matching when
    "debit (prepaid)" is present).
    """
    if val_lc not in text:
        return False

    # Skip if a longer value from the same column is also present
    for other in all_values:
        other_lc = other.lower()
        if other_lc != val_lc and val_lc in other_lc and other_lc in text:
            return False

    # Require the value to appear with non-alpha/non-underscore boundaries
    for m in _re.finditer(_re.escape(val_lc), text):
        start, end = m.start(), m.end()
        before = text[start - 1] if start > 0 else " "
        after  = text[end]       if end < len(text) else " "
        if (before.isalpha() or before == "_"):
            continue
        if (after.isalpha() or after == "_"):
            continue
        return True

    return False


def validate_extremum_claims(
    insights: list[dict],
    df: pd.DataFrame,
    semantics: dict,
) -> tuple[list[dict], list[str]]:
    """
    For each insight that claims an extremum about a categorical group,
    verify against the ground-truth in df. Drop insights whose claim is
    contradicted by the data.

    Returns (kept_insights, dropped_reasons).
    """
    kept: list[dict] = []
    dropped: list[str] = []

    cats = [c for c, s in semantics.items() if s.tag == "categorical_meaningful"]
    nums = [c for c, s in semantics.items() if s.tag in ("monetary", "numeric_meaningful")]

    # Precompute ground-truth extremums for every (cat, num) pair
    extremums: dict[tuple, dict] = {}
    for g in cats:
        for t in nums:
            if not pd.api.types.is_numeric_dtype(df[t]):
                continue
            means = df.groupby(g, dropna=False)[t].mean().dropna()
            if len(means) >= 2:
                extremums[(g, t)] = {
                    "top": str(means.idxmax()),
                    "bot": str(means.idxmin()),
                }

    for ins in insights:
        text = ((ins.get("text") or "") + " " + (ins.get("title") or "")).lower()

        # Fast path: no extremum word → keep unconditionally
        if not any(w in text for w in EXTREMUM_WORDS):
            kept.append(ins)
            continue

        contradicted = False
        reason = ""

        for (cat, num), ext in extremums.items():
            if contradicted:
                break

            # Only check if this numeric column is mentioned in the insight
            num_lc = num.lower()
            if num_lc not in text and num_lc.replace("_", " ") not in text:
                continue

            # Only validate this (cat, num) pair if the categorical column
            # name is explicitly mentioned — prevents cross-pair false positives
            # (e.g. "credit" in "credit limit" triggering card_type validation).
            cat_lc = cat.lower()
            cat_mentioned = (
                cat_lc in text
                or cat_lc.replace("_", " ") in text
            )
            if not cat_mentioned:
                continue

            # Sort values longest-first: "Debit (Prepaid)" before "Debit"
            cat_values_sorted = sorted(
                df[cat].astype(str).unique().tolist(),
                key=len,
                reverse=True,
            )

            for word in EXTREMUM_WORDS:
                if word not in text:
                    continue
                is_top = word in _TOP_WORDS
                is_bot = word in _BOT_WORDS

                for val in cat_values_sorted:
                    val_lc = val.lower()
                    if len(val_lc) < 2:
                        continue

                    if not _value_in_text(val_lc, text, cat_values_sorted):
                        continue

                    # Insight mentions `val` as the extremum for this pair.
                    # Check against ground truth.
                    if is_top and val != ext["top"]:
                        contradicted = True
                        reason = (
                            f"Dropped: claims '{val}' is {word} for "
                            f"{cat}×{num}, but ground truth says "
                            f"'{ext['top']}' is."
                        )
                        break
                    if is_bot and val != ext["bot"]:
                        contradicted = True
                        reason = (
                            f"Dropped: claims '{val}' is {word} for "
                            f"{cat}×{num}, but ground truth says "
                            f"'{ext['bot']}' is."
                        )
                        break

                if contradicted:
                    break

        if contradicted:
            dropped.append(reason)
        else:
            kept.append(ins)

    return kept, dropped
