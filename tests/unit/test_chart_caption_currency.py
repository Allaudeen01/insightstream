"""
Wave 3 tests for chart caption currency formatting (Task 7.4).

Validates that _generate_chart_summary uses MetricStore.format(fmt='currency')
for monetary columns and falls back to f"{v:,.2f}" when semantics=None.
"""
import sys
import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

from analyzer import _generate_chart_summary
from render.metric_store import MetricStore, MetricKey


# ── helpers ───────────────────────────────────────────────────────────────────

class _S:
    def __init__(self, tag):
        self.tag = tag


def _make_store(col: str, mean_val: float) -> MetricStore:
    store = MetricStore()
    store.put(MetricKey(col, "mean"), mean_val)
    return store


def _make_df_with_groups(group_col: str, target_col: str,
                          groups: dict) -> pd.DataFrame:
    """Build a DataFrame with categorical groups and numeric target."""
    rows = []
    for grp, vals in groups.items():
        for v in vals:
            rows.append({group_col: grp, target_col: v})
    return pd.DataFrame(rows)


# ── Task 7.4: Currency formatting tests ──────────────────────────────────────

def test_monetary_bar_chart_uses_currency_format():
    """
    Aggregated bar chart with monetary y_col should show $18,558 not 18558.23.
    """
    df = _make_df_with_groups("card_type", "credit_limit", {
        "Debit":          [18558.0] * 100,
        "Credit":         [26000.0] * 100,
        "Debit (Prepaid)": [64.0] * 100,
    })
    sem = {
        "card_type":    _S("categorical_meaningful"),
        "credit_limit": _S("monetary"),
    }
    store = _make_store("credit_limit", 18558.0)

    caption = _generate_chart_summary(
        "bar", "card_type", "credit_limit", df,
        semantics=sem, store=store,
    )

    # Should contain $ sign (currency format)
    assert "$" in caption, f"Expected $ in caption, got: {caption!r}"
    # Should NOT contain raw float like 18558.23 or 26000.00
    assert "18558.23" not in caption
    assert "26000.00" not in caption


def test_non_monetary_bar_chart_uses_plain_format():
    """
    Bar chart with non-monetary y_col should use plain f'{v:,.2f}' format.
    """
    df = _make_df_with_groups("card_type", "num_cards_issued", {
        "Debit":  [1.5] * 100,
        "Credit": [2.0] * 100,
    })
    sem = {
        "card_type":        _S("categorical_meaningful"),
        "num_cards_issued": _S("numeric_meaningful"),
    }
    store = _make_store("num_cards_issued", 1.75)

    caption = _generate_chart_summary(
        "bar", "card_type", "num_cards_issued", df,
        semantics=sem, store=store,
    )

    # No $ sign for non-monetary
    assert "$" not in caption, f"Unexpected $ in non-monetary caption: {caption!r}"


def test_fallback_when_semantics_none():
    """
    When semantics=None, caption uses plain f'{v:,.2f}' format (no currency).
    """
    df = _make_df_with_groups("card_type", "credit_limit", {
        "Debit":  [18558.0] * 50,
        "Credit": [26000.0] * 50,
    })

    caption = _generate_chart_summary(
        "bar", "card_type", "credit_limit", df,
        semantics=None, store=None,
    )

    # No $ sign when semantics is None
    assert "$" not in caption, f"Unexpected $ when semantics=None: {caption!r}"
    # Should still produce a valid string
    assert isinstance(caption, str)
    assert len(caption) > 0


def test_fallback_when_store_none():
    """
    When store=None but semantics is provided, falls back to plain format.
    """
    df = _make_df_with_groups("card_type", "credit_limit", {
        "Debit":  [18558.0] * 50,
        "Credit": [26000.0] * 50,
    })
    sem = {
        "card_type":    _S("categorical_meaningful"),
        "credit_limit": _S("monetary"),
    }

    caption = _generate_chart_summary(
        "bar", "card_type", "credit_limit", df,
        semantics=sem, store=None,
    )

    # No $ sign when store is None (can't look up MetricKey)
    assert "$" not in caption, f"Unexpected $ when store=None: {caption!r}"
    assert isinstance(caption, str)


def test_histogram_monetary_column_no_crash():
    """
    Histogram of a monetary column should not crash and should produce a caption.
    """
    import numpy as np
    df = pd.DataFrame({
        "credit_limit": np.random.normal(18000, 5000, 200).tolist(),
    })
    sem = {"credit_limit": _S("monetary")}
    store = _make_store("credit_limit", 18000.0)

    caption = _generate_chart_summary(
        "histogram", "credit_limit", None, df,
        semantics=sem, store=store,
    )

    assert isinstance(caption, str)
    assert len(caption) > 0


def test_scatter_monetary_y_col():
    """
    Scatter plot with monetary y_col should produce a valid caption.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "num_cards_issued": rng.integers(1, 4, 200).tolist(),
        "credit_limit":     rng.normal(18000, 5000, 200).tolist(),
    })
    sem = {
        "num_cards_issued": _S("numeric_meaningful"),
        "credit_limit":     _S("monetary"),
    }
    store = _make_store("credit_limit", 18000.0)

    caption = _generate_chart_summary(
        "scatter", "num_cards_issued", "credit_limit", df,
        semantics=sem, store=store,
    )

    assert isinstance(caption, str)
    assert len(caption) > 0


def test_full_number_not_k_suffix():
    """
    Currency format must use full number ($18,558) not K-suffix ($18.6K).
    This ensures consistency with segmentation headlines in the same report.
    """
    df = _make_df_with_groups("card_type", "credit_limit", {
        "Debit":  [18558.0] * 100,
        "Credit": [26000.0] * 100,
    })
    sem = {
        "card_type":    _S("categorical_meaningful"),
        "credit_limit": _S("monetary"),
    }
    # Store with the actual mean value
    store = MetricStore()
    store.put(MetricKey("credit_limit", "mean"), 22279.0)  # average of both groups

    caption = _generate_chart_summary(
        "bar", "card_type", "credit_limit", df,
        semantics=sem, store=store,
    )

    # K-suffix should NOT appear
    assert "K" not in caption or "$" not in caption.split("K")[0][-5:], (
        f"K-suffix formatting found in caption: {caption!r}"
    )
