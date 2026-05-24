"""
Wave 1 tests for engine/analysis/outlier_profile.py

Tasks 3.2 (Property 5: Outlier Fraction Accuracy), 3.3 (unit tests)
"""
import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

from analysis.outlier_profile import profile_outliers, OutlierProfile


# ── helpers ───────────────────────────────────────────────────────────────────

class _S:
    def __init__(self, tag):
        self.tag = tag


def _cards_sem():
    return {
        "card_type":  _S("categorical_meaningful"),
        "card_brand": _S("categorical_meaningful"),
        "has_chip":   _S("categorical_meaningful"),
        "num_cards":  _S("numeric_meaningful"),
    }


def _cards_df(n=200):
    import numpy as np
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "card_type":  rng.choice(["Debit", "Credit", "Prepaid"], n).tolist(),
        "card_brand": rng.choice(["Visa", "Mastercard", "Amex"], n).tolist(),
        "has_chip":   rng.choice(["YES", "NO"], n).tolist(),
        "num_cards":  rng.integers(1, 4, n).tolist(),
    })


# ── Unit tests (Task 3.3) ─────────────────────────────────────────────────────

def test_returns_none_for_empty_indices():
    df = _cards_df()
    assert profile_outliers(df, _cards_sem(), []) is None


def test_returns_none_for_none_indices():
    df = _cards_df()
    assert profile_outliers(df, _cards_sem(), None) is None


def test_basic_profile_structure():
    """Returns OutlierProfile with correct fields for valid indices."""
    df = _cards_df(200)
    indices = list(df.index[:40])
    prof = profile_outliers(df, _cards_sem(), indices)

    assert prof is not None
    assert isinstance(prof, OutlierProfile)
    assert prof.n_outliers == 40
    assert isinstance(prof.modal_profile, dict)
    assert isinstance(prof.modal_pct, dict)
    assert isinstance(prof.narrative, str)
    assert len(prof.narrative) > 0


def test_pct_of_total_exact():
    """pct_of_total must equal len(indices) / len(df) exactly."""
    df = _cards_df(300)
    indices = list(df.index[:75])
    prof = profile_outliers(df, _cards_sem(), indices)
    assert prof.pct_of_total == 75 / 300


def test_modal_profile_keys_are_categorical_cols():
    """modal_profile only contains categorical_meaningful columns."""
    df = _cards_df(200)
    sem = _cards_sem()
    indices = list(df.index[:50])
    prof = profile_outliers(df, sem, indices)

    cat_cols = {c for c, s in sem.items() if s.tag == "categorical_meaningful"}
    for col in prof.modal_profile:
        assert col in cat_cols


def test_narrative_mentions_dominant_column():
    """When ≥70% of outliers share a value, narrative mentions it."""
    # Force all outliers to have card_type=Credit
    df = pd.DataFrame({
        "card_type":  ["Credit"] * 100 + ["Debit"] * 100,
        "card_brand": ["Visa"] * 200,
    })
    sem = {
        "card_type":  _S("categorical_meaningful"),
        "card_brand": _S("categorical_meaningful"),
    }
    # All 50 outliers are from the first 50 rows (all Credit)
    indices = list(df.index[:50])
    prof = profile_outliers(df, sem, indices)

    assert "card_type" in prof.modal_profile
    assert prof.modal_profile["card_type"] == "Credit"
    assert prof.modal_pct["card_type"] == 1.0
    assert "card_type" in prof.narrative
    assert "Credit" in prof.narrative


def test_narrative_omits_non_dominant_column():
    """Columns where <70% share the modal value are omitted from narrative."""
    import numpy as np
    rng = np.random.default_rng(1)
    # card_type is 60% Credit among outliers (below 70% threshold)
    n = 100
    df = pd.DataFrame({
        "card_type": ["Credit"] * 60 + ["Debit"] * 40,
    })
    sem = {"card_type": _S("categorical_meaningful")}
    indices = list(df.index)  # all rows as outliers
    prof = profile_outliers(df, sem, indices)

    # modal_pct = 0.6 < 0.7 → should NOT appear in narrative
    assert "card_type" not in prof.narrative or "60%" not in prof.narrative


def test_invalid_indices_filtered():
    """Invalid indices are silently filtered; profile computed on valid subset."""
    df = _cards_df(100)
    valid = list(df.index[:10])
    invalid = [999999, 1000000]
    prof = profile_outliers(df, _cards_sem(), valid + invalid)

    assert prof is not None
    assert prof.n_outliers == 10
    assert prof.pct_of_total == 10 / 100


def test_all_invalid_indices_returns_none():
    """If all indices are invalid, return None."""
    df = _cards_df(50)
    prof = profile_outliers(df, _cards_sem(), [999, 1000, 1001])
    assert prof is None


def test_high_cardinality_col_excluded():
    """Columns with >20 unique values are excluded from modal_profile."""
    n = 100
    df = pd.DataFrame({
        "card_type": ["Debit"] * n,
        "high_card": list(range(n)),  # 100 unique values
    })
    sem = {
        "card_type": _S("categorical_meaningful"),
        "high_card": _S("categorical_meaningful"),
    }
    indices = list(df.index[:20])
    prof = profile_outliers(df, sem, indices)

    assert "high_card" not in prof.modal_profile


# ── Property test (Task 3.2) ──────────────────────────────────────────────────

from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes


@given(
    df_size=st.integers(min_value=10, max_value=200),
    n_outliers=st.integers(min_value=1, max_value=9),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_outlier_fraction_accuracy(df_size, n_outliers):
    """Property 5: pct_of_total == len(anomaly_indices) / len(df) exactly."""
    assume(n_outliers <= df_size)

    df = pd.DataFrame({
        "cat": ["A"] * df_size,
        "num": list(range(df_size)),
    })
    sem = {"cat": _S("categorical_meaningful"), "num": _S("numeric_meaningful")}
    indices = list(df.index[:n_outliers])

    prof = profile_outliers(df, sem, indices)
    assert prof is not None
    assert prof.pct_of_total == n_outliers / df_size
