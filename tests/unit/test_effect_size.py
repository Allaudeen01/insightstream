"""
Wave 1 tests for engine/analysis/effect_size.py

Tasks 2.2 (Property 1: Bounds), 2.3 (Property 2: Ordering),
       2.4 (Property 10: Immutability), 2.5 (unit tests)
"""
import sys
import os
import math

import numpy as np
import pandas as pd
import pytest

# ── path setup (mirrors tests/unit/conftest.py) ───────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

from analysis.effect_size import compute_effect_sizes, EffectSize


# ── helpers ───────────────────────────────────────────────────────────────────

class _S:
    """Minimal ColumnSemantics stub."""
    def __init__(self, tag):
        self.tag = tag


def _sem(cat_cols, num_cols, extra_cols=None):
    s = {c: _S("categorical_meaningful") for c in cat_cols}
    s.update({c: _S("numeric_meaningful") for c in num_cols})
    if extra_cols:
        s.update(extra_cols)
    return s


# ── Unit tests (Task 2.5) ─────────────────────────────────────────────────────

def test_three_group_returns_valid_effect_size():
    """3-group DataFrame: η² ∈ [0,1], list sorted descending."""
    df = pd.DataFrame({
        "group": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
        "value": list(range(100)) + list(range(200, 300)) + list(range(400, 500)),
    })
    sem = _sem(["group"], ["value"])
    results = compute_effect_sizes(df, sem)

    assert len(results) == 1
    e = results[0]
    assert isinstance(e, EffectSize)
    assert 0.0 <= e.eta_squared <= 1.0
    assert 0.0 <= e.p_value <= 1.0
    assert e.is_significant  # large group separation → p ≈ 0
    assert e.group_col == "group"
    assert e.target_col == "value"


def test_sorted_descending_multiple_pairs():
    """Multiple (cat × num) pairs are sorted by eta_squared descending."""
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "cat_strong": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
        "cat_weak":   rng.choice(["X", "Y"], n).tolist(),
        # strong: large between-group variance
        "num_strong": list(range(100)) + list(range(500, 600)) + list(range(1000, 1100)),
        # weak: nearly identical groups
        "num_weak":   rng.normal(50, 1, n).tolist(),
    })
    sem = _sem(["cat_strong", "cat_weak"], ["num_strong", "num_weak"])
    results = compute_effect_sizes(df, sem)

    assert len(results) >= 2
    for i in range(len(results) - 1):
        assert results[i].eta_squared >= results[i + 1].eta_squared, (
            f"Not sorted at index {i}: {results[i].eta_squared} < {results[i+1].eta_squared}"
        )


def test_empty_pairs_returns_empty_list():
    """No categorical columns → returns [] not None."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    sem = _sem([], ["a", "b"])
    result = compute_effect_sizes(df, sem)
    assert result == []
    assert result is not None


def test_no_numeric_cols_returns_empty_list():
    """No numeric columns → returns []."""
    df = pd.DataFrame({"cat": ["A", "B", "C"]})
    sem = _sem(["cat"], [])
    result = compute_effect_sizes(df, sem)
    assert result == []


def test_identical_values_in_one_group_skipped_gracefully():
    """A group where all values are identical should not crash f_oneway."""
    df = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "value": [5.0] * 50 + list(range(50)),  # group A: all 5.0
    })
    sem = _sem(["group"], ["value"])
    # Should not raise; may return [] or a valid result
    result = compute_effect_sizes(df, sem)
    assert isinstance(result, list)
    for e in result:
        assert 0.0 <= e.eta_squared <= 1.0


def test_all_identical_values_returns_empty_or_zero():
    """When all values are identical across all groups, η² = 0 or pair is skipped."""
    df = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "value": [42.0] * 100,
    })
    sem = _sem(["group"], ["value"])
    result = compute_effect_sizes(df, sem)
    # Either skipped (empty) or eta_squared == 0
    for e in result:
        assert e.eta_squared == 0.0


def test_does_not_mutate_dataframe():
    """compute_effect_sizes must not mutate the input DataFrame."""
    df = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "value": list(range(100)),
    })
    shape_before = df.shape
    cols_before = df.columns.tolist()
    vals_before = df.values.tolist()

    sem = _sem(["group"], ["value"])
    compute_effect_sizes(df, sem)

    assert df.shape == shape_before
    assert df.columns.tolist() == cols_before
    assert df.values.tolist() == vals_before


def test_monetary_tag_included():
    """Columns tagged 'monetary' are included as numeric targets."""
    df = pd.DataFrame({
        "card_type": ["Debit"] * 100 + ["Credit"] * 100 + ["Prepaid"] * 100,
        "credit_limit": list(range(100)) + list(range(500, 600)) + [64.0] * 100,
    })
    sem = {
        "card_type": _S("categorical_meaningful"),
        "credit_limit": _S("monetary"),
    }
    results = compute_effect_sizes(df, sem)
    assert len(results) == 1
    assert results[0].target_col == "credit_limit"
    assert results[0].eta_squared > 0


# ── Property tests (Tasks 2.2, 2.3, 2.4) ─────────────────────────────────────

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes


def _make_semantics_for_df(df: pd.DataFrame) -> dict:
    """Assign semantics: first object col → categorical, rest numeric → numeric_meaningful."""
    sem = {}
    cat_assigned = False
    for col in df.columns:
        if df[col].dtype == object and not cat_assigned:
            sem[col] = _S("categorical_meaningful")
            cat_assigned = True
        elif pd.api.types.is_numeric_dtype(df[col]):
            sem[col] = _S("numeric_meaningful")
        else:
            sem[col] = _S("free_text")
    return sem


# Task 2.2 — Property 1: Effect Size Bounds
@given(
    data_frames(
        columns=[
            column("cat", dtype=str, elements=st.sampled_from(["A", "B", "C"])),
            column("num", dtype=float, elements=st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            )),
        ],
        index=range_indexes(min_size=10, max_size=200),
    )
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_effect_size_bounds(df):
    """Property 1: every eta_squared ∈ [0,1] and p_value ∈ [0,1]."""
    sem = _make_semantics_for_df(df)
    results = compute_effect_sizes(df, sem)
    for e in results:
        assert 0.0 <= e.eta_squared <= 1.0, f"eta_squared={e.eta_squared} out of [0,1]"
        assert 0.0 <= e.p_value <= 1.0, f"p_value={e.p_value} out of [0,1]"


# Task 2.3 — Property 2: Effect Size Ordering
@given(
    data_frames(
        columns=[
            column("cat", dtype=str, elements=st.sampled_from(["A", "B", "C"])),
            column("num1", dtype=float, elements=st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            )),
            column("num2", dtype=float, elements=st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            )),
        ],
        index=range_indexes(min_size=10, max_size=200),
    )
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_effect_size_ordering(df):
    """Property 2: returned list is sorted by eta_squared descending."""
    sem = _make_semantics_for_df(df)
    results = compute_effect_sizes(df, sem)
    for i in range(len(results) - 1):
        assert results[i].eta_squared >= results[i + 1].eta_squared


# Task 2.4 — Property 10: DataFrame Immutability
@given(
    data_frames(
        columns=[
            column("cat", dtype=str, elements=st.sampled_from(["A", "B", "C"])),
            column("num", dtype=float, elements=st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            )),
        ],
        index=range_indexes(min_size=5, max_size=100),
    )
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_dataframe_immutability(df):
    """Property 10: compute_effect_sizes does not mutate df."""
    shape_before = df.shape
    cols_before = df.columns.tolist()
    vals_before = df.values.tolist()

    sem = _make_semantics_for_df(df)
    compute_effect_sizes(df, sem)

    assert df.shape == shape_before
    assert df.columns.tolist() == cols_before
    assert df.values.tolist() == vals_before
