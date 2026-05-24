"""
Wave 1 tests for engine/analysis/limitations.py

Tasks 4.2 (Property 3: Completeness), 4.3 (Property 11: Alias Case-Insensitivity),
       4.4 (unit tests)
"""
import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

from analysis.limitations import (
    detect_limitations,
    CANONICAL_COLUMNS,
    Limitation,
)

# Total canonical concepts across all buckets
_ALL_CONCEPTS = (
    set(CANONICAL_COLUMNS["finance"])
    | set(CANONICAL_COLUMNS["temporal"])
    | set(CANONICAL_COLUMNS["entity"])
)
_N_TOTAL = len(_ALL_CONCEPTS)  # 5


# ── Unit tests (Task 4.4) ─────────────────────────────────────────────────────

def test_empty_df_finance_domain_returns_all_finance_limitations():
    """Empty DataFrame with FINANCE domain → all 5 concepts missing."""
    df = pd.DataFrame()
    lims = detect_limitations(df, {}, domain="FINANCE")
    concepts = {l.missing_concept for l in lims}
    assert "account_status" in concepts
    assert "utilization_rate" in concepts
    assert "transaction_amount" in concepts
    assert "transaction_date" in concepts
    assert "customer_id" in concepts
    assert len(lims) == _N_TOTAL


def test_non_finance_domain_skips_finance_bucket():
    """ENTERTAINMENT domain → finance concepts never appear."""
    df = pd.DataFrame()
    lims = detect_limitations(df, {}, domain="ENTERTAINMENT")
    finance_concepts = set(CANONICAL_COLUMNS["finance"])
    leaked = {l.missing_concept for l in lims} & finance_concepts
    assert leaked == set(), f"Finance concepts leaked: {leaked}"


def test_none_domain_skips_finance_bucket():
    """domain=None → finance concepts never appear."""
    df = pd.DataFrame()
    lims = detect_limitations(df, {}, domain=None)
    finance_concepts = set(CANONICAL_COLUMNS["finance"])
    leaked = {l.missing_concept for l in lims} & finance_concepts
    assert leaked == set()


def test_all_canonical_present_returns_empty_list():
    """DataFrame with all canonical columns → returns [] not None."""
    df = pd.DataFrame(columns=[
        "account_status", "transaction_date", "customer_id",
        "utilization_rate", "transaction_amount",
    ])
    result = detect_limitations(df, {}, domain="FINANCE")
    assert result == []
    assert result is not None


def test_alias_matching_status_for_account_status():
    """'Status' column matches account_status alias."""
    df = pd.DataFrame(columns=["Status"])
    lims = detect_limitations(df, {}, domain="FINANCE")
    concepts = {l.missing_concept for l in lims}
    assert "account_status" not in concepts, (
        "'Status' should match account_status alias"
    )


def test_alias_matching_client_id_for_customer_id():
    """'client_id' matches customer_id alias."""
    df = pd.DataFrame(columns=["client_id"])
    lims = detect_limitations(df, {}, domain="FINANCE")
    concepts = {l.missing_concept for l in lims}
    assert "customer_id" not in concepts


def test_alias_matching_date_for_transaction_date():
    """'date' matches transaction_date alias."""
    df = pd.DataFrame(columns=["date"])
    lims = detect_limitations(df, {}, domain="FINANCE")
    concepts = {l.missing_concept for l in lims}
    assert "transaction_date" not in concepts


def test_case_insensitive_matching():
    """Column names are matched case-insensitively."""
    df = pd.DataFrame(columns=["DATE", "CUSTOMER_ID"])
    lims = detect_limitations(df, {}, domain="FINANCE")
    concepts = {l.missing_concept for l in lims}
    assert "transaction_date" not in concepts, "DATE should match transaction_date"
    assert "customer_id" not in concepts, "CUSTOMER_ID should match customer_id"


def test_underscore_insensitive_matching():
    """Column names are matched underscore-insensitively."""
    df = pd.DataFrame(columns=["transactiondate", "customerid"])
    lims = detect_limitations(df, {}, domain="FINANCE")
    concepts = {l.missing_concept for l in lims}
    assert "transaction_date" not in concepts
    assert "customer_id" not in concepts


def test_does_not_mutate_df():
    """detect_limitations must not mutate the input DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    shape_before = df.shape
    cols_before = df.columns.tolist()
    detect_limitations(df, {}, domain="FINANCE")
    assert df.shape == shape_before
    assert df.columns.tolist() == cols_before


def test_temporal_bucket_always_checked():
    """transaction_date is checked regardless of domain."""
    for domain in [None, "ENTERTAINMENT", "SPORTS", "HR"]:
        df = pd.DataFrame()
        lims = detect_limitations(df, {}, domain=domain)
        concepts = {l.missing_concept for l in lims}
        assert "transaction_date" in concepts, (
            f"transaction_date should be missing for domain={domain!r}"
        )


def test_entity_bucket_always_checked():
    """customer_id is checked regardless of domain."""
    for domain in [None, "ENTERTAINMENT", "SPORTS"]:
        df = pd.DataFrame()
        lims = detect_limitations(df, {}, domain=domain)
        concepts = {l.missing_concept for l in lims}
        assert "customer_id" in concepts, (
            f"customer_id should be missing for domain={domain!r}"
        )


def test_finance_domain_variants():
    """All finance domain variants trigger the finance bucket."""
    for domain in ["FINANCE", "CREDIT", "FINANCE_CREDIT", "ECOMMERCE_TRANSACTIONS"]:
        df = pd.DataFrame()
        lims = detect_limitations(df, {}, domain=domain)
        concepts = {l.missing_concept for l in lims}
        assert "account_status" in concepts, (
            f"account_status should be missing for domain={domain!r}"
        )


def test_limitation_has_non_empty_impact():
    """Every returned Limitation has a non-empty missing_impact string."""
    df = pd.DataFrame()
    lims = detect_limitations(df, {}, domain="FINANCE")
    for lim in lims:
        assert isinstance(lim.missing_impact, str)
        assert len(lim.missing_impact) > 10


# ── Property tests (Tasks 4.2, 4.3) ──────────────────────────────────────────

from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st


def _count_present_canonical(df: pd.DataFrame, domain: str) -> int:
    """Count how many canonical concepts are present in df for the given domain."""
    from analysis.limitations import _is_present, CANONICAL_COLUMNS, _FINANCE_DOMAINS
    cols_norm = {c.lower().replace("_", "") for c in df.columns}
    count = 0
    domain_upper = (domain or "").upper().replace(" ", "_")

    buckets = ["temporal", "entity"]
    if domain_upper in _FINANCE_DOMAINS:
        buckets.append("finance")

    for bucket in buckets:
        for concept, meta in CANONICAL_COLUMNS[bucket].items():
            if _is_present(concept, meta["aliases"], cols_norm):
                count += 1
    return count


# Task 4.2 — Property 3: Limitations Completeness
@given(
    col_names=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"),
                                   whitelist_characters="_"),
            min_size=1, max_size=20,
        ),
        min_size=0, max_size=10,
        unique=True,
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_property_limitations_completeness(col_names):
    """
    Property 3: len(limitations) + count_present == total applicable concepts.
    Tested with FINANCE domain so all 5 concepts are in scope.
    """
    df = pd.DataFrame(columns=col_names) if col_names else pd.DataFrame()
    domain = "FINANCE"

    lims = detect_limitations(df, {}, domain=domain)
    present = _count_present_canonical(df, domain)

    # Total applicable concepts for FINANCE domain = all 5
    assert len(lims) + present == _N_TOTAL, (
        f"completeness violated: {len(lims)} missing + {present} present != {_N_TOTAL}"
    )


# Task 4.3 — Property 11: Alias Case-Insensitivity
@given(
    # Pick a random canonical alias and mutate its case/underscores
    alias_variant=st.one_of(
        # account_status aliases
        st.just("Status"), st.just("STATUS"), st.just("status"),
        st.just("AcctStatus"), st.just("ACCT_STATUS"), st.just("acct_status"),
        # transaction_date aliases
        st.just("Date"), st.just("DATE"), st.just("date"),
        st.just("TxnDate"), st.just("TXN_DATE"), st.just("txn_date"),
        # customer_id aliases
        st.just("ClientId"), st.just("CLIENT_ID"), st.just("client_id"),
        st.just("CustId"), st.just("CUST_ID"), st.just("cust_id"),
    )
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_alias_case_insensitivity(alias_variant):
    """
    Property 11: A DataFrame containing any alias variant does NOT include
    that concept in the returned limitations list.
    """
    df = pd.DataFrame(columns=[alias_variant])
    lims = detect_limitations(df, {}, domain="FINANCE")
    concepts = {l.missing_concept for l in lims}

    # Determine which concept this alias belongs to
    alias_lower = alias_variant.lower().replace("_", "")
    from analysis.limitations import CANONICAL_COLUMNS
    for bucket in CANONICAL_COLUMNS.values():
        for concept, meta in bucket.items():
            aliases_norm = [a.lower().replace("_", "") for a in meta["aliases"]]
            concept_norm = concept.lower().replace("_", "")
            if alias_lower in aliases_norm or alias_lower == concept_norm:
                assert concept not in concepts, (
                    f"Alias {alias_variant!r} should match {concept!r} "
                    f"but it still appears in limitations"
                )
