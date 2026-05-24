"""
Wiring Task A — coercion + semantics injected into analyze_dataset.
Tests run against the safe_fallback path (no GROQ_API_KEY needed).
"""
import os
import pandas as pd
import pytest

# Ensure no GROQ key so we get deterministic safe_fallback output
os.environ.pop("GROQ_API_KEY", None)

from analyzer import analyze_dataset


@pytest.fixture(scope="module")
def cards_result():
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    return analyze_dataset(df)


def test_result_has_semantics(cards_result):
    """Semantics must be attached to the result."""
    assert "semantics" in cards_result, (
        "analyze_dataset must return a 'semantics' key after Phase 2 wiring"
    )
    sem = cards_result["semantics"]
    assert "credit_limit" in sem
    assert "cvv" in sem


def test_credit_limit_not_flagged_as_missing(cards_result):
    """After coercion, credit_limit must not appear in data_quality as missing."""
    dq = cards_result.get("data_quality", [])
    missing_issues = [
        item for item in dq
        if item.get("column") == "credit_limit"
        and "missing" in item.get("issue", "").lower()
    ]
    assert missing_issues == [], (
        f"credit_limit should not be flagged as missing after coercion. Got: {missing_issues}"
    )


def test_cvv_tagged_as_random_token(cards_result):
    """CVV must be classified as random_token."""
    sem = cards_result.get("semantics", {})
    assert sem.get("cvv", {}).get("tag") == "random_token", (
        f"cvv should be random_token, got: {sem.get('cvv')}"
    )


def test_dark_web_tagged_as_degenerate(cards_result):
    """card_on_dark_web (all 'No') must be categorical_degenerate."""
    sem = cards_result.get("semantics", {})
    assert sem.get("card_on_dark_web", {}).get("tag") == "categorical_degenerate", (
        f"card_on_dark_web should be categorical_degenerate, got: {sem.get('card_on_dark_web')}"
    )


def test_dark_web_in_data_quality(cards_result):
    """card_on_dark_web must appear in data_quality with no-variance flag."""
    dq = cards_result.get("data_quality", [])
    degen_items = [
        item for item in dq
        if item.get("column") == "card_on_dark_web"
        and "variance" in item.get("issue", "").lower()
    ]
    assert degen_items, (
        f"card_on_dark_web should be in data_quality with 'no variance'. Got dq: {dq}"
    )
