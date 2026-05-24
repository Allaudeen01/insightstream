"""
Wiring Task B — semantics gates CVV/dark_web from charts and insights.
Tests run against the safe_fallback path (no GROQ_API_KEY needed).
"""
import os
import pandas as pd
import pytest

os.environ.pop("GROQ_API_KEY", None)

from analyzer import analyze_dataset


@pytest.fixture(scope="module")
def cards_result():
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    return analyze_dataset(df)


def test_cvv_not_charted(cards_result):
    """CVV must not appear as a chart column."""
    chart_metas = cards_result.get("chart_metas", [])
    chart_cols = [m.get("x_col", "") for m in chart_metas]
    assert "cvv" not in chart_cols, (
        f"CVV should not be charted. Chart columns: {chart_cols}"
    )


def test_dark_web_not_promoted(cards_result):
    """card_on_dark_web must not appear as CRITICAL or IMPORTANT finding."""
    insights = cards_result.get("insights", [])
    promoted = [
        f for f in insights
        if "dark_web" in str(f).lower()
        and f.get("impact", "").upper() in {"CRITICAL", "IMPORTANT"}
    ]
    assert promoted == [], (
        f"card_on_dark_web should not be promoted. Got: {promoted}"
    )


def test_dark_web_in_data_quality_not_findings(cards_result):
    """card_on_dark_web should be in data_quality, not in main findings."""
    dq = cards_result.get("data_quality", [])
    dq_cols = [item.get("column") for item in dq]
    assert "card_on_dark_web" in dq_cols, (
        "card_on_dark_web should appear in data_quality"
    )


def test_no_cvv_in_recommendations(cards_result):
    """No recommendation should mention CVV."""
    for r in cards_result.get("recommendations", []):
        body = str(r).lower()
        assert "cvv" not in body, f"Recommendation mentions CVV: {r!r}"
