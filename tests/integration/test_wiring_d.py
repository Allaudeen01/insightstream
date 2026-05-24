"""
Wiring Task D — segmentation, hypotheses, unit_notes surface in result.
Tests run against the safe_fallback path (no GROQ_API_KEY needed).
Note: segmentation requires credit_limit to be numeric. The safe_fallback
path still runs coercion + segmentation before falling back.
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


def test_prepaid_story_is_a_finding(cards_result):
    """card_type × credit_limit segmentation must appear as a CRITICAL finding."""
    insights = cards_result.get("insights", [])
    relevant = [
        f for f in insights
        if "card_type" in str(f).lower() or "prepaid" in str(f).lower()
        or ("credit_limit" in str(f).lower() and "card_type" in str(f).lower())
    ]
    assert relevant, (
        f"prepaid/card_type story missing from findings. "
        f"All titles: {[f.get('title') for f in insights]}"
    )
    top = relevant[0]
    assert top.get("impact", "").upper() == "CRITICAL", (
        f"prepaid story should be CRITICAL given 290x spread, got: {top.get('impact')}"
    )


def test_hypotheses_section_present(cards_result):
    """Hypotheses section must not be empty for cards_data."""
    hyps = cards_result.get("hypotheses", [])
    assert len(hyps) > 0, (
        "Hypotheses section should not be empty — "
        "cards_data has zero credit_limits that should trigger a hypothesis"
    )


def test_zero_credit_limit_hypothesis(cards_result):
    """The zero-credit_limit hypothesis must be present."""
    hyps = cards_result.get("hypotheses", [])
    assert any(
        "credit_limit" in h.get("observation", "").lower()
        and "zero" in h.get("observation", "").lower()
        for h in hyps
    ), f"Zero credit_limit hypothesis missing. Got: {[h.get('observation') for h in hyps]}"


def test_unit_of_analysis_note_present(cards_result):
    """client_id unit-of-analysis note must be present."""
    notes = cards_result.get("unit_notes", [])
    assert any(n.get("id_col") == "client_id" for n in notes), (
        f"client_id unit note missing. Notes: {notes}"
    )
