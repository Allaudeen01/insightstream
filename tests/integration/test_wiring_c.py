"""
Wiring Task C — MetricStore built, no placeholder leaks in output.
Tests run against the safe_fallback path (no GROQ_API_KEY needed).
"""
import os
import re
import pandas as pd
import pytest

os.environ.pop("GROQ_API_KEY", None)

from analyzer import analyze_dataset


@pytest.fixture(scope="module")
def cards_result():
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    return analyze_dataset(df)


def test_no_placeholders_leak_to_output(cards_result):
    """No unfilled {{metric:...}} placeholders should appear in output."""
    blob = str(cards_result)
    assert "{{metric:" not in blob, "Unfilled metric placeholder leaked to output"
    assert "{{fmt:" not in blob, "Unfilled format placeholder leaked to output"


def test_metric_consistency_cvv(cards_result):
    """If CVV mean is mentioned anywhere, all mentions must agree."""
    blob = str(cards_result).lower()
    cvv_means = re.findall(r"mean\s+cvv\s*(?:is|of|=|:)?\s*(\d+\.?\d*)", blob)
    assert len(set(cvv_means)) <= 1, (
        f"Inconsistent CVV mean values: {cvv_means}"
    )


def test_hypotheses_present(cards_result):
    """Hypotheses section must be present (even if empty list)."""
    assert "hypotheses" in cards_result, "Result must have 'hypotheses' key"


def test_unit_notes_present(cards_result):
    """Unit-of-analysis notes must be present."""
    assert "unit_notes" in cards_result, "Result must have 'unit_notes' key"
    notes = cards_result["unit_notes"]
    assert any(n["id_col"] == "client_id" for n in notes), (
        f"client_id unit note missing. Notes: {notes}"
    )
