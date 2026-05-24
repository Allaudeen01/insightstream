"""
Wiring Task E — narrative headers, balance language, voice register.
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


def test_narrative_titles_generated(cards_result):
    """Findings should have narrative_title fields."""
    insights = cards_result.get("insights", [])
    assert insights, "No insights in result"
    # At least some findings should have narrative_title
    with_titles = [f for f in insights if f.get("narrative_title")]
    assert with_titles, (
        "No findings have narrative_title. "
        "generate_narrative_headers should have been called."
    )


def test_narrative_titles_not_raw_column_names(cards_result):
    """Narrative titles should not be bare column names."""
    insights = cards_result.get("insights", [])
    bare_col_names = {
        "card brand distribution", "cvv distribution",
        "has chip distribution", "card_brand", "cvv", "has_chip",
    }
    for f in insights:
        title = (f.get("narrative_title") or f.get("title", "")).lower()
        assert title not in bare_col_names, (
            f"Bare column-name header: {title!r}"
        )


def test_no_relatively_balanced_for_card_brand(cards_result):
    """card_brand at 52% Mastercard must not be called 'relatively balanced'."""
    blob = str(cards_result).lower()
    if "card_brand" in blob:
        # Find all occurrences of "relatively balanced" and check proximity to card_brand
        import re
        for m in re.finditer(r"relatively\s+balanced", blob):
            window = blob[max(0, m.start() - 150): m.end() + 150]
            assert "card_brand" not in window, (
                f"card_brand at 52% described as 'relatively balanced': {window!r}"
            )


def test_no_top_5_for_4_category_column(cards_result):
    """'top 5' must not appear next to card_brand which has only 4 values."""
    blob = str(cards_result).lower()
    import re
    for m in re.finditer(r"top\s+5", blob):
        window = blob[max(0, m.start() - 100): m.end() + 100]
        assert "card_brand" not in window, (
            f"'top 5' used for card_brand which has only 4 values: {window!r}"
        )
