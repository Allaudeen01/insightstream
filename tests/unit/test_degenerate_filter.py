import pandas as pd
from analyzer import _drop_degenerate_only_insights
from classifiers.semantics import classify_dataframe


def test_drops_insight_only_about_dark_web():
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    sem = classify_dataframe(df)
    insights = [
        {
            "impact": "MINOR",
            "title": "Missing Data",
            "text": "The card_on_dark_web column has no variance, with all values being 'No'.",
        },
        {
            "impact": "CRITICAL",
            "title": "Prepaid story",
            "text": "card_type=Debit (Prepaid) has the lowest credit_limit.",
        },
    ]
    out = _drop_degenerate_only_insights(insights, sem, df)
    assert len(out) == 1
    assert out[0]["title"] == "Prepaid story"


def test_keeps_insight_mentioning_degenerate_alongside_meaningful():
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    sem = classify_dataframe(df)
    insights = [
        {
            "impact": "IMPORTANT",
            "title": "Cross-finding",
            "text": (
                "Even though card_on_dark_web shows no variance, card_type "
                "reveals significant differences in credit_limit."
            ),
        },
    ]
    out = _drop_degenerate_only_insights(insights, sem, df)
    # mentions degenerate + meaningful → kept
    assert len(out) == 1


def test_keeps_insight_with_no_column_references():
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    sem = classify_dataframe(df)
    insights = [
        {
            "impact": "MINOR",
            "title": "General note",
            "text": "The dataset contains 6,146 records across 13 columns.",
        },
    ]
    out = _drop_degenerate_only_insights(insights, sem, df)
    assert len(out) == 1  # no column refs → pass through
