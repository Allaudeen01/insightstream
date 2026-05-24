import pandas as pd
from utils.coerce import coerce_numeric
from classifiers.semantics import classify_dataframe
from analysis.extremum_validator import validate_extremum_claims


def _cards_df():
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    df["credit_limit"], _ = coerce_numeric(df["credit_limit"])
    return df


def test_drops_contradicting_mastercard_highest_claim():
    df = _cards_df()
    sem = classify_dataframe(df)
    # Reproduce the exact Phase 2 real-run bug
    insights = [
        {
            "impact": "CRITICAL",
            "title": "Key Takeaway",
            "text": (
                "The average credit limit for Mastercard users is $14,659.60, "
                "which is the highest among all card brands."
            ),
        },
        {
            "impact": "IMPORTANT",
            "title": "Card Brand Analysis",
            "text": (
                "The mean credit limit for Visa users is $14,737.33, "
                "which is the highest among all card brands."
            ),
        },
    ]
    kept, dropped = validate_extremum_claims(insights, df, sem)

    # Only one "highest" claim can survive — the one matching ground truth
    survivors = [
        i for i in kept
        if "highest" in i["text"].lower() and "card brand" in i["text"].lower()
    ]
    assert len(survivors) == 1, (
        f"Expected exactly one survivor, got {len(survivors)}: "
        f"{[i['text'][:60] for i in survivors]}"
    )
    assert "visa" in survivors[0]["text"].lower(), (
        f"Visa (the true extremum) should be kept, not Mastercard. "
        f"Got: {survivors[0]['text']}"
    )
    assert any("mastercard" in r.lower() for r in dropped), (
        f"Mastercard claim should be in dropped. Got: {dropped}"
    )


def test_keeps_correct_extremum_claim():
    df = _cards_df()
    sem = classify_dataframe(df)
    insights = [
        {
            "impact": "IMPORTANT",
            "title": "Prepaid story",
            "text": "Debit (Prepaid) has the lowest mean credit_limit among card types.",
        },
    ]
    kept, dropped = validate_extremum_claims(insights, df, sem)
    assert len(kept) == 1
    assert len(dropped) == 0


def test_passthrough_for_non_extremum_claims():
    df = _cards_df()
    sem = classify_dataframe(df)
    insights = [
        {
            "impact": "MINOR",
            "title": "Distribution",
            "text": "Credit limits range from $0 to $151,223.",
        }
    ]
    kept, _ = validate_extremum_claims(insights, df, sem)
    assert len(kept) == 1  # no extremum word → pass through unchanged
