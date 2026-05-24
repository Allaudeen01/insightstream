import pandas as pd
from utils.coerce import coerce_numeric
from classifiers.semantics import classify_dataframe
from analysis.hypotheses import detect_ambiguities


def test_zero_credit_limits_flagged():
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    df["credit_limit"], rpt = coerce_numeric(df["credit_limit"])
    sem = classify_dataframe(df)
    hyps = detect_ambiguities(df, sem, {"credit_limit": rpt})

    # cards_data has 31 zero values -> should trigger zero-cluster hypothesis
    assert any(
        "credit_limit" in h.observation and "zero" in h.observation.lower()
        for h in hyps
    ), f"Expected zero-credit_limit hypothesis. Got: {[h.observation for h in hyps]}"
