import pandas as pd
from utils.coerce import coerce_numeric
from classifiers.semantics import classify_dataframe
from analysis.segmentation import auto_segment_all, segment


def test_card_type_credit_limit_surfaces():
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    df["credit_limit"], _ = coerce_numeric(df["credit_limit"])
    sem = classify_dataframe(df)
    findings = auto_segment_all(df, sem)

    pair = next(
        ((f.group_col, f.target_col) for f in findings
         if f.group_col == "card_type" and f.target_col == "credit_limit"),
        None,
    )
    assert pair is not None, "card_type × credit_limit must surface as a significant segmentation"

    seg = next(f for f in findings if f.group_col == "card_type" and f.target_col == "credit_limit")
    assert seg.spread_ratio > 100, (
        f"Expected huge spread (prepaid ~$64 vs debit ~$18558), got {seg.spread_ratio:.1f}"
    )
    assert seg.min_group_n >= 30


def test_small_n_excluded():
    df = pd.DataFrame({
        "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 1000,
        "value": [1] * 5 + [100] * 5 + [50] * 1000,
    })

    class _S:
        def __init__(self, tag):
            self.tag = tag

    sem = {
        "group": _S("categorical_meaningful"),
        "value": _S("numeric_meaningful"),
    }
    findings = auto_segment_all(df, sem, min_n=30)
    assert all(f.min_group_n >= 30 for f in findings)


def test_monetary_target_uses_currency_format():
    """Task 2: monetary segmentation headlines must use {{fmt:currency:metric:...}}."""
    from render.metric_store import build_metric_store, MetricKey
    from render.metric_filler import fill_metrics

    df = pd.read_csv("tests/fixtures/cards_data.csv")
    df["credit_limit"], _ = coerce_numeric(df["credit_limit"])
    sem = classify_dataframe(df)

    findings = auto_segment_all(df, sem)
    pair = next(
        f for f in findings
        if f.group_col == "card_type" and f.target_col == "credit_limit"
    )

    assert "fmt:currency:metric:credit_limit" in pair.headline, (
        f"Monetary segmentation must use currency formatting. Got: {pair.headline}"
    )

    # End-to-end: fill the placeholders and assert $-formatted output
    store = build_metric_store(df, sem)
    # Ensure scoped means are present
    for val, sub in df.groupby("card_type", dropna=False):
        store.put(
            MetricKey("credit_limit", "mean", f"card_type={val}"),
            float(sub["credit_limit"].mean()),
        )
    filled = fill_metrics(pair.headline, store)
    assert "$" in filled, f"Filled headline must contain '$'. Got: {filled}"
    # Prepaid mean ~$64, Debit mean ~$18,558
    assert "$64" in filled or "$63" in filled, (
        f"Prepaid mean should render as ~$64. Got: {filled}"
    )
    assert "$18,5" in filled or "$18,4" in filled, (
        f"Debit mean should render as ~$18,5xx. Got: {filled}"
    )
