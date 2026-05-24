import pandas as pd
from analyzer import _filter_recommendations
from classifiers.semantics import classify_dataframe
from utils.coerce import coerce_numeric


def _cards():
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    df["credit_limit"], _ = coerce_numeric(df["credit_limit"])
    return df


def test_discover_value_reference_is_kept():
    df = _cards()
    sem = classify_dataframe(df)
    recs = [
        {"text": "Investigate the reasons behind the low credit limit of Discover users."}
    ]
    out = _filter_recommendations(recs, df, semantics=sem)
    assert len(out) == 1, (
        "Discover (a card_brand value) should anchor the recommendation"
    )


def test_column_name_reference_is_kept():
    df = _cards()
    sem = classify_dataframe(df)
    recs = [
        {"text": "Review credit_limit distribution for users with multiple cards."}
    ]
    out = _filter_recommendations(recs, df, semantics=sem)
    assert len(out) == 1


def test_unanchored_rec_is_dropped():
    df = _cards()
    sem = classify_dataframe(df)
    recs = [
        {"text": "Improve customer satisfaction across all segments."}
    ]
    out = _filter_recommendations(recs, df, semantics=sem)
    assert len(out) == 0


def test_cvv_value_not_an_anchor():
    """Values of random_token columns must NOT serve as anchors."""
    df = _cards()
    sem = classify_dataframe(df)
    # A rec that only references a CVV value (not the column name)
    # should be dropped — CVV values are not in value_to_col
    recs = [
        {"text": "Investigate accounts with security code 123 for fraud patterns."}
    ]
    out = _filter_recommendations(recs, df, semantics=sem)
    # "123" is a CVV value but random_token values are excluded from value_to_col
    # and "security code" / "123" are not column names → should be dropped
    assert len(out) == 0, (
        "Random-token values must not anchor recommendations"
    )


def test_mastercard_value_reference_is_kept():
    """Mastercard is a value of card_brand — should anchor a recommendation."""
    df = _cards()
    sem = classify_dataframe(df)
    recs = [
        {"text": "Analyse Mastercard holders to understand their higher credit limits."}
    ]
    out = _filter_recommendations(recs, df, semantics=sem)
    assert len(out) == 1
