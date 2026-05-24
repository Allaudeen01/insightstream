import pandas as pd
from utils.coerce import coerce_numeric
from classifiers.semantics import classify_dataframe


def test_cards_data_classification():
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    # Apply currency coercion so credit_limit becomes numeric (monetary tag)
    df["credit_limit"], _ = coerce_numeric(df["credit_limit"])
    sem = classify_dataframe(df)

    assert sem["id"].tag == "identifier"
    assert sem["client_id"].tag == "identifier"
    assert sem["card_number"].tag in ("identifier", "random_token")
    assert sem["cvv"].tag == "random_token"
    assert sem["card_on_dark_web"].tag == "categorical_degenerate"
    assert sem["credit_limit"].tag == "monetary"
    assert sem["card_brand"].tag == "categorical_meaningful"
    assert sem["card_type"].tag == "categorical_meaningful"
    assert sem["has_chip"].tag == "categorical_meaningful"
    assert sem["acct_open_date"].tag == "temporal"
    assert sem["expires"].tag == "temporal"
    assert sem["year_pin_last_changed"].tag == "temporal"


def test_degenerate_detection():
    s = pd.Series(["No"] * 1000)
    df = pd.DataFrame({"card_on_dark_web": s})
    sem = classify_dataframe(df)
    assert sem["card_on_dark_web"].tag == "categorical_degenerate"
