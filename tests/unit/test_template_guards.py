import pandas as pd
from render.template_guards import balance_descriptor, safe_top_n, describe_distribution


def test_card_brand_dominated():
    counts = pd.Series({
        "Mastercard": 3209,
        "Visa": 2326,
        "Amex": 402,
        "Discover": 209,
    })
    assert balance_descriptor(counts) == "dominated"
    desc = describe_distribution("card_brand", counts)
    assert "dominated" in desc
    assert "balanced" not in desc


def test_safe_top_n_clamps():
    s = pd.Series(["a", "b", "c", "a", "b"])  # only 3 unique
    _, n_used = safe_top_n(s, 5)
    assert n_used == 3


def test_balanced_distribution():
    counts = pd.Series({"A": 100, "B": 105, "C": 98})
    assert balance_descriptor(counts) == "balanced"
    desc = describe_distribution("test_col", counts)
    assert "balanced" in desc


def test_uneven_distribution():
    # 500/1200 = 41.7% — below 50% threshold, ratio = 500/100 = 5 — below 10
    counts = pd.Series({"A": 500, "B": 400, "C": 300})
    assert balance_descriptor(counts) == "uneven"


def test_single_valued():
    counts = pd.Series({"No": 1000})
    assert balance_descriptor(counts) == "single-valued"


def test_describe_distribution_no_balanced_for_dominated():
    # Mastercard at 52% should NOT say "balanced"
    counts = pd.Series({"Mastercard": 3209, "Visa": 2326, "Amex": 402, "Discover": 209})
    desc = describe_distribution("card_brand", counts)
    assert "relatively balanced" not in desc.lower()
    assert "balanced" not in desc.lower()
