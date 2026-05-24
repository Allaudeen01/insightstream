import pandas as pd
from utils.coerce import coerce_numeric


def test_currency_dollar():
    s = pd.Series(["$24295", "$21968", "$46414", "$28"])
    out, rpt = coerce_numeric(s)
    assert out.tolist() == [24295.0, 21968.0, 46414.0, 28.0]
    assert rpt["success_rate"] == 1.0
    assert rpt["detected_format"] == "currency"


def test_thousands_separator():
    s = pd.Series(["1,234.56", "2,000", "999"])
    out, _ = coerce_numeric(s)
    assert out.tolist() == [1234.56, 2000.0, 999.0]


def test_parens_negative():
    s = pd.Series(["(100)", "200", "(50)"])
    out, _ = coerce_numeric(s)
    assert out.tolist() == [-100.0, 200.0, -50.0]


def test_percent():
    s = pd.Series(["10%", "25%", "100%"])
    out, rpt = coerce_numeric(s)
    assert out.tolist() == [0.1, 0.25, 1.0]
    assert rpt["detected_format"] == "percent"


def test_scale_suffix():
    s = pd.Series(["10K", "2.5M", "1B"])
    out, rpt = coerce_numeric(s)
    assert out.tolist() == [10000.0, 2500000.0, 1000000000.0]
    assert rpt["detected_format"] == "scale_suffix"


def test_mixed_failures_reported():
    s = pd.Series(["$100", "$200", "not a number", "$300"])
    out, rpt = coerce_numeric(s)
    assert rpt["success_rate"] == 0.75
    assert "not a number" in rpt["sample_failures"]
