import pytest
from render.metric_store import MetricStore, MetricKey
from render.metric_filler import fill_metrics, assert_no_bare_numerics_for, UnknownMetricError


def test_fill_basic():
    store = MetricStore()
    store.put(MetricKey("cvv", "mean"), 506.22)
    out = fill_metrics("Mean CVV is {{metric:cvv.mean}}.", store)
    assert out == "Mean CVV is 506.22."


def test_fill_currency():
    store = MetricStore()
    store.put(MetricKey("credit_limit", "mean"), 14347.49)
    out = fill_metrics("Mean is {{fmt:currency:metric:credit_limit.mean}}.", store)
    assert out == "Mean is $14,347."


def test_missing_metric_raises():
    store = MetricStore()
    with pytest.raises(UnknownMetricError):
        fill_metrics("Mean is {{metric:cvv.mean}}.", store)


def test_consistency_two_mentions_match():
    store = MetricStore()
    store.put(MetricKey("cvv", "mean"), 506.22)
    text = "First mention {{metric:cvv.mean}}. Later: mean CVV is {{metric:cvv.mean}}."
    out = fill_metrics(text, store)
    # both mentions resolved from same store -> impossible to disagree
    assert out.count("506.22") == 2


def test_bare_numeric_guard():
    with pytest.raises(ValueError):
        assert_no_bare_numerics_for("Mean cvv is 506.22.", ["cvv"])
