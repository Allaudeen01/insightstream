"""
Regression tests for domain classification.
Tests the rule-based fast path (no LLM call needed for these cases).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "engine"))

import pandas as pd
import pytest
from classifiers.domain import (
    classify_domain, get_domain_template,
    _rule_based_classify, DOMAIN_TEMPLATE_MAP,
)


class _StubLLMClient:
    """Stub LLM client — should never be called when rule-based path fires."""
    def chat(self):
        raise AssertionError("LLM should not be called for rule-based classifications")
    class completions:
        @staticmethod
        def create(**kwargs):
            raise AssertionError("LLM should not be called for rule-based classifications")


def test_celebrity_classified_as_people_catalog(celebrity_df):
    """Celebrity dataset with name+gender+known_for_department → PEOPLE_CATALOG."""
    result = classify_domain(celebrity_df, _StubLLMClient())
    assert result["category"] == "PEOPLE_CATALOG", (
        f"Expected PEOPLE_CATALOG, got {result['category']!r}. "
        f"Reason: {result.get('reason')}"
    )
    assert result["confidence"] >= 0.7


def test_ecommerce_not_classified_as_people(ecommerce_df):
    """Ecommerce dataset should NOT be PEOPLE_CATALOG."""
    result = _rule_based_classify(ecommerce_df)
    assert result != "PEOPLE_CATALOG", "Ecommerce should not be PEOPLE_CATALOG"


def test_financial_timeseries_rule():
    """Dataset with open/high/low/close/volume → FINANCIAL_TIMESERIES."""
    df = pd.DataFrame({
        "date":   ["2024-01-01", "2024-01-02"],
        "open":   [100.0, 101.0],
        "high":   [105.0, 106.0],
        "low":    [99.0, 100.0],
        "close":  [103.0, 104.0],
        "volume": [1000000, 1100000],
    })
    result = _rule_based_classify(df)
    assert result == "FINANCIAL_TIMESERIES"


def test_get_domain_template_people():
    """PEOPLE_CATALOG template has correct title and forbidden charts."""
    tmpl = get_domain_template("PEOPLE_CATALOG")
    assert tmpl["report_title"] == "People Dataset Analysis Report"
    assert tmpl["primary_entity_label"] == "Total People"
    assert "pareto_revenue" in tmpl["forbidden_chart_types"]
    assert "sales_funnel" in tmpl["forbidden_chart_types"]


def test_get_domain_template_fallback():
    """Unknown domain falls back to GENERIC_TABULAR."""
    tmpl = get_domain_template("UNKNOWN_DOMAIN_XYZ")
    assert tmpl["report_title"] == "Data Analysis Report"
    assert tmpl["primary_entity_label"] == "Total Records"


def test_all_categories_have_required_keys():
    """Every domain template has the four required keys."""
    required = {"report_title", "primary_entity_label",
                "forbidden_chart_types", "required_sections"}
    for domain, tmpl in DOMAIN_TEMPLATE_MAP.items():
        missing = required - set(tmpl.keys())
        assert not missing, f"{domain} template missing keys: {missing}"
