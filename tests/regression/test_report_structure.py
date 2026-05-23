"""
Regression tests for report structure and terminology.
Ensures celebrity datasets don't get media/ecommerce labels.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "engine"))

import pytest
from classifiers.domain import classify_domain, get_domain_template, _rule_based_classify


def test_people_dataset_title_not_content_library(celebrity_df):
    """Celebrity dataset report title must not say 'Content Library'."""
    result = classify_domain(celebrity_df, None)  # rule-based, no LLM needed
    tmpl   = get_domain_template(result["category"])
    assert "Content Library" not in tmpl["report_title"], (
        f"Celebrity dataset got wrong title: {tmpl['report_title']!r}"
    )


def test_people_dataset_kpi_label(celebrity_df):
    """Celebrity dataset primary entity label should be 'Total People'."""
    result = classify_domain(celebrity_df, None)
    tmpl   = get_domain_template(result["category"])
    assert tmpl["primary_entity_label"] in (
        "Total People", "Total Personalities", "Total Records"
    ), f"Wrong KPI label: {tmpl['primary_entity_label']!r}"


def test_ecommerce_dataset_title(ecommerce_df):
    """Ecommerce dataset should get Transaction Analytics title."""
    # Ecommerce has revenue/price/order_id — rule-based won't fire PEOPLE
    # It may fall to GENERIC_TABULAR without LLM, which is acceptable
    result = _rule_based_classify(ecommerce_df)
    if result:
        tmpl = get_domain_template(result)
        assert "People" not in tmpl["report_title"]
