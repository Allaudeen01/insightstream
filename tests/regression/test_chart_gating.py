"""
Regression tests for chart gating.
Ensures forbidden chart types are blocked for each domain.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "engine"))

import pytest
from classifiers.domain import get_domain_template, DOMAIN_TEMPLATE_MAP


def test_no_revenue_charts_for_people_catalog():
    """PEOPLE_CATALOG must block pareto_revenue, sales_funnel, gmv_trend."""
    tmpl     = get_domain_template("PEOPLE_CATALOG")
    forbidden = set(tmpl["forbidden_chart_types"])
    assert "pareto_revenue" in forbidden
    assert "sales_funnel"   in forbidden
    assert "gmv_trend"      in forbidden
    assert "revenue_treemap" in forbidden


def test_ecommerce_allows_revenue_charts():
    """ECOMMERCE_TRANSACTIONS must NOT block revenue charts."""
    tmpl     = get_domain_template("ECOMMERCE_TRANSACTIONS")
    forbidden = set(tmpl["forbidden_chart_types"])
    assert "pareto_revenue" not in forbidden
    assert "sales_funnel"   not in forbidden


def test_chart_gate_logic(celebrity_df):
    """Simulate chart gating: forbidden charts are skipped for PEOPLE_CATALOG."""
    from classifiers.domain import classify_domain
    result   = classify_domain(celebrity_df, None)
    tmpl     = get_domain_template(result["category"])
    forbidden = set(tmpl["forbidden_chart_types"])

    charts = [
        {"id": "pareto_revenue",  "title": "Revenue Pareto",    "type": "pareto_revenue"},
        {"id": "sales_funnel",    "title": "Sales Funnel",      "type": "sales_funnel"},
        {"id": "dept_bar",        "title": "Dept Distribution", "type": "bar"},
        {"id": "popularity_hist", "title": "Popularity Dist",   "type": "histogram"},
    ]

    allowed = [
        ch for ch in charts
        if (ch.get("type") or ch.get("id")) not in forbidden
    ]
    blocked = [
        ch for ch in charts
        if (ch.get("type") or ch.get("id")) in forbidden
    ]

    assert len(allowed) == 2, f"Expected 2 allowed charts, got {len(allowed)}"
    assert len(blocked) == 2, f"Expected 2 blocked charts, got {len(blocked)}"
    assert all(ch["id"] in ("dept_bar", "popularity_hist") for ch in allowed)


def test_all_domains_have_forbidden_list():
    """Every domain template has a forbidden_chart_types list (may be empty)."""
    for domain, tmpl in DOMAIN_TEMPLATE_MAP.items():
        assert isinstance(tmpl["forbidden_chart_types"], list), (
            f"{domain} forbidden_chart_types must be a list"
        )
