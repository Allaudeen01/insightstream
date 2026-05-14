"""
Tests for _rule_rfm_segmentation routing:
  - n < 50 unique customers  → _rfm_simple_tiers (3-tier path)
  - n >= 50 unique customers → full RFM quintile path

insight_engine is mocked by conftest.py to let main.py import cleanly.
We bypass that mock by loading the real module under a private alias via
importlib.util so the actual BusinessRuleEngine class is available here.
"""
import sys
import os
import importlib.util

import pytest
import polars as pl

_ENGINE_FILE = os.path.join(os.path.dirname(__file__), "..", "engine", "insight_engine.py")


def _load_real_engine():
    """Load the real insight_engine.py under alias '_ie_real', bypassing the conftest mock."""
    spec = importlib.util.spec_from_file_location("_ie_real", _ENGINE_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ie_real"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules["_ie_real"]
        raise
    return mod


try:
    _ie_real = _load_real_engine()
    BusinessRuleEngine = _ie_real.BusinessRuleEngine
    DataProfile = _ie_real.DataProfile
    _LOAD_OK = True
    _LOAD_ERR = ""
except Exception as exc:
    _LOAD_OK = False
    _LOAD_ERR = str(exc)


def _make_rfm_df(n_customers: int) -> tuple:
    """Build a minimal e-commerce DataFrame with n_customers unique customers, 2 orders each."""
    customers = [f"C{i:04d}" for i in range(n_customers) for _ in range(2)]
    dates     = ["2024-01-15", "2024-03-20"] * n_customers
    # Vary revenues so quintile scoring has something to work with
    revenues  = [float(20 + (i % 30) * 5) for i in range(n_customers * 2)]
    df = pl.DataFrame({
        "CustomerID":  customers,
        "InvoiceDate": dates,
        "TotalPrice":  revenues,
    })
    profile = DataProfile(row_count=n_customers * 2, col_count=3)
    profile.identifiers    = []
    profile.categoricals   = ["CustomerID"]
    profile.temporals      = ["InvoiceDate"]
    profile.numericals     = ["TotalPrice"]
    profile.revenue_col    = "TotalPrice"
    profile.price_col      = "TotalPrice"
    profile.qty_col        = None
    profile.date_col       = "InvoiceDate"
    profile.category_col   = None
    profile.geographic_col = None
    profile.return_col     = None
    return df, profile


@pytest.mark.skipif(not _LOAD_OK, reason=f"insight_engine unavailable: {_LOAD_ERR}")
class TestRfmThreeTierFallback:
    def setup_method(self):
        self.engine = BusinessRuleEngine()

    def test_rfm_uses_3tier_when_below_50_customers(self):
        df, profile = _make_rfm_df(n_customers=42)
        results = self.engine._rule_rfm_segmentation(df, profile)
        assert len(results) == 1, f"Expected 1 insight, got {len(results)}"
        ins = results[0]
        desc_lower = ins.description.lower()
        assert "simplified" in desc_lower or "3-tier" in desc_lower, (
            f"Expected 3-tier/simplified mention in description.\n"
            f"Got: {ins.description[:400]}"
        )
        assert "n=42" in ins.description, (
            f"Expected 'n=42' in description.\n"
            f"Got: {ins.description[:400]}"
        )

    def test_rfm_full_quintiles_when_above_50_customers(self):
        df, profile = _make_rfm_df(n_customers=60)
        results = self.engine._rule_rfm_segmentation(df, profile)
        assert len(results) == 1, f"Expected 1 insight, got {len(results)}"
        ins = results[0]
        assert "rfm" in ins.title.lower(), (
            f"Expected 'RFM' in title for full quintile path.\n"
            f"Got title: {ins.title}"
        )
        assert "simplified" not in ins.description.lower(), (
            f"Full RFM path should not mention 'simplified'.\n"
            f"Got: {ins.description[:400]}"
        )
