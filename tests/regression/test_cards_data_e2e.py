"""
tests/regression/test_cards_data_e2e.py
────────────────────────────────────────
Real acceptance gate for Phase 2 wiring.
Calls analyze_dataset() directly on cards_data.csv and asserts that:
  - All old bugs are gone (N1-N5)
  - All new features are present (P1-P5)

Does NOT require GROQ_API_KEY — runs against the safe_fallback path which
still executes all Phase 2 modules (coercion, semantics, segmentation,
hypotheses, unit_notes, narrative headers).
"""
import os
import re
import json
import sys
import pandas as pd
import pytest

# Ensure engine is on path
ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "engine")
sys.path.insert(0, ENGINE_DIR)

# No GROQ key — deterministic safe_fallback path
os.environ.pop("GROQ_API_KEY", None)

from analyzer import analyze_dataset


@pytest.fixture(scope="module")
def report():
    """Run the full pipeline on cards_data.csv once for all tests."""
    df = pd.read_csv("tests/fixtures/cards_data.csv")
    return analyze_dataset(df, force_refresh=True)


@pytest.fixture(scope="module")
def full_text(report):
    """Serialized lowercase text of the entire report for regex searches."""
    return json.dumps(report, default=str).lower()


# ── Positive assertions ────────────────────────────────────────────────────

def test_p1_prepaid_finding_is_critical(report):
    """[P1] card_type × credit_limit segmentation is a CRITICAL finding."""
    findings = report.get("insights", [])
    prepaid_finding = next(
        (f for f in findings
         if "card_type" in str(f).lower() or "prepaid" in str(f).lower()),
        None,
    )
    assert prepaid_finding is not None, (
        f"[P1] prepaid/card_type finding missing. "
        f"Titles: {[f.get('title') for f in findings]}"
    )
    assert prepaid_finding.get("impact", "").upper() == "CRITICAL", (
        f"[P1] prepaid finding should be CRITICAL, got {prepaid_finding.get('impact')}"
    )


def test_p2_dark_web_in_data_quality_not_findings(report):
    """[P2] card_on_dark_web flagged in data_quality, NOT as CRITICAL/IMPORTANT finding."""
    dq = report.get("data_quality", [])
    dq_blob = json.dumps(dq).lower()
    assert "dark_web" in dq_blob and ("no variance" in dq_blob or "degenerate" in dq_blob), (
        f"[P2] card_on_dark_web should be in data_quality with no-variance flag. Got: {dq}"
    )
    findings = report.get("insights", [])
    promoted = [
        f for f in findings
        if "dark_web" in str(f).lower()
        and f.get("impact", "").upper() in {"CRITICAL", "IMPORTANT"}
    ]
    assert promoted == [], (
        f"[P2] card_on_dark_web must not appear as CRITICAL/IMPORTANT finding. Got: {promoted}"
    )


def test_p3_zero_credit_limit_hypothesis(report):
    """[P3] hypotheses section has zero-credit_limit entry."""
    hyps = report.get("hypotheses", [])
    assert any(
        "credit_limit" in h.get("observation", "").lower()
        and "zero" in h.get("observation", "").lower()
        for h in hyps
    ), f"[P3] zero credit_limit hypothesis missing. Got: {[h.get('observation') for h in hyps]}"


def test_p4_unit_of_analysis_note(report):
    """[P4] unit-of-analysis note present for client_id."""
    notes = report.get("unit_notes", [])
    assert any(n.get("id_col") == "client_id" for n in notes), (
        f"[P4] client_id unit-of-analysis note missing. Notes: {notes}"
    )


def test_p5_cvv_mean_consistent(full_text):
    """[P5] If CVV mean is mentioned, all mentions must agree."""
    cvv_numbers = re.findall(r"mean\s+cvv\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)", full_text)
    assert len(set(cvv_numbers)) <= 1, (
        f"[P5] inconsistent CVV mean values: {cvv_numbers}"
    )


# ── Negative assertions (old bugs gone) ───────────────────────────────────

def test_n1_no_credit_limit_missing_claim(report, full_text):
    """[N1] no '% missing' claim about credit_limit."""
    cl_idx = full_text.find("credit_limit")
    if cl_idx >= 0:
        window = full_text[max(0, cl_idx - 100): cl_idx + 300]
        assert "% missing" not in window, (
            f"[N1] credit_limit still flagged as missing: {window!r}"
        )
    # Also check data_quality
    dq = report.get("data_quality", [])
    missing_dq = [
        item for item in dq
        if item.get("column") == "credit_limit"
        and "missing" in item.get("issue", "").lower()
    ]
    assert missing_dq == [], (
        f"[N1] credit_limit in data_quality as missing: {missing_dq}"
    )


def test_n2_no_cvv_in_key_takeaway_or_recommendations(report):
    """[N2] no 'mean CVV' in key takeaway or recommendations."""
    # Key takeaway is the first insight titled "Key Takeaway"
    insights = report.get("insights", [])
    key_takeaway = next(
        (f.get("text", "") for f in insights if "key takeaway" in f.get("title", "").lower()),
        "",
    )
    assert "cvv" not in key_takeaway.lower(), (
        f"[N2] CVV in key takeaway: {key_takeaway!r}"
    )
    for r in report.get("recommendations", []):
        assert "cvv" not in str(r).lower(), f"[N2] CVV in recommendation: {r!r}"


def test_n3_no_cvv_in_recommendations(report):
    """[N3] no recommendation referring to CVV values."""
    for r in report.get("recommendations", []):
        body = str(r).lower()
        assert "cvv" not in body, f"[N3] Recommendation mentions CVV: {r!r}"


def test_n4_no_top_5_for_4_category_column(full_text):
    """[N4] no 'top 5' phrase when card_brand has 4 categories."""
    for m in re.finditer(r"top\s+5", full_text):
        window = full_text[max(0, m.start() - 100): m.end() + 100]
        assert "card_brand" not in window, (
            f"[N4] 'top 5' is impossible — card_brand has 4 values. Context: {window!r}"
        )


def test_n5_no_unfilled_placeholders(report):
    """[N5] no unfilled {{metric:...}} placeholders in output."""
    blob = json.dumps(report, default=str)
    assert "{{metric:" not in blob, "[N5] unfilled metric placeholder in output"
    assert "{{fmt:" not in blob, "[N5] unfilled format placeholder in output"
