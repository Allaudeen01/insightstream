"""
Wave 5 tests for build_from_assets new PDF sections.

Tasks 10.2 (Property 7: Hypothesis Rendering Completeness)
       10.3 (Property 8: Unit Notes Rendering Completeness)
       10.6 (Property 9: Limitations Rendering Completeness)
       10.7 (Unit tests: empty inputs produce no section text)

Uses pdfminer.six for text extraction (installed as a dev dependency).
"""
import sys
import os
import io
import tempfile
from pathlib import Path

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "engine")
sys.path.insert(0, ENGINE_DIR)

# Stub heavy side-effect modules that report_generator imports at module load
# Do NOT stub report_generator itself — we need the real UnifiedReportGenerator
from unittest.mock import MagicMock
for _mod in ["database", "session_cache", "insight_engine", "google", "google.genai",
             "google.generativeai"]:
    sys.modules.setdefault(_mod, MagicMock())

# insight_engine needs specific attributes
_ie = sys.modules["insight_engine"]
for _name in ("ColumnClassifier", "MetricComputer", "BusinessRuleEngine",
              "InsightNarrator", "SmartChartRecommender", "AnomalyDetector",
              "run_insight_engine", "RecommendationEngine", "StrategicBriefBuilder",
              "validate_dataframe", "auto_clean_dataframe",
              "_CURRENCY_SYMBOL", "_CURRENCY_EXPLICIT", "_set_currency_symbol"):
    if not hasattr(_ie, _name):
        setattr(_ie, _name, MagicMock())

# The parent conftest (tests/conftest.py) stubs report_generator as a MagicMock.
# Remove that stub so we can import the real module for PDF rendering tests.
sys.modules.pop("report_generator", None)

from report_generator import UnifiedReportGenerator


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_pdf(**kwargs) -> str:
    """
    Call build_from_assets with minimal valid inputs and return extracted PDF text.
    Accepts keyword overrides for any parameter.
    """
    gen = UnifiedReportGenerator()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        out_path = f.name

    defaults = dict(
        output_path=out_path,
        charts=[],
        kpis={},
        ai_summary="",
        insights=[{"title": "Test Finding", "text": "Some insight text.", "impact": "IMPORTANT"}],
        recommendations=[],
        text_blocks=[],
        title="Test Report",
        project_name="Test",
        template="modern",
        session_id="test",
        df=None,
        domain_id="general",
        hypotheses=[],
        unit_notes=[],
        synthesis="",
        limitations=[],
    )
    defaults.update(kwargs)

    try:
        gen.build_from_assets(**defaults)
        # Read bytes before cleanup so pdfminer can parse them
        pdf_bytes = Path(out_path).read_bytes()
        return _extract_text_from_bytes(pdf_bytes)
    finally:
        try:
            os.unlink(out_path)
        except Exception:
            pass


def _extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfminer."""
    import io as _io
    from pdfminer.high_level import extract_text
    return extract_text(_io.BytesIO(pdf_bytes))


# ── Unit tests (Task 10.7) ────────────────────────────────────────────────────

def test_empty_hypotheses_no_open_questions():
    """Empty hypotheses → no 'Open Questions' section in PDF."""
    text = _build_pdf(hypotheses=[])
    assert "Open Questions" not in text, (
        "Empty hypotheses should produce no 'Open Questions' section"
    )


def test_empty_unit_notes_no_context():
    """Empty unit_notes → no 'Context' callout in PDF."""
    text = _build_pdf(unit_notes=[])
    assert "Context:" not in text, (
        "Empty unit_notes should produce no 'Context' callout"
    )


def test_empty_synthesis_no_synthesis_box():
    """Empty synthesis → no 'Synthesis' section in PDF."""
    text = _build_pdf(synthesis="")
    assert "Synthesis" not in text, (
        "Empty synthesis should produce no Synthesis box"
    )


def test_empty_limitations_no_what_we_dont_know():
    """Empty limitations → no 'What We Don't Know' section in PDF."""
    text = _build_pdf(limitations=[])
    assert "What We Don" not in text, (
        "Empty limitations should produce no 'What We Don't Know' section"
    )


def test_non_empty_hypotheses_renders_open_questions():
    """Non-empty hypotheses → 'Open Questions' appears in PDF."""
    hyps = [
        {
            "observation": "31 of 'credit_limit' values are exactly zero.",
            "candidates": ["Prepaid accounts", "Sentinel for missing"],
            "disambiguating_info": "Cross-tabulate with account_status.",
        }
    ]
    text = _build_pdf(hypotheses=hyps)
    assert "Open Questions" in text, (
        "Non-empty hypotheses should render 'Open Questions' section"
    )


def test_non_empty_unit_notes_renders_context():
    """Non-empty unit_notes → 'Context' appears in PDF."""
    notes = [
        {
            "id_col": "client_id",
            "rows": 6146,
            "entities": 2000,
            "rows_per_entity": 3.07,
            "note": "Rows are at card level. Each client_id has on average 3.1 rows.",
        }
    ]
    text = _build_pdf(unit_notes=notes)
    assert "Context" in text, (
        "Non-empty unit_notes should render 'Context' callout"
    )


def test_non_empty_synthesis_renders_synthesis():
    """Non-empty synthesis → 'Synthesis' appears in PDF."""
    text = _build_pdf(synthesis="card_type explains 23% of credit_limit variance.")
    assert "Synthesis" in text, (
        "Non-empty synthesis should render Synthesis box"
    )


def test_non_empty_limitations_renders_what_we_dont_know():
    """Non-empty limitations → 'What We Don't Know' appears in PDF."""
    lims = [
        {"concept": "account_status", "impact": "Cannot distinguish active from closed accounts."},
        {"concept": "transaction_date", "impact": "Cannot perform time-series analysis."},
    ]
    text = _build_pdf(limitations=lims)
    assert "What We Don" in text, (
        "Non-empty limitations should render 'What We Don't Know' section"
    )


def test_hypothesis_observation_in_pdf():
    """Hypothesis observation text appears in the PDF."""
    hyps = [
        {
            "observation": "UniqueObservationTextXYZ",
            "candidates": ["Candidate A", "Candidate B"],
            "disambiguating_info": "Check the data.",
        }
    ]
    text = _build_pdf(hypotheses=hyps)
    assert "UniqueObservationTextXYZ" in text, (
        "Hypothesis observation text should appear in PDF"
    )


def test_limitation_concept_in_pdf():
    """Limitation concept name appears in the PDF (title-cased)."""
    lims = [{"concept": "account_status", "impact": "Cannot distinguish active accounts."}]
    text = _build_pdf(limitations=lims)
    # concept is title-cased: "account_status" → "Account Status"
    assert "Account" in text, (
        "Limitation concept should appear in PDF (title-cased)"
    )


def test_synthesis_text_in_pdf():
    """Synthesis paragraph text appears in the PDF."""
    text = _build_pdf(synthesis="SYNTHESISMARKER card_type explains variance.")
    assert "SYNTHESISMARKER" in text, (
        "Synthesis text should appear in PDF"
    )


def test_unit_note_text_in_pdf():
    """Unit note text appears in the PDF."""
    notes = [
        {
            "id_col": "client_id",
            "rows": 6146,
            "entities": 2000,
            "rows_per_entity": 3.07,
            "note": "UniqueNoteTextDEF rows are at card level.",
        }
    ]
    text = _build_pdf(unit_notes=notes)
    assert "UniqueNoteTextDEF" in text, (
        "Unit note text should appear in PDF"
    )


# ── Property tests (Tasks 10.2, 10.3, 10.6) ──────────────────────────────────

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Use safe text alphabet to avoid XML/PDF encoding issues
_SAFE_TEXT = st.text(
    min_size=1, max_size=50,
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"),
                           whitelist_characters=" "),
)

_hyp_strategy = st.fixed_dictionaries({
    "observation":        _SAFE_TEXT,
    "candidates":         st.lists(_SAFE_TEXT, min_size=1, max_size=3),
    "disambiguating_info": _SAFE_TEXT,
})

_unit_note_strategy = st.fixed_dictionaries({
    "id_col":          st.text(min_size=1, max_size=20,
                               alphabet=st.characters(whitelist_categories=("Ll", "Lu"))),
    "rows":            st.integers(min_value=1, max_value=100000),
    "entities":        st.integers(min_value=1, max_value=10000),
    "rows_per_entity": st.floats(min_value=1.0, max_value=100.0,
                                  allow_nan=False, allow_infinity=False),
    "note":            _SAFE_TEXT,
})

_limitation_strategy = st.fixed_dictionaries({
    "concept": st.text(min_size=1, max_size=30,
                       alphabet=st.characters(whitelist_categories=("Ll", "Lu"))),
    "impact":  _SAFE_TEXT,
})


# Task 10.2 — Property 7: Hypothesis Rendering Completeness
@given(
    hypotheses=st.lists(_hyp_strategy, min_size=1, max_size=3),
)
@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_property_hypothesis_rendering_completeness(hypotheses):
    """
    Property 7: For any non-empty hypotheses list, the PDF contains 'Open Questions'.
    """
    text = _build_pdf(hypotheses=hypotheses)
    assert "Open Questions" in text, (
        f"PDF must contain 'Open Questions' for non-empty hypotheses. "
        f"Got {len(hypotheses)} hypotheses."
    )


# Task 10.3 — Property 8: Unit Notes Rendering Completeness
@given(
    unit_notes=st.lists(_unit_note_strategy, min_size=1, max_size=3),
)
@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_property_unit_notes_rendering_completeness(unit_notes):
    """
    Property 8: For any non-empty unit_notes list, the PDF contains 'Context'.
    """
    text = _build_pdf(unit_notes=unit_notes)
    assert "Context" in text, (
        f"PDF must contain 'Context' for non-empty unit_notes. "
        f"Got {len(unit_notes)} notes."
    )


# Task 10.6 — Property 9: Limitations Rendering Completeness
@given(
    limitations=st.lists(_limitation_strategy, min_size=1, max_size=3),
)
@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_property_limitations_rendering_completeness(limitations):
    """
    Property 9: For any non-empty limitations list, the PDF contains 'What We Don'.
    """
    text = _build_pdf(limitations=limitations)
    assert "What We Don" in text, (
        f"PDF must contain 'What We Don't Know' for non-empty limitations. "
        f"Got {len(limitations)} limitations."
    )
