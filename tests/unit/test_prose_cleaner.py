"""
Wave 1 tests for engine/render/prose_cleaner.py

Tasks 6.2 (Property 4: Prose Cleanliness), 6.3 (unit tests)
"""
import re
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

from render.prose_cleaner import clean_prose_artifacts

_COLVAL = re.compile(r'\b\w+=\S+')


# ── Unit tests (Task 6.3) ─────────────────────────────────────────────────────

def test_card_type_debit_prepaid_cleaned():
    """'card_type=Debit (Prepaid)' → no COLUMN=VALUE pattern remains."""
    text = "card_type=Debit (Prepaid) has the highest mean credit_limit"
    result = clean_prose_artifacts(text)
    assert not _COLVAL.search(result), f"Artifact remains: {result!r}"
    # Value should still be present in some form
    assert "Debit" in result or "debit" in result


def test_has_chip_yes_cleaned():
    """'has_chip=YES' → natural-language output, no artifact."""
    text = "has_chip=YES accounts for 89% of records"
    result = clean_prose_artifacts(text)
    assert not _COLVAL.search(result), f"Artifact remains: {result!r}"
    assert "YES" in result or "yes" in result


def test_card_brand_visa_cleaned():
    """'card_brand=Visa' → cleaned."""
    text = "card_brand=Visa leads with average Weekly Sales"
    result = clean_prose_artifacts(text)
    assert not _COLVAL.search(result), f"Artifact remains: {result!r}"
    assert "Visa" in result


def test_multiple_artifacts_all_cleaned():
    """Multiple COLUMN=VALUE patterns in one string are all cleaned."""
    text = "card_type=Credit and card_brand=Mastercard dominate the dataset"
    result = clean_prose_artifacts(text)
    assert not _COLVAL.search(result), f"Artifact remains: {result!r}"


def test_plain_text_unchanged():
    """Text with no COLUMN=VALUE patterns is returned unchanged."""
    text = "No artifacts here — plain text with numbers 42 and symbols."
    result = clean_prose_artifacts(text)
    assert result == text


def test_empty_string_returns_empty():
    assert clean_prose_artifacts("") == ""


def test_none_like_empty_string():
    """Passing empty string doesn't raise."""
    assert clean_prose_artifacts("") == ""


def test_regex_special_chars_in_value_no_exception():
    """Values with regex special characters must not raise."""
    tricky_cases = [
        "col=(value+with*special[chars])",
        "col=value.with.dots",
        "col=value^caret",
        "col=value$dollar",
        "col=value|pipe",
        "col=value?question",
    ]
    for text in tricky_cases:
        result = clean_prose_artifacts(text)
        assert isinstance(result, str), f"Expected str for {text!r}, got {type(result)}"


def test_segmentation_headline_cleaned():
    """Full segmentation headline template is cleaned."""
    text = (
        "card_type=Debit has the highest mean credit_limit ($18,558), "
        "vs card_type=Debit (Prepaid) at ($64) — a 288.0x spread."
    )
    result = clean_prose_artifacts(text)
    assert not _COLVAL.search(result), f"Artifact remains: {result!r}"
    # Numbers should survive
    assert "18,558" in result or "18558" in result
    assert "64" in result


def test_url_like_patterns_handled():
    """Patterns that look like URLs (http=...) are handled without crash."""
    text = "See http=example.com for details"
    result = clean_prose_artifacts(text)
    assert isinstance(result, str)


def test_natural_language_output_readable():
    """Output should be human-readable (no raw underscores from column name)."""
    text = "card_type=Credit shows high credit_limit"
    result = clean_prose_artifacts(text)
    # "card_type" should become "card type" (underscores removed)
    assert "card_type=Credit" not in result


# ── Property test (Task 6.2) ──────────────────────────────────────────────────

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


@given(
    text=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs",),  # exclude surrogates
        ),
        max_size=500,
    )
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_property_prose_cleanliness(text):
    """
    Property 4: For any string, clean_prose_artifacts returns a string
    containing no substring matching \\b\\w+=\\S+.
    """
    result = clean_prose_artifacts(text)
    assert isinstance(result, str), "Must return str"
    assert not _COLVAL.search(result), (
        f"COLUMN=VALUE artifact remains after cleaning.\n"
        f"Input:  {text!r}\n"
        f"Output: {result!r}"
    )
