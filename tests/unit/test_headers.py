"""
tests/unit/test_headers.py
───────────────────────────
Validates narrative header generation.
Uses rule-based fallback (no LLM required) for deterministic testing.
"""
from llm.headers import generate_narrative_headers, has_verb


_SYNTHETIC_FINDINGS = [
    {
        "title": "card_type × credit_limit",
        "body": (
            "Prepaid cards have a mean credit limit of $64, "
            "vs $18,558 for debit cards — a 290x spread. "
            "This is a CRITICAL finding."
        ),
    },
    {
        "title": "card_brand distribution",
        "body": (
            "Mastercard accounts for 52% of records, "
            "dominating the card_brand distribution."
        ),
    },
    {
        "title": "num_cards_issued × credit_limit",
        "body": (
            "There is a moderate negative correlation (r = -0.31) "
            "between num_cards_issued and credit_limit."
        ),
    },
]


def test_header_count_matches_findings():
    headers = generate_narrative_headers(_SYNTHETIC_FINDINGS)
    assert len(headers) == len(_SYNTHETIC_FINDINGS)


def test_headers_contain_verb():
    headers = generate_narrative_headers(_SYNTHETIC_FINDINGS)
    for i, h in enumerate(headers):
        assert has_verb(h), (
            f"Header {i} has no recognisable verb: {h!r}"
        )


def test_headers_word_count():
    headers = generate_narrative_headers(_SYNTHETIC_FINDINGS)
    for i, h in enumerate(headers):
        words = h.split()
        assert 5 <= len(words) <= 20, (
            f"Header {i} word count {len(words)} out of range [5,20]: {h!r}"
        )


def test_headers_not_just_column_names():
    headers = generate_narrative_headers(_SYNTHETIC_FINDINGS)
    # Headers should not be bare column names like "card_type"
    for h in headers:
        assert "_" not in h or " " in h, (
            f"Header looks like a raw column name: {h!r}"
        )
