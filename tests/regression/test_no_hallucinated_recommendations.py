"""
Regression test: no hallucinated domain statistics in recommendations.

v373 introduced TV-MA 36%, 42% post-2020 — fake stats from wrong domain
template. This test prevents that regression.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "engine"))

import pytest
import pandas as pd
from classifiers.domain import (
    classify_domain, get_domain_template, DOMAIN_FORBIDDEN_CONCEPTS,
)
from analyzer import _filter_recommendations, validate_recommendation


# ── Forbidden phrases that must never appear in celebrity recommendations ─────
CELEBRITY_FORBIDDEN_PHRASES = [
    "tv-ma", "tv-pg", "tv-14", "r-rated", "mature content",
    "release date", "post-2020", "post-2019", "newer releases",
    "local originals", "international growth", "regional content",
    "subscriber", "churn", "household account", "kids profile",
    "licensing newer", "catalogue refresh", "content ratings",
    "episode count", "seasons", "box office",
]


def test_no_hallucinated_domain_statistics(celebrity_df):
    """
    Simulate the v373 regression: hallucinated TV-MA/streaming recs for
    a celebrity dataset must be removed by the filter.
    """
    # These are the exact hallucinated recs that appeared in v373
    hallucinated_recs = [
        {"text": "Increase TV-MA content to 36% of the catalogue to match viewer demand.", "timeframe": "Next quarter", "owner": "Content team", "impact": "Important"},
        {"text": "Focus on post-2020 releases which account for 42% of popularity scores.", "timeframe": "Next 30 days", "owner": "Catalogue team", "impact": "Critical"},
        {"text": "Expand local originals from India and UK to drive regional content growth.", "timeframe": "Next quarter", "owner": "Regional team", "impact": "Important"},
        {"text": "Introduce kids profiles and parental controls to capture family subscribers.", "timeframe": "Next 14 days", "owner": "Product team", "impact": "Important"},
        # This one is valid — references actual columns
        {"text": "Analyse the known_for_department distribution: Acting dominates at 93% of records.", "timeframe": "Next 30 days", "owner": "Analytics", "impact": "Important"},
    ]

    filtered = _filter_recommendations(
        hallucinated_recs,
        celebrity_df.columns.tolist(),
        domain="PEOPLE_CATALOG",
    )

    # Only the valid rec should survive
    assert len(filtered) == 1, (
        f"Expected 1 valid rec, got {len(filtered)}. "
        f"Kept: {[r['text'][:60] for r in filtered]}"
    )
    assert "known_for_department" in filtered[0]["text"].lower()

    # Verify none of the forbidden phrases appear in kept recs
    all_text = " ".join(r.get("text", "") for r in filtered).lower()
    for phrase in CELEBRITY_FORBIDDEN_PHRASES:
        assert phrase not in all_text, (
            f"Hallucinated concept {phrase!r} found in kept recommendations"
        )


def test_validate_recommendation_rejects_hallucinated_stats():
    """validate_recommendation rejects percentages not in computed_stats."""
    computed_stats = {"acting_pct": 0.93, "directing_pct": 0.04}

    # 36% is not in computed_stats (acting=93%, directing=4%)
    bad_rec = "Increase TV-MA content to 36% of the catalogue."
    assert not validate_recommendation(bad_rec, ["known_for_department"], computed_stats), \
        "36% is not in computed_stats — should be rejected"

    # 93% IS in computed_stats
    good_rec = "Acting department represents 93% of all records."
    assert validate_recommendation(good_rec, ["known_for_department"], computed_stats), \
        "93% is in computed_stats — should be accepted"


def test_domain_forbidden_concepts_populated():
    """PEOPLE_CATALOG must have forbidden concepts defined."""
    forbidden = DOMAIN_FORBIDDEN_CONCEPTS.get("PEOPLE_CATALOG", [])
    assert len(forbidden) >= 5, "PEOPLE_CATALOG should have at least 5 forbidden concepts"
    assert any("tv-ma" in c.lower() for c in forbidden), "TV-MA must be forbidden for PEOPLE_CATALOG"
    assert any("subscriber" in c.lower() for c in forbidden), "subscriber must be forbidden"


def test_domain_forbidden_block_in_prompt(celebrity_df):
    """The prompt for a celebrity dataset must include the forbidden concepts block."""
    from analyzer import _build_context, _build_data_quality, _generate_prompt

    context = _build_context(celebrity_df)
    quality = _build_data_quality(celebrity_df)
    prompt  = _generate_prompt(context, quality, domain="PEOPLE_CATALOG")

    assert "PEOPLE_CATALOG" in prompt, "Domain name should appear in prompt"
    assert "TV-MA" in prompt or "tv-ma" in prompt.lower(), \
        "TV-MA should be listed as forbidden in the prompt"
    assert "subscriber" in prompt.lower(), \
        "'subscriber' should be listed as forbidden in the prompt"
