import pytest
from render.voice import assert_register


def test_low_must_hedge():
    with pytest.raises(ValueError):
        assert_register("This is definitely caused by fraud.", "LOW")
    # Should not raise
    assert_register(
        "This could reflect fraud, but without timestamps we cannot distinguish.",
        "LOW",
    )


def test_medium_cannot_be_pure_declarative():
    with pytest.raises(ValueError):
        assert_register("Prepaid dominates the segment.", "MEDIUM")
    # Should not raise
    assert_register("Prepaid appears to dominate the segment.", "MEDIUM")


def test_high_allows_declarative():
    # HIGH confidence — no restriction
    assert_register("Prepaid dominates the segment.", "HIGH")
    assert_register("This confirms the pattern.", "HIGH")
