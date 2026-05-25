"""
Wave 4 tests for _generate_synthesis (Task 8.6).

Property 6: Synthesis Non-Crash — _generate_synthesis always returns a str
and never raises, regardless of findings content or client behaviour.
"""
import sys
import os
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

from analyzer import _generate_synthesis


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_client(response_text: str):
    """Return a mock Groq client that returns response_text."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


def _raising_client(exc=RuntimeError("API error")):
    """Return a mock Groq client that raises on every call."""
    client = MagicMock()
    client.chat.completions.create.side_effect = exc
    return client


def _findings(n: int) -> list[dict]:
    return [
        {"title": f"Finding {i}", "text": f"Some insight text {i} with number {i * 100}."}
        for i in range(1, n + 1)
    ]


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_returns_empty_for_fewer_than_2_findings():
    """Returns '' immediately when len(findings) < 2 — no LLM call."""
    client = _mock_client("should not be called")
    assert _generate_synthesis([], client, "any-model") == ""
    assert _generate_synthesis([{"title": "A", "text": "B"}], client, "any-model") == ""
    client.chat.completions.create.assert_not_called()


def test_returns_string_on_success():
    """Returns the LLM response text as a plain string."""
    client = _mock_client("card_type explains 23% of variance. Segment before modeling.")
    result = _generate_synthesis(_findings(3), client, "test-model")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "card_type" in result


def test_returns_empty_string_on_api_error():
    """Returns '' when the Groq API raises — never re-raises."""
    client = _raising_client(RuntimeError("rate limit"))
    result = _generate_synthesis(_findings(3), client, "test-model")
    assert isinstance(result, str)
    assert result == ""


def test_returns_empty_string_on_timeout():
    """Returns '' on timeout exception."""
    import socket
    client = _raising_client(socket.timeout("timed out"))
    result = _generate_synthesis(_findings(3), client, "test-model")
    assert result == ""


def test_discards_json_looking_response():
    """Discards responses that look like JSON (starts with { or [)."""
    client = _mock_client('{"synthesis": "some text"}')
    result = _generate_synthesis(_findings(3), client, "test-model")
    assert result == ""


def test_uses_top_8_findings_only():
    """Only the first 8 findings are passed to the LLM."""
    client = _mock_client("synthesis text")
    _generate_synthesis(_findings(20), client, "test-model")
    call_args = client.chat.completions.create.call_args
    prompt = call_args[1]["messages"][0]["content"]
    # Findings 1-8 should appear, finding 9+ should not
    assert "Finding 8" in prompt
    assert "Finding 9" not in prompt


def test_returns_string_for_exactly_2_findings():
    """Works correctly at the minimum boundary of 2 findings."""
    client = _mock_client("Two findings point to the same conclusion.")
    result = _generate_synthesis(_findings(2), client, "test-model")
    assert isinstance(result, str)
    assert len(result) > 0


# ── Property test (Task 8.6) ──────────────────────────────────────────────────

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


_finding_strategy = st.fixed_dictionaries({
    "title": st.text(min_size=1, max_size=50),
    "text":  st.text(min_size=1, max_size=200),
})


@given(
    findings=st.lists(_finding_strategy, min_size=2, max_size=10),
    use_raising_client=st.booleans(),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_synthesis_non_crash(findings, use_raising_client):
    """
    Property 6: _generate_synthesis always returns a str and never raises,
    regardless of findings content or whether the client raises.
    """
    if use_raising_client:
        client = _raising_client()
    else:
        client = _mock_client("A synthesis paragraph with some text.")

    result = _generate_synthesis(findings, client, "test-model")
    assert isinstance(result, str), f"Expected str, got {type(result)}"
