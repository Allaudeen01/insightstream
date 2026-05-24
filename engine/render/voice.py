"""
engine/render/voice.py
───────────────────────
Confidence-tier register enforcement.
Ensures findings are written in the correct prose register for their
confidence level — preventing declarative assertions in uncertain findings.

Confidence tiers:
  HIGH   — Direct measurement, n ≥ 100, spread > 5x, no parse issues
           Register: Declarative ("X is", "X dominates")
  MEDIUM — n 30–100, spread 2–5x, or cross-column inference
           Register: Hedged ("appears to", "suggests", "tends to")
  LOW    — n < 30, parse issues, or multiple plausible interpretations
           Register: Hypothesis ("could reflect", "without X we cannot distinguish")
"""
from __future__ import annotations

DECLARATIVE_WORDS = [
    "dominates", "is the", "are the", "shows that", "confirms",
]
HEDGED_WORDS = [
    "appears to", "suggests", "tends to", "likely", "indicates",
]
HYPOTHESIS_WORDS = [
    "could reflect", "may be explained", "without", "cannot distinguish",
    "would need", "could indicate",
]


def assert_register(text: str, confidence: str) -> None:
    """
    Raise ValueError if the text's prose register doesn't match the confidence tier.

    LOW  → must contain at least one hypothesis-register phrase
    MEDIUM → must not be purely declarative (must contain at least one hedge)
    HIGH → no restriction (declarative is fine)
    """
    t = text.lower()
    if confidence == "LOW":
        if not any(w in t for w in HYPOTHESIS_WORDS):
            raise ValueError(
                f"LOW-confidence finding must use hypothesis register "
                f"(e.g., 'could reflect', 'without X we cannot distinguish'): {text!r}"
            )
    elif confidence == "MEDIUM":
        if any(w in t for w in DECLARATIVE_WORDS) and not any(w in t for w in HEDGED_WORDS):
            raise ValueError(
                f"MEDIUM-confidence finding is too declarative — "
                f"add hedging language ('appears to', 'suggests', etc.): {text!r}"
            )
