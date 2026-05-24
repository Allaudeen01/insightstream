"""
engine/llm/headers.py
──────────────────────
Generates newspaper-style narrative section headers for findings.
Uses the Groq LLM with a strict prompt; falls back to a rule-based
generator if the API is unavailable.
"""
from __future__ import annotations

import os
import re
from typing import Optional


# ── Verb list for validation ──────────────────────────────────────────────────
_VERB_LIST = [
    "is", "are", "has", "have", "shows", "show", "reveals", "reveal",
    "dominates", "dominate", "drives", "drive", "masks", "mask",
    "signals", "signal", "outperforms", "surges", "collapses",
    "diverges", "clusters", "skews", "inflates", "deflates",
    "appears", "suggests", "indicates", "contains", "accounts",
    "represents", "makes", "leads", "follows", "exceeds", "falls",
    "rises", "drops", "grows", "declines", "increases", "decreases",
    "tells", "hides", "look", "looks", "check", "investigating",
    "worth", "watching",
]


def has_verb(header: str) -> bool:
    """Check if a header contains at least one verb from the known list."""
    words = set(header.lower().split())
    return any(v in words for v in _VERB_LIST)


def _rule_based_header(title: str, body: str) -> str:
    """
    Generate a narrative header from a finding title + body using simple rules.
    Always produces a header containing at least one verb.
    """
    # If body mentions a huge spread (segmentation finding)
    spread_match = re.search(r"(\d+)x\s+spread", body, re.IGNORECASE)
    if spread_match:
        vs_match = re.search(
            r"(\w[\w\s]+?)\s+(?:cards?|accounts?|customers?).*?vs\.?\s+(\w[\w\s]+?)\s+(?:cards?|accounts?)",
            body, re.IGNORECASE,
        )
        if vs_match:
            a = vs_match.group(1).strip().title()
            b = vs_match.group(2).strip().title()
            return f"{a} Cards Look Similar to {b} — Until You Check the Limit"
        return f"A {spread_match.group(1)}x Spread Reveals Hidden Segmentation"

    # If body mentions a dominant pattern
    dominant_match = re.search(
        r"'?(\w[\w\s]+?)'?\s+(?:accounts for|dominates|is the largest)",
        body, re.IGNORECASE,
    )
    if dominant_match:
        entity = dominant_match.group(1).strip()
        return f"{entity} Dominates — But the Distribution Tells a Deeper Story"

    # If body mentions a correlation
    corr_match = re.search(
        r"(strong|moderate|weak)\s+(positive|negative)\s+correlation",
        body, re.IGNORECASE,
    )
    if corr_match:
        strength = corr_match.group(1).title()
        direction = corr_match.group(2).title()
        clean = title.replace("_", " ").title()
        return f"{clean}: A {strength} {direction} Link Suggests a Pattern"

    # If body mentions a trend
    trend_match = re.search(r"(increasing|decreasing|stable)\s+trend", body, re.IGNORECASE)
    if trend_match:
        trend = trend_match.group(1).title()
        clean = title.replace("_", " ").title()
        return f"{clean} Shows an {trend} Trend Worth Watching"

    # Default: always includes a verb
    clean = title.replace("_", " ").title()
    return f"{clean} Reveals Patterns Worth Investigating"


def generate_narrative_headers(
    findings: list[dict],
    groq_client=None,
    model: str = "llama-3.1-8b-instant",
) -> list[str]:
    """
    Generate one narrative header per finding.

    Args:
        findings: list of dicts with 'title' and 'body' keys
        groq_client: optional Groq client; if None, uses rule-based fallback
        model: Groq model to use

    Returns:
        list of header strings, one per finding (same order)
    """
    if not findings:
        return []

    # Try LLM if client provided
    if groq_client is not None:
        try:
            findings_text = "\n".join(
                f"{i+1}. Title: {f.get('title','')}\n   Body: {f.get('body','')[:200]}"
                for i, f in enumerate(findings)
            )
            prompt = f"""You are writing newspaper-style section headers for a data analysis report.

For each finding below, write a 5–12 word header that:
- Names the entity or pattern (not the column name)
- Implies the takeaway, not just the topic
- Uses an active verb when possible
- Includes a tension word ("but", "yet", "however") when the finding is counterintuitive

Bad: "Card Type Distribution"
Good: "Prepaid Cards Look Identical to Debit — Until You Check the Limit"

Bad: "Credit Limit by Number of Cards Issued"
Good: "More Cards, Lower Limits: An Inverted Relationship"

Output ONLY the header strings, one per line, in the order given. No numbering, no explanation.

Findings:
{findings_text}"""

            resp = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            lines = [
                line.strip()
                for line in resp.choices[0].message.content.strip().split("\n")
                if line.strip()
            ]
            # Validate: must have same count as findings, each 5–15 words
            if len(lines) == len(findings):
                validated = []
                for idx, line in enumerate(lines):
                    words = line.split()
                    if 5 <= len(words) <= 15:
                        validated.append(line)
                    else:
                        validated.append(_rule_based_header(
                            findings[idx].get("title", ""),
                            findings[idx].get("body", ""),
                        ))
                return validated
        except Exception as e:
            print(f"[headers] LLM header generation failed ({e}), using rule-based fallback")

    # Rule-based fallback
    return [
        _rule_based_header(f.get("title", ""), f.get("body", ""))
        for f in findings
    ]
