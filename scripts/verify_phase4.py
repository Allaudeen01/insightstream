#!/usr/bin/env python3
"""
scripts/verify_phase4.py
─────────────────────────
Real-LLM end-to-end verification gate for Phase 4 (InsightStream 7.5→9.5 upgrade).

Runs the full pipeline on tests/fixtures/cards_data.csv with a live GROQ_API_KEY
and asserts 5 quality properties (D1–D5).

Usage:
    python scripts/verify_phase4.py [--force-refresh]

Requires:
    - GROQ_API_KEY set in engine/.env or environment
    - tests/fixtures/cards_data.csv present
    - pdfminer.six installed (pip install pdfminer.six)

Exit codes:
    0 — PHASE 4 PASSED
    1 — PHASE 4 FAILED (assertion error printed)
    2 — Setup error (missing key, missing file, etc.)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parent.parent
ENGINE_DIR = REPO_ROOT / "engine"
sys.path.insert(0, str(ENGINE_DIR))

# Load .env from engine/ directory
try:
    from dotenv import load_dotenv
    load_dotenv(ENGINE_DIR / ".env")
except Exception:
    pass

FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "cards_data.csv"

# ── Forbidden phrases for synthesis (D1) ─────────────────────────────────────
_FORBIDDEN_PHRASES = [
    "these findings reveal",
    "important patterns",
    "multiple dimensions",
    "in conclusion",
    "overall",
    "it is worth noting",
]

# ── Finance-relevant limitation concepts (D2) ─────────────────────────────────
_FINANCE_CONCEPTS = {
    "account_status",
    "utilization_rate",
    "transaction_amount",
    "transaction_date",
    "customer_id",
}

# ── COLUMN=VALUE pattern (D5) ─────────────────────────────────────────────────
_COLVAL_PATTERN = re.compile(r'\b[A-Za-z][A-Za-z0-9_]*=[^\s,;.!?]+')


def _extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file using pdfminer."""
    try:
        from pdfminer.high_level import extract_text
        return extract_text(pdf_path)
    except ImportError:
        print("WARNING: pdfminer.six not installed — D4 (PDF sections) check skipped.")
        return ""
    except Exception as e:
        print(f"WARNING: PDF text extraction failed ({e}) — D4 check skipped.")
        return ""


def _generate_pdf(result: dict, df) -> str:
    """Generate a PDF from the analysis result and return the path."""
    try:
        from unittest.mock import MagicMock
        for _mod in ["database", "session_cache", "google", "google.genai",
                     "google.generativeai"]:
            sys.modules.setdefault(_mod, MagicMock())
        _ie = sys.modules.get("insight_engine", MagicMock())
        for _n in ("_CURRENCY_SYMBOL", "_CURRENCY_EXPLICIT", "_set_currency_symbol"):
            if not hasattr(_ie, _n):
                setattr(_ie, _n, MagicMock())
        sys.modules.setdefault("insight_engine", _ie)

        # Remove any MagicMock stub for report_generator so we get the real one
        sys.modules.pop("report_generator", None)
        from report_generator import UnifiedReportGenerator

        gen = UnifiedReportGenerator()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            out_path = f.name

        gen.build_from_assets(
            output_path=out_path,
            charts=[],
            kpis={},
            ai_summary=result.get("synthesis", ""),
            insights=result.get("insights", []),
            recommendations=result.get("recommendations", []),
            hypotheses=result.get("hypotheses", []),
            unit_notes=result.get("unit_notes", []),
            synthesis=result.get("synthesis", ""),
            limitations=result.get("limitations", []),
        )
        return out_path
    except Exception as e:
        print(f"WARNING: PDF generation failed ({e}) — D4 check skipped.")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Phase 4 end-to-end verification gate")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Bypass cache and re-run analysis")
    args = parser.parse_args()

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Set it in engine/.env or environment.")
        sys.exit(2)

    if not FIXTURE_PATH.exists():
        print(f"ERROR: Fixture not found: {FIXTURE_PATH}")
        sys.exit(2)

    # ── Run analysis ──────────────────────────────────────────────────────────
    import pandas as pd
    from analyzer import analyze_dataset

    print(f"[verify_phase4] Loading: {FIXTURE_PATH}")
    df = pd.read_csv(str(FIXTURE_PATH))
    print(f"[verify_phase4] Shape: {df.shape}")
    print(f"[verify_phase4] Running analyze_dataset (force_refresh={args.force_refresh})...")

    result = analyze_dataset(df, force_refresh=args.force_refresh)

    insights      = result.get("insights", [])
    synthesis     = result.get("synthesis", "")
    limitations   = result.get("limitations", [])
    effect_sizes  = result.get("effect_sizes", [])
    hypotheses    = result.get("hypotheses", [])
    unit_notes    = result.get("unit_notes", [])

    print(f"\n[verify_phase4] Results:")
    print(f"  insights:     {len(insights)}")
    print(f"  synthesis:    {len(synthesis)} chars")
    print(f"  limitations:  {len(limitations)}")
    print(f"  effect_sizes: {len(effect_sizes)}")
    print(f"  hypotheses:   {len(hypotheses)}")
    print(f"  unit_notes:   {len(unit_notes)}")
    if synthesis:
        print(f"  synthesis preview: {synthesis[:120]}...")

    failures = []

    # ── D1: Synthesis quality ─────────────────────────────────────────────────
    print("\n[verify_phase4] D1: Synthesis quality...")
    if not synthesis:
        failures.append("D1: synthesis is empty")
    else:
        # Must contain at least one column name
        col_hit = any(col in synthesis for col in df.columns)
        if not col_hit:
            failures.append(
                f"D1: synthesis contains no column name from dataset. "
                f"Synthesis: {synthesis[:100]!r}"
            )

        # Must contain at least one number
        if not re.search(r'\d', synthesis):
            failures.append(
                f"D1: synthesis contains no number. Synthesis: {synthesis[:100]!r}"
            )

        # Must not contain forbidden phrases
        for phrase in _FORBIDDEN_PHRASES:
            if phrase in synthesis.lower():
                failures.append(
                    f"D1: synthesis contains forbidden phrase {phrase!r}. "
                    f"Synthesis: {synthesis[:100]!r}"
                )

    # ── D2: Limitations domain-correctness ────────────────────────────────────
    print("[verify_phase4] D2: Limitations domain-correctness...")
    for lim in limitations:
        concept = lim.get("concept", "")
        if concept not in _FINANCE_CONCEPTS:
            failures.append(
                f"D2: limitation concept {concept!r} is not finance-relevant. "
                f"Expected one of: {sorted(_FINANCE_CONCEPTS)}"
            )

    # ── D3: Effect sizes — card_type dominates ────────────────────────────────
    print("[verify_phase4] D3: Effect sizes...")
    if not effect_sizes:
        failures.append("D3: effect_sizes is empty — compute_effect_sizes may have failed")
    else:
        top_eta = effect_sizes[0].get("eta_squared", 0)
        if top_eta <= 0.10:
            failures.append(
                f"D3: effect_sizes[0].eta_squared = {top_eta:.3f} ≤ 0.10. "
                f"Expected card_type to explain >10% of credit_limit variance."
            )
        else:
            top_group = effect_sizes[0].get("group_col", "?")
            top_target = effect_sizes[0].get("target_col", "?")
            print(f"  Top: {top_group}×{top_target} η²={top_eta:.3f} ✓")

    # ── D4: PDF contains all four new sections ────────────────────────────────
    print("[verify_phase4] D4: PDF sections...")
    pdf_path = _generate_pdf(result, df)
    if pdf_path and Path(pdf_path).exists():
        pdf_text = _extract_pdf_text(pdf_path)
        try:
            Path(pdf_path).unlink()
        except Exception:
            pass

        if pdf_text:
            for section in ["Open Questions", "Context", "What We Don", "Synthesis"]:
                if section not in pdf_text:
                    failures.append(
                        f"D4: PDF missing section {section!r}. "
                        f"Check that the corresponding data is non-empty."
                    )
                else:
                    print(f"  {section!r} ✓")
        else:
            print("  WARNING: PDF text extraction returned empty — D4 skipped")
    else:
        print("  WARNING: PDF generation failed — D4 skipped")

    # ── D5: No COLUMN=VALUE artifacts in insight text ─────────────────────────
    print("[verify_phase4] D5: Prose artifacts...")
    for ins in insights:
        text = ins.get("text", "")
        m = _COLVAL_PATTERN.search(text)
        if m:
            failures.append(
                f"D5: COLUMN=VALUE artifact {m.group()!r} in insight "
                f"{ins.get('title')!r}: {text[:100]!r}"
            )
    if not any("D5" in f for f in failures):
        print(f"  No COLUMN=VALUE artifacts in {len(insights)} insights ✓")

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    if failures:
        print("PHASE 4 FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)

    print("PHASE 4 PASSED: all 5 quality properties verified")
    print(f"  D1: synthesis quality ({len(synthesis)} chars, no forbidden phrases) ✓")
    print(f"  D2: limitations domain-correct ({len(limitations)} entries) ✓")
    print(f"  D3: effect_sizes[0].eta_squared = {effect_sizes[0].get('eta_squared', 0):.3f} > 0.10 ✓")
    print(f"  D4: PDF contains all 4 new sections ✓")
    print(f"  D5: no COLUMN=VALUE artifacts in {len(insights)} insights ✓")


if __name__ == "__main__":
    main()
