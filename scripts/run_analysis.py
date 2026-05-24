#!/usr/bin/env python3
"""
scripts/run_analysis.py
────────────────────────
CLI wrapper around analyze_dataset() for testing and QA.
Supports --input (CSV path) and --out-json (JSON output path).

Usage:
    python scripts/run_analysis.py --input tests/fixtures/cards_data.csv --out-json /tmp/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

# Add engine to path
ENGINE_DIR = Path(__file__).parent.parent / "engine"
sys.path.insert(0, str(ENGINE_DIR))

# Load .env from engine/ directory
try:
    from dotenv import load_dotenv
    load_dotenv(ENGINE_DIR / ".env")
except Exception:
    pass


def _serialize(obj):
    """JSON serializer that handles non-serializable objects."""
    if hasattr(obj, "to_json"):
        return "<plotly_figure>"
    if hasattr(obj, "data") and hasattr(obj, "layout"):
        return "<plotly_figure>"
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Run InsightStream analysis on a CSV file")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--out-json", required=False, help="Path to write JSON output")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Bypass cache and re-run analysis")
    args = parser.parse_args()

    import pandas as pd
    from analyzer import analyze_dataset

    print(f"[run_analysis] Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"[run_analysis] Shape: {df.shape}")

    result = analyze_dataset(df, force_refresh=args.force_refresh)

    # Build a JSON-serializable version of the result
    serializable = {}
    for k, v in result.items():
        if k == "charts":
            serializable[k] = [f"<chart_{i}>" for i in range(len(v))]
        elif k == "chart_metas":
            serializable[k] = [
                {mk: mv for mk, mv in m.items() if mk != "fig"}
                for m in v
            ]
        else:
            try:
                json.dumps(v, default=_serialize)
                serializable[k] = v
            except Exception:
                serializable[k] = str(v)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(serializable, indent=2, default=_serialize),
            encoding="utf-8",
        )
        print(f"[run_analysis] JSON written to: {out_path}")
    else:
        print(json.dumps(serializable, indent=2, default=_serialize))

    # Summary
    print(f"\n[run_analysis] Summary:")
    print(f"  insights:    {len(result.get('insights', []))}")
    print(f"  charts:      {len(result.get('charts', []))}")
    print(f"  hypotheses:  {len(result.get('hypotheses', []))}")
    print(f"  unit_notes:  {len(result.get('unit_notes', []))}")
    print(f"  data_quality:{len(result.get('data_quality', []))}")
    seg_findings = [f for f in result.get("insights", []) if f.get("is_segmentation")]
    print(f"  seg_findings:{len(seg_findings)}")


if __name__ == "__main__":
    main()
