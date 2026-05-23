"""
verify_financial_filter.py
==========================
Standalone verification script for the financial language filter.
Run with: python engine/verify_financial_filter.py

Tests:
  1. _detect_financial_language_risk returns 'sports' for PSL-like data
  2. _filter_recommendations removes financial recs for sports datasets
  3. _filter_recommendations keeps financial recs for financial datasets
  4. Column-name exception: 'revenue' allowed when it's an actual column
  5. Health/pandemic keywords always removed regardless of dataset type
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import importlib
import pandas as pd

# Force reload to pick up latest code (bypasses .pyc cache)
import analyzer
importlib.reload(analyzer)

from analyzer import (
    _detect_financial_language_risk,
    _filter_recommendations,
    _FINANCIAL_KEYWORDS,
    _FINANCIAL_COL_SIGNALS,
    _FORBIDDEN_REC_KEYWORDS,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_failures = []

def check(condition, label):
    if condition:
        print(f"  {PASS}: {label}")
    else:
        print(f"  {FAIL}: {label}")
        _failures.append(label)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Task 1: Verify components exist ===")
# ─────────────────────────────────────────────────────────────────────────────

check(len(_FINANCIAL_KEYWORDS) >= 5,
      f"_FINANCIAL_KEYWORDS has {len(_FINANCIAL_KEYWORDS)} entries")
check("revenue" in _FINANCIAL_KEYWORDS,
      "'revenue' in _FINANCIAL_KEYWORDS")
check("profit margin" in _FINANCIAL_KEYWORDS,
      "'profit margin' in _FINANCIAL_KEYWORDS")

check(len(_FINANCIAL_COL_SIGNALS) >= 5,
      f"_FINANCIAL_COL_SIGNALS has {len(_FINANCIAL_COL_SIGNALS)} entries")
check("revenue" in _FINANCIAL_COL_SIGNALS,
      "'revenue' in _FINANCIAL_COL_SIGNALS")
check("profit" in _FINANCIAL_COL_SIGNALS,
      "'profit' in _FINANCIAL_COL_SIGNALS")

check("case fatality" in _FORBIDDEN_REC_KEYWORDS,
      "'case fatality' in _FORBIDDEN_REC_KEYWORDS (health terms always forbidden)")
check("revenue" not in _FORBIDDEN_REC_KEYWORDS,
      "'revenue' NOT in _FORBIDDEN_REC_KEYWORDS (moved to _FINANCIAL_KEYWORDS)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Task 2a: PSL-like sports DataFrame ===")
# ─────────────────────────────────────────────────────────────────────────────

df_psl = pd.DataFrame({
    "match_id":      range(1, 51),
    "venue":         ["National Stadium"] * 30 + ["Gaddafi Stadium"] * 20,
    "batting_team":  ["Karachi Kings"] * 25 + ["Lahore Qalandars"] * 25,
    "bowling_team":  ["Lahore Qalandars"] * 25 + ["Karachi Kings"] * 25,
    "extras_type":   ["wide", "no_ball", "bye", "leg_bye", "penalty"] * 10,
    "dismissal_kind":["caught", "bowled", "lbw", "run_out", "stumped"] * 10,
    "winner":        ["Karachi Kings"] * 30 + ["Lahore Qalandars"] * 20,
    "win_by":        [10, 20, 5, 15, 8] * 10,
})

risk = _detect_financial_language_risk(df_psl)
check(risk == "sports",
      f"_detect_financial_language_risk(df_psl) == 'sports' (got {risk!r})")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Task 2b: Filter removes financial recs for sports dataset ===")
# ─────────────────────────────────────────────────────────────────────────────

mixed_recs = [
    # GOOD — references actual columns
    {"text": "Analyse venue concentration: National Stadium hosts 60% of matches. Consider distributing matches to Gaddafi Stadium to reduce venue dependency.", "timeframe": "Next 30 days", "owner": "Operations", "impact": "Important"},
    {"text": "Review dismissal_kind patterns — 'caught' accounts for 40% of dismissals. Batting teams should focus on reducing aerial shots.", "timeframe": "Next 14 days", "owner": "Coaching staff", "impact": "Important"},
    # BAD — financial hallucination
    {"text": "Increase revenue by monetizing match data and improving product pricing strategy for broadcast rights.", "timeframe": "Next quarter", "owner": "Finance", "impact": "Critical"},
    {"text": "Improve profit margins by reducing cost of match operations and optimizing ROI on venue bookings.", "timeframe": "Next 30 days", "owner": "Management", "impact": "Important"},
    # BAD — health hallucination
    {"text": "Monitor case fatality rate and recovery rate across all venues to ensure player safety.", "timeframe": "Ongoing", "owner": "Health Ministry", "impact": "Critical"},
]

filtered = _filter_recommendations(mixed_recs, df_psl.columns.tolist())
check(len(filtered) == 2,
      f"2 good recs kept out of 5 (got {len(filtered)})")
check(all("revenue" not in r["text"].lower() for r in filtered),
      "No 'revenue' in kept recs")
check(all("profit" not in r["text"].lower() for r in filtered),
      "No 'profit' in kept recs")
check(all("case fatality" not in r["text"].lower() for r in filtered),
      "No 'case fatality' in kept recs")
check(any("venue" in r["text"].lower() for r in filtered),
      "Venue rec kept")
check(any("dismissal" in r["text"].lower() for r in filtered),
      "Dismissal rec kept")

print(f"  Kept recs:")
for r in filtered:
    print(f"    - {r['text'][:80]}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Task 2c: Financial dataset keeps financial recs ===")
# ─────────────────────────────────────────────────────────────────────────────

df_fin = pd.DataFrame({
    "revenue":  [100000, 200000, 150000],
    "profit":   [10000,  20000,  15000],
    "product":  ["Widget A", "Widget B", "Widget C"],
    "region":   ["North", "South", "East"],
    "quarter":  ["Q1", "Q2", "Q3"],
})

risk_fin = _detect_financial_language_risk(df_fin)
check(risk_fin is None,
      f"_detect_financial_language_risk(df_fin) == None (got {risk_fin!r})")

fin_recs = [
    {"text": "Increase revenue by 15% in Q3 by expanding product distribution to the North region.", "timeframe": "Next quarter", "owner": "Sales", "impact": "Critical"},
    {"text": "Reduce cost of goods to improve profit margin from 10% to 15% across all product lines.", "timeframe": "Next 30 days", "owner": "Finance", "impact": "Important"},
]
filtered_fin = _filter_recommendations(fin_recs, df_fin.columns.tolist())
check(len(filtered_fin) == 2,
      f"Financial dataset keeps 2/2 financial recs (got {len(filtered_fin)})")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Task 2d: Column-name exception ===")
# ─────────────────────────────────────────────────────────────────────────────

# 'revenue' is a column name — rec mentioning it should be allowed
df_with_revenue_col = pd.DataFrame({
    "revenue":    [100, 200, 300],
    "department": ["HR", "Sales", "IT"],
    "quarter":    ["Q1", "Q2", "Q3"],
})
col_exception_recs = [
    {"text": "The revenue column shows a 50% increase in Q3 — investigate department-level drivers.", "timeframe": "Next 30 days", "owner": "Finance", "impact": "Important"},
]
filtered_col = _filter_recommendations(col_exception_recs, df_with_revenue_col.columns.tolist())
check(len(filtered_col) == 1,
      f"Column-name exception: 'revenue' allowed when it's a column (got {len(filtered_col)})")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Task 2e: Health keywords always removed ===")
# ─────────────────────────────────────────────────────────────────────────────

# Even for a financial dataset, health hallucinations should be removed
health_rec = [
    {"text": "Monitor case fatality rate and recovery rate across all revenue streams.", "timeframe": "Ongoing", "owner": "Health Ministry", "impact": "Critical"},
]
filtered_health = _filter_recommendations(health_rec, df_fin.columns.tolist())
check(len(filtered_health) == 0,
      "Health keywords removed even for financial dataset")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if _failures:
    print(f"RESULT: {len(_failures)} test(s) FAILED:")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {5 + len(mixed_recs) + 2 + 1 + 1} checks PASSED")
    print("Financial language filter is working correctly.")
    print("=" * 60)
