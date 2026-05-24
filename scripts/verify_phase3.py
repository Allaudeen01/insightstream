"""
scripts/verify_phase3.py
─────────────────────────
Verifies that all 4 Phase 3 defects are fixed in the real LLM output.
Run after: python scripts/run_analysis.py --input tests/fixtures/cards_data.csv --out-json C:/temp/phase3_final.json
"""
import json
import re
import sys

try:
    with open("C:/temp/phase3_final.json", encoding="utf-8") as f:
        d = json.load(f)
except FileNotFoundError:
    print("ERROR: C:/temp/phase3_final.json not found.")
    print("Run first: python scripts/run_analysis.py --input tests/fixtures/cards_data.csv --out-json C:/temp/phase3_final.json")
    sys.exit(1)

insights = d.get("insights", [])
recs     = d.get("recommendations", [])
blob     = json.dumps(d).lower()
failures = []

# ── D1: No two "highest" claims contradict each other on card_brand × credit_limit ──
highest_brand_claims = []
for ins in insights:
    text = ins.get("text", "").lower()
    if "highest" in text and "card brand" in text:
        for brand in ("mastercard", "visa", "amex", "discover"):
            if brand in text:
                highest_brand_claims.append(brand)
if len(set(highest_brand_claims)) > 1:
    failures.append(
        f"D1: Multiple brands claimed 'highest' for credit_limit: {highest_brand_claims}"
    )

# ── D2: Segmentation headlines use $-formatted numbers ──────────────────────
seg_insights = [
    i for i in insights
    if "card_type" in i.get("text", "").lower()
    and "spread" in i.get("text", "").lower()
]
for s in seg_insights:
    if "$" not in s["text"]:
        failures.append(
            f"D2: Segmentation headline missing $-format: {s['text'][:100]}"
        )

# ── D3: card_on_dark_web does NOT appear in insights ────────────────────────
for ins in insights:
    combined = (ins.get("text", "") + ins.get("title", "")).lower()
    if "card_on_dark_web" in combined:
        failures.append(
            f"D3: card_on_dark_web appears in insight: {ins.get('title')!r}"
        )

# ── D4: At least 2 recommendations survive ──────────────────────────────────
if len(recs) < 2:
    failures.append(
        f"D4: Only {len(recs)} recommendation(s) survived; filter likely still too tight"
    )

# ── D5 (soft): at least one insight has narrative_title ─────────────────────
narrative_count = sum(1 for i in insights if i.get("narrative_title"))
if narrative_count == 0:
    print("WARN D5: No narrative_title fields present (optional polish task)")

# ── Report ───────────────────────────────────────────────────────────────────
if failures:
    print("PHASE 3 FAILED:")
    for f in failures:
        print("  ", f)
    sys.exit(1)

print("PHASE 3 PASSED: all 4 defects fixed")
print(f"  insights:              {len(insights)}")
print(f"  recommendations:       {len(recs)}")
print(f"  narrative_titles:      {narrative_count}/{len(insights)}")
print(f"  hypotheses:            {len(d.get('hypotheses', []))}")
print(f"  unit_notes:            {len(d.get('unit_notes', []))}")
print(f"  data_quality entries:  {len(d.get('data_quality', []))}")
