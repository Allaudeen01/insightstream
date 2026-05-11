"""
Verification Script for TIER 1 Enhancements
============================================

This script verifies that all Tier 1 enhancements have been properly applied
to the insight engine code.
"""

import re
from pathlib import Path

def verify_enhancements():
    """Check that all Tier 1 enhancements are present in the code."""
    
    insight_engine_path = Path("engine/insight_engine.py")
    
    if not insight_engine_path.exists():
        print("❌ ERROR: insight_engine.py not found")
        return False
    
    content = insight_engine_path.read_text(encoding='utf-8')
    
    print("=" * 70)
    print("TIER 1 ENHANCEMENTS VERIFICATION")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Check for ColumnCoverageTracker class
    print("Enhancement 1.1: Column Coverage Tracker")
    print("-" * 70)
    
    if "class ColumnCoverageTracker:" in content:
        print("✅ ColumnCoverageTracker class present")
    else:
        print("❌ ColumnCoverageTracker class missing")
        all_passed = False
    
    if "def mark(self, *cols: str):" in content:
        print("✅ mark() method present")
    else:
        print("❌ mark() method missing")
        all_passed = False
    
    if "high_value_missed" in content:
        print("✅ High-value column detection present")
    else:
        print("❌ High-value column detection missing")
        all_passed = False
    
    if "coverage = ColumnCoverageTracker(df, profile)" in content:
        print("✅ Coverage tracker initialized in run_insight_engine")
    else:
        print("❌ Coverage tracker not initialized")
        all_passed = False
    
    if 'result["column_coverage"]' in content:
        print("✅ Coverage report added to result")
    else:
        print("❌ Coverage report not added to result")
        all_passed = False
    
    print()
    
    # Check for Enhanced Temporal Analysis
    print("Enhancement 1.2: Enhanced Time-Series Module")
    print("-" * 70)
    
    if "TIER 1.2" in content:
        print("✅ Tier 1.2 markers present")
    else:
        print("❌ Tier 1.2 markers missing")
        all_passed = False
    
    if "slope, intercept = np.polyfit" in content:
        print("✅ Trend slope calculation present")
    else:
        print("❌ Trend slope calculation missing")
        all_passed = False
    
    if "trend_direction" in content and "growing" in content:
        print("✅ Trend direction detection present")
    else:
        print("❌ Trend direction detection missing")
        all_passed = False
    
    if "seasonality_cv" in content:
        print("✅ Seasonality detection present")
    else:
        print("❌ Seasonality detection missing")
        all_passed = False
    
    if re.search(r'score=9\.0.*TIER 1\.2', content, re.DOTALL):
        print("✅ Temporal score boosted to 9.0")
    else:
        print("❌ Temporal score not boosted")
        all_passed = False
    
    if "Trend: {slope_pct:+.1f}%/mo" in content or "trend_slope_pct" in content:
        print("✅ Trend percentage in output")
    else:
        print("❌ Trend percentage missing from output")
        all_passed = False
    
    print()
    
    # Check for SanityChecker
    print("Enhancement 5.6: Sanity Checker")
    print("-" * 70)
    
    if "class SanityChecker:" in content:
        print("✅ SanityChecker class present")
    else:
        print("❌ SanityChecker class missing")
        all_passed = False
    
    if "_check_entity_confusion" in content:
        print("✅ Entity confusion check present")
    else:
        print("❌ Entity confusion check missing")
        all_passed = False
    
    if "_check_magnitude" in content:
        print("✅ Magnitude sanity check present")
    else:
        print("❌ Magnitude sanity check missing")
        all_passed = False
    
    if "_check_count_consistency" in content:
        print("✅ Count consistency check present")
    else:
        print("❌ Count consistency check missing")
        all_passed = False
    
    if "checker = SanityChecker(df, profile)" in content:
        print("✅ Sanity checker initialized in run_insight_engine")
    else:
        print("❌ Sanity checker not initialized")
        all_passed = False
    
    if "checker.check_all(compressed_insights, metrics)" in content:
        print("✅ Sanity checker wired into pipeline")
    else:
        print("❌ Sanity checker not wired into pipeline")
        all_passed = False
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 ALL TIER 1 ENHANCEMENTS VERIFIED SUCCESSFULLY!")
        print("\nImplemented Enhancements:")
        print("1. ✅ Column Coverage Tracker (Tier 1.1)")
        print("2. ✅ Enhanced Time-Series Module (Tier 1.2)")
        print("3. ✅ Sanity Checker (Tier 5.6)")
        print("\nNext steps:")
        print("1. Test with product-sales-region dataset")
        print("2. Review column coverage report")
        print("3. Verify temporal insights show trend and seasonality")
        print("4. Check sanity checker logs for blocked/flagged insights")
    else:
        print("\n⚠️  SOME ENHANCEMENTS ARE MISSING - REVIEW REQUIRED")
    
    print()
    return all_passed

if __name__ == "__main__":
    verify_enhancements()
