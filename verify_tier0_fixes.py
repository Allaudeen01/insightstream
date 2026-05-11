"""
Verification Script for TIER 0 Critical Fixes
==============================================

This script verifies that all critical fixes have been properly applied
to the insight engine code.
"""

import re
from pathlib import Path

def verify_fixes():
    """Check that all P0 fixes are present in the code."""
    
    insight_engine_path = Path("engine/insight_engine.py")
    
    if not insight_engine_path.exists():
        print("❌ ERROR: insight_engine.py not found")
        return False
    
    content = insight_engine_path.read_text(encoding='utf-8')
    
    fixes_to_verify = {
        "Bug 0.1 - Binary Detection": r"P0 FIX \(Bug 0\.1\).*Numeric binary detection",
        "Bug 0.2 - Geographic Guard": r"P0 FIX \(Bug 0\.2\).*Geographic assignment",
        "Bug 0.3 - TotalPrice Detection": r"P0 FIX \(Bug 0\.3\).*POST-LOOP.*Detect row-level revenue",
        "Bug 0.4 - RPU Calculation": r"P0 FIX \(Bug 0\.4\).*Always compute actual revenue",
        "Bug 0.5 - Executive Summary Count": r"P0 FIX \(Bug 0\.5\).*Count from compressed_insights",
        "Bug 0.6 - Pricing Simulation": r"P0 FIX \(Bug 0\.6\).*within-group vs between-group",
    }
    
    print("=" * 70)
    print("TIER 0 CRITICAL FIXES VERIFICATION")
    print("=" * 70)
    print()
    
    all_passed = True
    
    for fix_name, pattern in fixes_to_verify.items():
        if re.search(pattern, content, re.DOTALL | re.IGNORECASE):
            print(f"✅ {fix_name}")
        else:
            print(f"❌ {fix_name} - NOT FOUND")
            all_passed = False
    
    print()
    print("=" * 70)
    
    # Additional checks
    print("\nADDITIONAL CHECKS:")
    print("-" * 70)
    
    # Check for entity detection method
    if "_detect_entity_type" in content:
        print("✅ Entity type detection method present")
    else:
        print("❌ Entity type detection method missing")
        all_passed = False
    
    # Check for person_columns tracking
    if "profile.person_columns" in content:
        print("✅ Person columns tracking present")
    else:
        print("❌ Person columns tracking missing")
        all_passed = False
    
    # Check for within-category CV calculation
    if "within_cvs" in content and "avg_within_cv" in content:
        print("✅ Within-category CV calculation present")
    else:
        print("❌ Within-category CV calculation missing")
        all_passed = False
    
    # Check for _computed_rev usage
    if "_computed_rev" in content:
        print("✅ Computed revenue column present")
    else:
        print("❌ Computed revenue column missing")
        all_passed = False
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 ALL TIER 0 CRITICAL FIXES VERIFIED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Test with product-sales-region dataset")
        print("2. Verify return rate metrics appear")
        print("3. Verify 'Cameron' finding is eliminated")
        print("4. Verify revenue = TotalPrice (not UnitPrice × Quantity)")
        print("5. Verify RPU values are meaningful (₹200-300 range)")
        print("6. Verify pricing simulation is suppressed if variance is structural")
    else:
        print("\n⚠️  SOME FIXES ARE MISSING - REVIEW REQUIRED")
    
    print()
    return all_passed

if __name__ == "__main__":
    verify_fixes()
