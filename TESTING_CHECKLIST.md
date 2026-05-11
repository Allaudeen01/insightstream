# Testing Checklist - All Fixes & Enhancements

## 🎯 Quick Start

Upload your product-sales-region dataset and verify each fix is working correctly.

---

## ✅ Tier 0: Critical Fixes Testing

### Fix 0.1: Binary Detection
- [ ] **Test**: Upload dataset with "Returned" column (Int64, values: 0, 1)
- [ ] **Expected**: Return rate appears in executive summary
- [ ] **Look for**: "Return Rate: 24.8%" or similar
- [ ] **Status**: ⬜ Not Tested | ✅ Passed | ❌ Failed

### Fix 0.2: Geographic Protection  
- [ ] **Test**: Upload dataset with "Region" and "RegionManager" columns
- [ ] **Expected**: Insights use Region values (North/South/East/West/Central)
- [ ] **Look for**: NO mentions of person names (Cameron/Eric/Ryan) in geographic context
- [ ] **Status**: ⬜ Not Tested | ✅ Passed | ❌ Failed

### Fix 0.3: TotalPrice Detection
- [ ] **Test**: Upload dataset with "UnitPrice", "Quantity", "TotalPrice" columns
- [ ] **Expected**: Revenue = sum(TotalPrice), not sum(UnitPrice × Quantity)
- [ ] **Look for**: Total Revenue ≈ ₹43.80L (not ₹47.28L)
- [ ] **Status**: ⬜ Not Tested | ✅ Passed | ❌ Failed

### Fix 0.4: RPU Calculation
- [ ] **Test**: Check "Volume-Value Decoupling" insight
- [ ] **Expected**: RPU values in ₹200-300 range
- [ ] **Look for**: "East generates highest revenue per unit (₹287)" (not ₹31)
- [ ] **Status**: ⬜ Not Tested | ✅ Passed | ❌ Failed

### Fix 0.5: Executive Summary Count
- [ ] **Test**: Count findings in "Strategic Findings & Key Results" section
- [ ] **Expected**: Executive summary count matches number of findings shown
- [ ] **Look for**: "4 high-impact findings" if 4 findings shown
- [ ] **Status**: ⬜ Not Tested | ✅ Passed | ❌ Failed

### Fix 0.6: Pricing Simulation
- [ ] **Test**: Check "Pricing Standardization" insight
- [ ] **Expected**: Either suppressed or shows correct value
- [ ] **Look for**: "variance is structural" message if within-category CV ≈ overall CV
- [ ] **Status**: ⬜ Not Tested | ✅ Passed | ❌ Failed

---

## ✅ Tier 1: Enhancements Testing

### Enhancement 1.1: Column Coverage Tracker
- [ ] **Test**: Upload dataset with 15+ columns
- [ ] **Expected**: Coverage report in API response
- [ ] **Check**: `result["column_coverage"]` in API response
- [ ] **Verify**: 
  - [ ] `coverage_pct` is computed correctly
  - [ ] `high_value_missed` flags columns like "Discount", "Salesperson"
  - [ ] Warning appears if high-value columns missed
- [ ] **Status**: ⬜ Not Tested | ✅ Passed | ❌ Failed

### Enhancement 1.2: Enhanced Temporal Analysis
- [ ] **Test**: Upload dataset with date column and revenue
- [ ] **Expected**: Rich temporal insight with trend and seasonality
- [ ] **Verify**:
  - [ ] Trend direction appears (growing/declining/flat)
  - [ ] Monthly growth rate (%) appears
  - [ ] Seasonality detection if applicable
  - [ ] Score is 9.0 (not 7.5)
  - [ ] Impact is Critical if gap > 30% or |slope| > 5%
- [ ] **Status**: ⬜ Not Tested | ✅ Passed | ❌ Failed

### Enhancement 5.6: Sanity Checker
- [ ] **Test**: Upload dataset with person and region columns
- [ ] **Expected**: No person names in geographic insights
- [ ] **Check**: Logs for "[SANITY CHECKER]" messages
- [ ] **Verify**:
  - [ ] Blocked insights don't appear in report
  - [ ] Flagged insights have confidence warnings
  - [ ] No entity confusion in final insights
- [ ] **Status**: ⬜ Not Tested | ✅ Passed | ❌ Failed

---

## 🔍 Detailed Verification

### Executive Summary Metrics
- [ ] Total Revenue: ₹43.80L (not ₹47.28L)
- [ ] Average Order Value: ₹2.9K (not ₹3.2K)
- [ ] Return Rate: 24.8% (visible, not hidden)
- [ ] High-Impact Findings: Count matches report (e.g., "4 findings")

### Strategic Findings
- [ ] Regional Analysis: Uses region names (North/South/East/West/Central)
- [ ] NO person names: No Cameron, Eric, Ryan, Sophie, Wendy in geographic context
- [ ] Volume-Value: RPU values meaningful (₹200-300 range, not ₹30)
- [ ] Pricing: Simulation suppressed or shows correct methodology

### New Features
- [ ] Column Coverage: Report shows analyzed vs. untouched columns
- [ ] Temporal Insights: Include trend direction and growth rate
- [ ] Seasonality: Detected if CV > 0.15 across calendar months
- [ ] Sanity Checks: Logs show validation results

---

## 🐛 Known Issues to Watch For

### Red Flags (Should NOT Appear)
- ❌ Person names in geographic insights (Cameron, Eric, Ryan, etc.)
- ❌ RPU values < ₹100 (likely wrong calculation)
- ❌ Revenue > sum(TotalPrice) from data (indicates UnitPrice×Qty used)
- ❌ Executive summary count ≠ findings shown
- ❌ Missing return rate when "Returned" column exists

### Green Flags (Should Appear)
- ✅ Return rate visible in executive summary
- ✅ Only region names in geographic insights
- ✅ RPU values in ₹200-300 range
- ✅ Revenue matches sum(TotalPrice)
- ✅ Count matches report content
- ✅ Column coverage report present
- ✅ Temporal insights show trend
- ✅ Sanity checker logs present

---

## 📊 API Response Validation

### Check These Fields in API Response:
```json
{
  "computed_metrics": {
    "total_revenue": { "value": 4380000 },  // Should be ~₹43.80L
    "return_rate": { "value": 24.8 }        // Should be visible
  },
  "column_coverage": {
    "total_columns": 19,
    "analyzed_columns": 8,
    "coverage_pct": 42.1,
    "high_value_missed": ["Discount", "Salesperson"]
  },
  "strategic_brief": [
    // Should have 4 insights if executive summary says "4 findings"
  ],
  "warnings": [
    // May include column coverage warnings
    // May include sanity checker warnings
  ]
}
```

---

## 🔧 Troubleshooting

### If Return Rate Not Showing:
1. Check if "Returned" column exists in dataset
2. Verify column dtype is Int64 or Boolean
3. Check values are {0, 1} or {True, False}
4. Look for `[PROFILE] Numeric columns detected` in logs

### If Person Names Still Appearing:
1. Check logs for `[EntityDetection]` messages
2. Verify person columns detected correctly
3. Check sanity checker logs for blocked insights
4. Review geographic_col assignment in logs

### If Revenue Still Wrong:
1. Check if "TotalPrice" column exists
2. Look for `[SubRole] Promoted 'TotalPrice' to revenue_col` in logs
3. Verify correlation > 0.8 with UnitPrice × Quantity
4. Check computed_metrics.total_revenue in API response

### If RPU Still Nonsensical:
1. Verify revenue_col is set correctly
2. Check if _computed_rev is being used
3. Look for RPU calculation in Volume-Value insight
4. Verify values are in ₹200-300 range

---

## 📝 Test Results Template

```
Test Date: _______________
Dataset: product-sales-region
Tester: _______________

TIER 0 FIXES:
[ ] 0.1 Binary Detection - Status: _______
[ ] 0.2 Geographic Protection - Status: _______
[ ] 0.3 TotalPrice Detection - Status: _______
[ ] 0.4 RPU Calculation - Status: _______
[ ] 0.5 Executive Summary Count - Status: _______
[ ] 0.6 Pricing Simulation - Status: _______

TIER 1 ENHANCEMENTS:
[ ] 1.1 Column Coverage - Status: _______
[ ] 1.2 Temporal Analysis - Status: _______
[ ] 5.6 Sanity Checker - Status: _______

OVERALL STATUS: ⬜ All Passed | ⬜ Some Failed | ⬜ Not Tested

NOTES:
_________________________________________________
_________________________________________________
_________________________________________________
```

---

## 🎉 Success Criteria

### All Tests Pass When:
1. ✅ Return rate visible (24.8%)
2. ✅ No person names in geographic insights
3. ✅ Revenue = ₹43.80L (correct)
4. ✅ RPU values meaningful (₹200-300)
5. ✅ Count matches report (4 findings)
6. ✅ Column coverage report present
7. ✅ Temporal insights show trend
8. ✅ Sanity checker logs clean

### Ready for Production When:
- All Tier 0 fixes verified ✅
- All Tier 1 enhancements verified ✅
- No red flags detected ✅
- All green flags present ✅

---

**Created**: May 7, 2026
**Status**: Ready for Testing
**Next**: Upload dataset and start testing!
