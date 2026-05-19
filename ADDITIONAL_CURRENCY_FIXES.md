# Additional Hardcoded Currency Fixes

## Status: ✅ COMPLETE

## Overview
Fixed 4 additional locations in `report_generator.py` where ₹ was hardcoded instead of using the detected currency symbol.

---

## Fixes Applied

### Fix 1: Lines 1799-1802 - Chart Annotation Formatter
**Location**: `smart_fmt()` function in chart generation

**Before**:
```python
def smart_fmt(x, _):
    if x >= 1e7:   return f"₹{x/1e7:.1f}Cr"
    if x >= 1e5:   return f"₹{x/1e5:.1f}L"
    if x >= 1e3:   return f"₹{x/1e3:.0f}K"
    return f"₹{x:.0f}"
```

**After**:
```python
_sym = getattr(self, '_currency_symbol', '₹')

def smart_fmt(x, _):
    if _sym == '₹':
        if x >= 1e7:   return f"₹{x/1e7:.1f}Cr"
        if x >= 1e5:   return f"₹{x/1e5:.1f}L"
        if x >= 1e3:   return f"₹{x/1e3:.0f}K"
        return f"₹{x:.0f}"
    else:
        if x >= 1e9:   return f"{_sym}{x/1e9:.1f}B"
        if x >= 1e6:   return f"{_sym}{x/1e6:.1f}M"
        if x >= 1e3:   return f"{_sym}{x/1e3:.0f}K"
        return f"{_sym}{x:.0f}"
```

**Impact**: Chart annotations now use correct currency symbol

---

### Fix 2: Line 2643 - Regional Stats Table
**Location**: Regional statistics median formatting

**Before**:
```python
region_stats_df[f"Median {target_metric}"] = region_stats_df[f"Median {target_metric}"].apply(
    lambda v: f"₹{v:,.0f}" if _use_currency else f"{v:,.1f}"
)
```

**After**:
```python
_sym = getattr(self, '_currency_symbol', '₹')
region_stats_df[f"Median {target_metric}"] = region_stats_df[f"Median {target_metric}"].apply(
    lambda v: f"{_sym}{v:,.0f}" if _use_currency else f"{v:,.1f}"
)
```

**Impact**: Regional statistics tables use correct currency

---

### Fix 3: Line 3161 - HR Domain Median Income
**Location**: HR KPIs median monthly income

**Before**:
```python
hr_kpis["Median Monthly Income"] = (
    f"₹{avg_income:,.0f}" if avg_income > 100 else f"{avg_income:.1f}"
)
```

**After**:
```python
_sym = getattr(self, '_currency_symbol', '₹')
hr_kpis["Median Monthly Income"] = (
    f"{_sym}{avg_income:,.0f}" if avg_income > 100 else f"{avg_income:.1f}"
)
```

**Impact**: HR reports show correct currency for income

---

### Fix 4: Line 3472 - Build From Assets Regional Stats
**Location**: Regional statistics in `build_from_assets()` method

**Before**:
```python
region_stats_df[f"Median {target_metric}"] = region_stats_df[f"Median {target_metric}"].apply(
    lambda v: f"₹{v:,.0f}" if _use_currency_bfa else f"{v:,.1f}"
)
```

**After**:
```python
_sym = getattr(self, '_currency_symbol', '₹')
region_stats_df[f"Median {target_metric}"] = region_stats_df[f"Median {target_metric}"].apply(
    lambda v: f"{_sym}{v:,.0f}" if _use_currency_bfa else f"{v:,.1f}"
)
```

**Impact**: PDF regional statistics use correct currency

---

## Summary of All Currency Fixes

### Previously Fixed (Commit b700209)
1. ✅ Text replacement (lines 1472-1473)
2. ✅ Replacement tuples (lines 1622-1625)
3. ✅ `_fmt_inr()` method (lines 1111-1114)
4. ✅ `_find_revenue()` method (uses detected currency)

### This Commit
5. ✅ Chart annotation formatter (lines 1799-1802)
6. ✅ Regional stats table (line 2643)
7. ✅ HR median income (line 3161)
8. ✅ Build from assets regional stats (line 3472)

---

## Verification

✅ File compiles without errors  
✅ All hardcoded ₹ replaced with `_sym`  
✅ Backward compatible (defaults to ₹)  
✅ Uses detected currency symbol

---

## Testing Checklist

### Chart Annotations
- [ ] Upload UK dataset with GBP
- [ ] Generate chart with revenue annotations
- [ ] Verify annotations show £ (not ₹)

### Regional Statistics
- [ ] Upload multi-region dataset with GBP
- [ ] Check regional stats table
- [ ] Verify median values show £

### HR Reports
- [ ] Upload HR dataset with USD
- [ ] Check Median Monthly Income KPI
- [ ] Verify shows $ (not ₹)

### PDF Export
- [ ] Export PDF with GBP dataset
- [ ] Check regional statistics section
- [ ] Verify all currency values show £

---

## Complete Currency Fix Locations

| Line | Location | Fixed |
|------|----------|-------|
| 1111-1114 | `_fmt_inr()` method | ✅ |
| 1133-1143 | `_find_revenue()` method | ✅ |
| 1472-1473 | Text replacement | ✅ |
| 1622-1625 | Replacement tuples | ✅ |
| 1799-1802 | Chart annotation formatter | ✅ |
| 2643 | Regional stats table | ✅ |
| 3161 | HR median income | ✅ |
| 3472 | Build from assets regional stats | ✅ |

---

## Commit Message
```
fix: additional hardcoded currency locations

- Fix chart annotation formatter to use detected currency
- Fix regional stats table formatting
- Fix HR median income KPI formatting
- Fix build_from_assets regional stats formatting
- All locations now use self._currency_symbol
```

**Status**: ✅ ALL HARDCODED CURRENCIES FIXED
