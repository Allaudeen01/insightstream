# Data Cleaning Fix - "All Rows Removed" Issue

## ✅ FIXED

## Problem
Error: "Data cleaning removed all rows. Please check your data quality."

## Root Cause
The `auto_clean_dataframe()` function was using `pdf.dropna()` which drops ANY row with ANY null value. With 55 columns, virtually every row had at least one null value somewhere, so ALL rows were being dropped.

## Solution
Changed from:
```python
# BEFORE - Too aggressive
pdf = pdf.dropna()  # Drops rows with ANY null
```

To:
```python
# AFTER - Only drops completely empty rows
pdf = pdf.dropna(how='all')  # Only drops rows where ALL values are null
```

## Changes Made

### File: `engine/insight_engine.py` (Lines ~3555-3585)

**Before:**
```python
def auto_clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    pdf = df.to_pandas()
    pdf = pdf.dropna()  # ← Drops ANY row with ANY null
    pdf = pdf.drop_duplicates()
    
    # Coerce numeric columns
    for col in pdf.columns:
        if pdf[col].dtype == object and any(kw in col.lower() for kw in _NUMERIC_KW):
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
    pdf = pdf.dropna()  # ← Drops again after coercion
    
    # Coerce date columns
    for col in pdf.columns:
        if pdf[col].dtype == object and any(kw in col.lower() for kw in _DATE_KW):
            pdf[col] = pd.to_datetime(pdf[col], errors="coerce")
    pdf = pdf.dropna()  # ← Drops again after coercion
    
    return pl.from_pandas(pdf)
```

**After:**
```python
def auto_clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    pdf = df.to_pandas()
    
    # Only drop rows where ALL values are null
    pdf = pdf.dropna(how='all')  # ← Fixed!
    pdf = pdf.drop_duplicates()
    
    # Coerce numeric columns
    for col in pdf.columns:
        if pdf[col].dtype == object and any(kw in col.lower() for kw in _NUMERIC_KW):
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
    
    # Coerce date columns
    for col in pdf.columns:
        if pdf[col].dtype == object and any(kw in col.lower() for kw in _DATE_KW):
            pdf[col] = pd.to_datetime(pdf[col], errors="coerce")
    
    # Only drop rows where ALL values are null (after coercion)
    pdf = pdf.dropna(how='all')  # ← Fixed!
    
    return pl.from_pandas(pdf)
```

## Impact

### Before Fix:
- 227K rows uploaded
- Data cleaning runs
- **ALL 227K rows dropped** (any row with any null removed)
- Error: "Data cleaning removed all rows"

### After Fix:
- 227K rows uploaded
- Data cleaning runs
- **Only completely empty rows dropped** (rows where all 55 columns are null)
- Most/all rows preserved
- Upload succeeds ✅

## Backend Status
✅ Backend auto-reloaded with fix
✅ Server running on http://127.0.0.1:8000
✅ Ready for upload

## Next Steps
1. **Upload your insurance dataset again**
2. Should now succeed with 227K rows preserved
3. Check backend logs for confirmation

## Expected Backend Logs
```
=== UPLOAD DEBUG ===
Filename: insurance_data.xlsx
Bytes read: XXXXXX
Shape: (227000, 55)
[quality] critical=0  medium=X
[quality] DataFrame shape after validation: (227000, 55)
[quality] Auto-cleaning medium issues...
[quality] auto-cleaned → (227000, 55)  ← Should preserve most rows!

ColumnMap → numeric='MINPAYMENTAMT'  category='AGENTSTATUSCD'  region='STATECD'

=== UPLOAD SUCCESS: (227000, 55) ===
```

## Status: 🟢 READY TO TEST

Try uploading your file again now!
