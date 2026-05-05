# Task 9: Debug Logging Added for Column Detection

## Problem
Backend logs show only `ULIPSTATUS` is detected as numeric column, but `MINPAYMENTAMT` and `Vintage` should also be numeric.

## Root Cause Analysis
1. **Excel Parsing**: Columns are read as strings first, then converted to numeric using `pd.to_numeric()` with 50% threshold
2. **Threshold Issue**: If less than 50% of values can be converted to numeric, column stays as string
3. **Likely Causes for MINPAYMENTAMT/Vintage**:
   - Many null/empty values
   - Some non-numeric values (like "N/A", "-", or text)
   - Formatted as text in Excel
   - Mixed data types

## Changes Made

### 1. Force Numeric Conversion for Known Patterns (`engine/main.py`)
**Location**: Lines 440-470 (Excel parsing logic)

**Added**:
- `FORCE_NUMERIC_PATTERNS` list with keywords: "amt", "amount", "payment", "vintage", "tenure", "age", "years", "commission", "premium", etc.
- Lower threshold (10% instead of 50%) for columns matching these patterns
- Debug logging to show:
  - `[FORCE NUMERIC]` - successful conversion with counts
  - `[FORCE NUMERIC FAILED]` - failed conversion with reason
  - `[FORCE NUMERIC ERROR]` - exception details

### 2. Numeric Column Detection Logging (`engine/main.py`)
**Location**: Lines 475-478

**Added**:
- `[NUMERIC COLS DETECTED]` - shows which columns are detected as numeric after Polars conversion
- Shows first 10 columns for readability

### 3. DataProfile Numeric Logging (`engine/insight_engine.py`)
**Location**: Lines 170-172

**Added**:
- `[PROFILE] Numeric columns detected:` - shows which columns are classified as numeric by DataProfile
- Shows first 10 columns for readability

### 4. Fallback Chart Debug Logging (`engine/insight_engine.py`)
**Location**: Lines 3130-3133

**Added**:
- `[FALLBACK DEBUG] profile.numericals:` - shows numeric columns from profile
- `[FALLBACK DEBUG] num_cols passed:` - shows numeric columns passed to fallback function
- Helps identify where columns are being lost in the pipeline

## Expected Debug Output

When user uploads the insurance dataset, we should now see:

```
[FORCE NUMERIC] MINPAYMENTAMT → 150000/227270 values converted
[FORCE NUMERIC] Vintage → 200000/227270 values converted
[NUMERIC COLS DETECTED] 3 columns: ['ULIPSTATUS', 'MINPAYMENTAMT', 'Vintage']
[PROFILE] Numeric columns detected: ['ULIPSTATUS', 'MINPAYMENTAMT', 'Vintage']
[FALLBACK DEBUG] profile.numericals: ['ULIPSTATUS', 'MINPAYMENTAMT', 'Vintage']
[FALLBACK DEBUG] num_cols passed: ['ULIPSTATUS', 'MINPAYMENTAMT', 'Vintage']
[FALLBACK] Filtered numeric columns: ['MINPAYMENTAMT', 'Vintage']
[FALLBACK] Priority nums: ['MINPAYMENTAMT']
```

## Next Steps

1. **User Action Required**: Restart backend to load new code
   ```bash
   cd engine && python -m uvicorn main:app --port 8000 --reload
   ```

2. **Re-upload Dataset**: Upload the insurance agent dataset again

3. **Check Logs**: Look for the new debug messages to see:
   - Which columns are being force-converted
   - How many values are successfully converted
   - Which columns make it through to fallback charts

4. **If Still Failing**: The debug logs will tell us exactly where the problem is:
   - If `[FORCE NUMERIC FAILED]` appears → columns have too many non-numeric values
   - If `[NUMERIC COLS DETECTED]` doesn't show them → Polars conversion issue
   - If `[PROFILE]` doesn't show them → DataProfile classification issue
   - If `[FALLBACK]` filters them out → ID blacklist is too aggressive

## Files Modified
- `engine/main.py` (lines 440-478)
- `engine/insight_engine.py` (lines 170-172, 3130-3133)
