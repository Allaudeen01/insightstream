# Final Fix - Insight Engine Fallback Charts

## ✅ ROOT CAUSE FOUND AND FIXED

## Problem
The ID blacklist was only in `report_generator.py` but NOT in `insight_engine.py`. The fallback chart generation in the insight engine was using ALL numeric columns including IDs like `LACLIENTNUMBER`.

## Solution Applied

### Added ID Blacklist to Insight Engine
**File**: `engine/insight_engine.py`  
**Function**: `_add_fallback_charts()` (Lines ~3125-3165)

```python
# ID column blacklist - exclude these from fallback charts
ID_KEYWORDS = [
    "num", "number", "id", "code", "cd", "ifsc", "pin", "pincode",
    "adhaar", "aadhaar", "account", "mobile", "contact", "license",
    "tax", "payee", "employee", "agent", "branch", "application",
    "laclient", "parent", "recruited", "partner_code", "channel_code",
    "sub_channel_code", "payee_code", "account_payee", "mapped"
]

# Filter out ID columns from num_cols
filtered_num_cols = []
for c in num_cols:
    col_lower = c.lower().replace(" ", "").replace("_", "")
    if not any(id_kw.replace("_", "") in col_lower for id_kw in ID_KEYWORDS):
        filtered_num_cols.append(c)
```

### Added "amt" and "payment" to Priority Keywords
```python
# Priority 1: revenue/sales/amount/price columns
priority_nums = [
    c for c in filtered_num_cols
    if any(k in c.lower() for k in ["sales", "amount", "amt", "payment", "revenue", "price", "profit"])
]
```

### Added Debug Logging
```python
print(f"[FALLBACK] Filtered numeric columns: {filtered_num_cols[:5]}")
print(f"[FALLBACK] Priority nums: {priority_nums}")
print(f"[FALLBACK] Ordered nums: {ordered_nums}")
```

## Expected Results

### Before Fix:
```
[CHART SUPPRESSED] Fallback LACLIENTNUMBER by CHANNELCD — variance too low
```
- Tries to use `LACLIENTNUMBER` (ID)
- Falls back to `EMPLOYEECD` (ID)
- Falls back to `SUB_CHANNEL_CODE` (code)

### After Fix:
```
[FALLBACK] Filtered numeric columns: ['MINPAYMENTAMT', 'Vintage', ...]
[FALLBACK] Priority nums: ['MINPAYMENTAMT']
[FALLBACK] Ordered nums: ['MINPAYMENTAMT', 'Vintage']
```
- Skips all ID columns
- Detects `MINPAYMENTAMT` as priority (contains "amt" and "payment")
- Uses `MINPAYMENTAMT` and `Vintage` for charts

## Testing Instructions

### Step 1: Upload File Again (Fresh Session)
Since you already have session `705ce5f8-7cf5-46cf-9823-212f4a8d99f1`, you need to either:
- **Option A**: Upload the file again to create a new session
- **Option B**: Navigate to Insights page and it should regenerate with new code

### Step 2: Watch Backend Logs
Look for these NEW log lines:
```
[FALLBACK] Filtered numeric columns: ['MINPAYMENTAMT', 'Vintage']
[FALLBACK] Priority nums: ['MINPAYMENTAMT']
[FALLBACK] Ordered nums: ['MINPAYMENTAMT', 'Vintage']
```

### Step 3: Generate Report
- Navigate to Insights
- Click "Export PDF"
- Check the new report

### Step 4: Verify Report Content
- ✅ No more "EMPLOYEECD by CHANNELCD"
- ✅ Should show "MINPAYMENTAMT by CHANNELCD" or similar
- ✅ No more "SUB_CHANNEL_CODE Distribution"
- ✅ Should show payment-related insights

## Status: 🟢 READY TO TEST

Backend reloaded with the fix. Try one of these:

1. **Refresh the Insights page** (session `705ce5f8...`) - it should regenerate with new logic
2. **Upload file again** to create a completely fresh session
3. **Generate a new report** and check if it uses MINPAYMENTAMT

Share the backend logs after you navigate to Insights or generate the report!
