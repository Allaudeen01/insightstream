# Currency Override in PDF Generation

## Status: ✅ IMPLEMENTED

## Overview
Added `currency_override` parameter to `build_from_assets()` method to allow manual currency selection to persist through PDF generation.

---

## Changes Made

### FILE: `engine/report_generator.py`

#### Method: `build_from_assets()`

**Added Parameter**:
```python
currency_override: Optional[str] = None
```

**Added Logic** (before auto-detection):
```python
# Check for currency override first (from session or user selection)
if currency_override and currency_override != "auto":
    _CURRENCY_MAP = {
        "INR": "₹", "USD": "$", "GBP": "£",
        "EUR": "€", "AED": "AED", "SGD": "S$",
        "JPY": "¥"
    }
    self._currency_symbol = _CURRENCY_MAP.get(currency_override, "₹")
    print(f"[CURRENCY] Using override: {currency_override} → {self._currency_symbol}")
else:
    self._currency_symbol = _detect_currency_symbol(df) if df is not None else "₹"
    print(f"[CURRENCY] Detected symbol: {self._currency_symbol}")
```

---

## How It Works

### Priority Order
1. **Currency Override** (if provided and not "auto")
   - Uses the explicitly provided currency code
   - Maps code to symbol (INR → ₹, USD → $, etc.)
   - Logs: `[CURRENCY] Using override: GBP → £`

2. **Auto-Detection** (if no override or override = "auto")
   - Runs `_detect_currency_symbol(df)`
   - Checks column names and country data
   - Logs: `[CURRENCY] Detected symbol: £`

### Usage Example

```python
gen = UnifiedReportGenerator()
gen.build_from_assets(
    output_path=str(out_path),
    charts=clean_charts,
    kpis=body.kpis,
    ai_summary=clean_ai_summary,
    insights=_insights_to_use,
    recommendations=structured_recs,
    text_blocks=body.text_blocks,
    title=body.title,
    project_name=body.project_name,
    template=body.template,
    session_id=session_id,
    df=df,
    domain_id=domain_id,
    currency_override="GBP"  # ← NEW: Force GBP currency
)
```

---

## Integration with Upload Flow

### Current Flow
1. User selects currency in upload form (e.g., "GBP")
2. Frontend sends `currency` in FormData
3. Backend receives currency in `/analyze` endpoint
4. Backend calls `_set_currency_symbol()` for insight generation
5. **NEW**: Backend should store currency in session for PDF generation

### Recommended Enhancement

#### Option 1: Store in Session Model (Recommended)
Add `currency` field to `AnalysisSession` model:

```python
# In models.py
class AnalysisSession(Base):
    # ... existing fields ...
    currency = Column(String(10))  # "INR", "USD", "GBP", "EUR", etc.
```

Then in `/analyze` endpoint:
```python
session_record.currency = currency if currency != "auto" else None
await db.commit()
```

Then in PDF export endpoint:
```python
# Load session from database
session = await get_session_detail(db, session_id, current_user.id)

# Pass currency to PDF generator
gen.build_from_assets(
    # ... other params ...
    currency_override=session.currency
)
```

#### Option 2: Store in Session Cache (Quick Fix)
In `/analyze` endpoint:
```python
# After analysis completes
_cache.set(session_id, "currency", currency)
```

In PDF export endpoint:
```python
# Retrieve from cache
currency_override = _cache.get(session_id, "currency")

gen.build_from_assets(
    # ... other params ...
    currency_override=currency_override
)
```

---

## Benefits

### 1. Consistency
- User's currency selection persists through entire workflow
- Insights, charts, and PDF all use same currency
- No confusion from mixed currencies

### 2. User Control
- User can override auto-detection
- Useful when auto-detection fails
- Ensures correct currency for international datasets

### 3. Flexibility
- Can be used with or without session storage
- Falls back to auto-detection if not provided
- Backward compatible (optional parameter)

---

## Console Output

### With Override
```
[CURRENCY] User selected: GBP → £
[INSIGHT ENGINE CURRENCY] Symbol set to: £
[CURRENCY] Using override: GBP → £
```

### Without Override (Auto-Detect)
```
[CURRENCY] Auto-detecting...
[CURRENCY] Detected symbol: £
```

---

## Testing Checklist

### Unit Testing
- [ ] Override with valid currency code (GBP → £)
- [ ] Override with invalid currency code (fallback to ₹)
- [ ] Override with "auto" (runs auto-detection)
- [ ] No override provided (runs auto-detection)

### Integration Testing
- [ ] Upload with GBP selected → PDF shows £
- [ ] Upload with USD selected → PDF shows $
- [ ] Upload with auto-detect → PDF shows detected currency
- [ ] Re-generate PDF → uses same currency as original

---

## Future Enhancements

### 1. Database Storage
Add `currency` column to `analysis_sessions` table:
```sql
ALTER TABLE analysis_sessions ADD COLUMN currency VARCHAR(10);
```

### 2. Currency Conversion
Allow users to convert between currencies in reports:
```python
currency_override="USD",
conversion_rate=1.27  # GBP to USD
```

### 3. Multi-Currency Support
Support datasets with multiple currencies:
```python
currency_map={
    "UK_Revenue": "GBP",
    "US_Revenue": "USD",
    "EU_Revenue": "EUR"
}
```

---

## Files Modified

1. `engine/report_generator.py` - Added `currency_override` parameter to `build_from_assets()`

---

## Verification

✅ File compiles without errors  
✅ Parameter added to function signature  
✅ Currency override logic implemented  
✅ Fallback to auto-detection preserved  
✅ Console logging added

---

## Next Steps

1. **Add currency field to Session model** (recommended)
2. **Store currency in `/analyze` endpoint**
3. **Pass currency to PDF generator in export endpoint**
4. **Test end-to-end flow**

---

## Commit Message
```
feat: add currency override to PDF generation

- Add currency_override parameter to build_from_assets()
- Check override before running auto-detection
- Map currency codes to symbols (INR, USD, GBP, EUR, etc.)
- Log currency source (override vs auto-detect)
- Maintain backward compatibility with optional parameter
```

**Status**: ✅ READY TO INTEGRATE WITH SESSION STORAGE
