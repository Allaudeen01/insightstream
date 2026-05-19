# Currency Selector Feature - Complete

## Status: ✅ IMPLEMENTED

## Overview
Added a currency selector to the upload flow, allowing users to manually select their currency or use auto-detection.

---

## Changes Made

### FILE 1: Frontend - `web/app/upload/page.tsx`

#### 1. Added Currency State
```typescript
const [currency, setCurrency] = useState("auto");
```

#### 2. Added Currency Options Constant
```typescript
const CURRENCIES = [
    { code: "auto", symbol: "AUTO", label: "Auto-detect" },
    { code: "INR",  symbol: "₹",   label: "₹ INR — Indian Rupee" },
    { code: "USD",  symbol: "$",   label: "$ USD — US Dollar" },
    { code: "GBP",  symbol: "£",   label: "£ GBP — British Pound" },
    { code: "EUR",  symbol: "€",   label: "€ EUR — Euro" },
    { code: "AED",  symbol: "AED", label: "AED — UAE Dirham" },
    { code: "SGD",  symbol: "S$",  label: "S$ SGD — Singapore Dollar" },
    { code: "JPY",  symbol: "¥",   label: "¥ JPY — Japanese Yen" },
];
```

#### 3. Added Currency Selector UI
Placed before the dropzone:
```tsx
<div className="mt-6">
    <label className="block text-sm font-medium text-zinc-700 mb-2">
        Currency
    </label>
    <select
        value={currency}
        onChange={(e) => setCurrency(e.target.value)}
        className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2.5 text-sm text-zinc-900 focus:outline-none focus:ring-2 focus:ring-[#6d5ef5] focus:border-transparent"
    >
        {CURRENCIES.map(c => (
            <option key={c.code} value={c.code}>
                {c.label}
            </option>
        ))}
    </select>
    <p className="text-xs text-zinc-500 mt-1.5">
        Select the currency your data is in. Auto-detect works for most datasets.
    </p>
</div>
```

#### 4. Pass Currency to API
```typescript
const formData = new FormData();
formData.append("file", file);
formData.append("currency", currency);  // ← NEW
if (sheetName) formData.append("sheet_name", sheetName);
```

---

### FILE 2: Backend - `engine/routers/analyze.py`

#### 1. Added Currency Parameter
```python
@router.post("/analyze")
async def analyze(
    file: UploadFile,
    currency: str = Form("auto"),  # ← NEW
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
```

#### 2. Override Currency if User Specified
```python
# Override currency if user specified
if currency != "auto":
    _CURRENCY_MAP = {
        "INR": "₹", "USD": "$", "GBP": "£",
        "EUR": "€", "AED": "AED", "SGD": "S$",
        "JPY": "¥",
    }
    sym = _CURRENCY_MAP.get(currency, "₹")
    # Import and set in both engines
    from insight_engine import _set_currency_symbol
    _set_currency_symbol(sym)
    print(f"[CURRENCY] User selected: {currency} → {sym}")
else:
    print(f"[CURRENCY] Auto-detecting...")
```

---

## User Experience

### Upload Flow
1. User navigates to `/upload`
2. **NEW**: User sees currency selector dropdown (defaults to "Auto-detect")
3. User can select specific currency (₹, $, £, €, AED, S$, ¥)
4. User uploads file
5. Backend receives currency preference
6. If currency != "auto", backend overrides auto-detection
7. All monetary values in report use selected currency

### Currency Options
| Code | Symbol | Label |
|------|--------|-------|
| auto | AUTO | Auto-detect |
| INR | ₹ | ₹ INR — Indian Rupee |
| USD | $ | $ USD — US Dollar |
| GBP | £ | £ GBP — British Pound |
| EUR | € | € EUR — Euro |
| AED | AED | AED — UAE Dirham |
| SGD | S$ | S$ SGD — Singapore Dollar |
| JPY | ¥ | ¥ JPY — Japanese Yen |

---

## Technical Details

### Frontend Implementation
- **State Management**: Uses React `useState` hook
- **UI Component**: Native HTML `<select>` with Tailwind styling
- **Form Data**: Currency sent as form field alongside file
- **Default Value**: "auto" (preserves existing auto-detection behavior)

### Backend Implementation
- **Parameter**: `currency: str = Form("auto")`
- **Currency Map**: Maps currency codes to symbols
- **Engine Integration**: Calls `_set_currency_symbol()` from `insight_engine`
- **Logging**: Prints selected currency to console for debugging
- **Fallback**: If currency code not recognized, defaults to ₹

### Auto-Detection Behavior
- When `currency = "auto"` (default):
  - Backend runs existing auto-detection logic
  - Checks column names for currency keywords
  - Checks country column for UK/US records
  - Falls back to ₹ if no signals found

- When user selects specific currency:
  - Auto-detection is bypassed
  - Selected currency is used for all monetary values
  - Ensures consistency across entire report

---

## Benefits

### 1. User Control
- Users can override auto-detection if it's wrong
- Useful for datasets without clear currency signals
- Prevents currency mismatches

### 2. International Support
- Supports 7 major currencies
- Easy to add more currencies in the future
- Covers most common business scenarios

### 3. Backward Compatible
- Default "auto" preserves existing behavior
- No breaking changes for existing users
- Existing datasets continue to work

### 4. Better UX
- Clear, visible currency selector
- Helpful hint text
- Professional dropdown UI

---

## Testing Checklist

### Frontend Testing
- [ ] Currency selector appears on upload page
- [ ] Dropdown shows all 8 currency options
- [ ] Default value is "Auto-detect"
- [ ] Selecting currency updates state
- [ ] Currency value is sent with file upload

### Backend Testing
- [ ] Analyze endpoint accepts currency parameter
- [ ] Auto-detect works when currency="auto"
- [ ] Manual selection overrides auto-detection
- [ ] Console shows: `[CURRENCY] User selected: GBP → £`
- [ ] Report uses selected currency symbol

### Integration Testing
- [ ] Upload UK dataset with "Auto-detect" → £
- [ ] Upload UK dataset with "USD" selected → $
- [ ] Upload US dataset with "GBP" selected → £
- [ ] Upload generic dataset with "INR" selected → ₹

---

## Example Console Output

### Auto-Detect Mode
```
[CURRENCY] Auto-detecting...
[CURRENCY] Detected symbol: £
[INSIGHT ENGINE CURRENCY] Symbol set to: £
```

### Manual Selection Mode
```
[CURRENCY] User selected: GBP → £
[INSIGHT ENGINE CURRENCY] Symbol set to: £
```

---

## Files Modified

1. `web/app/upload/page.tsx` - Added currency selector UI and state
2. `engine/routers/analyze.py` - Added currency parameter and override logic

---

## Future Enhancements

### Potential Improvements
1. **Remember Last Selection**: Store user's currency preference in localStorage
2. **Smart Defaults**: Pre-select currency based on user's location (IP geolocation)
3. **More Currencies**: Add CNY (¥), CAD ($), AUD ($), etc.
4. **Currency Conversion**: Allow users to convert between currencies
5. **Per-Column Currency**: Support datasets with multiple currencies

### Database Integration
Consider storing currency preference in session record:
```python
session_record.currency = currency if currency != "auto" else None
```

This would allow:
- Remembering currency for re-analysis
- Showing currency in session list
- Filtering sessions by currency

---

## Verification

✅ Frontend compiles without errors  
✅ Backend compiles without errors  
✅ Currency selector UI implemented  
✅ Currency parameter added to API  
✅ Currency override logic implemented  
✅ Console logging added for debugging

---

## Next Steps

1. **Restart Backend**: Kill all Python processes and restart
2. **Restart Frontend**: Restart Next.js dev server
3. **Test Upload Flow**: Upload file with different currency selections
4. **Verify Console**: Check for currency selection logs
5. **Verify Report**: Ensure selected currency appears in PDF

---

## Commit Message
```
feat: add currency selector to upload flow

- Add currency dropdown to upload page (auto, INR, USD, GBP, EUR, AED, SGD, JPY)
- Pass currency selection to backend via form data
- Override auto-detection when user selects specific currency
- Maintain backward compatibility with auto-detect default
```

**Status**: ✅ READY TO TEST
