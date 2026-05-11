# 🚀 Quick Start - Test the Fixes

## ⚡ 3-Minute Test

### 1. Open Frontend
```
http://localhost:3000
```

### 2. Upload NEW File
- Click "New analysis"
- Upload Customer-Purchase-History.csv
- **IMPORTANT:** Must be a NEW upload, not previous session

### 3. Watch Backend Console
Look for this:
```
✅ V2 ENGINE ACTIVE — 2026-05-09 BUILD
=== COLUMN MAPPING ===
revenue_col: TotalPrice
category_col: ProductCategory
...
[RULE OK] domain_detection → 1 insights
[RULE OK] revenue_by_category → 1 insights
...
[INSIGHT ENGINE] FINAL: 8 insights
```

### 4. Check Insights Page
Should see:
- ✅ 6-8 insight cards
- ✅ Executive summary
- ✅ No errors

## ✅ Success = You See

1. **Version marker** in console
2. **Column mapping** with actual column names
3. **Multiple "[RULE OK]"** messages (6-8)
4. **8 insights** on insights page

## ❌ Problem = You See

1. **No version marker** → Old code still running
2. **"MISSING" in column mapping** → Column detection failed
3. **"[RULE FAIL]"** messages → Rule crashing
4. **Only 1-2 insights** → Thresholds too strict

## 🔧 Quick Fix

If old code still running:
```bash
# Stop backend (Ctrl+C)
cd engine
rm -rf __pycache__
python main.py
```

## 📚 Full Documentation

- `COMPLETE_FIX_SUMMARY.md` - Everything in one place
- `VERIFY_NEW_CODE_ACTIVE.md` - Detailed verification
- `INSIGHT_GENERATION_IMPROVEMENTS.md` - Technical details

---

**Backend:** http://localhost:8000 ✅
**Frontend:** http://localhost:3000 ✅
**Status:** Ready to test! 🎉
