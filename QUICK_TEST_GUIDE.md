# Quick Test Guide - Insurance Dataset

## 🚀 Start Here

### 1. Restart Backend
```bash
cd engine
python -m uvicorn main:app --port 8000 --reload
```

### 2. Clear Browser
- Press **Ctrl+Shift+Delete**
- Clear cache and local storage
- Or use **Incognito mode**

### 3. Upload File
Go to: http://localhost:3000/upload

---

## 📊 What to Look For

### Backend Terminal (Should Show):
```
=== UPLOAD DEBUG ===
Filename: insurance_data.xlsx
Bytes read: XXXXXX
Shape: (227000, 55)  ← Should be 227000, not 0!

ColumnMap → numeric='MINPAYMENTAMT'  ← Should NOT be None!
            category='AGENTSTATUSCD'  ← Should be a status column
            region='STATECD'          ← Should be a state column

=== UPLOAD SUCCESS: (227000, 55) ===
```

### Frontend (Should Show):
- **ROWS: 227000** (not 0!)
- **COLUMNS: 55**
- **Quality Score: A or B**
- **"Continue to EDA" button enabled**

---

## ❌ If Still Shows "0 ROWS"

### Check Backend Logs For:

**Option A: Critical Quality Issues**
```
[quality] CRITICAL ISSUES FOUND - returning early without session
[quality] Issues: [...]
```
→ Data has critical quality problems

**Option B: All Rows Removed**
```
[quality] ERROR: All rows removed during cleaning!
```
→ Cleaning logic too aggressive

**Option C: Parse Failed**
```
=== UPLOAD FAILED: ValueError: ...
```
→ File format issue

---

## 🔍 Quick Diagnostics

### Test 1: Small Sample
```bash
# Create 100-row test file
head -101 insurance_data.csv > test_small.csv
```
Upload `test_small.csv` - if this works, it's a size/timeout issue.

### Test 2: Check File
```bash
# Check file size
ls -lh insurance_data.xlsx

# Check first few rows
head -5 insurance_data.csv
```

### Test 3: Browser Console
- Press **F12**
- Go to **Network** tab
- Upload file
- Click `/upload` request
- Check **Response** tab

---

## ✅ Expected Results

### After Upload:
- ✅ Shows 227K rows
- ✅ Shows 55 columns
- ✅ Quality score A or B
- ✅ Can continue to EDA

### After Report Generation:
- ✅ 8-10 page PDF
- ✅ Charts show meaningful data
- ✅ No ID numbers in charts
- ✅ State-wise breakdowns present
- ✅ Agent status distribution shown

---

## 📞 If Issues Persist

Share these from backend logs:
1. The `=== UPLOAD DEBUG ===` section
2. The `ColumnMap →` line
3. Any `[quality]` error messages
4. The `=== UPLOAD FAILED` section (if present)

---

## 🎯 Key Fixes Applied

1. ✅ Added "amt" to revenue keywords → detects MINPAYMENTAMT
2. ✅ Added "status", "cd" to category keywords → detects status columns
3. ✅ Blacklisted 14 ID column patterns → excludes ID numbers
4. ✅ Added insurance_agents domain → optimized thresholds
5. ✅ Disabled sampling → analyzes all 227K rows
6. ✅ Enhanced logging → shows exact failure point

---

## Status: 🟢 READY TO TEST
