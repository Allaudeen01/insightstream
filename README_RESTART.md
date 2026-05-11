# 🚀 Quick Restart Guide

## Problem
Your report shows only 2 insights because the backend is running old code.

## Solution (2 Minutes)

### Step 1: Run Restart Script
```powershell
cd c:\Users\ALI\Downloads\insightstream_-ai-data-analyst
.\restart_backend.ps1
```

### Step 2: Start Backend
```powershell
.\.venv\Scripts\Activate.ps1
python engine/main.py
```

### Step 3: Look for This
```
✅ V2 ENGINE ACTIVE — 2026-05-09 BUILD
```

### Step 4: Upload New File
Go to http://localhost:3000 and upload a file you haven't used before.

### Step 5: Verify
You should see **6-8 insights** instead of 2!

---

## What to Expect

**Before Restart** (Current):
- ❌ Only 2 insights
- ❌ No version marker
- ❌ Old code running

**After Restart** (Expected):
- ✅ 6-8 insights
- ✅ Version marker in console
- ✅ New code with all fixes

---

## Files Created for You

1. **`restart_backend.ps1`** - Automated restart script
2. **`DIAGNOSIS_COMPLETE.md`** - Full technical details
3. **`RESTART_BACKEND_NOW.md`** - Step-by-step instructions
4. **`README_RESTART.md`** - This quick guide

---

**Ready?** Run the restart script now! 🚀
