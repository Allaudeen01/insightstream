# How to Verify New Version is Running

## ✅ Backend is Running
- **URL**: http://localhost:8000
- **Process ID**: 16184
- **Status**: Responding to health checks

## ✅ Frontend is Running
- **URL**: http://localhost:3000
- **Status**: Ready

---

## 🔍 How to Verify New Code is Loaded

### Method 1: Check Backend Logs During Upload

When you upload a file and it's being processed, watch Terminal ID 7 for these NEW markers:

**Look for these in the logs:**
```
[EntityDetection] 'RegionManager' is a PERSON column
[SubRole] Promoted 'TotalPrice' to revenue_col
[SANITY CHECKER] X issues found
```

**If you see these, the NEW version is running! ✅**

**If you DON'T see these, the OLD version is running ❌**

---

### Method 2: Check the Report Output

After generating a report, check for these indicators:

#### ✅ NEW Version Indicators:
1. **Return Rate Visible**: Executive summary shows "Return Rate: 24.8%"
2. **No Person Names**: Geographic insights use only region names (North/South/East/West/Central)
3. **Correct Revenue**: Total Revenue ≈ ₹43.80L (not ₹47.28L)
4. **Meaningful RPU**: Revenue-per-unit values in ₹200-300 range (not ₹31)
5. **Matching Count**: Executive summary says "4 findings" and 4 findings are shown
6. **Column Coverage**: API response includes `column_coverage` field
7. **Rich Temporal**: Temporal insights include trend direction (growing/declining/flat)

#### ❌ OLD Version Indicators:
1. **No Return Rate**: Executive summary doesn't show return rate
2. **Person Names**: "Cameron shows highest variability" appears
3. **Wrong Revenue**: Total Revenue = ₹47.28L
4. **Nonsensical RPU**: "East RPU = ₹31"
5. **Wrong Count**: Says "8 findings" but only 4 shown
6. **No Coverage**: No `column_coverage` in API response
7. **Weak Temporal**: Temporal insights don't show trend

---

### Method 3: Check File Modification Time

The insight_engine.py file should have been modified recently:

```powershell
Get-Item engine/insight_engine.py | Select-Object Name, Length, LastWriteTime
```

**Expected**:
- **Length**: 237,469 bytes
- **LastWriteTime**: May 7, 2026, 1:24 AM (or later)

---

### Method 4: Search for Fix Markers in Loaded Module

After uploading a file, check if the backend logs show any of these:
- `P0 FIX (Bug 0.1)` markers
- `TIER 1.1` markers
- `TIER 1.2` markers
- `TIER 5.6` markers

---

## 🧪 Quick Test Procedure

1. **Upload your product-sales-region dataset**
2. **Watch Terminal ID 7** (backend logs) during processing
3. **Look for NEW markers**:
   - `[EntityDetection]` messages
   - `[SubRole]` messages
   - `[SANITY CHECKER]` messages

4. **Check the generated report**:
   - Return rate visible? ✅
   - No "Cameron"? ✅
   - Revenue = ₹43.80L? ✅
   - RPU meaningful? ✅
   - Count matches? ✅

5. **If ALL checks pass** → NEW version is running! 🎉
6. **If ANY check fails** → OLD version still cached 😞

---

## 🔧 If Old Version Still Running

### Force Reload Steps:

1. **Stop all processes**:
   ```powershell
   # Kill all Python processes
   Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
   
   # Stop frontend
   # Press Ctrl+C in Terminal ID 5
   ```

2. **Clear ALL caches**:
   ```powershell
   # Clear Python cache
   Remove-Item -Path "engine/__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
   Remove-Item -Path "engine/*.pyc" -Force -ErrorAction SilentlyContinue
   
   # Clear pip cache (if needed)
   pip cache purge
   ```

3. **Verify file is correct**:
   ```powershell
   # Check file size
   (Get-Item engine/insight_engine.py).Length
   # Should be: 237469
   
   # Check for fixes
   Select-String -Path "engine/insight_engine.py" -Pattern "P0 FIX" | Measure-Object
   # Should find multiple matches
   ```

4. **Restart with fresh Python**:
   ```powershell
   # Start backend
   python -u engine/main.py
   
   # Start frontend (in separate terminal)
   npm run dev
   ```

5. **Test again** with the procedure above

---

## 📊 Expected vs Actual

### Expected Behavior (NEW Version):

**Executive Summary**:
```
The Sales system is operating at a scale of 1,500 records. No single 
numeric driver dominates the data — variance is distributed across 
multiple variables. Risk assessment identifies 4 high-impact findings 
requiring leadership review.

Return Rate: 24.8%
```

**Strategic Findings**:
1. Central Dominates in 1/5 Regions ✅
2. Volume–Value Decoupling: East (₹287) vs North ✅
3. Pricing Not Standardized — High Cost Variability ✅
4. Simulation: Suppressed (variance is structural) ✅

**NO "Cameron" anywhere** ✅

### Actual Behavior (OLD Version):

**Executive Summary**:
```
Risk assessment identifies 8 high-impact findings requiring leadership review.

(No return rate shown)
```

**Strategic Findings**:
1. Cameron shows highest variability ❌
2. East RPU = ₹31 ❌
3. Pricing opportunity: ₹57.4K ❌

---

## 🎯 Definitive Test

**Upload a file and check these 3 things:**

1. **Backend logs show**: `[EntityDetection]` or `[SANITY CHECKER]`
   - ✅ NEW version
   - ❌ OLD version

2. **Report shows**: Return Rate in executive summary
   - ✅ NEW version
   - ❌ OLD version

3. **Report shows**: "Cameron" in any insight
   - ✅ OLD version (BUG!)
   - ❌ NEW version (FIXED!)

---

**Current Status**: Backend restarted with cache cleared
**Next Step**: Upload a file and watch the logs!
