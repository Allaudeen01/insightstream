# 🚀 Quick Test Guide - Chart Fix

**Time**: 1 minute  
**Goal**: See real charts in PDF

---

## Step 1: Restart Backend (10 seconds)

**In the backend terminal**, press **Ctrl+C** to stop, then:

```powershell
python engine/main.py
```

**Wait for**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 2: Export PDF (30 seconds)

1. Open http://localhost:3000
2. Upload a file (or use existing session)
3. Go to Insights page
4. Click "Export PDF"
5. **Watch backend console** for chart messages

---

## Step 3: Check PDF (20 seconds)

Open the downloaded PDF and look at **pages 4-5**:

### ✅ SUCCESS:
- See actual bar charts
- See actual pie charts
- No placeholder text

### ❌ STILL BROKEN:
- See "Revenue by Product" text
- See "⚠ Chart skipped" messages
- No actual images

---

## 🔍 What to Look For in Console

### During PDF export, you should see:

```
[Chart 1/4] No base64, attempting Plotly conversion
[Plotly Convert] Successfully converted chart to /tmp/...
[Chart 2/4] No base64, attempting Plotly conversion
[Plotly Convert] Successfully converted chart to /tmp/...
```

**If you see this** → Charts are rendering ✅  
**If you don't see this** → Old code still running ❌

---

## 🐛 If Charts Still Don't Render

### Try:
1. **Clear Python cache**:
   ```powershell
   Remove-Item -Path "engine\__pycache__" -Recurse -Force
   python engine/main.py
   ```

2. **Check file was saved**:
   - Open `engine/report_generator.py`
   - Search for `_convert_plotly_to_png`
   - Should be there at line ~2065

3. **Install kaleido** (optional):
   ```powershell
   pip install kaleido
   ```

---

## ✅ Success = Real Charts!

**Before**:
```
Revenue by Product
⚠ Chart skipped
```

**After**:
```
Revenue by Product
[BAR CHART IMAGE]
📊 Total Revenue breakdown
```

---

**Ready?** Restart backend and test! 🎨
