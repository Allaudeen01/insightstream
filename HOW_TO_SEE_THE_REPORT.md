# 📊 How to See the Report - Step by Step

**Backend Status**: ✅ Running on http://localhost:8000  
**Frontend Status**: ✅ Should be on http://localhost:3000  
**V2 Engine**: ✅ Deployed and ready

---

## 🚀 Quick Steps (2 Minutes)

### Step 1: Open Your Browser
```
Navigate to: http://localhost:3000
```

### Step 2: Upload a File

**Option A - New Analysis:**
1. Click "New Analysis" or "Upload" button
2. Select a CSV or Excel file from your computer
3. Click "Upload" or "Analyze"

**Option B - Use Existing Session:**
1. If you see a list of previous uploads, click on one
2. Navigate to the "Insights" tab
3. Click "Export PDF" to download the report

### Step 3: View Insights

After upload completes:
1. Click the "Insights" tab (or it may auto-navigate)
2. You should see multiple insight cards
3. Scroll through to see all insights

### Step 4: Download PDF Report

1. Look for "Export PDF" or "Generate Report" button
2. Click it
3. Wait for PDF generation (5-10 seconds)
4. PDF will download automatically
5. Open the PDF to see the full report

---

## 🔍 What You Should See

### On the Insights Page:
- ✅ **6-8 insight cards** with titles and descriptions
- ✅ **Executive summary** at the top with key metrics
- ✅ **Charts and visualizations** (may be placeholders)
- ✅ **No error messages** or "500" errors

### In the PDF Report:
- ✅ **7-10 pages** of content
- ✅ **Multiple insights** with detailed analysis
- ✅ **Charts and graphs** showing trends
- ✅ **Recommendations** section
- ✅ **Professional formatting** with headers and sections

---

## 🐛 If You See Errors

### "Insights fetch failed: 500"
This means the backend crashed during insight generation.

**What to do:**
1. Check the backend console (where you ran `python engine/main.py`)
2. Look for error messages or stack traces
3. Copy the error and share it

### "No insights found" or Only 1-2 Insights
This means the rules aren't firing properly.

**What to do:**
1. Check backend console for version marker: "✅ V2 ENGINE ACTIVE"
2. If you don't see it, the old code is still running
3. Follow the restart instructions below

### Blank Page or Loading Forever
This means the frontend can't connect to the backend.

**What to do:**
1. Verify backend is running: Open http://localhost:8000/health
2. Should see: `{"status":"ok"}`
3. If not, restart the backend

---

## 🔄 How to Restart Backend (If Needed)

### Step 1: Stop Current Backend
In the terminal where backend is running:
- Press `Ctrl+C` to stop

Or kill the process:
```powershell
Stop-Process -Id 15296
```

### Step 2: Clear Python Cache
```powershell
cd c:\Users\ALI\Downloads\insightstream_-ai-data-analyst\engine
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
```

### Step 3: Start Backend
```powershell
cd c:\Users\ALI\Downloads\insightstream_-ai-data-analyst
.\.venv\Scripts\Activate.ps1
python engine/main.py
```

### Step 4: Verify It Started
Look for:
```
✅ V2 ENGINE ACTIVE — 2026-05-09 BUILD
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 📸 What to Look For in Backend Console

When you upload a file and it's being analyzed, you should see:

```
======================================================================
✅ V2 ENGINE ACTIVE — 2026-05-09 BUILD
✅ Enhanced error handling, lowered thresholds, safety nets active
======================================================================

=== COLUMN MAPPING ===
revenue_col: TotalPrice
price_col: UnitPrice
qty_col: Quantity
category_col: ProductCategory
geographic_col: Region
date_col: OrderDate
return_col: ReturnStatus

[RULE OK] domain_detection → 1 insights
[RULE OK] revenue_by_category → 1 insights
[RULE OK] return_rate_by_category → 1 insights
[RULE OK] strong_correlation → 1 insights
[RULE OK] revenue_by_segment → 2 insights
[RULE OK] top_performers → 1 insights
[RULE OK] time_series_analyzer → 1 insights

[INSIGHT ENGINE] FINAL: 8 insights
```

**Key Things:**
- ✅ Version marker appears (confirms new code is running)
- ✅ Column mapping shows actual column names (not "MISSING")
- ✅ Multiple "[RULE OK]" messages (6-8 rules firing)
- ✅ "FINAL: 6-8 insights" (not just 1-2)

---

## 🎯 Testing Checklist

Use this to verify everything is working:

### Backend Health:
- [ ] Backend running on port 8000
- [ ] http://localhost:8000/health returns `{"status":"ok"}`
- [ ] No error messages in backend console

### Frontend Access:
- [ ] http://localhost:3000 loads successfully
- [ ] Upload page is accessible
- [ ] Can select and upload files

### Insight Generation:
- [ ] Version marker appears in backend console
- [ ] Column mapping shows actual columns
- [ ] Multiple [RULE OK] messages appear
- [ ] "FINAL: 6-8 insights" message appears

### Insights Page:
- [ ] 6-8 insight cards displayed
- [ ] Executive summary shows metrics
- [ ] No 500 errors
- [ ] Charts render (even if placeholders)

### PDF Export:
- [ ] "Export PDF" button works
- [ ] PDF downloads successfully
- [ ] PDF has 7-10 pages
- [ ] PDF contains actual insights (not errors)
- [ ] Charts appear in PDF

---

## 💡 Tips

### Use a Fresh File
If you've uploaded a file before, the results might be cached. Try uploading a different file to see fresh results.

### Watch the Backend Console
Keep the backend console visible while uploading. This helps you see what's happening in real-time.

### Check Browser Console
If the frontend has issues, open browser DevTools (F12) and check the Console tab for errors.

### Test with Sample Data
If you don't have a file handy, create a simple CSV:
```csv
Date,Product,Sales,Quantity,Region
2024-01-01,Widget A,1000,10,North
2024-01-02,Widget B,1500,15,South
2024-01-03,Widget A,1200,12,East
2024-01-04,Widget C,800,8,West
2024-01-05,Widget B,1800,18,North
```

---

## 🆘 Still Having Issues?

If you're still stuck, provide these details:

1. **Backend Console Output**: Copy the last 50 lines
2. **Frontend Error**: Screenshot or copy error message
3. **Browser Console**: Any errors in DevTools Console tab
4. **What You See**: Describe what happens when you upload a file
5. **File Info**: What type of file (CSV/Excel), how many rows/columns

---

## ✅ Success Criteria

You'll know everything is working when:

1. ✅ Backend shows version marker during upload
2. ✅ Insights page shows 6-8 insight cards
3. ✅ PDF downloads with 7-10 pages of content
4. ✅ No error messages anywhere
5. ✅ Charts and visualizations appear

---

**Ready to test!** 🚀

**Quick Start**: Open http://localhost:3000 → Upload file → Check Insights → Export PDF

---

## 📞 Need Help?

If something doesn't work:
1. Copy the error message
2. Copy backend console output
3. Describe what you see
4. Share it for debugging

The V2 engine is deployed and ready. Just need to verify it's working by uploading a file!
