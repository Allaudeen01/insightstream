# 🚀 Quick Reference - See Your Report Now

**Backend**: ✅ Running on http://localhost:8000  
**Frontend**: http://localhost:3000  
**Status**: Ready to test

---

## 📊 To See the Report (30 Seconds)

### 1. Open Browser
```
http://localhost:3000
```

### 2. Upload File
- Click "Upload" or "New Analysis"
- Select any CSV or Excel file
- Click "Analyze"

### 3. View Insights
- Wait for processing (10-30 seconds)
- Insights page should auto-load
- See 6-8 insight cards

### 4. Export PDF
- Click "Export PDF" button
- Download opens automatically
- Open PDF to see full report

---

## 🔍 What to Look For

### Backend Console (While Uploading):
```
✅ V2 ENGINE ACTIVE — 2026-05-09 BUILD
=== COLUMN MAPPING ===
[RULE OK] domain_detection → 1 insights
[RULE OK] revenue_by_category → 1 insights
...
[INSIGHT ENGINE] FINAL: 8 insights
```

### Insights Page:
- ✅ 6-8 insight cards
- ✅ Executive summary
- ✅ No errors

### PDF:
- ✅ 7-10 pages
- ✅ Multiple insights
- ✅ Charts included

---

## 🐛 Quick Fixes

### "500 Error" or "Failed to load"
```powershell
# Restart backend
cd c:\Users\ALI\Downloads\insightstream_-ai-data-analyst
.\.venv\Scripts\Activate.ps1
python engine/main.py
```

### "No version marker in console"
```powershell
# Clear cache and restart
cd engine
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
python main.py
```

### "Only 1-2 insights"
- Upload a DIFFERENT file (avoid cached results)
- Check backend console for [RULE FAIL] messages
- Share console output for debugging

---

## ✅ Success Checklist

- [ ] Backend shows version marker
- [ ] Column mapping appears
- [ ] Multiple [RULE OK] messages
- [ ] 6-8 insights on page
- [ ] PDF downloads successfully
- [ ] No error messages

---

## 📞 Need Help?

**Share These:**
1. Backend console output (copy/paste)
2. Error messages (screenshot)
3. Number of insights you see
4. What happens when you upload

---

## 📚 Full Documentation

- `HOW_TO_SEE_THE_REPORT.md` - Detailed step-by-step guide
- `SUMMARY_ALL_FIXES_DEPLOYED.md` - Complete fix summary
- `CURRENT_STATUS_AND_NEXT_STEPS.md` - Troubleshooting guide

---

**Ready!** Open http://localhost:3000 and upload a file! 🚀
