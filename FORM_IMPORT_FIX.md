# Form Import Fix

## Issue
```
NameError: name 'Form' is not defined
```

## Root Cause
The `Form` class from FastAPI was not imported in `engine/routers/analyze.py`, but was used in the function signature:
```python
currency: str = Form("auto"),
```

## Fix Applied
Added `Form` to the FastAPI imports:

**Before**:
```python
from fastapi import APIRouter, UploadFile, Depends, HTTPException
```

**After**:
```python
from fastapi import APIRouter, UploadFile, Depends, HTTPException, Form
```

## Verification
✅ File compiles without errors  
✅ Committed: `a9aa1a2`  
✅ Pushed to remote

## Server Ready
The server should now start without errors. Run:
```bash
cd c:\Users\ALI\Downloads\insightstream_-ai-data-analyst\engine
python main.py
```

Expected output:
```
[FONT] OK Registered DejaVuSans (INR supported)
Session directory created: ...
Database initialized.
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```
