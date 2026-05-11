# Insights 500 Error - Fix Applied

## Problem
The insights page was showing a "500 Internal Server Error" when trying to load insights for certain datasets. This was caused by the insight engine taking too long to process large or complex datasets, causing the request to hang indefinitely.

## Root Cause
The `run_insight_engine()` function was taking more than 60 seconds to process certain datasets, causing the HTTP request to timeout or hang. The backend had no timeout protection, so it would either:
1. Block indefinitely
2. Return a 500 error without a helpful message

## Fixes Applied

### 1. Backend Timeout Protection (`engine/main.py`)
- Added a 60-second timeout to the `/insights/{session_id}` endpoint
- If insight generation takes longer than 60 seconds, the endpoint now returns a **504 Gateway Timeout** error with a helpful message
- Added comprehensive error logging to help debug future issues
- Improved error handling in the `load_session()` function

### 2. Frontend Error Handling (`web/app/insights/page.tsx`)
- Added detection for 504 timeout errors
- Display a user-friendly error message explaining the issue
- Suggest uploading a smaller file or using background processing

### 3. Session Loading Improvements (`engine/main.py`)
- Added validation to check if session files exist before attempting to load them
- Better error messages for missing or corrupted session data

## How to Apply the Fix

### Option 1: Restart the Backend Server
The backend server needs to be restarted to apply the changes:

```bash
# Stop the current server (Ctrl+C in the terminal where it's running)
# Then restart it:
cd engine
python main.py
```

### Option 2: Wait for Auto-Reload
If you're running the server with `--reload` flag, it should automatically detect the changes and reload. Check the terminal for:
```
WARNING:  WatchFiles detected changes in 'main.py'. Reloading...
INFO:     Started server process [XXXX]
INFO:     Application startup complete.
```

## Testing the Fix

1. **Restart the backend server** (see above)
2. **Refresh the browser** page showing the error
3. The page should now either:
   - Load successfully (if the insight generation completes within 60 seconds)
   - Show a user-friendly timeout message (if it takes longer than 60 seconds)

## For Users Experiencing Timeouts

If you see the timeout error message, you have several options:

1. **Upload a smaller dataset**: Try filtering your data to fewer rows before uploading
2. **Use background processing**: Go to the Dashboard and click the "Analyze" button, which processes insights in the background
3. **Wait and retry**: Sometimes the first request is slower; subsequent requests use cached results

## Technical Details

### Changes Made

**File: `engine/main.py`**
- Line ~1119: Added timeout protection using `concurrent.futures.ThreadPoolExecutor`
- Line ~1145: Added 504 error response for timeouts
- Line ~1180: Enhanced error logging with full traceback
- Line ~162: Improved `load_session()` with better validation

**File: `web/app/insights/page.tsx`**
- Line ~123: Added 504 status code detection
- Line ~302: Added timeout error message display

### Why 60 Seconds?
- Most datasets process in under 10 seconds
- 60 seconds provides a reasonable buffer for larger datasets
- Prevents indefinite hanging while still allowing complex analysis
- Users can use background processing for datasets that need more time

## Monitoring

To monitor insight generation performance:
```bash
# Watch the backend logs
tail -f engine/backend_run.log | grep -E "INSIGHT ENGINE|ERROR|Timeout"
```

## Future Improvements

Consider these enhancements:
1. Add progress indicators for long-running insight generation
2. Implement streaming responses to show partial results
3. Add dataset size warnings before analysis
4. Optimize the insight engine for better performance on large datasets
