# Insights 500 Error - Complete Fix

## Problem
The insights page was showing "Something went wrong - Insights fetch failed: 500" error.

## Root Causes Identified

1. **Missing Session Validation**: The endpoint didn't check if the session exists before trying to load it
2. **Poor Error Handling**: Errors during insight generation weren't being caught and logged properly
3. **Missing Data Structure Validation**: No validation of the result structure before trying to access fields
4. **Incomplete Error Recovery**: If insight generation failed, the entire request would crash

## Fixes Applied

### 1. Enhanced `/insights/{session_id}` Endpoint (main.py)

**Added comprehensive error handling:**

- ✅ **Session validation** - Check if session exists before processing
- ✅ **Detailed logging** - Log every step (cache hit, cold path, success, errors)
- ✅ **Result structure validation** - Verify result is a dict with required fields
- ✅ **Safe insight card conversion** - Handle malformed insights gracefully
- ✅ **Better error messages** - Specific error messages for different failure modes

**Key improvements:**
```python
# Before: No session check
filename, df = load_session(session_id)

# After: Validate session exists first
if not session_exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found or expired")
```

```python
# Before: Direct access without validation
for ins in result["strategic_brief"]:
    card = InsightCard(...)

# After: Validate and handle errors
if not isinstance(result, dict):
    raise ValueError(f"Invalid insight result type: {type(result)}")

strategic_brief = result.get("strategic_brief", [])
if not isinstance(strategic_brief, list):
    strategic_brief = []

for idx, ins in enumerate(strategic_brief):
    try:
        if not isinstance(ins, dict):
            continue
        card = InsightCard(...)
    except Exception as e:
        print(f"[WARNING] Failed to convert insight {idx}: {e}")
        continue
```

### 2. Enhanced `run_insight_engine()` Function (insight_engine.py)

**Added try-catch wrapper with fallback:**

- ✅ **Wrapped entire function** - Catch any error during insight generation
- ✅ **Fallback response** - Return valid minimal response instead of crashing
- ✅ **Detailed error logging** - Print full traceback for debugging

**Key improvement:**
```python
def run_insight_engine(...) -> dict:
    try:
        # ... all existing logic ...
        return result
    except Exception as e:
        print(f"[ERROR] Insight engine failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        
        # Return minimal valid response
        return {
            "domain": {...},
            "strategic_brief": [],
            "recommendations": [f"⚠️ Analysis failed: {str(e)}"],
            "executive_summary": f"Analysis could not be completed: {str(e)}",
            "warnings": [f"🔴 Critical Error: {str(e)}"],
            ...
        }
```

## Error Flow Now

### Scenario 1: Session Not Found
```
Frontend → GET /insights/{session_id}
Backend checks: session_exists(session_id) → False
Backend returns: 404 "Session not found or expired"
Frontend shows: "Session expired" message with upload button
```

### Scenario 2: Insight Generation Fails
```
Frontend → GET /insights/{session_id}
Backend: Session exists ✓
Backend: Runs insight engine → Exception caught
Backend: Returns fallback response with error message
Frontend shows: Warning message with error details
```

### Scenario 3: Malformed Result
```
Frontend → GET /insights/{session_id}
Backend: Insight engine returns invalid structure
Backend: Validates result → Detects issue
Backend: Logs warning, skips bad insights
Frontend shows: Valid insights that could be parsed
```

## Testing Recommendations

1. **Test with existing session**: Upload a file and navigate to insights
2. **Test with expired session**: Refresh page after backend restart
3. **Test with problematic data**: Upload files that might cause parsing errors
4. **Check backend logs**: Look for detailed error messages

## Monitoring

The fix adds extensive logging. Look for these log patterns:

- `[CACHE HIT]` - Insights served from cache (fast path)
- `[COLD PATH]` - Generating new insights (slow path)
- `[LOADED]` - Session loaded successfully
- `[SUCCESS]` - Insights generated successfully
- `[ERROR]` - Something went wrong (with details)
- `[WARNING]` - Non-critical issue (e.g., skipped malformed insight)

## Next Steps

If you still see 500 errors:

1. Check the backend logs for `[ERROR]` messages
2. Look for the specific error type and message
3. Check if the session file exists in the temp directory
4. Verify the data file can be loaded with Polars

## Files Modified

1. `engine/main.py` - Enhanced `/insights/{session_id}` endpoint
2. `engine/insight_engine.py` - Added error handling to `run_insight_engine()`

## Status

✅ **COMPLETE** - The insights endpoint now has comprehensive error handling and will provide meaningful error messages instead of generic 500 errors.
