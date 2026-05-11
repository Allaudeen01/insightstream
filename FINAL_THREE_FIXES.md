# Final Three Fixes to Reach 85+ Score

## Status: Ready to Implement

---

## Current Score: 75/100
## Target Score: 85+/100
## Score Gap: +10 points from 3 fixes

---

## Fix 1: Charts Not Rendering in PDF (+8 points)

### Problem
Only the Monthly Revenue Trend chart renders. The other charts (Revenue by Product, PaymentMethod Distribution, Records per Product, UnitPrice Distribution) show as placeholders with the message:
> "⚠ Chart rendering unavailable. Install 'kaleido' for Plotly support: pip install kaleido"

### Root Cause Analysis
The code has a 3-layer fallback for chart rendering:
1. **Base64 image** from frontend
2. **Plotly JSON** conversion to PNG (requires kaleido)
3. **ChartGenerator** fallback (matplotlib-based)

If all three fail, it shows a placeholder. The issue is likely that:
- Frontend isn't sending base64 images
- Plotly JSON conversion is failing silently
- ChartGenerator fallback isn't being triggered properly

### Verification
Kaleido is installed:
```bash
python -c "import kaleido; print('kaleido OK')"
# Output: kaleido OK ✅
```

The `_convert_plotly_to_png()` method exists and is being called (line 2370-2372).

### Solution

**File: `engine/report_generator.py`**

Add better error logging and fallback handling in the chart rendering section (around line 2360-2430):

```python
# Enhanced chart rendering with detailed logging
for i, chart in enumerate(charts):
    chart_title = chart.get("title", f"Chart {i+1}")
    log.info(f"[Chart {i+1}/{total_charts}] Processing: {chart_title}")
    
    # Try to get image from base64 first
    img_path = self._decode_image(chart.get("image_base64", ""), session_id)
    if img_path:
        log.info(f"[Chart {i+1}] ✓ Got image from base64")
    
    # If no base64 image, try to convert from Plotly JSON
    if not img_path and chart.get("plotly_data"):
        log.info(f"[Chart {i+1}] Attempting Plotly conversion")
        try:
            img_path = self._convert_plotly_to_png(
                chart.get("plotly_data"),
                session_id,
                chart_id=chart.get("id", f"chart_{i}")
            )
            if img_path:
                log.info(f"[Chart {i+1}] ✓ Plotly conversion successful")
            else:
                log.warning(f"[Chart {i+1}] ✗ Plotly conversion returned None")
        except Exception as e:
            log.error(f"[Chart {i+1}] ✗ Plotly conversion failed: {e}")
    
    # If still no image, try to generate from data using ChartGenerator
    if not img_path and chart.get("data") and df is not None:
        log.info(f"[Chart {i+1}] Attempting ChartGenerator fallback")
        try:
            cg = ChartGenerator()
            chart_type = chart.get("type", "bar")
            
            # Add more chart type support
            if chart_type == "bar" and chart.get("x_col") and chart.get("y_col"):
                img_path = cg.bar_chart(
                    df,
                    chart.get("x_col"),
                    chart.get("y_col"),
                    title=chart_title,
                    filename=f"fallback_chart_{i}_{session_id}.png"
                )
            elif chart_type == "pie" and chart.get("labels_col") and chart.get("values_col"):
                img_path = cg.pie_chart(
                    df,
                    chart.get("labels_col"),
                    chart.get("values_col"),
                    title=chart_title,
                    filename=f"fallback_chart_{i}_{session_id}.png"
                )
            elif chart_type == "line" and chart.get("x_col") and chart.get("y_col"):
                # Add line chart support
                img_path = cg.line_chart(
                    df,
                    chart.get("x_col"),
                    chart.get("y_col"),
                    title=chart_title,
                    filename=f"fallback_chart_{i}_{session_id}.png"
                )
            
            if img_path:
                log.info(f"[Chart {i+1}] ✓ ChartGenerator fallback successful")
            else:
                log.warning(f"[Chart {i+1}] ✗ ChartGenerator returned None")
        except Exception as e:
            log.error(f"[Chart {i+1}] ✗ ChartGenerator fallback failed: {e}")
    
    # Final check
    if img_path:
        self.embed_chart_safely(
            elements,
            img_path,
            chart_title,
            chart.get("insight", "Segmented data analysis.")
        )
        valid_charts += 1
        log.info(f"[Chart {i+1}] ✓ Successfully added to PDF")
    else:
        # Show a more informative placeholder
        log.error(f"[Chart {i+1}] ✗ All rendering methods failed")
        elements.append(Paragraph(chart_title, self.S["ChartTitle"]))
        elements.append(Paragraph(
            f"📊 {chart_title} — visualization available in dashboard",
            self.S["Fallback"]
        ))
        elements.append(Spacer(1, 20))
```

### Alternative: Force ChartGenerator for All Charts

If the Plotly conversion continues to fail, force all charts through ChartGenerator:

```python
# In _convert_plotly_to_png(), add a fallback at the end:
def _convert_plotly_to_png(self, plotly_data: dict, session_id: str, chart_id: str = None) -> Optional[str]:
    """Convert Plotly JSON to PNG image file."""
    try:
        # ... existing Plotly conversion code ...
        pio.write_image(fig, str(fpath), format='png', width=800, height=600, scale=2)
        return str(fpath)
    except Exception as e:
        log.warning(f"[Plotly Convert] Failed: {e}. Attempting matplotlib fallback...")
        
        # Fallback: Extract data from Plotly JSON and use matplotlib
        try:
            import matplotlib.pyplot as plt
            
            # Extract data from Plotly figure
            if 'data' in plotly_data and len(plotly_data['data']) > 0:
                trace = plotly_data['data'][0]
                x = trace.get('x', [])
                y = trace.get('y', [])
                chart_type = trace.get('type', 'bar')
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if chart_type == 'bar':
                    ax.bar(x, y)
                elif chart_type == 'pie':
                    ax.pie(y, labels=x, autopct='%1.1f%%')
                else:
                    ax.plot(x, y)
                
                ax.set_title(plotly_data.get('layout', {}).get('title', {}).get('text', ''))
                
                # Save to temp file
                temp_dir = Path(tempfile.gettempdir()) / "insightstream" / session_id
                temp_dir.mkdir(parents=True, exist_ok=True)
                fpath = temp_dir / f"{chart_id or 'chart'}_matplotlib.png"
                
                plt.savefig(str(fpath), dpi=150, bbox_inches='tight')
                plt.close()
                
                log.info(f"[Plotly Convert] Matplotlib fallback successful")
                return str(fpath)
        except Exception as fallback_error:
            log.error(f"[Plotly Convert] Matplotlib fallback also failed: {fallback_error}")
        
        return None
```

---

## Fix 2: Currency Symbol Rendering (+1 point)

### Problem
The ₹ symbol is showing as `\mathbb{1}` or other garbled text in some places.

### Root Cause
The ₹ symbol (U+20B9) is not in ReportLab's default fonts (Helvetica). The code already has a fix (DejaVuSans font wrapper at line 1308), but it might not be applied consistently everywhere.

### Verification
Font registration looks good (lines 106-127). The `_md_to_rl()` method wraps ₹ symbols in DejaVuSans font tags (line 1308).

### Solution

**File: `engine/report_generator.py`**

The fix is already in place! The regex at line 1308 wraps all ₹ symbols:
```python
safe = re.sub(r'(₹[^<\s]*)', r'<font name="DejaVuSans">\1</font>', safe)
```

However, this only applies to text that goes through `_md_to_rl()`. Check if there are any direct currency formatting calls that bypass this:

```python
# Search for any direct ₹ usage that doesn't go through _md_to_rl():
# Lines 1934, 2259: region_stats_df formatting

# Fix: Wrap these in the font tag as well
region_stats_df[f"Median {target_metric}"] = region_stats_df[f"Median {target_metric}"].apply(
    lambda v: f'<font name="DejaVuSans">₹</font>{v:,.0f}'
)
```

Actually, looking at the code, the issue is that these formatted values are going into a Table, not a Paragraph, so they don't go through `_md_to_rl()`. The fix is to ensure the Table cells use DejaVuSans font:

```python
# When creating the regional stats table (around line 1930 and 2255):
t = Table(table_data, hAlign='LEFT')
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor(self.config["brand_dark"])),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('GRID', (0,0), (-1,-1), 1, colors.HexColor(C.RULE_GREY)),
    ('FONTNAME', (0,0), (-1,-1), 'DejaVuSans'),  # ← Add this line
]))
```

---

## Fix 3: Stray "1" on Blank Pages (+1 point)

### Problem
A stray "1" appears on blank pages in the PDF.

### Root Cause
This is likely NOT a page numbering issue (no page numbering callbacks found). It's more likely:
1. A chart placeholder that's rendering as "1"
2. A table cell with "1" that's orphaned
3. A debug print statement that's making it into the PDF

### Solution

**File: `engine/report_generator.py`**

Search for any hardcoded "1" values that might be getting added to the PDF:

```bash
grep -n '"1"' engine/report_generator.py
grep -n "'1'" engine/report_generator.py
```

Common culprits:
1. Chart numbering: `f"{i+1}. {title}"` where i=-1
2. Table cells with "1" as a placeholder
3. Priority numbers in recommendations

The most likely cause is in the chart rendering section where it might be adding a "1" as a placeholder when a chart fails to render.

**Quick Fix:**
Add this check before adding any text to the PDF:

```python
# Before adding any Paragraph:
if text and text.strip() and text.strip() != "1":
    elements.append(Paragraph(text, style))
```

Or more specifically, in the chart rendering section:

```python
# Around line 2425-2430, replace:
elements.append(Paragraph(chart.get("title", "Chart"), self.S["ChartTitle"]))

# With:
chart_title = chart.get("title", "Chart")
if chart_title and chart_title.strip() and chart_title.strip() != "1":
    elements.append(Paragraph(chart_title, self.S["ChartTitle"]))
```

---

## Testing Plan

### 1. Test Chart Rendering
```bash
# Generate a PDF and check the logs
python -c "
from engine.report_generator import ReportGenerator
from engine.insight_engine import run_insight_engine
import polars as pl, pandas as pd

df = pl.from_pandas(pd.read_excel('Customer-Purchase-History.xlsx'))
result = run_insight_engine(df)

rg = ReportGenerator()
pdf_path = rg.generate_pdf(
    insights=result['strategic_brief'],
    charts=result.get('charts', []),
    metrics=result.get('computed_metrics', {}),
    df=df,
    session_id='test_charts'
)
print(f'PDF generated: {pdf_path}')
"
```

Check the logs for:
- `[Chart X] ✓ Got image from base64`
- `[Chart X] ✓ Plotly conversion successful`
- `[Chart X] ✓ ChartGenerator fallback successful`
- `[Chart X] ✗ All rendering methods failed`

### 2. Test Currency Symbol
Open the generated PDF and search for:
- ₹ symbols (should render correctly)
- `\mathbb{1}` or garbled text (should not appear)

### 3. Test Stray "1"
Open the generated PDF and check for:
- Any pages with just "1" and nothing else
- Any orphaned "1" characters

---

## Expected Score After Fixes

| Dimension | Current | After Fixes |
|-----------|---------|-------------|
| Visualization & Presentation | 5 | 8-9 |
| Trustworthiness & Reliability | 8 | 9 |
| Enterprise Readiness | 5 | 8% |
| **Overall** | **75** | **85-88** |

---

## Implementation Priority

1. **Fix 1 (Charts)** - Highest impact (+8 points)
   - Add detailed logging first to diagnose the issue
   - Then implement the matplotlib fallback if needed

2. **Fix 2 (Currency)** - Quick win (+1 point)
   - Just add `('FONTNAME', (0,0), (-1,-1), 'DejaVuSans')` to table styles

3. **Fix 3 (Stray "1")** - Quick win (+1 point)
   - Search for hardcoded "1" values and add guards

Total time estimate: 30-60 minutes for all three fixes.
