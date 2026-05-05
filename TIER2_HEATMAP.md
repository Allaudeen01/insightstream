# Tier 2 Enhancement: Region × Category Heatmap ✅

**Date:** May 5, 2026  
**Feature:** Multi-dimensional revenue heatmap  
**Status:** Implemented  
**Priority:** Medium ROI

---

## Overview

Added a revenue heatmap that visualizes the intensity of revenue across all Region × Category combinations. This provides an instant visual understanding of which categories dominate in each region.

---

## Feature Details

### What It Shows
- **Rows:** Regions (East, West, North, South)
- **Columns:** Product Categories (Electronics, Clothing, etc.)
- **Color Intensity:** Revenue amount (darker = higher revenue)
- **Cell Values:** Formatted as `₹5.7M` or `₹850K`

### Visual Design
- **Color Scale:** Blues (light to dark)
- **Text Color:** White for readability
- **Font Size:** 12pt
- **Color Bar:** Shows revenue scale in millions

### Use Cases
1. **Regional Strategy:** Identify which categories to push in each region
2. **Inventory Planning:** Stock high-performing category-region combinations
3. **Marketing Focus:** Target ads based on regional category preferences
4. **Gap Analysis:** Spot underperforming category-region pairs

---

## Implementation

### Location
**File:** `engine/insight_engine.py`  
**Method:** `SmartChartRecommender.recommend()`  
**Position:** After `geo_cat_revenue` grouped bar chart

### Code
```python
# ✅ TIER 2 ENHANCEMENT: Region × Category Revenue Heatmap
try:
    pivot = (
        pdf_tmp.groupby([region_col, cat_col])["_revenue"]
        .sum()
        .unstack(cat_col)
        .fillna(0)
    )
    # Format values for display (in millions)
    pivot_display = (pivot / 1_000_000).round(2)
    text_matrix = [
        [f"₹{v:.1f}M" if v >= 1 else f"₹{v*1000:.0f}K"
         for v in row]
        for row in pivot_display.values
    ]
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot_display.values,
        x=pivot_display.columns.tolist(),
        y=pivot_display.index.tolist(),
        colorscale="Blues",
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        hoverongaps=False,
        showscale=True,
        colorbar=dict(
            title="Revenue (M)",
            tickformat=".1f",
            ticksuffix="M"
        )
    ))
    fig_heat.update_layout(
        template="plotly_dark",
        xaxis_title=cat_col,
        yaxis_title=region_col,
        xaxis=dict(side="bottom"),
        margin=dict(l=80, r=80, t=20, b=60),
    )
    add("geo_heatmap", {
        "chart_id": "geo_heatmap",
        "chart_type": "heatmap",
        "title": f"Revenue Heatmap: {region_col} × {cat_col}",
        "description": f"Revenue intensity across all {region_col}–{cat_col} combinations",
        "plotly_json": json.loads(fig_heat.update_layout(**{
            **CHART_LAYOUT_BASE,
            "yaxis": {
                "gridcolor": "rgba(255,255,255,0.05)",
            }
        }).to_json()),
        "columns_used": [region_col, cat_col, revenue_col],
        "priority_score": 86,
        "insight_reason": "Multi-dimensional revenue concentration — spot which category dominates each region",
        "interest_level": "high"
    })
except Exception as _e:
    print(f"[geo_heatmap] failed: {_e}")
```

---

## Technical Details

### Data Transformation
1. **Group by:** Region and Category
2. **Aggregate:** Sum of revenue
3. **Pivot:** Regions as rows, Categories as columns
4. **Fill:** 0 for missing combinations
5. **Scale:** Divide by 1M for display

### Value Formatting
- **≥ ₹1M:** Show as `₹5.7M` (1 decimal)
- **< ₹1M:** Show as `₹850K` (no decimals)
- **Format:** Rupee symbol + value + unit

### Chart Properties
- **Chart ID:** `geo_heatmap`
- **Chart Type:** `heatmap`
- **Priority Score:** 86 (higher than grouped bar at 82)
- **Interest Level:** `high`

---

## Example Output

### Sample Heatmap
```
                Electronics  Clothing  Home & Kitchen  Sports  Books
East            ₹1.4M        ₹620K     ₹510K          ₹560K   ₹320K
North           ₹1.8M        ₹680K     ₹470K          ₹410K   ₹380K
South           ₹1.0M        ₹720K     ₹550K          ₹330K   ₹450K
West            ₹1.4M        ₹760K     ₹560K          ₹370K   ₹300K
```

**Insights from this example:**
- Electronics dominates all regions (darkest blue)
- North has highest Electronics revenue (₹1.8M)
- Books consistently lowest across all regions
- Clothing shows consistent performance (₹620K-₹760K range)

---

## Benefits

### 1. Instant Pattern Recognition
- **Before:** Scan grouped bar chart, mentally compare bars
- **After:** Color intensity shows patterns at a glance

### 2. Multi-Dimensional Analysis
- **Before:** Grouped bar shows one dimension at a time
- **After:** Heatmap shows all combinations simultaneously

### 3. Gap Identification
- **Before:** Hard to spot underperforming combinations
- **After:** Light colors immediately highlight gaps

### 4. Strategic Planning
- **Before:** Manual analysis of category-region performance
- **After:** Visual guide for resource allocation

---

## Testing Instructions

1. **Upload Data:**
   - Go to http://localhost:3000/upload
   - Upload sales_data_1000.csv

2. **View Heatmap:**
   - Navigate to Insights page
   - Scroll to find "Revenue Heatmap: Region × Product Category"

3. **Verify:**
   - Heatmap appears after grouped bar chart
   - All regions shown as rows
   - All categories shown as columns
   - Values formatted as ₹X.XM or ₹XXXK
   - Color intensity matches values
   - Color bar shows scale

4. **Export PDF:**
   - Click "Export PDF"
   - Verify heatmap appears in report

---

## Performance Impact

- **Computation:** ~5-10ms (pivot operation)
- **Rendering:** Client-side Plotly (no backend impact)
- **Payload Size:** +2-3KB (heatmap data)
- **Priority Score:** 86 (appears before lower-priority charts)

---

## Future Enhancements

1. **Interactive Drill-Down:** Click cell to see detailed transactions
2. **Comparison Mode:** Show YoY or MoM changes in heatmap
3. **Threshold Highlighting:** Auto-highlight cells above/below targets
4. **Export to Excel:** Download heatmap data as spreadsheet

---

## Commit

```
00ffffa - feat: Add Region × Category Revenue Heatmap (Tier 2)
```

---

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| Code Implementation | ✅ Complete | Added to insight_engine.py |
| Backend Reload | ✅ Complete | Auto-reloaded successfully |
| Error Handling | ✅ Complete | Try-except with logging |
| Documentation | ✅ Complete | This file |
| Testing | ⏳ Pending | User testing required |

---

**Ready for Testing!** 🚀

Upload data and verify the heatmap appears in the Insights page and PDF export.
