# P0: Verification Layer Implementation Plan

## Status: Verification Layer Created ✅

The verification layer (`engine/verifier.py`) has been created with the following components:

### 1. MetricVerifier
- ✅ `verify_revenue_calculation()` - Validates total revenue against source data
- ✅ `verify_aov_calculation()` - Validates Average Order Value
- ✅ `verify_percentage_claim()` - Validates percentage claims (return rate, etc.)

### 2. EntitySemanticVerifier
- ✅ `verify_entity_type()` - Prevents "Cameron" being treated as category
- ✅ Person/Place/Category/ID detection logic

### 3. StatisticalSignificanceVerifier
- ✅ `verify_group_difference()` - T-test validation
- ✅ `verify_within_group_variance()` - CV-based pricing inconsistency validation

### 4. BusinessPlausibilityVerifier
- ✅ `verify_revenue_impact_realism()` - Prevents impossible impact claims
- ✅ `verify_percentage_range()` - Validates percentage bounds

### 5. ContradictionDetector
- ✅ `detect_contradictions()` - Finds contradictory insights

### 6. InsightVerifier (Main Orchestrator)
- ✅ `validate_insight()` - Comprehensive single insight validation
- ✅ `verify_all_insights()` - Batch verification with filtering

## Next Steps (In Priority Order)

### WEEK 1 - Stop the Bleeding

#### Day 1-2: Integrate Verification into Pipeline
- [ ] Modify `report_generator.py::_derive_kpis()` to call `verify_kpis()`
- [ ] Add verification results to report metadata
- [ ] Log verification failures

#### Day 3-4: Fix Entity Detection
- [ ] Enhance `insight_engine.py::ColumnClassifier._detect_sub_roles()`
- [ ] Add entity type detection (person/place/category/ID)
- [ ] Use `EntitySemanticVerifier` for validation

#### Day 5: Column Coverage Tracking
- [ ] Add column importance scoring to `ColumnMap`
- [ ] Track which columns generated insights
- [ ] Enforce minimum coverage for high-importance columns

### WEEK 2 - Add Missing Analysis

#### Day 1-2: Returns Analysis
- [ ] Create `insight_engine.py::_rule_returns_analysis()`
- [ ] Detect return rate patterns
- [ ] Cross-reference with discount/shipping/category

#### Day 3-4: Temporal Intelligence
- [ ] Fix `_rule_temporal_peaks()` to always fire on date columns
- [ ] Add trend detection (30-month analysis)
- [ ] Add seasonality detection

#### Day 5: Salesperson Ranking
- [ ] Create `_rule_salesperson_ranking()`
- [ ] Detect person columns vs category columns
- [ ] Generate performance rankings

### WEEK 3 - Fix Bad Reasoning

#### Day 1-2: Pricing Inconsistency Fix
- [ ] Enhance `_rule_pricing_inconsistency()`
- [ ] Add within-group CV validation
- [ ] Use `StatisticalSignificanceVerifier`

#### Day 3-4: Real Prioritization
- [ ] Rewrite `_rank_insights()`
- [ ] Use evidence-based confidence scoring
- [ ] Not everything is CRITICAL

#### Day 5: Recommendation Intelligence
- [ ] Fix `report_generator.py::_build_section_7_recommendations()`
- [ ] Vary timeframe by insight type
- [ ] Vary owner by insight domain

## Architecture Changes

### Current Pipeline
```
dataset → metrics → insights → report
```

### New Pipeline (With Verification)
```
dataset → metrics → insights → VERIFICATION → filtered_insights → report
                                    ↓
                            contradiction_detection
                            confidence_recalibration
                            suppression
```

### Integration Points

1. **In `run_insight_engine()`**:
   ```python
   from verifier import InsightVerifier
   
   # After generating insights
   verifier = InsightVerifier()
   filtered_insights, verifications = verifier.verify_all_insights(
       insights, df, context={'total_revenue': metrics['total_revenue']}
   )
   ```

2. **In `_derive_kpis()`**:
   ```python
   from verifier import verify_kpis
   
   kpis = {...}  # Calculate KPIs
   
   # Verify before returning
   verification_results = verify_kpis(kpis, df, cm)
   
   # Log failures
   for metric, result in verification_results.items():
       if not result.passed:
           print(f"[VERIFICATION FAILED] {metric}: {result.reason}")
   ```

3. **In `build()` (report generation)**:
   ```python
   # Add verification metadata to report
   if verification_results:
       elements.append(Paragraph(
           "Note: All metrics verified against source data",
           styles['BodyText']
       ))
   ```

## Column Importance Scoring

Add to `ColumnMap.__init__()`:

```python
self.column_importance = {}

for col in df.columns:
    score = 0
    col_lower = col.lower()
    
    # Revenue columns: highest importance
    if any(kw in col_lower for kw in ['revenue', 'sales', 'amount', 'price']):
        score = 10
    
    # Return/refund columns: critical business signal
    elif any(kw in col_lower for kw in ['return', 'refund']):
        score = 10
    
    # Discount columns: pricing intelligence
    elif 'discount' in col_lower:
        score = 9
    
    # Date columns: temporal intelligence
    elif any(kw in col_lower for kw in ['date', 'time', 'month', 'year']):
        score = 10
    
    # Person columns: performance analysis
    elif self._is_person_column(col):
        score = 7
    
    # Category/segment columns: segmentation
    elif self._is_category_column(col):
        score = 6
    
    # ID columns: low importance
    elif 'id' in col_lower:
        score = 2
    
    self.column_importance[col] = score
```

## Success Metrics

After implementation, the system should:

1. **Zero Revenue Hallucinations**: All revenue figures verified ±1%
2. **Zero Entity Confusion**: "Cameron" correctly identified as person
3. **100% Column Coverage**: All importance=10 columns analyzed
4. **Confidence Calibration**: Not everything marked CRITICAL
5. **Contradiction-Free**: No contradictory insights in same report

## Files to Modify

1. ✅ `engine/verifier.py` - Created
2. ⏳ `engine/insight_engine.py` - Needs entity detection, column coverage, new rules
3. ⏳ `engine/report_generator.py` - Needs KPI verification, recommendation intelligence
4. ⏳ `engine/main.py` - Needs verification integration

## Testing Strategy

After each fix:
1. Run on sales dataset
2. Check verification logs
3. Validate no hallucinations
4. Confirm all high-importance columns analyzed
5. Verify confidence scores make sense

## Current Status

- Verification layer: ✅ Complete
- Integration: ⏳ Not started
- Entity detection: ⏳ Not started
- Column coverage: ⏳ Not started
- Missing rules: ⏳ Not started

**Next Action**: Integrate verification into `_derive_kpis()` and `run_insight_engine()`
