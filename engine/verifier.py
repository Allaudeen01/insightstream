"""
Verification Layer for InsightStream
=====================================
This module provides validation, contradiction detection, and confidence
recalibration for all generated insights and metrics.

Architecture:
    dataset → metrics → insights → VERIFICATION → report
    
Without this layer, the system generates conclusions before proving them.
"""

import polars as pl
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class ValidationResult:
    """Result of validating a single aspect"""
    passed: bool
    confidence: float  # 0.0 to 1.0
    reason: str
    evidence: Dict[str, Any]


@dataclass
class InsightVerification:
    """Complete verification result for an insight"""
    insight_id: str
    original_confidence: float
    adjusted_confidence: float
    validations: List[ValidationResult]
    contradictions: List[str]
    should_suppress: bool
    warnings: List[str]


class MetricVerifier:
    """Validates that computed metrics are mathematically correct"""
    
    @staticmethod
    def verify_revenue_calculation(
        df: pl.DataFrame,
        claimed_revenue: float,
        revenue_col: Optional[str] = None,
        tolerance: float = 0.01
    ) -> ValidationResult:
        """
        Verify that claimed total revenue matches actual data.
        This is CRITICAL for enterprise trust.
        """
        if revenue_col is None:
            # Try to find revenue column
            candidates = []
            for col in df.columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in ['revenue', 'sales', 'amount', 'total', 'price']):
                    if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                        candidates.append(col)
            
            if not candidates:
                return ValidationResult(
                    passed=False,
                    confidence=0.0,
                    reason="No revenue column found in dataset",
                    evidence={"columns": df.columns}
                )
            revenue_col = candidates[0]
        
        try:
            actual_revenue = df[revenue_col].sum()
            diff_pct = abs(actual_revenue - claimed_revenue) / actual_revenue if actual_revenue > 0 else 1.0
            
            passed = diff_pct <= tolerance
            confidence = max(0.0, 1.0 - diff_pct)
            
            return ValidationResult(
                passed=passed,
                confidence=confidence,
                reason=f"Revenue {'matches' if passed else 'MISMATCH'}: claimed ₹{claimed_revenue:,.0f} vs actual ₹{actual_revenue:,.0f} (diff: {diff_pct*100:.1f}%)",
                evidence={
                    "claimed": claimed_revenue,
                    "actual": actual_revenue,
                    "column": revenue_col,
                    "diff_pct": diff_pct
                }
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                confidence=0.0,
                reason=f"Revenue verification failed: {str(e)}",
                evidence={"error": str(e)}
            )
    
    @staticmethod
    def verify_aov_calculation(
        df: pl.DataFrame,
        claimed_aov: float,
        revenue_col: Optional[str] = None,
        order_col: Optional[str] = None,
        tolerance: float = 0.05
    ) -> ValidationResult:
        """Verify Average Order Value calculation"""
        try:
            # Find revenue column
            if revenue_col is None:
                for col in df.columns:
                    if any(kw in col.lower() for kw in ['revenue', 'sales', 'amount', 'price']):
                        if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                            revenue_col = col
                            break
            
            if revenue_col is None:
                return ValidationResult(False, 0.0, "No revenue column found", {})
            
            # Calculate actual AOV
            total_revenue = df[revenue_col].sum()
            
            # Count unique orders if order_col exists, otherwise count rows
            if order_col and order_col in df.columns:
                order_count = df[order_col].n_unique()
            else:
                order_count = len(df)
            
            actual_aov = total_revenue / order_count if order_count > 0 else 0
            diff_pct = abs(actual_aov - claimed_aov) / actual_aov if actual_aov > 0 else 1.0
            
            passed = diff_pct <= tolerance
            confidence = max(0.0, 1.0 - diff_pct)
            
            return ValidationResult(
                passed=passed,
                confidence=confidence,
                reason=f"AOV {'matches' if passed else 'MISMATCH'}: claimed ₹{claimed_aov:,.0f} vs actual ₹{actual_aov:,.0f}",
                evidence={
                    "claimed": claimed_aov,
                    "actual": actual_aov,
                    "total_revenue": total_revenue,
                    "order_count": order_count,
                    "diff_pct": diff_pct
                }
            )
        except Exception as e:
            return ValidationResult(False, 0.0, f"AOV verification failed: {str(e)}", {"error": str(e)})
    
    @staticmethod
    def verify_percentage_claim(
        df: pl.DataFrame,
        claimed_pct: float,
        numerator_col: str,
        denominator_col: Optional[str] = None,
        tolerance: float = 0.02
    ) -> ValidationResult:
        """Verify percentage claims (e.g., return rate, discount rate)"""
        try:
            if numerator_col not in df.columns:
                return ValidationResult(False, 0.0, f"Column {numerator_col} not found", {})
            
            # Calculate actual percentage
            if denominator_col:
                numerator = df[numerator_col].sum()
                denominator = df[denominator_col].sum()
            else:
                # Assume binary flag
                numerator = df[numerator_col].sum()
                denominator = len(df)
            
            actual_pct = (numerator / denominator * 100) if denominator > 0 else 0
            diff = abs(actual_pct - claimed_pct)
            
            passed = diff <= tolerance * 100
            confidence = max(0.0, 1.0 - diff / 100)
            
            return ValidationResult(
                passed=passed,
                confidence=confidence,
                reason=f"Percentage {'matches' if passed else 'MISMATCH'}: claimed {claimed_pct:.1f}% vs actual {actual_pct:.1f}%",
                evidence={
                    "claimed": claimed_pct,
                    "actual": actual_pct,
                    "numerator": numerator,
                    "denominator": denominator
                }
            )
        except Exception as e:
            return ValidationResult(False, 0.0, f"Percentage verification failed: {str(e)}", {"error": str(e)})


class EntitySemanticVerifier:
    """Validates that entities are correctly classified (person vs place vs category)"""
    
    # Known person name patterns
    PERSON_INDICATORS = {
        'first_names': {'john', 'jane', 'michael', 'sarah', 'david', 'emily', 'cameron', 'alex', 'chris'},
        'titles': {'mr', 'mrs', 'ms', 'dr', 'prof'},
        'suffixes': {'jr', 'sr', 'ii', 'iii'}
    }
    
    # Known place patterns
    PLACE_INDICATORS = {
        'regions': {'north', 'south', 'east', 'west', 'central', 'northeast', 'northwest', 'southeast', 'southwest'},
        'place_types': {'city', 'state', 'country', 'region', 'district', 'zone'}
    }
    
    @classmethod
    def verify_entity_type(
        cls,
        value: str,
        claimed_type: str  # 'person', 'place', 'category', 'id'
    ) -> ValidationResult:
        """
        Verify that an entity is correctly classified.
        Critical for preventing "Cameron" being treated as a category.
        """
        value_lower = str(value).lower().strip()
        
        # Check for person indicators
        is_person = (
            value_lower in cls.PERSON_INDICATORS['first_names'] or
            any(title in value_lower for title in cls.PERSON_INDICATORS['titles']) or
            any(value_lower.endswith(suffix) for suffix in cls.PERSON_INDICATORS['suffixes'])
        )
        
        # Check for place indicators
        is_place = (
            any(region in value_lower for region in cls.PLACE_INDICATORS['regions']) or
            any(ptype in value_lower for ptype in cls.PLACE_INDICATORS['place_types'])
        )
        
        # Determine actual type
        if is_person:
            actual_type = 'person'
            confidence = 0.8
        elif is_place:
            actual_type = 'place'
            confidence = 0.7
        elif value_lower.isdigit() or (len(value) > 8 and any(c.isdigit() for c in value)):
            actual_type = 'id'
            confidence = 0.9
        else:
            actual_type = 'category'
            confidence = 0.5
        
        passed = (actual_type == claimed_type)
        
        return ValidationResult(
            passed=passed,
            confidence=confidence if passed else 0.2,
            reason=f"Entity '{value}' is {actual_type}, {'matches' if passed else 'NOT'} claimed type '{claimed_type}'",
            evidence={
                "value": value,
                "claimed_type": claimed_type,
                "actual_type": actual_type,
                "is_person": is_person,
                "is_place": is_place
            }
        )


class StatisticalSignificanceVerifier:
    """Validates that insights are statistically significant"""
    
    @staticmethod
    def verify_group_difference(
        group_a_values: List[float],
        group_b_values: List[float],
        claimed_significant: bool,
        alpha: float = 0.05
    ) -> ValidationResult:
        """Verify that claimed group differences are statistically significant"""
        try:
            if len(group_a_values) < 2 or len(group_b_values) < 2:
                return ValidationResult(
                    passed=False,
                    confidence=0.0,
                    reason="Insufficient data for statistical test",
                    evidence={"n_a": len(group_a_values), "n_b": len(group_b_values)}
                )
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group_a_values, group_b_values)
            is_significant = p_value < alpha
            
            passed = (is_significant == claimed_significant)
            confidence = 1.0 - p_value if is_significant else p_value
            
            return ValidationResult(
                passed=passed,
                confidence=confidence,
                reason=f"Statistical test: p={p_value:.4f}, {'significant' if is_significant else 'not significant'}",
                evidence={
                    "p_value": p_value,
                    "t_statistic": t_stat,
                    "is_significant": is_significant,
                    "claimed_significant": claimed_significant,
                    "alpha": alpha
                }
            )
        except Exception as e:
            return ValidationResult(False, 0.0, f"Statistical test failed: {str(e)}", {"error": str(e)})
    
    @staticmethod
    def verify_within_group_variance(
        df: pl.DataFrame,
        group_col: str,
        value_col: str,
        claimed_inconsistent: bool,
        cv_threshold: float = 0.3
    ) -> ValidationResult:
        """
        Verify pricing inconsistency claims by checking within-group coefficient of variation.
        Critical for preventing false "pricing inconsistency" claims.
        """
        try:
            pdf = df.to_pandas()
            grouped = pdf.groupby(group_col)[value_col]
            
            # Calculate CV for each group
            cvs = []
            for name, group in grouped:
                if len(group) > 1:
                    cv = group.std() / group.mean() if group.mean() > 0 else 0
                    cvs.append(cv)
            
            if not cvs:
                return ValidationResult(False, 0.0, "No groups with sufficient data", {})
            
            avg_cv = np.mean(cvs)
            max_cv = np.max(cvs)
            
            is_inconsistent = max_cv > cv_threshold
            passed = (is_inconsistent == claimed_inconsistent)
            
            confidence = max_cv if is_inconsistent else (1.0 - max_cv)
            
            return ValidationResult(
                passed=passed,
                confidence=confidence,
                reason=f"Within-group CV: avg={avg_cv:.2f}, max={max_cv:.2f}, threshold={cv_threshold}",
                evidence={
                    "avg_cv": avg_cv,
                    "max_cv": max_cv,
                    "threshold": cv_threshold,
                    "is_inconsistent": is_inconsistent,
                    "claimed_inconsistent": claimed_inconsistent,
                    "n_groups": len(cvs)
                }
            )
        except Exception as e:
            return ValidationResult(False, 0.0, f"Variance check failed: {str(e)}", {"error": str(e)})


class BusinessPlausibilityVerifier:
    """Validates that insights make business sense"""
    
    @staticmethod
    def verify_revenue_impact_realism(
        claimed_impact: float,
        total_revenue: float,
        max_reasonable_pct: float = 0.5
    ) -> ValidationResult:
        """Verify that claimed revenue impact is realistic"""
        impact_pct = claimed_impact / total_revenue if total_revenue > 0 else 0
        
        is_realistic = impact_pct <= max_reasonable_pct
        confidence = 1.0 - min(1.0, impact_pct / max_reasonable_pct)
        
        return ValidationResult(
            passed=is_realistic,
            confidence=confidence,
            reason=f"Impact is {impact_pct*100:.1f}% of total revenue ({'realistic' if is_realistic else 'UNREALISTIC'})",
            evidence={
                "claimed_impact": claimed_impact,
                "total_revenue": total_revenue,
                "impact_pct": impact_pct,
                "max_reasonable_pct": max_reasonable_pct
            }
        )
    
    @staticmethod
    def verify_percentage_range(
        claimed_pct: float,
        min_val: float = 0.0,
        max_val: float = 100.0
    ) -> ValidationResult:
        """Verify that percentage is in valid range"""
        is_valid = min_val <= claimed_pct <= max_val
        
        return ValidationResult(
            passed=is_valid,
            confidence=1.0 if is_valid else 0.0,
            reason=f"Percentage {claimed_pct:.1f}% is {'valid' if is_valid else 'INVALID'} (range: {min_val}-{max_val})",
            evidence={
                "claimed_pct": claimed_pct,
                "min_val": min_val,
                "max_val": max_val
            }
        )


class ContradictionDetector:
    """Detects contradictions between insights"""
    
    @staticmethod
    def detect_contradictions(insights: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
        """
        Detect contradictory insights.
        Returns list of (insight_idx_1, insight_idx_2, contradiction_reason)
        """
        contradictions = []
        
        for i, insight_a in enumerate(insights):
            for j, insight_b in enumerate(insights[i+1:], start=i+1):
                # Check for opposite claims about same entity
                if ContradictionDetector._are_contradictory(insight_a, insight_b):
                    reason = ContradictionDetector._explain_contradiction(insight_a, insight_b)
                    contradictions.append((i, j, reason))
        
        return contradictions
    
    @staticmethod
    def _are_contradictory(insight_a: Dict, insight_b: Dict) -> bool:
        """Check if two insights contradict each other"""
        # Example: "Category A is best" vs "Category B is best"
        text_a = insight_a.get('insight', '').lower()
        text_b = insight_b.get('insight', '').lower()
        
        # Check for opposite superlatives
        if ('highest' in text_a or 'best' in text_a) and ('highest' in text_b or 'best' in text_b):
            # If they mention different entities, might be contradictory
            return True
        
        return False
    
    @staticmethod
    def _explain_contradiction(insight_a: Dict, insight_b: Dict) -> str:
        """Explain why two insights contradict"""
        return f"Insights claim different entities are 'best' or 'highest'"


class InsightVerifier:
    """Main verification orchestrator"""
    
    def __init__(self):
        self.metric_verifier = MetricVerifier()
        self.entity_verifier = EntitySemanticVerifier()
        self.stats_verifier = StatisticalSignificanceVerifier()
        self.business_verifier = BusinessPlausibilityVerifier()
        self.contradiction_detector = ContradictionDetector()
    
    def validate_insight(
        self,
        insight: Dict[str, Any],
        df: pl.DataFrame,
        context: Dict[str, Any]
    ) -> InsightVerification:
        """
        Validate a single insight with comprehensive checks.
        This is the MOST IMPORTANT function in the verification layer.
        """
        validations = []
        warnings = []
        
        # 1. Check metric consistency
        if 'revenue' in insight.get('insight', '').lower():
            if 'total_revenue' in context:
                val = self.metric_verifier.verify_revenue_calculation(
                    df,
                    context['total_revenue'],
                    context.get('revenue_col')
                )
                validations.append(val)
        
        # 2. Check entity semantics
        # (Would need to extract entities from insight text)
        
        # 3. Check statistical support
        # (Would need to extract claims and verify)
        
        # 4. Check business plausibility
        if 'impact' in insight:
            val = self.business_verifier.verify_revenue_impact_realism(
                insight['impact'],
                context.get('total_revenue', 0)
            )
            validations.append(val)
        
        # 5. Aggregate confidence
        original_confidence = insight.get('confidence', 0.5)
        
        if validations:
            avg_validation_confidence = np.mean([v.confidence for v in validations])
            adjusted_confidence = original_confidence * avg_validation_confidence
        else:
            adjusted_confidence = original_confidence
        
        # 6. Determine if should suppress
        should_suppress = (
            adjusted_confidence < 0.55 or
            any(not v.passed for v in validations)
        )
        
        return InsightVerification(
            insight_id=insight.get('id', 'unknown'),
            original_confidence=original_confidence,
            adjusted_confidence=adjusted_confidence,
            validations=validations,
            contradictions=[],
            should_suppress=should_suppress,
            warnings=warnings
        )
    
    def verify_all_insights(
        self,
        insights: List[Dict[str, Any]],
        df: pl.DataFrame,
        context: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[InsightVerification]]:
        """
        Verify all insights and return filtered list + verification results.
        """
        verifications = []
        filtered_insights = []
        
        # Verify each insight
        for insight in insights:
            verification = self.validate_insight(insight, df, context)
            verifications.append(verification)
            
            if not verification.should_suppress:
                # Update confidence
                insight['confidence'] = verification.adjusted_confidence
                filtered_insights.append(insight)
        
        # Detect contradictions
        contradictions = self.contradiction_detector.detect_contradictions(filtered_insights)
        
        # Mark contradictory insights
        for idx_a, idx_b, reason in contradictions:
            verifications[idx_a].contradictions.append(f"Contradicts insight {idx_b}: {reason}")
            verifications[idx_b].contradictions.append(f"Contradicts insight {idx_a}: {reason}")
        
        return filtered_insights, verifications


def verify_kpis(
    kpis: Dict[str, Any],
    df: pl.DataFrame,
    column_map: Any
) -> Dict[str, ValidationResult]:
    """
    Verify all KPIs against source data.
    This should be called BEFORE generating the report.
    """
    verifier = MetricVerifier()
    results = {}
    
    # Verify total revenue
    if 'total_revenue' in kpis:
        results['total_revenue'] = verifier.verify_revenue_calculation(
            df,
            kpis['total_revenue'],
            column_map.revenue if hasattr(column_map, 'revenue') else None
        )
    
    # Verify AOV
    if 'aov' in kpis:
        results['aov'] = verifier.verify_aov_calculation(
            df,
            kpis['aov'],
            column_map.revenue if hasattr(column_map, 'revenue') else None
        )
    
    return results
