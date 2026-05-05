"""
InsightStream — Smart Insight Engine
=====================================
Replaces the basic EDA insight generator with a production-grade
business intelligence engine.

Components:
  1. ColumnClassifier    — detects IDs, numerics, categoricals, temporal, binary
  2. MetricComputer      — derives Revenue, Return Rate, AOV, Delivery Delay, etc.
  3. BusinessRuleEngine  — threshold-based rule evaluation → structured insights
  4. InsightNarrator     — plain-English descriptions for non-technical users
  5. SmartChartRecommender — context-aware, meaningful chart suggestions
  6. AnomalyDetector     — IQR / pattern-based anomaly detection
  7. EdgeCaseHandler     — small dataset, missing values, skew warnings
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import polars as pl
import pandas as pd
import numpy as np

from report_generator import TEMPLATES

log = logging.getLogger(__name__)

pd.set_option('display.max_colwidth', None)


import functools
import time

def log_rule(func):
    """Decorator that logs entry, exit, count, and duration of each insight rule."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rule_name = func.__name__
        start = time.time()
        print(f"[RULE START] {rule_name}")
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000
            count = len(result) if isinstance(result, list) else (1 if result else 0)
            status = "[FIRED]" if count > 0 else "[SUPPRESSED] (no qualifying segments)"
            print(f"[RULE END]   {rule_name} -> {count} insights | {duration:.1f}ms | {status}")
            return result
        except Exception as e:
            duration = (time.time() - start) * 1000
            print(f"[RULE FAIL]  {rule_name} -> ERROR: {e} | {duration:.1f}ms")
            raise
    return wrapper

# Global chart layout base to prevent duplicate titles and ensure styling
CHART_LAYOUT_BASE = {
    "title": {"text": ""},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#94a3b8"},
    "xaxis": {"gridcolor": "rgba(255,255,255,0.05)"},
    "yaxis": {"gridcolor": "rgba(255,255,255,0.05)"},
    "legend": {"bgcolor": "rgba(0,0,0,0)"},
    "margin": {"l": 60, "r": 30, "t": 20, "b": 60},
}

# ============================================================
# 1. DATA TYPES & SHARED STRUCTURES
# ============================================================

IDENTIFIER_PATTERNS = re.compile(
    r"(^|_)(id|key|code|uuid|guid|index|idx|num|no|ref|pk|sk)(_|$)",
    re.IGNORECASE,
)

REVENUE_KEYWORDS  = {"revenue", "sales", "income", "turnover", "gmv", "gross"}
PRICE_KEYWORDS    = {"price", "cost", "amount", "value", "fee", "charge", "rate", "spend"}
QTY_KEYWORDS      = {"quantity", "qty", "units", "count", "volume", "items", "pieces"}
RETURN_KEYWORDS   = {"return", "returned", "refund", "refunded", "chargeback", "cancelled"}
DATE_KEYWORDS     = {"date", "time", "day", "month", "year", "created", "updated",
                     "ordered", "shipped", "delivered", "at"}
DELIVERY_KEYWORDS = {"delivery", "shipping", "days", "lead", "duration", "delay", "transit"}
CATEGORY_KEYWORDS = {"category", "type", "segment", "class", "group", "department",
                     "product", "item", "brand", "sku", "region", "city", "country",
                     "state", "zone", "area", "channel", "payment", "method", "status"}


@dataclass
class ColumnProfile:
    name: str
    role: str  # "identifier" | "numerical" | "categorical" | "temporal" | "binary" | "text"
    sub_role: str = ""  # e.g., "price", "quantity", "return_flag", "date_order"
    n_unique: int = 0
    missing_pct: float = 0.0
    sample_values: list = field(default_factory=list)


@dataclass
class DataProfile:
    row_count: int
    col_count: int
    identifiers: list[str] = field(default_factory=list)
    numericals: list[str] = field(default_factory=list)
    categoricals: list[str] = field(default_factory=list)
    temporals: list[str] = field(default_factory=list)
    binaries: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    profiles: dict[str, ColumnProfile] = field(default_factory=dict)

    # Derived‐metric targets
    price_col: Optional[str] = None
    qty_col: Optional[str] = None
    revenue_col: Optional[str] = None  # explicit revenue col if found
    return_col: Optional[str] = None
    date_col: Optional[str] = None
    delivery_days_col: Optional[str] = None
    category_col: Optional[str] = None  # best categorical for grouping
    geographic_col: Optional[str] = None  # city / region / country


@dataclass
class BusinessInsight:
    title: str
    description: str            # "What is happening" summary
    why_it_matters: str = ""    # "Why it matters" context
    evidence: str = ""          # "Supporting evidence"
    decision_implication: str = "" # "Decision Implication (Step 4.4)"
    impact: str = "medium"      # "🔴 Critical" | "🟠 Important" | "🟢 Minor"
    recommendation: str = ""    
    is_unexpected: bool = False # "Unexpected Insight (Step 5)"
    confidence_label: str = "medium"
    confidence_explanation: str = "" # "Step 8: Detailed explanation"
    score: float = 0.0          
    chart_type: str | None = None
    chart_data: dict | None = None
    qualified_segments: list[str] = field(default_factory=list)
    excluded_segments: list[str] = field(default_factory=list)
    rule_type: str = "general"

@dataclass
class ComputedMetric:
    name: str
    value: float
    formatted: str
    description: str


# ============================================================
# 2. COLUMN CLASSIFIER
# ============================================================

class ColumnClassifier:
    """Classify every column into a role, and detect sub-roles for computing metrics."""

    def classify(self, df: pl.DataFrame) -> DataProfile:
        profile = DataProfile(row_count=len(df), col_count=len(df.columns))

        for col in df.columns:
            cp = self._classify_column(df, col)
            profile.profiles[col] = cp

            if cp.role == "identifier":
                profile.identifiers.append(col)
            elif cp.role == "numerical":
                profile.numericals.append(col)
            elif cp.role == "categorical":
                profile.categoricals.append(col)
            elif cp.role == "temporal":
                profile.temporals.append(col)
            elif cp.role == "binary":
                profile.binaries.append(col)
            else:
                profile.texts.append(col)

        # Detect sub-roles for metric computation
        self._detect_sub_roles(df, profile)
        return profile

    # ------------------------------------------------------------------ #
    def _classify_column(self, df: pl.DataFrame, col: str) -> ColumnProfile:
        series = df[col]
        dtype = series.dtype
        n_unique = series.n_unique()
        n_total = len(series)
        missing_pct = round(series.null_count() / max(n_total, 1) * 100, 1)
        sample = [v for v in series.drop_nulls().head(5).to_list()]
        col_lower = col.lower()

        # ── Temporal ──────────────────────────────────────────────────────
        if dtype in (pl.Date, pl.Datetime, pl.Duration, pl.Time):
            return ColumnProfile(col, "temporal", n_unique=n_unique,
                                 missing_pct=missing_pct, sample_values=sample)

        # ── Try to parse string as date ────────────────────────────────────
        if dtype == pl.Utf8:
            if self._looks_like_date_col(col_lower, series):
                return ColumnProfile(col, "temporal", n_unique=n_unique,
                                     missing_pct=missing_pct, sample_values=sample)

        is_numeric = dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32,
                               pl.Int16, pl.Int8, pl.UInt64, pl.UInt32,
                               pl.UInt16, pl.UInt8)

        # ── Identifier (numeric) ─────────────────────────────────────────
        if is_numeric:
            uniqueness_ratio = n_unique / max(n_total, 1)
            if IDENTIFIER_PATTERNS.search(col_lower) or (
                uniqueness_ratio > 0.95 and n_unique > 50
                and self._is_sequential_or_high_card(series)
            ):
                return ColumnProfile(col, "identifier", n_unique=n_unique,
                                     missing_pct=missing_pct, sample_values=sample)
            return ColumnProfile(col, "numerical", n_unique=n_unique,
                                 missing_pct=missing_pct, sample_values=sample)

        # ── Identifier (string) ──────────────────────────────────────────
        if dtype == pl.Utf8:
            uniqueness_ratio = n_unique / max(n_total, 1)
            if IDENTIFIER_PATTERNS.search(col_lower) and uniqueness_ratio > 0.5:
                return ColumnProfile(col, "identifier", n_unique=n_unique,
                                     missing_pct=missing_pct, sample_values=sample)

            if uniqueness_ratio >= 0.95 and n_unique > 50:
                return ColumnProfile(col, "identifier", n_unique=n_unique,
                                     missing_pct=missing_pct, sample_values=sample)

            # If the column NAME suggests it's numeric (price/amount/cost/etc.)
            # AND most values can be parsed as numbers after stripping currency
            # symbols, treat it as numeric. This handles "price_string" columns
            # in messy CSVs where prices were stored as strings.
            numeric_name_kws = ("price", "amount", "cost", "value", "revenue",
                                "salary", "fee", "rate")
            if any(kw in col_lower for kw in numeric_name_kws):
                try:
                    cleaned = (series.dropna().astype(str)
                                     .str.replace(r"[^\d.\-]", "", regex=True))
                    parsed = pd.to_numeric(cleaned, errors="coerce")
                    parse_rate = parsed.notna().sum() / max(len(cleaned), 1)
                    if parse_rate >= 0.8:
                        return ColumnProfile(col, "numerical", n_unique=n_unique,
                                             missing_pct=missing_pct, sample_values=sample)
                except Exception:
                    pass

            # ── Binary ────────────────────────────────────────────────────
            if n_unique <= 2:
                return ColumnProfile(col, "binary", n_unique=n_unique,
                                     missing_pct=missing_pct, sample_values=sample)

            # ── High-cardinality text ──────────────────────────────────────
            if uniqueness_ratio > 0.5 and n_unique > 50:
                return ColumnProfile(col, "text", n_unique=n_unique,
                                     missing_pct=missing_pct, sample_values=sample)

            return ColumnProfile(col, "categorical", n_unique=n_unique,
                                 missing_pct=missing_pct, sample_values=sample)

        # ── Boolean → binary ──────────────────────────────────────────────
        if dtype == pl.Boolean:
            return ColumnProfile(col, "binary", n_unique=n_unique,
                                 missing_pct=missing_pct, sample_values=sample)

        return ColumnProfile(col, "text", n_unique=n_unique,
                             missing_pct=missing_pct, sample_values=sample)

    # ------------------------------------------------------------------ #
    def _looks_like_date_col(self, col_lower: str, series: pl.Series) -> bool:
        if not any(kw in col_lower for kw in DATE_KEYWORDS):
            return False
        sample = series.drop_nulls().head(10).to_list()
        date_pattern = re.compile(
            r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}"
        )
        hits = sum(1 for v in sample if date_pattern.search(str(v)))
        return hits >= len(sample) * 0.7

    def _is_sequential_or_high_card(self, series: pl.Series) -> bool:
        """True if the series looks like an auto-increment ID."""
        try:
            numeric = series.drop_nulls().cast(pl.Float64)
            if len(numeric) < 2:
                return False
            diffs = numeric.diff().drop_nulls()
            # Mostly sequential (monotonically increasing by ~1)
            return (diffs > 0).mean() > 0.8
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def _detect_sub_roles(self, df: pl.DataFrame, profile: DataProfile) -> None:
        """Assign semantic sub-roles to drive metric computation.
        
        IMPORTANT: REVENUE_KEYWORDS must be checked BEFORE PRICE_KEYWORDS
        because columns like 'Sales Amount' contain 'amount' (a PRICE keyword)
        but are actually revenue. Checking revenue first prevents Price×Qty
        double-counting that inflates revenue by ~400×.
        """
        for col in profile.numericals:
            cl = col.lower()
            # Check revenue FIRST — 'Sales Amount' matches both revenue ('sales')
            # and price ('amount'). Revenue must win to avoid Price×Qty inflation.
            if any(k in cl for k in REVENUE_KEYWORDS):
                if profile.revenue_col is None:
                    profile.revenue_col = col
            elif any(k in cl for k in PRICE_KEYWORDS):
                if profile.price_col is None:
                    profile.price_col = col
            elif any(k in cl for k in QTY_KEYWORDS):
                if profile.qty_col is None:
                    profile.qty_col = col
            elif any(k in cl for k in DELIVERY_KEYWORDS):
                if profile.delivery_days_col is None:
                    profile.delivery_days_col = col

        for col in profile.binaries:
            cl = col.lower()
            if any(k in cl for k in RETURN_KEYWORDS):
                profile.return_col = col

        for col in profile.temporals:
            cl = col.lower()
            if any(k in cl for k in DATE_KEYWORDS):
                if profile.date_col is None:
                    profile.date_col = col

        # Best categorical for grouping: prefer known category keywords
        for col in profile.categoricals:
            cl = col.lower()
            if any(k in cl for k in CATEGORY_KEYWORDS):
                n_unique = profile.profiles[col].n_unique
                if 2 <= n_unique <= 20:
                    if profile.category_col is None:
                        profile.category_col = col
                    if any(k in cl for k in {"city", "region", "state", "country", "area", "zone"}):
                        profile.geographic_col = col
        # Fallback
        if not profile.category_col and profile.categoricals:
            profile.category_col = min(
                profile.categoricals,
                key=lambda c: abs(profile.profiles[c].n_unique - 5)
            )

# ============================================================
# 2.5 DOMAIN DETECTION ENGINE (Step 1)
# ============================================================

class DomainDetector:
    """Identify the dataset domain using weighted column keyword matching."""
    
    def detect_domain(self, df) -> tuple[str, float]:
        """
        Weighted keyword scoring for domain detection.
        Returns (domain_name, confidence_score).
        Deterministic — same input always returns same output.
        """
        # Normalize all column names: lowercase, strip
        cols = [str(c).lower().strip() for c in df.columns]
        cols_joined = " ".join(cols)

        DOMAIN_KEYWORDS = {
            "ecommerce": [
                "price", "revenue", "order", "payment", "cart",
                "delivery", "returned", "product", "category",
                "discount", "quantity", "customer", "shipping", "sku"
            ],
            "sales": [
                "sale", "profit", "salesperson",
                "sale_date", "quantity_sold",
                "quantity sold", "product_category",
                "sales_amount", "sales amount",
            ],
            "healthcare": [
                "patient", "diagnosis", "blood", "bio", "clinical",
                "insurance", "doctor", "hospital", "treatment",
                "medication", "symptom", "icd"
            ],
            "finance": [
                "profit", "balance", "equity", "asset", "liability",
                "ledger", "debit", "credit", "interest", "loan",
                "portfolio", "stock", "dividend"
            ],
            "hr": [
                "salary", "employee", "department", "attrition",
                "tenure", "hire", "manager", "performance_review",
                "leave", "promotion", "designation"
            ],
        }

        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            hits = 0
            for kw in keywords:
                # Match if keyword appears in any column name
                if any(kw in c for c in cols):
                    hits += 1
            # Score = hits / total keywords for that domain
            scores[domain] = hits / len(keywords)

        # Tie-breaking: boost sales score using column-level evidence
        actual_cols_lower = [c.lower() for c in df.columns]

        # "profit" is a strong sales/retail signal, absent from ecommerce
        if "profit" in actual_cols_lower:
            scores["sales"] = scores.get("sales", 0) + 0.3

        # "salesperson" / "sales_person" → definitively sales domain
        if any("salesperson" in c or "sales_person" in c for c in actual_cols_lower):
            scores["sales"] = scores.get("sales", 0) + 0.5

        # "order" + "discount" without "product" → ecommerce signal
        has_order   = any("order"    in c for c in actual_cols_lower)
        has_product = any("product"  in c for c in actual_cols_lower)
        has_discount = any("discount" in c for c in actual_cols_lower)
        if has_order and has_discount and not has_product:
            scores["ecommerce"] = scores.get("ecommerce", 0) + 0.2

        # Re-pick winner after tie-breaking adjustments
        best_domain = max(scores, key=scores.get)
        best_score  = scores[best_domain]

        # Log for debugging
        print(f"[DOMAIN DETECTOR] Scores: {scores}")
        print(f"[DOMAIN DETECTOR] Winner: {best_domain} @ {best_score:.2f}")

        if best_score >= 0.35:
            return best_domain, best_score
        return "general", best_score

    def detect(self, df) -> dict:
        best_domain, best_score = self.detect_domain(df)
        confidence = "high" if best_score > 0.6 else "medium" if best_score >= 0.35 else "low"
        name = "Generic Dataset" if best_domain == "general" else best_domain.title()
        
        log.info(f"DOMAIN_ENGINE: Detected domain '{best_domain}' with {confidence} confidence.")
        return {
            "name": name,
            "confidence": confidence,
            "reason": f"Weighted score {best_score:.2f}",
            "id": best_domain
        }

def detect_domain(df) -> str:
    """Legacy wrapper for DomainDetector."""
    detector = DomainDetector()
    domain, _ = detector.detect_domain(df)
    return domain


# ============================================================
# 2.6 KEY DRIVER ANALYZER (Step 2)
# ============================================================

class KeyDriverAnalyzer:
    """Analyze correlations to detect key drivers of main target variables."""

    def analyze(self, df: pl.DataFrame, profile: DataProfile, domain_id: str = "general") -> dict:
        num_cols = [c for c in profile.numericals if c not in profile.identifiers]
        if len(num_cols) < 2:
            return {"drivers": [], "matrix": {}}
            
        # Get thresholds from TEMPLATES
        template = TEMPLATES.get(domain_id, TEMPLATES["general"])
        high_threshold = template.get("high_correlation_threshold", 0.70)
        secondary_threshold = template.get("secondary_threshold", 0.40)

        # Select target variable (Domain Target > Revenue > Profit > First numerical)
        target = template.get("target_metric")
        if not target or target not in df.columns:
            target = profile.revenue_col or next((c for c in num_cols if "profit" in c.lower()), num_cols[0])
        
        drivers = []
        try:
            # Simple correlation matrix
            pdf = df.select(num_cols).to_pandas()
            corr_matrix = pdf.corr()
            
            target_corrs = corr_matrix[target].abs().sort_values(ascending=False)
            
            for col, r_val in target_corrs.items():
                if col == target: continue
                
                raw_corr = corr_matrix.loc[target, col]
                if abs(r_val) >= high_threshold:
                    strength = "Strong Driver"
                    priority = "🔴"
                elif abs(r_val) >= secondary_threshold:
                    strength = "Moderate Driver"
                    priority = "🟠"
                else:
                    strength = "Weak Signal"
                    priority = "🟢"
                    
                drivers.append({
                    "column": col,
                    "target": target,
                    "r": round(raw_corr, 2),
                    "strength": strength,
                    "priority": priority,
                    "impact": "positive" if raw_corr > 0 else "negative"
                })
            
            # Step 5: DETECT NON-OBVIOUS / SURPRISING PATTERNS
            # Logic: Check if expected relationships are missing or inverted
            expected_pairs = [
                (profile.price_col, profile.revenue_col, "positive"),
                (profile.qty_col, profile.revenue_col, "positive"),
                (profile.delivery_days_col, profile.return_col, "positive")
            ]
            
            for c1, c2, exp in expected_pairs:
                if not c1 or not c2 or c1 not in num_cols or c2 not in num_cols: continue
                r = corr_matrix.loc[c1, c2]
                
                # Surprise 1: Weak relationship where strong expected
                if exp == "positive" and abs(r) < 0.2:
                    drivers.append({
                        "column": f"{c1} vs {c2}",
                        "is_surprise": True,
                        "type": "Weak Linkage",
                        "r": round(r, 2),
                        "description": f"Unexpectedly weak linkage between {c1} and {c2}. These variables usually move in tandem."
                    })
                # Surprise 2: Inverted relationship
                elif exp == "positive" and r < -0.3:
                    drivers.append({
                        "column": f"{c1} vs {c2}",
                        "is_surprise": True,
                        "type": "Inverted Logic",
                        "r": round(r, 2),
                        "description": f"Anomaly: {c1} is inversely correlated with {c2}, violating standard business logic."
                    })
                    
        except Exception:
            pass
            
        return {"target": target, "drivers": drivers[:8]}

# ============================================================
# 2.7 DECISION INTELLIGENCE SYNTHESIZER (Step 4 & 5)
# ============================================================

class DecisionIntelligenceSynthesizer:
    """Merge related signals into 2-4 high-quality strategic insights."""

    def synthesize(self, insights: list[BusinessInsight], drivers: dict, domain_id: str = "general") -> list[BusinessInsight]:
        if not insights:
            return []
            
        template = TEMPLATES.get(domain_id, TEMPLATES["general"])
            
        # 1. Detect Anomalies from Driver Analysis (Step 5)
        for d in drivers.get("drivers", []):
            if d.get("is_surprise"):
                # Build human-readable title from the column pair and r-value
                raw_col = d.get("column", "Variable A vs Variable B")
                col_parts = raw_col.split(" vs ")
                c1_label = col_parts[0].replace("_", " ").title() if col_parts else "Variable A"
                c2_label = col_parts[1].replace("_", " ").title() if len(col_parts) > 1 else "Variable B"
                r_val = d.get("r", 0)
                if d.get("type") == "Inverted Logic":
                    insight_title = f"{c1_label} Moves Inversely with {c2_label} (r={r_val:.2f})"
                else:
                    direction = "negative link" if r_val < 0 else "weak positive link"
                    insight_title = f"{c1_label} and {c2_label} Are Decoupled ({direction}, r={r_val:.2f})"

                insights.append(BusinessInsight(
                    title=insight_title,
                    description=d["description"],
                    why_it_matters="When core variables decouple, it usually indicates either a data quality issue or a fundamental breakdown in expected business logic.",
                    evidence=f"Correlation r={r_val:.2f} (expected strong positive)",
                    decision_implication="Audit the data ingestion pipeline for these two variables. If valid, investigate why standard drivers are failing to influence outcomes.",
                    recommendation="Exclude these decoupled variables from predictive models to reduce noise. Audit data ingestion for this variable pair before the next planning cycle.",
                    impact="Critical",
                    is_unexpected=True,
                    rule_type="surprise"
                ))

        # 2. Insight Compression (Merge by Topic)
        compressed = []
        topics = {
            "revenue": [
                "revenue_dominance", "worst_revenue", "profit_dominance", "margin_divergence",
                "revenue_by_region", "revenue_by_category", "revenue_by_product",
                "revenue_by_customer_gender", "revenue_by_discount",
                "top_performers_product", "top_performers_category", "top_performers_region",
            ],
            "quality": ["perfect_quality", "high_return_rate", "payment_risk", "delivery_delay_risk"],
            "discovery": ["dominance", "correlation_matrix", "domain_detection"],
            "distribution": ["skewed_distribution"],
            "discount": ["discount_impact"],
            "demographic": ["demographic_split_gender", "demographic_split_customer_gender"],
            "temporal": ["temporal_peaks"],
        }

        for name, rules in topics.items():
            topic_insights = [i for i in insights if i.rule_type in rules]
            if not topic_insights: continue

            # Pick the single highest impact/score insight for each topic
            topic_insights.sort(key=lambda x: x.score, reverse=True)
            best = topic_insights[0]

            # Enrich the best insight with context from others if they exist
            if len(topic_insights) > 1:
                others = [i.title for i in topic_insights[1:3]]
                best.evidence += f" | Also supported by secondary signals in: {', '.join(others)}."

            compressed.append(best)

        # Universal passthrough: rule_types not covered by any topic get through as-is
        # This ensures future rules are never silently dropped by the topic map.
        claimed_rule_types = {rt for rules in topics.values() for rt in rules}
        already_compressed = {id(i) for i in compressed}
        for ins in insights:
            if ins.rule_type not in claimed_rule_types and id(ins) not in already_compressed:
                compressed.append(ins)
            
        # 3. Fallback logic & Contradiction Guard
        # Check both emoji impact and string severity for symmetry
        high_impact_insights = [
            i for i in compressed 
            if "🔴" in str(i.impact) 
            or str(i.impact).lower() == "high" 
            or str(i.impact).lower() == "critical"
        ]
        
        # Also check for high-impact drivers
        high_threshold = template.get("high_correlation_threshold", 0.70)
        has_high_driver = any(abs(d.get('r', 0)) >= high_threshold for d in drivers.get('drivers', []))

        if not compressed and not has_high_driver:
            print("WARNING SYNT: No high-impact insights or drivers found, using fallback.")
            compressed.append(BusinessInsight(
                title="No Significant Insights Detected",
                description="The current analytical session did not exhibit patterns meeting the high-impact strategic threshold.",
                why_it_matters="Data homogeneity or limited variability may be preventing the isolation of distinct business drivers.",
                evidence=f"No correlations exceeded the {high_threshold} domain threshold.",
                decision_implication="Collect more granular data or refine the analytical focus variables to isolate deeper trends.",
                impact="🟢 Minor",
                rule_type="fallback"
            ))
        elif high_impact_insights or has_high_driver:
            # If we have high impact insights or drivers, ensure no fallback exists
            # This "suppress if high-impact" rule is essential for clean narrative
            compressed = [i for i in compressed if i.rule_type != "fallback"]
            
        # Always include temporal_peaks if it fired
        temporal_insights = [i for i in insights if getattr(i, 'rule_type', '') == 'temporal_peaks']
        already_included = any(getattr(i, 'rule_type', '') == 'temporal_peaks' for i in compressed)
        if temporal_insights and not already_included:
            compressed.insert(2, temporal_insights[0])
        return compressed[:4]

# ============================================================

# 3. METRIC COMPUTER
# ============================================================

class MetricComputer:
    """Compute derived business metrics from classified columns."""

    def compute(self, df: pl.DataFrame, profile: DataProfile) -> dict[str, ComputedMetric]:
        metrics: dict[str, ComputedMetric] = {}

        # ── Revenue / value summary ──────────────────────────────────────
        revenue_series = self._compute_revenue_series(df, profile)
        # Only call it "Revenue" when we have evidence of transactions
        # (explicit revenue column OR price * quantity). If we only have
        # a price column, it's a catalog — call it differently.
        is_true_revenue = (profile.revenue_col is not None
                           or (profile.price_col is not None and profile.qty_col is not None))
        if revenue_series is not None and is_true_revenue:
            total_rev = float(revenue_series.sum())
            avg_rev   = float(revenue_series.mean())
            metrics["total_revenue"] = ComputedMetric(
                name="Total Revenue",
                value=total_rev,
                formatted=_fmt_currency(total_rev),
                description=f"Sum of all transaction values across {len(df):,} records"
            )
            metrics["avg_order_value"] = ComputedMetric(
                name="Average Order Value",
                value=avg_rev,
                formatted=_fmt_currency(avg_rev),
                description="Revenue per transaction"
            )
        elif revenue_series is not None and profile.price_col:
            # Catalog data: use same keys as transactional path so the
            # frontend KPI serialization stays consistent. Only the labels
            # and descriptions differ to be honest about the data shape.
            total_val = float(revenue_series.sum())
            avg_val   = float(revenue_series.mean())
            price_label = profile.price_col.replace("_", " ").title()
            metrics["total_revenue"] = ComputedMetric(
                name=f"Total {price_label}",
                value=total_val,
                formatted=_fmt_currency(total_val),
                description=f"Sum of {profile.price_col} across {len(df):,} records"
            )
            metrics["avg_order_value"] = ComputedMetric(
                name=f"Average {price_label}",
                value=avg_val,
                formatted=_fmt_currency(avg_val),
                description=f"Average {profile.price_col} per record"
            )
            # Attach the series to the profile for downstream use
            profile._revenue_series = revenue_series  # type: ignore[attr-defined]
        else:
            profile._revenue_series = None  # type: ignore[attr-defined]

        # ── Return Rate ───────────────────────────────────────────────────
        if profile.return_col:
            try:
                ret_series = df[profile.return_col]
                # Handle Yes/No, 1/0, True/False
                if ret_series.dtype == pl.Boolean:
                    return_count = int(ret_series.filter(ret_series).len())
                elif ret_series.dtype == pl.Utf8:
                    positives = ["yes", "true", "1", "returned", "refund"]
                    return_count = int(
                        ret_series.drop_nulls()
                        .str.strip_chars()
                        .str.to_lowercase()
                        .is_in(positives)
                        .sum()
                    )
                else:
                    return_count = int((ret_series.drop_nulls() > 0).sum())
                total = len(df)
                rate = return_count / max(total, 1) * 100
                metrics["return_rate"] = ComputedMetric(
                    name="Overall Return Rate",
                    value=rate,
                    formatted=f"{rate:.1f}%",
                    description=f"{return_count:,} returns out of {total:,} orders"
                )
                profile._return_count_series = self._get_binary_flag(df, profile.return_col)  # type: ignore[attr-defined]
            except Exception:
                profile._return_count_series = None  # type: ignore[attr-defined]
        else:
            profile._return_count_series = None  # type: ignore[attr-defined]

        # ── Total Orders ──────────────────────────────────────────────────
        metrics["Records"] = ComputedMetric(
            name="Total Records",
            value=float(len(df)),
            formatted=f"{len(df):,}",
            description="Total number of records in the dataset"
        )

        return metrics

    # ------------------------------------------------------------------ #
    def _compute_revenue_series(
        self, df: pl.DataFrame, profile: DataProfile
    ) -> Optional[pl.Series]:
        # If an explicit revenue column exists
        if profile.revenue_col:
            return df[profile.revenue_col].drop_nulls()

        # Compute Price × Quantity
        if profile.price_col and profile.qty_col:
            try:
                return (df[profile.price_col] * df[profile.qty_col]).drop_nulls()
            except Exception:
                pass

        # Fallback: just the price column
        if profile.price_col:
            return df[profile.price_col].drop_nulls()

        return None

    def _get_binary_flag(self, df: pl.DataFrame, col: str) -> pl.Series:
        series = df[col]
        if series.dtype == pl.Boolean:
            return series.cast(pl.Int8).fill_null(0)
        if series.dtype == pl.Utf8:
            positives = ["yes", "true", "1", "returned", "refund"]
            return (
                series.str.strip_chars()
                .str.to_lowercase()
                .is_in(positives)
                .cast(pl.Int8)
                .fill_null(0)
            )
        return (series > 0).cast(pl.Int8).fill_null(0)


# ============================================================
# 4. BUSINESS RULE ENGINE
# ============================================================

class BusinessRuleEngine:
    """Evaluate threshold-based business rules and emit structured insights."""

    REVENUE_CONCENTRATION_THRESHOLD = 0.35   # >35% revenue from one category → risk
    HIGH_RETURN_RATE_MULTIPLIER     = 1.5    # cat return rate > 1.5× global → issue
    CORRELATION_RISK_THRESHOLD      = 0.4    # |corr(delivery, returns)| > 0.4 → risk
    DOMINANCE_THRESHOLD             = 0.35   # one value > 35% → flag for specific categories

    @staticmethod
    def _smart_plural(word: str) -> str:
        """Naive English pluralizer for column-name labels."""
        w = word.strip()
        if not w:
            return w
        lower = w.lower()
        if lower.endswith('y') and lower[-2] not in 'aeiou':
            return w[:-1] + 'ies'
        if lower.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return w + 'es'
        return w + 's'

    # ── Tautology Detector ──────────────────────────────────────────────
    def is_derived_column(self, df: pl.DataFrame, col_a: str, col_b: str) -> bool:
        """
        Detects if col_b is a mathematical transform of col_a (or vice versa).
        Returns True if derivation is detected — the correlation should be suppressed.

        Checks for:
        1. col_b == col_a * other_numeric_col   (e.g. Revenue = Price * Quantity)
        2. col_b == col_a + other_numeric_col
        3. col_b == col_a / other_numeric_col
        4. Identical values (col_b is a copy or rename of col_a)
        """
        try:
            a = df[col_a].to_numpy().astype(float)
            b = df[col_b].to_numpy().astype(float)

            # Skip if either column has nulls or is constant
            if np.isnan(a).any() or np.isnan(b).any():
                return False
            if np.std(a) == 0 or np.std(b) == 0:
                return False

            # Check 1: identical
            if np.allclose(a, b, rtol=1e-3):
                return True

            # Check 2: b = a * other_col for some other numeric column
            numeric_cols = [c for c in df.columns
                           if df[c].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
                           and c not in (col_a, col_b)]

            for other in numeric_cols:
                o = df[other].to_numpy().astype(float)
                if np.isnan(o).any() or np.std(o) == 0:
                    continue

                # Test multiplication: a * o ≈ b
                with np.errstate(divide='ignore', invalid='ignore'):
                    product = a * o
                    if np.std(product) > 0:
                        corr = np.corrcoef(product, b)[0, 1]
                        if corr > 0.99:
                            print(f"[TAUTOLOGY DETECTED] {col_b} ≈ {col_a} × {other}")
                            return True

                    # Test division: a / o ≈ b
                    ratio = np.divide(a, o, out=np.zeros_like(a), where=o != 0)
                    if np.std(ratio) > 0:
                        corr = np.corrcoef(ratio, b)[0, 1]
                        if corr > 0.99:
                            print(f"[TAUTOLOGY DETECTED] {col_b} ≈ {col_a} / {other}")
                            return True

            return False

        except Exception as e:
            print(f"[is_derived_column] Error checking {col_a}/{col_b}: {e}")
            return False

    def _compute_confidence(self, df: pl.DataFrame) -> tuple[str, float, str]:
        """Return confidence label, weight, and strict formatting text."""
        rows = len(df)
        if rows > 500:
            return "high", 1.0, f"Based on a high sample size (>500 records)"
        elif rows >= 100:
            return "medium", 0.7, f"Based on {rows} total records"
        else:
            return "low", 0.4, f"Based on exactly {rows} orders in the entire dataset"

    def generate_insights(
        self,
        df: pl.DataFrame,
        profile: DataProfile,
        metrics: dict[str, ComputedMetric],
        domain: str = "general"
    ) -> tuple[list[BusinessInsight], list[str]]:
        """Return (insights, warnings)."""
        self._self_diagnostic()
        
        print(f"\n{'='*60}")
        print(f"[INSIGHT ENGINE] Domain: {domain} | Shape: {df.shape}")
        print(f"{'='*60}\n")

        all_insights: list[BusinessInsight] = []
        warnings: list[str] = []
        pdf = df.to_pandas()

        # ── EXISTING rules ────────────────────────────────────────────────
        rev_series = getattr(profile, "_revenue_series", None)
        if rev_series is not None and profile.category_col:
            all_insights.extend(self._rule_revenue_by_category(df, pdf, profile, rev_series))

        ret_series = getattr(profile, "_return_count_series", None)
        if ret_series is not None and profile.category_col and "return_rate" in metrics:
            global_rate = metrics["return_rate"].value
            all_insights.extend(self._rule_return_rate_by_category(df, pdf, profile, ret_series, global_rate))

        if ret_series is not None:
            all_insights.extend(self._rule_high_return_rate_alert(df, profile, ret_series))
            
        if profile.category_col and ret_series is not None:
            all_insights.extend(self._rule_payment_return_correlation(df, pdf, profile, ret_series))

        if len(profile.numericals) >= 2:
            all_insights.extend(self._rule_strong_correlation_insight(df, profile))

        all_insights.extend(self._rule_outlier_alert(df, profile))

        # ── ✅ NEW SEGMENT RULES (Fix 3) ──────────────────────────────────
        all_insights.extend(self._rule_revenue_by_segment(df, domain))
        all_insights.extend(self._rule_top_performers(df, domain))
        all_insights.extend(self._rule_skewed_distribution_alert(df, domain))
        all_insights.extend(self._rule_discount_impact(df, domain))
        all_insights.extend(self._rule_demographic_split(df, domain))
        all_insights.extend(self._rule_temporal_peaks(df))

        # ── Post-Processing ──────────────────────────────────────────────
        all_insights = self._deduplicate(all_insights)
        all_insights = self._inject_contradictions(all_insights)

        print(f"\n[INSIGHT ENGINE] FINAL: {len(all_insights)} insights\n")
        return all_insights, warnings

    def evaluate(self, df, profile, metrics) -> tuple[list[BusinessInsight], list[str]]:
        """Backwards compatibility wrapper for run_insight_engine."""
        domain = getattr(profile, "domain_id", "general")
        return self.generate_insights(df, profile, metrics, domain=domain)

    # ------------------------------------------------------------------ #
    # Rule implementations
    # ------------------------------------------------------------------ #

    @log_rule
    def _rule_revenue_by_category(
        self,
        df: pl.DataFrame,
        pdf: pd.DataFrame,
        profile: DataProfile,
        rev_series: pl.Series,
    ) -> list[BusinessInsight]:
        insights = []
        cat = profile.category_col
        try:
            # Attach revenue series to df temporarily
            rev_col_name = "Revenue (₹)"
            price_col = profile.price_col or profile.revenue_col
            qty_col   = profile.qty_col

            if price_col and qty_col:
                pdf_tmp = pdf.copy()
                pdf_tmp[rev_col_name] = pdf[price_col].fillna(0) * pdf[qty_col].fillna(0)
            elif price_col:
                pdf_tmp = pdf.copy()
                pdf_tmp[rev_col_name] = pdf[price_col].fillna(0)
            else:
                return insights

            grouped = (
                pdf_tmp.groupby(cat)[rev_col_name].sum()
                .reset_index()
                .sort_values(rev_col_name, ascending=False)
            )
            total_rev = grouped[rev_col_name].sum()
            if total_rev == 0:
                return insights

            top = grouped.iloc[0]
            top_pct = top[rev_col_name] / total_rev * 100
            top_name = str(top[cat])
            top_val  = top[rev_col_name]

            if top_pct > self.REVENUE_CONCENTRATION_THRESHOLD * 100:
                high_concentration = grouped[grouped[rev_col_name] / total_rev > self.REVENUE_CONCENTRATION_THRESHOLD]
                qual_str = ", ".join([f"{str(row[cat])} ({_fmt_currency(row[rev_col_name])}, {row[rev_col_name]/total_rev*100:.0f}%)" for _, row in high_concentration.iterrows()])
                non_qual = grouped[grouped[rev_col_name] / total_rev <= self.REVENUE_CONCENTRATION_THRESHOLD]
                excl_str = "None"
                if not non_qual.empty:
                    excl_str = ", ".join([f"{str(row[cat])} ({_fmt_currency(row[rev_col_name])}, {row[rev_col_name]/total_rev*100:.0f}%)" for _, row in non_qual.head(3).iterrows()]) + (" and others" if len(non_qual) > 3 else "")

                insights.append(BusinessInsight(
                    title=f"Strategic Revenue Concentration: {top_name}",
                    description=f"{top_name} effectively controls {top_pct:.0f}% of total portfolio revenue, indicating high market dominance but severe systemic risk.",
                    why_it_matters="Over-concentration in a single segment leaves the bottom line vulnerable to niche market shifts or supply chain disruptions.",
                    evidence=f"Concentration Index: {top_pct:.1f}% | Top Performing Segment: {top_name} ({_fmt_currency(top_val)})",
                    decision_implication="Execute an immediate diversification strategy. Reallocate 15-20% of marketing spend towards growing secondary segments to mitigate single-source failure risk.",
                    impact="🔴 Critical",
                    recommendation=f"Prioritize growth in under-indexed segments like {excl_str.split(' ')[0]}.",
                    rule_type="revenue_dominance"
                ))
            else:
                # Check for moderate concentration (25–35%)
                bottom = grouped.iloc[-1]
                
                if top_pct > 25:
                    dist_title = f"Emerging Market Leader: {top_name}"
                    dist_desc = f"{top_name} is gaining healthy momentum with {top_pct:.0f}% share. Positive growth indicators detected."
                    dist_why = "A single leader at this level indicates a successful product-market fit but requires monitoring to prevent future dependency risk."
                    dist_evidence = f"Lead segment share: {top_pct:.0f}% | Dataset: {len(df):,} rows"
                    dist_dec = f"Nurture {top_name} to maintain leadership while beginning to seed growth in {str(bottom[cat])} to ensure balanced portfolio evolution."
                    dist_impact = "🟠 Important"
                else:
                    dist_title = f"Balanced Portfolio Distribution: {cat}"
                    dist_desc = f"Revenue is efficiently distributed across {cat} segments, maximizing operational stability."
                    dist_why = "A diversified portfolio is the gold standard for risk mitigation and suggests broad market appeal."
                    dist_evidence = f"Max segment skew: {top_pct:.0f}% | {len(grouped)} active segments."
                    dist_dec = "Maintain current allocation. Leverage the stability of this portfolio to experiment with high-margin niche segments."
                    dist_impact = "🟢 Minor"

                insights.append(BusinessInsight(
                    title=dist_title,
                    description=dist_desc,
                    why_it_matters=dist_why,
                    evidence=dist_evidence,
                    decision_implication=dist_dec,
                    impact=dist_impact,
                    recommendation=f"Protect market share for {top_name}. Investigate leakage in {str(bottom[cat])}.",
                    rule_type="worst_revenue"
                ))
        except Exception:
            pass
        return insights

    @log_rule
    def _rule_return_rate_by_category(
        self,
        df: pl.DataFrame,
        pdf: pd.DataFrame,
        profile: DataProfile,
        ret_series: pl.Series,
        global_rate: float,
    ) -> list[BusinessInsight]:
        insights = []
        cat = profile.category_col
        ret_col = profile.return_col
        try:
            # Revert to raw count aggregation to ensure absolute correctness per segment
            # Compute total orders and total returns per group
            ret_flag = self._get_binary_flag_pd(pdf[ret_col])
            
            # Using groupby and agg to get mathematically exact figures per entity without global filters
            agg_df = pdf.copy()
            agg_df["_ret_flag"] = ret_flag
            grouped = agg_df.groupby(cat).agg(
                total_orders=(cat, "count"),
                returned_count=("_ret_flag", "sum")
            ).reset_index()
            
            # Mathematical evaluation
            grouped["return_rate"] = grouped["returned_count"] / grouped["total_orders"]
            grouped = grouped.sort_values(["return_rate", "total_orders"], ascending=[True, False])
            
            # Use Human Readable columns for charts!
            grouped_for_chart = grouped.rename(columns={
                "returned_count": "Return Count",
                "total_orders": "Order Count",
                "return_rate": "Rate (%)"
            })
            
            chart_data = {
                "labels": grouped_for_chart[cat].tolist(),
                "values": [round(v * 100, 1) for v in grouped_for_chart["Rate (%)"].tolist()],
                "title": f"Return Rate by {cat} (%)",
                "y_label": "Rate (%)",
                "reference_line": round(global_rate, 1)
            }

            # 1. 0% Return logic (Perfect Quality)
            perfect_entities = grouped[grouped["returned_count"] == 0]
            if not perfect_entities.empty:
                # Format: List items ascending (already sorted by rate ascending, then volume)
                qual_str = ", ".join([f"{row[cat]} ({row['returned_count']}/{row['total_orders']} returns, 0%)" for _, row in perfect_entities.iterrows()])
                non_perfect = grouped[grouped["returned_count"] > 0]
                excl_str = "None"
                if not non_perfect.empty:
                    excl_str = ", ".join([f"{row[cat]} ({row['returned_count']}/{row['total_orders']}, {row['return_rate']*100:.1f}%)" for _, row in non_perfect.head(3).iterrows()])
                    if len(non_perfect) > 3:
                        excl_str += " and others"

                insights.append(BusinessInsight(
                    title=f"Segment Quality Excellence: {perfect_entities.iloc[0][cat]}",
                    description="Highest fidelity quality standards detected in specific strategic segments, resulting in a zero-defect (0% return) record.",
                    why_it_matters="Segments with zero returns represent a 'Gold Standard' operational blueprint. Analyzing their success provides a roadmap for reducing costs across the broader catalog.",
                    evidence=f"Zero-return segments: {perfect_entities.iloc[0][cat]} | Dataset: {len(df):,} records.",
                    decision_implication="Conduct an internal audit of the fulfillment and QA protocols for these perfect segments. Scale these exact processes to high-risk categories to reduce global return rates.",
                    impact="🟢 Minor",
                    recommendation="Replicate the fulfillment workflow of these segments across the catalog.",
                    rule_type="perfect_quality"
                ))

            # 2. High Return Rate logic (Fix 5: Added Rate > 1.5x AND Count > 5)
            high_entities = grouped[
                (grouped["return_rate"] > (global_rate / 100.0) * self.HIGH_RETURN_RATE_MULTIPLIER) & 
                (grouped["returned_count"] > 5)
            ]
            # Ensure return rate > 0
            high_entities = high_entities.sort_values("return_rate", ascending=False)
            
            if not high_entities.empty:
                qual_str = ", ".join([f"{row[cat]} ({row['returned_count']}/{row['total_orders']} returns, {row['return_rate']*100:.1f}%)" for _, row in high_entities.head(3).iterrows()])
                if len(high_entities) > 3:
                    qual_str += " and others"
                    
                normal_entities = grouped[(grouped["return_rate"] <= (global_rate / 100.0) * self.HIGH_RETURN_RATE_MULTIPLIER) & (grouped["total_orders"] > 0)]
                excl_str = "None"
                if not normal_entities.empty:
                    excl_str = ", ".join([f"{row[cat]} ({row['returned_count']}/{row['total_orders']}, {row['return_rate']*100:.1f}%)" for _, row in normal_entities.head(3).iterrows()])

                insights.append(BusinessInsight(
                    title=f"Critical Quality Degradation in {cat}",
                    description=f"Specific segments within {cat} are experiencing outsized return rates, severely exceeding the global average of {global_rate:.1f}%.",
                    why_it_matters="High return rates are a direct indicator of customer dissatisfaction, manufacturing defects, or misleading product descriptions. This creates a severe drag on net margins.",
                    evidence=f"Highest risk segments: {qual_str} | Variance Level: >1.5x global average.",
                    decision_implication="Immediately suspend marketing for the highest-return SKUs and update product descriptions to better align expectations. Investigate the logistics chain for possible damage during transit.",
                    impact="🔴 Critical",
                    recommendation="Audit product descriptions and fulfillment quality for the flagged segments.",
                    rule_type="high_return_rate"
                ))

        except Exception:
            pass
        return insights

    @log_rule
    def _rule_delivery_vs_returns(
        self,
        df: pl.DataFrame,
        pdf: pd.DataFrame,
        profile: DataProfile,
        ret_series: pl.Series,
    ) -> list[BusinessInsight]:
        insights = []
        try:
            del_col = profile.delivery_days_col
            ret_col = profile.return_col
            sub = pdf[[del_col, ret_col]].dropna()
            if len(sub) < 20:
                return insights
            ret_flag = self._get_binary_flag_pd(sub[ret_col])
            corr = np.corrcoef(sub[del_col].astype(float), ret_flag)[0, 1]
            if abs(corr) > 0.4:
                direction = "positive" if corr > 0 else "negative"
                insights.append(BusinessInsight(
                    title="Operational Risk: Delivery-Induced Returns",
                    description="A significant tactical relationship detected where delivery latency is directly driving higher return velocities.",
                    why_it_matters="When delivery times exceed customer expectations, the probability of 'buyer's remorse' or competitive substitution increases, leading to a direct loss in net revenue.",
                    evidence=f"r-value: {corr:.2f} | Strength: Moderate-to-High | Correlation Direction: {direction}.",
                    decision_implication="Negotiate stricter delivery SLAs with logistics partners for high-volume routes. Introduce 'Express' options for segments with the highest sensitivity to latency.",
                    impact="🔴 Critical",
                    recommendation="Reduce delivery latency to protect transaction integrity.",
                    rule_type="delivery_delay_risk"
                ))
        except Exception:
            pass
        return insights

    @log_rule
    def _rule_categorical_dominance(
        self, df: pl.DataFrame, profile: DataProfile
    ) -> list[BusinessInsight]:
        insights = []
        for col in profile.categoricals:
            if col == profile.category_col:
                continue  # Already analysed above
            try:
                counts = df[col].value_counts()
                top_val = counts.head(1)
                top_name = top_val[col].item()
                top_count = top_val["count"].item()
                pct = top_count / len(df)
                
                # Check payment methods against a lower dominance threshold (35%)
                cl_lower = col.lower()
                is_payment_related = any(k in cl_lower for k in {"payment", "method", "channel"})
                threshold = self.DOMINANCE_THRESHOLD if is_payment_related else 0.70
                
                if pct > threshold:
                    counts_filtered = counts[counts["count"] > len(df) * threshold]
                    qual_str = ", ".join([f"{str(row[col])} ({row['count']} rows, {row['count']/len(df)*100:.0f}%)" for _, row in counts_filtered.iterrows()])
                    non_qual = counts[counts["count"] <= len(df) * threshold]
                    excl_str = "None"
                    if not non_qual.empty:
                        excl_str = ", ".join([f"{str(row[col])} ({row['count']} rows, {row['count']/len(df)*100:.0f}%)" for _, row in non_qual.head(3).iterrows()]) + (" and others" if len(non_qual) > 3 else "")

                    if is_payment_related:
                        title = f"Dominant Payment Method: {top_name}"
                        desc = f"{top_name} accounts for {pct*100:.0f}% of all {col} records."
                        why = f"Heavy reliance on a single payment method creates systemic vulnerability and checkout friction for users preferring other options."
                        rec = f"Optimize the checkout flow specifically for {top_name}. Simultaneously, introduce and promote alternative {col} options to reduce dependency risk."
                    else:
                        title = f"{top_name} Dominates {col}"
                        desc = f"{top_name} makes up {pct*100:.0f}% of the volume in {col}."
                        why = f"Over-concentration in one categorical attribute exposes the business to single points of failure."
                        rec = f"Diversify offerings in {col} to broaden appeal and mitigate reliance on {top_name}."

                    insights.append(BusinessInsight(
                        title=title,
                        description=desc,
                        why_it_matters=why,
                        evidence=f"Concentration: {pct*100:.1f}% | Dominant Value: {top_name}.",
                        decision_implication="Diversify the categorical portfolio. Reducing dominance from >35% to ~20% will significantly improve systemic resilience against segment-specific shocks.",
                        impact="🟠 Important",
                        recommendation=rec,
                        rule_type="dominance"
                    ))
            except Exception:
                pass
        return insights

    @log_rule
    def _rule_top_geographic_performance(
        self,
        df: pl.DataFrame,
        pdf: pd.DataFrame,
        profile: DataProfile,
        rev_series: pl.Series,
    ) -> list[BusinessInsight]:
        insights = []
        geo_col = profile.geographic_col
        if not geo_col:
            return insights
        price_col = profile.price_col or profile.revenue_col
        qty_col   = profile.qty_col
        try:
            pdf_tmp = pdf.copy()
            if price_col and qty_col:
                pdf_tmp["Revenue (₹)"] = pdf[price_col].fillna(0) * pdf[qty_col].fillna(0)
            elif price_col:
                pdf_tmp["Revenue (₹)"] = pdf[price_col].fillna(0)
            else:
                return insights
            grouped = (
                pdf_tmp.groupby(geo_col)["Revenue (₹)"].sum()
                .reset_index()
                .sort_values("Revenue (₹)", ascending=False)
            )
            if len(grouped) < 2:
                return insights
            top = grouped.iloc[0]
            bottom = grouped.iloc[-1]
            total = grouped["Revenue (₹)"].sum()
            qual_str = f"{str(top[geo_col])} ({_fmt_currency(top['Revenue (₹)'])} , {top['Revenue (₹)']/total*100:.0f}%)"
            excl_str = f"{str(bottom[geo_col])} ({_fmt_currency(bottom['Revenue (₹)'])} , {bottom['Revenue (₹)']/total*100:.0f}%)"

            insights.append(BusinessInsight(
                title=f"Geographic Revenue Skew: {top[geo_col]}",
                description=f"Significant regional performance variance detected, with {top[geo_col]} driving disproportionate revenue compared to {bottom[geo_col]}.",
                why_it_matters="Unbalanced regional performance suggests untapped potential in laggard regions or an over-reliance on a single geographic market.",
                evidence=f"Total Regional Revenue: {_fmt_currency(total)} | Leader Share: {top_pct:.1f}%",
                decision_implication="Launch targeted localized growth campaigns in bottom-performing regions. Simultaneously, optimize logistics hubs in {top[geo_col]} to maintain dominance.",
                impact="🟠 Important",
                recommendation=f"Address regional imbalances via localized strategic initiatives.",
                rule_type="worst_revenue"
            ))
        except Exception:
            pass
        return insights



    @log_rule
    def _rule_numeric_correlations(
        self, df: pl.DataFrame, profile: DataProfile
    ) -> list[BusinessInsight]:
        insights = []
        cols = [c for c in profile.numericals if c not in profile.identifiers][:5]
        if len(cols) < 2:
            return insights
        try:
            seen_pairs: set[tuple] = set()
            for i, c1 in enumerate(cols):
                for c2 in cols[i + 1:]:
                    # Symmetric dedup
                    pair_key = tuple(sorted([c1, c2]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    corr = df.select(pl.corr(c1, c2)).item()
                    if corr is None:
                        continue
                    if abs(corr) < 0.7:
                        continue

                    # ⛔ TAUTOLOGY GUARD — suppress derived-column correlations
                    if self.is_derived_column(df, c1, c2):
                        print(f"[SUPPRESSED] {c1}↔{c2} is a derived relationship, not an insight")
                        continue

                    direction = "increases" if corr > 0 else "decreases"
                    impact = "Critical" if abs(corr) >= 0.85 else "Important"
                    insights.append(BusinessInsight(
                        title=f"Systemic Linkage: {c1} & {c2}",
                        description=f"Strong decision-relevant linkage detected where {c1} acts as a reliable predictor for {c2} behavior.",
                        why_it_matters="Predictive accuracy is highest when variables are strongly coupled. This linkage should be the foundation of any forecasting models.",
                        evidence=f"Correlation coefficient: {corr:.2f} | Strength: {'High' if abs(corr) >= 0.7 else 'Moderate'}",
                        decision_implication=f"Incorporate both {c1} and {c2} as primary features in all future predictive modeling efforts. Use {c1} as a leading indicator for {c2} performance.",
                        impact=impact,
                        recommendation=f"Foundation for predictive modeling identified via {c1}/{c2} linkage.",
                        rule_type="correlation_matrix"
                    ))
        except Exception:
            pass
        return insights

    def _find_profit_col(self, profile: DataProfile) -> str | None:
        """Find the profit column using heuristics."""
        for col in profile.numericals:
            if "profit" in col.lower() or "margin" in col.lower() or "earnings" in col.lower():
                return col
        return None

    @log_rule
    def _rule_profit_by_category(
        self, df: pl.DataFrame, pdf: pd.DataFrame, profile: DataProfile, profit_col: str
    ) -> list[BusinessInsight]:
        insights = []
        cat = profile.category_col
        try:
            grouped = pdf.groupby(cat)[profit_col].sum().sort_values(ascending=False).reset_index()
            total_profit = grouped[profit_col].sum()
            
            # Use same DOMINANCE_THRESHOLD logic on profit
            if total_profit > 0:
                top_name = str(grouped.iloc[0][cat])
                top_val = grouped.iloc[0][profit_col]
                top_pct = (top_val / total_profit) * 100
                
                if top_pct > (self.DOMINANCE_THRESHOLD * 100):
                    qual_str = f"{top_name} ({_fmt_currency(top_val)}, {top_pct:.0f}%)"
                    bottom = grouped.iloc[-1]
                    excl_str = f"{str(bottom[cat])} ({_fmt_currency(bottom[profit_col])}, {bottom[profit_col]/total_profit*100:.0f}%)"
                    insights.append(BusinessInsight(
                        title=f"Profit Channel Risk: {top_name}",
                        description=f"{top_name} generates {top_pct:.0f}% of total profit, creating a high-sensitivity exposure profile for the entire organization.",
                        why_it_matters="Profit concentration is more critical than revenue concentration. A minor margin compression in {top_name} would have a non-linear negative impact on total earnings.",
                        evidence=f"Profit Concentration: {top_pct:.1f}% | Total Portfolio Earnings: {_fmt_currency(total_profit)}.",
                        decision_implication="Diversify high-margin product offerings. Perform a stress test on the cost structure of {top_name} to identify where margins can be fortified.",
                        impact="🔴 Critical",
                        recommendation="Diversify high-margin streams to insulate the organization from segment-specific shocks.",
                        rule_type="profit_dominance"
                    ))
        except Exception:
            pass
        return insights

    @log_rule
    def _rule_profit_margin_insight(
        self, df: pl.DataFrame, pdf: pd.DataFrame, profile: DataProfile, 
        profit_col: str, rev_col: str
    ) -> list[BusinessInsight]:
        insights = []
        cat = profile.category_col
        try:
            grouped = pdf.groupby(cat).agg({profit_col: 'sum', rev_col: 'sum'}).reset_index()
            grouped['margin'] = (grouped[profit_col] / grouped[rev_col]) * 100
            
            # Find divergence: high revenue but low margin vs low revenue but high margin
            sorted_by_rev = grouped.sort_values(rev_col, ascending=False)
            highest_rev_cat = sorted_by_rev.iloc[0]
            highest_margin_cat = grouped.sort_values('margin', ascending=False).iloc[0]
            
            if highest_rev_cat[cat] != highest_margin_cat[cat] and highest_margin_cat['margin'] > highest_rev_cat['margin'] * 1.5:
                # The highest margin category is different from the highest revenue, and its margin is 1.5x better
                h_rev_name = highest_rev_cat[cat]
                h_mar_name = highest_margin_cat[cat]
                
                insights.append(BusinessInsight(
                    title="Margin Divergence Anomaly Detected",
                    description=f"{h_rev_name} leads revenue volume, but {h_mar_name} holds significantly higher quality profit margins ({highest_margin_cat['margin']:.1f}%).",
                    why_it_matters="An efficiency gap exists. Revenue growth is being pursued in segments where profitability is lower, indicating a possible misallocation of resources.",
                    evidence=f"Margin Alpha for {h_mar_name}: {highest_margin_cat['margin'] - highest_rev_cat['margin']:.1f}% over high-volume leader.",
                    decision_implication="Pivot marketing focus and capital allocation towards the higher-margin {h_mar_name} segment. Reassess the cost-to-serve for the high-volume {h_rev_name}.",
                    impact="🟠 Important",
                    recommendation=f"Align volume growth with high-margin profitability for maximum ROI.",
                    rule_type="margin_divergence"
                ))
        except Exception:
            pass
        return insights

    @log_rule
    def _rule_domain_detection(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """Detect if the dataset belongs to a specific industry domain using centralized logic."""
        domain_id = detect_domain(df)
        domain_name = domain_id.replace("_", " ").title()
        if domain_id == "general":
            domain_name = "General Business"
            
        return [BusinessInsight(
            title=f"Domain Intelligence Detected: {domain_name}",
            description=f"InsightStream has identified this dataset as {domain_name} data based on specific column signatures and TEMPLATES mapping.",
            why_it_matters="Applying domain-specific heuristics allows for more accurate target variable identification and risk modeling.",
            evidence=f"Detected signatures matching '{domain_id}' patterns.",
            impact="🟢 Minor",
            recommendation="Review the Strategic Brief section for industry-aligned operational suggestions.",
            rule_type="domain_detection"
        )]

    @log_rule
    def _rule_payment_correlation(self, df: pl.DataFrame, pdf: pd.DataFrame, profile: DataProfile, ret_series: pl.Series) -> list[BusinessInsight]:
        """Check for correlations between payment methods and returns."""
        pay_col = next((c for c in profile.categoricals if any(k in c.lower() for k in {"payment", "method", "channel"})), None)
        if not pay_col or ret_series is None:
            return []
            
        try:
            pdf_tmp = pdf.copy()
            pdf_tmp["_is_returned"] = self._get_binary_flag_pd(pdf[profile.return_col])
            stats = pdf_tmp.groupby(pay_col)["_is_returned"].agg(["mean", "count"]).reset_index()
            stats = stats[stats["count"] > 20].sort_values("mean", ascending=False)
            
            if len(stats) > 1 and stats.iloc[0]["mean"] > stats["mean"].mean() * 1.5:
                top_pay = stats.iloc[0]
                return [BusinessInsight(
                    title=f"Payment Risk: {top_pay[pay_col]}",
                    description=f"Orders using {top_pay[pay_col]} have a significantly higher return rate ({top_pay['mean']*100:.1f}%) compared to other methods.",
                    impact="medium",
                    recommendation=f"Investigate if checkout friction or fraud logic for {top_pay[pay_col]} is contributing to these returns.",
                    chart_type="bar",
                    chart_data={
                        "labels": stats[pay_col].tolist(),
                        "values": [round(v*100, 1) for v in stats["mean"].tolist()],
                        "title": f"Return Rate by Payment Method (%)"
                    },
                    rule_type="payment_risk"
                )]
        except:
            pass
        return []

    @log_rule
    def _rule_correlation_matrix(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """Find the top 3 strongest numerical correlations (Fix 5).
        Tautological (derived-column) pairs are excluded."""
        num_cols = profile.numericals
        if len(num_cols) < 2: return []
        
        try:
            # We use polars for fast correlation matrix
            df_num = df.select(num_cols).drop_nulls().head(5000)
            if len(df_num) < 50: return []
            
            corrs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    correlation = df_num.select(pl.corr(c1, c2)).item()
                    if np.isnan(correlation):
                        continue

                    # ⛔ TAUTOLOGY GUARD — skip derived-column pairs
                    if abs(correlation) >= 0.7 and self.is_derived_column(df, c1, c2):
                        print(f"[SUPPRESSED MATRIX] {c1}↔{c2} is derived, excluding from top-3")
                        continue

                    corrs.append(((c1, c2), abs(correlation), correlation))
            
            corrs.sort(key=lambda x: x[1], reverse=True)
            top_3 = corrs[:3]
            
            if not top_3: return []
            
            items = [f"{c[0][0]} vs {c[0][1]} (r={c[2]:.2f})" for c in top_3]
            return [BusinessInsight(
                title="Strongest Numerical Links",
                description=f"We detected strong mathematical links between: " + " • ".join(items),
                impact="medium",
                recommendation="Use these linked variables together in your predictive modeling for better accuracy.",
                rule_type="correlation_matrix"
            )]
        except:
            pass
        return []

    # ====================================================================
    # NEW SEGMENT-LEVEL RULES (FIX 3)
    # ====================================================================

    @log_rule
    def _rule_revenue_by_segment(self, df: pl.DataFrame, domain: str) -> list[BusinessInsight]:
        """Revenue concentration analysis: top vs bottom segment by total revenue."""
        insights = []
        revenue_col = self._find_column(df, ["revenue", "sales", "amount", "total"])
        if not revenue_col:
            return insights

        segment_cols = self._find_segment_columns(df, max_cardinality=20)
        for seg_col in segment_cols:
            try:
                grouped = (
                    df.group_by(seg_col)
                    .agg([
                        pl.col(revenue_col).sum().alias("total_rev"),
                        pl.col(revenue_col).count().alias("n_records"),
                    ])
                    .sort("total_rev", descending=True)
                )
                if grouped.height < 2: continue

                rows = grouped.to_dicts()
                top, bottom = rows[0], rows[-1]
                total_rev = sum(r["total_rev"] for r in rows)
                if total_rev == 0: continue

                top_share = (top["total_rev"] / total_rev) * 100
                bottom_share = (bottom["total_rev"] / total_rev) * 100
                gap_pct = top_share - bottom_share

                if gap_pct < 5: continue

                insights.append(BusinessInsight(
                    title=f"Revenue Concentration: {seg_col}",
                    impact="🔴 Critical" if gap_pct > 15 else "🟠 Important",
                    description=(
                        f"WHAT: {top[seg_col]} leads with {self._format_inr(top['total_rev'])} "
                        f"({top_share:.1f}% of total revenue) while {bottom[seg_col]} trails at "
                        f"{self._format_inr(bottom['total_rev'])} ({bottom_share:.1f}%). "
                        f"WHY: A {gap_pct:.1f}pp revenue gap across {seg_col} segments reveals "
                        f"non-uniform performance. "
                        f"EVIDENCE: {grouped.height} segments analyzed across "
                        f"{sum(r['n_records'] for r in rows):,} records."
                    ),
                    recommendation=(
                        f"Audit operations in {bottom[seg_col]} to identify whether the gap "
                        f"is demand-driven or execution-driven. If execution: replicate the {top[seg_col]} playbook."
                    ),
                    rule_type=f"revenue_by_{seg_col.lower()}",
                    qualified_segments=[str(top[seg_col])],
                    excluded_segments=[str(bottom[seg_col])]
                ))
            except Exception: pass
        return insights

    @log_rule
    def _rule_top_performers(self, df: pl.DataFrame, domain: str) -> list[BusinessInsight]:
        """Identify top 3 performers in high-cardinality categorical columns (e.g. Product)."""
        insights = []
        revenue_col = self._find_column(df, ["revenue", "sales", "amount"])
        if not revenue_col: return insights

        candidate_cols = []
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                unique_count = df[col].n_unique()
                if 5 <= unique_count <= 50:
                    candidate_cols.append((col, unique_count))

        for col, n_unique in candidate_cols:
            try:
                grouped = (
                    df.group_by(col)
                    .agg(pl.col(revenue_col).sum().alias("total_rev"))
                    .sort("total_rev", descending=True)
                )
                rows = grouped.to_dicts()
                if len(rows) < 5: continue

                top3 = rows[:3]
                total = sum(r["total_rev"] for r in rows)
                if total == 0: continue
                top3_share = (sum(r["total_rev"] for r in top3) / total) * 100
                if top3_share < 35: continue

                top3_names = ", ".join(str(r[col]) for r in top3)
                col_label = col.replace('_', ' ').title()
                col_plural = self._smart_plural(col_label)
                insights.append(BusinessInsight(
                    title=f"Top 3 {col_plural} Drive {top3_share:.0f}% of Revenue",
                    impact="Critical" if top3_share > 60 else "Important",
                    description=(
                        f"WHAT: Out of {n_unique} {col_plural.lower()}, just 3 ({top3_names}) "
                        f"account for {top3_share:.1f}% of total revenue. "
                        f"EVIDENCE: Top 3 = {self._format_inr(sum(r['total_rev'] for r in top3))} "
                        f"out of {self._format_inr(total)} total."
                    ),
                    recommendation=f"Run a feature/promotion playbook focused on {top3_names} for next quarter.",
                    rule_type=f"top_performers_{col.lower()}",
                    qualified_segments=[str(r[col]) for r in top3]
                ))
            except Exception: pass
        return insights

    @log_rule
    def _rule_skewed_distribution_alert(self, df: pl.DataFrame, domain: str) -> list[BusinessInsight]:
        """Fires when a numeric column has mean/median ratio > 2.0 (heavy skew)."""
        insights = []
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]]
        for col in numeric_cols:
            try:
                series = df[col].drop_nulls()
                if series.len() < 30: continue
                mean_val, median_val = float(series.mean()), float(series.median())
                if median_val == 0 or mean_val == 0: continue
                ratio = mean_val / median_val
                if ratio < 2.0: continue

                insights.append(BusinessInsight(
                    title=f"{col} Distribution is Heavily Right-Skewed",
                    impact="🟠 Important",
                    description=(
                        f"WHAT: {col} has mean {self._format_inr(mean_val)} but median "
                        f"{self._format_inr(median_val)} — a {ratio:.1f}× gap. "
                        f"WHY: A small number of high-value records are pulling the average up."
                    ),
                    evidence=f"Mean/median ratio of {ratio:.2f} indicates strong right-skew.",
                    recommendation=f"Switch dashboards to display median {col} ({self._format_inr(median_val)}) instead of mean.",
                    rule_type="skewed_distribution",
                    qualified_segments=[col]
                ))
            except Exception: pass
        return insights

    @log_rule
    def _rule_discount_impact(self, df: pl.DataFrame, domain: str) -> list[BusinessInsight]:
        """Analyzes whether discounts actually drive higher revenue per record."""
        insights = []
        discount_col = self._find_column(df, ["discount", "promo", "offer"])
        revenue_col = self._find_column(df, ["revenue", "sales", "amount"])
        if not (discount_col and revenue_col): return insights

        try:
            df_with_bucket = df.with_columns(
                pl.when(pl.col(discount_col) == 0).then(pl.lit("None"))
                .when(pl.col(discount_col) <= 10).then(pl.lit("Low (1-10%)"))
                .when(pl.col(discount_col) <= 20).then(pl.lit("Medium (11-20%)"))
                .otherwise(pl.lit("High (>20%)"))
                .alias("discount_bucket")
            )
            grouped = (
                df_with_bucket.group_by("discount_bucket")
                .agg([pl.col(revenue_col).mean().alias("avg_rev"), pl.col(revenue_col).count().alias("n")])
                .sort("avg_rev", descending=True)
            )
            rows = grouped.to_dicts()
            if len(rows) < 2: return insights

            highest, lowest = rows[0], rows[-1]
            gap_pct = ((highest["avg_rev"] - lowest["avg_rev"]) / lowest["avg_rev"]) * 100 if lowest["avg_rev"] > 0 else 0
            if abs(gap_pct) < 10: return insights

            counterintuitive = "High" in highest["discount_bucket"] or "Medium" in highest["discount_bucket"]
            insights.append(BusinessInsight(
                title="Discount Tiers Show Uneven Revenue Impact",
                impact="🔴 Critical" if abs(gap_pct) > 30 else "🟠 Important",
                description=(
                    f"Average revenue per order varies by discount tier. "
                    f"'{highest['discount_bucket']}' tier averages {self._format_inr(highest['avg_rev'])} "
                    f"vs '{lowest['discount_bucket']}' at {self._format_inr(lowest['avg_rev'])} — a {gap_pct:.0f}% gap."
                ),
                recommendation="Run a controlled discount A/B test to isolate margin impact.",
                rule_type="discount_impact",
                qualified_segments=[highest["discount_bucket"]],
                excluded_segments=[lowest["discount_bucket"]]
            ))
        except Exception: pass
        return insights

    @log_rule
    def _rule_demographic_split(self, df: pl.DataFrame, domain: str) -> list[BusinessInsight]:
        """Analyzes binary demographic columns (e.g. Gender) for revenue/engagement gaps."""
        insights = []
        revenue_col = self._find_column(df, ["revenue", "sales", "amount"])
        if not revenue_col: return insights
        binary_cols = [c for c in df.columns if df[c].dtype == pl.Utf8 and df[c].n_unique() == 2]

        for col in binary_cols:
            try:
                grouped = (
                    df.group_by(col)
                    .agg([
                        pl.col(revenue_col).sum().alias("total_rev"),
                        pl.col(revenue_col).mean().alias("avg_rev"),
                        pl.col(revenue_col).count().alias("n"),
                    ])
                )
                rows = grouped.to_dicts()
                if len(rows) != 2: continue
                a, b = rows[0], rows[1]
                if a["total_rev"] < b["total_rev"]: a, b = b, a
                gap_pct = ((a["total_rev"] - b["total_rev"]) / b["total_rev"]) * 100 if b["total_rev"] > 0 else 0
                if abs(gap_pct) < 5: continue

                insights.append(BusinessInsight(
                    title=f"{col} Revenue Split: {gap_pct:.0f}% Gap",
                    impact="🟠 Important" if abs(gap_pct) < 25 else "🔴 Critical",
                    description=(
                        f"WHAT: {a[col]} segment generates {self._format_inr(a['total_rev'])} total "
                        f"({a['n']:,} orders) vs {b[col]} at {self._format_inr(b['total_rev'])} — a {gap_pct:.0f}% gap."
                    ),
                    recommendation=f"Run a paid-acquisition test targeted specifically at {b[col]} for 30 days.",
                    rule_type=f"demographic_split_{col.lower()}",
                    qualified_segments=[str(a[col])],
                    excluded_segments=[str(b[col])]
                ))
            except Exception: pass
        return insights

    # --- Helpers ---
    def _find_column(self, df: pl.DataFrame, keywords: list) -> str | None:
        """Find first column whose name contains any keyword (case-insensitive)."""
        for col in df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in keywords):
                return col
        return None

    def _find_segment_columns(self, df: pl.DataFrame, max_cardinality: int = 20) -> list:
        """Find categorical columns suitable for segmenting (low cardinality, not identifiers)."""
        segments = []
        for col in df.columns:
            if df[col].dtype != pl.Utf8: continue
            n_unique = df[col].n_unique()
            if 2 <= n_unique <= max_cardinality:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in ["id", "uuid", "key"]): continue
                segments.append(col)
        return segments

    def _format_inr(self, value: float) -> str:
        """Format number as Indian Rupee with smart scaling."""
        try: value = float(value)
        except (TypeError, ValueError): return "₹0"
        if value >= 1_00_00_000: return f"₹{value/1_00_00_000:.2f} Cr"
        if value >= 1_00_000: return f"₹{value/1_00_000:.2f} L"
        if value >= 1_000: return f"₹{value/1_000:.1f}K"
        return f"₹{value:,.0f}"

    @log_rule
    def _rule_temporal_peaks(self, df: pl.DataFrame) -> list[BusinessInsight]:
        """Detect monthly revenue peaks and troughs from date + revenue columns."""
        date_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["date", "time", "month", "period", "day"])),
            None,
        )
        rev_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["revenue", "sales", "amount", "total", "value"])),
            None,
        )
        if not date_col or not rev_col:
            return []
        try:
            # Parse date — handle both native date types and strings
            try:
                if df.schema.get(date_col) in (pl.Date, pl.Datetime):
                    df_parsed = df.with_columns(
                        pl.col(date_col).cast(pl.Date).alias("_parsed_date")
                    )
                else:
                    # Use pandas for robust parsing — handles DD/MM/YYYY, MM-DD-YYYY, mixed
                    raw_dates = df[date_col].to_pandas()
                    parsed_dates = pd.to_datetime(raw_dates, errors="coerce", dayfirst=True)
                    if parsed_dates.isna().all():
                        print(f"[temporal_peaks] all dates unparseable in {date_col}")
                        return []
                    df_parsed = df.with_columns(
                        pl.Series("_parsed_date", parsed_dates.dt.date.values).alias("_parsed_date")
                    )
            except Exception as e:
                print(f"[temporal_peaks] date parse failed: {e}")
                return []
            df_parsed = df_parsed.filter(pl.col("_parsed_date").is_not_null())
            if df_parsed.height < 30:
                return []

            monthly = (
                df_parsed
                .with_columns(pl.col("_parsed_date").dt.truncate("1mo").alias("_month"))
                .group_by("_month")
                .agg(pl.col(rev_col).cast(pl.Float64).sum().alias("monthly_rev"))
                .sort("_month")
            )
            if monthly.height < 2:
                return []

            months   = monthly["_month"].to_list()
            revenues = monthly["monthly_rev"].to_list()

            # Peak/trough on FULL data
            peak_idx   = revenues.index(max(revenues))
            trough_idx = revenues.index(min(revenues))
            peak_month   = months[peak_idx].strftime("%B")
            trough_month = months[trough_idx].strftime("%B")
            peak_val   = revenues[peak_idx]
            trough_val = revenues[trough_idx]
            pct_gap = ((peak_val - trough_val) / peak_val) * 100

            # Center the chart window on the peak month
            MAX_CHART_MONTHS = 12
            half  = MAX_CHART_MONTHS // 2
            start = max(0, peak_idx - half)
            end   = min(len(months), start + MAX_CHART_MONTHS)
            start = max(0, end - MAX_CHART_MONTHS)
            display_months   = months[start:end]
            display_revenues = revenues[start:end]

            mom_parts = [
                f"{m.strftime('%b')}={self._format_inr(r)}"
                for m, r in zip(display_months, display_revenues)
            ]
            mom_str = ("..." if len(months) > MAX_CHART_MONTHS else "") + " → ".join(mom_parts)

            # Chart uses display window only; full data kept in evidence string
            chart_monthly_data = [
                (m.strftime("%Y-%m"), r) for m, r in zip(display_months, display_revenues)
            ]

            return [BusinessInsight(
                title=f"Revenue Peaked in {peak_month}, Troughed in {trough_month}",
                description=(
                    f"Monthly revenue shows clear peaks and troughs. "
                    f"{peak_month} was the strongest month at {self._format_inr(peak_val)}, "
                    f"while {trough_month} was the weakest at {self._format_inr(trough_val)} "
                    f"({pct_gap:.0f}% gap). Trend: {mom_str}."
                ),
                why_it_matters=(
                    "Temporal concentration creates cash flow risk and signals seasonality "
                    "that should inform inventory and marketing planning."
                ),
                evidence=(
                    f"Peak: {peak_month} ({self._format_inr(peak_val)}) | "
                    f"Trough: {trough_month} ({self._format_inr(trough_val)}) | "
                    f"Gap: {pct_gap:.1f}%"
                ),
                impact="Important",
                recommendation=(
                    f"Investigate the {trough_month} dip — determine if it is seasonal, "
                    f"promotional, or operational. Pre-position inventory and marketing "
                    f"spend ahead of {peak_month} next cycle."
                ),
                rule_type="temporal_peaks",
                score=7.5,
                chart_data={
                    "monthly_data": chart_monthly_data,
                    "peak_month": peak_month,
                    "peak_val": peak_val,
                    "trough_month": trough_month,
                    "trough_val": trough_val,
                    "pct_gap": round(pct_gap, 1),
                },
            )]
        except Exception as e:
            print(f"[temporal_peaks] error: {e}")
            return []

    @log_rule
    def _rule_high_return_rate_alert(self, df: pl.DataFrame, profile: DataProfile, ret_series: pl.Series) -> list[BusinessInsight]:
        """Fires when overall return rate exceeds 15%."""
        if ret_series is None: return []
        rate = ret_series.mean()
        if rate < 0.15: return []
        return [BusinessInsight(
            title="High Systemic Return Rate",
            impact="🔴 Critical",
            description=f"The dataset shows an overall return rate of {rate*100:.1f}%, which is above the 15% healthy threshold.",
            recommendation="Immediate audit of product quality and return reasons is required to preserve margins.",
            rule_type="high_return_rate"
        )]

    @log_rule
    def _rule_payment_return_correlation(self, df: pl.DataFrame, pdf: pd.DataFrame, profile: DataProfile, ret_series: pl.Series) -> list[BusinessInsight]:
        """Mapping to _rule_payment_correlation."""
        return self._rule_payment_correlation(df, pdf, profile, ret_series)

    @log_rule
    def _rule_strong_correlation_insight(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """Mapping to _rule_numeric_correlations."""
        return self._rule_numeric_correlations(df, profile)

    @log_rule
    def _rule_outlier_alert(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """Detect extreme outliers in numerical columns."""
        insights = []
        for col in profile.numericals:
            try:
                s = df[col].drop_nulls()
                if s.len() < 10: continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                upper = q3 + 3 * iqr
                outliers = s.filter(pl.col(col) > upper)
                if outliers.len() > 0:
                    insights.append(BusinessInsight(
                        title=f"Extreme Outliers in {col}",
                        impact="🟠 Important",
                        description=f"Detected {outliers.len()} extreme outlier values in {col} that are significantly above the normal range.",
                        recommendation="Verify if these records are data entry errors or represent a distinct high-value segment.",
                        rule_type="outlier_detection"
                    ))
            except Exception: pass
        return insights

    def _deduplicate(self, insights: list[BusinessInsight]) -> list[BusinessInsight]:
        """Remove duplicate insights based on title."""
        seen = set()
        unique = []
        for ins in insights:
            if ins.title not in seen:
                seen.add(ins.title)
                unique.append(ins)
        return unique

    def _inject_contradictions(self, insights: list[BusinessInsight]) -> list[BusinessInsight]:
        """Ensure consistency (e.g. don't say 'No Insights' if rules fired)."""
        if len(insights) > 0:
            # Filter out any 'No Significant Insights' placeholders if they exist
            return [i for i in insights if "No Significant Insights" not in i.title]
        return insights

    def _self_diagnostic(self) -> None:
        """Prints a diagnostic report of which rules and guards are wired up."""
        ok  = "[OK]"
        nok = "[--]"
        try:
            print("\n" + "=" * 70)
            print("[DIAG] INSIGHT ENGINE SELF-DIAGNOSTIC")
            print("=" * 70)

            has_tautology_guard = hasattr(self, "is_derived_column")
            print(f"  [{ok if has_tautology_guard else nok}] is_derived_column() method exists")

            has_skew_rule = hasattr(self, "_rule_skewed_distribution_alert")
            print(f"  [{ok if has_skew_rule else nok}] _rule_skewed_distribution_alert() exists")

            rule_methods = [m for m in dir(self) if m.startswith("_rule_")]
            print(f"  [>>] Found {len(rule_methods)} rule methods:")
            for m in rule_methods:
                method = getattr(self, m)
                is_wrapped = hasattr(method, "__wrapped__")
                marker = f"{ok} wrapped" if is_wrapped else f"{nok} NOT wrapped"
                print(f"     {marker}: {m}")

            import inspect
            try:
                src = inspect.getsource(self._rule_strong_correlation_insight)
                calls_guard = "is_derived_column" in src or "_rule_numeric_correlations" in src
                print(f"  [{ok if calls_guard else nok}] _rule_strong_correlation_insight wired to guard")
            except Exception as e:
                print(f"  [!!] Could not inspect: {e}")

            try:
                src = inspect.getsource(self.generate_insights)
                for rule in rule_methods:
                    wired = rule in src
                    print(f"     [{ok if wired else nok}] generate_insights() calls {rule}()")
            except Exception as e:
                print(f"  [!!] Could not inspect generate_insights: {e}")

            print("=" * 70 + "\n")
        except Exception as e:
            print(f"[DIAG] self-diagnostic failed: {e}")


class StrategicBriefBuilder:
    """Builds an executive brief using ONLY real column names from the dataset.
    Zero LLM, zero hallucination, deterministic output."""

    DOMAIN_LABELS = {
        "ecommerce": "Ecommerce",
        "sales":     "Sales",
        "retail":    "Retail",
        "healthcare": "Healthcare",
        "finance":   "Finance",
        "hr":        "Human Resources",
        "general":   "General Business",
    }

    def __init__(self, domain: str, df: pl.DataFrame, insights: list, corr_matrix=None):
        self.domain = domain
        self.df = df
        self.insights = insights
        self.corr_matrix = corr_matrix

    def build(self) -> str:
        domain_label = self.DOMAIN_LABELS.get(self.domain, "General Business")
        n_records = self.df.height

        # 1. Find the strongest non-tautological numeric driver from corr matrix
        driver_col, target_col, r_value = self._find_top_driver()

        # 2. Count critical risk insights
        critical_count = 0
        for i in self.insights:
            impact = i.get("impact", "") if isinstance(i, dict) else getattr(i, "impact", "")
            if "🔴" in str(impact) or "High" in str(impact):
                critical_count += 1

        # 3. Find segment with biggest revenue gap (if any)
        segment_finding = self._find_top_segment_finding()

        # Build paragraph
        lines = []
        lines.append(
            f"The {domain_label} system is operating at a scale of {n_records:,} records."
        )

        if driver_col and target_col:
            lines.append(
                f"Internal logic shows {driver_col} as the primary numeric driver of {target_col} "
                f"(correlation: {r_value:+.2f})."
            )
        else:
            lines.append(
                "No single numeric driver dominates the data — variance is distributed across multiple variables."
            )

        if segment_finding:
            lines.append(segment_finding)
        temporal_finding = self._find_temporal_finding()
        if not temporal_finding:
            temporal_finding = self._find_temporal_finding_direct()
        if temporal_finding:
            lines.append(temporal_finding)

        if critical_count > 0:
            lines.append(
                f"Risk assessment identifies {critical_count} high-impact "
                f"{'finding' if critical_count == 1 else 'findings'} requiring leadership review."
            )
        else:
            lines.append(
                "Current dataset shows no high-impact anomalies — system is operating within expected bounds."
            )

        # Strategic implication
        if driver_col:
            lines.append(
                f"Strategic implication: forecasting and optimization efforts should center on {driver_col} "
                f"as the leading indicator."
            )

        return " ".join(lines)

    def _is_tautology(self, col_a: str, col_b: str) -> bool:
        """Returns True if col_a and col_b are likely derived from each other."""
        tautologies = [
            ("price", "revenue"), ("quantity", "revenue"), ("sales", "revenue"),
            ("mrp", "price"), ("total", "amount"), ("subtotal", "total")
        ]
        a, b = col_a.lower(), col_b.lower()
        for t1, t2 in tautologies:
            if (t1 in a and t2 in b) or (t1 in b and t2 in a):
                return True
        return False

    def _find_top_driver(self) -> tuple:
        """Find strongest non-tautological correlation. Returns (driver, target, r) or (None, None, None)."""
        if self.corr_matrix is None:
            return None, None, None

        try:
            best_pair = (None, None, 0.0)
            # handle pandas or polars
            if hasattr(self.corr_matrix, "columns"):
                cols = list(self.corr_matrix.columns)
            else:
                return None, None, None

            for i, col_a in enumerate(cols):
                for col_b in cols[i + 1:]:
                    # Handle pandas loc
                    if hasattr(self.corr_matrix, "loc"):
                        r = float(self.corr_matrix.loc[col_a, col_b])
                    else:
                        continue

                    if abs(r) > abs(best_pair[2]) and abs(r) < 0.98: # Skip tautologies/perfect corr
                        if self._is_tautology(col_a, col_b):
                            continue
                        best_pair = (col_a, col_b, r)

            if abs(best_pair[2]) >= 0.3:
                return best_pair
        except Exception as e:
            print(f"[BRIEF] _find_top_driver error: {e}")

        return None, None, None

    def _find_temporal_finding(self) -> str:
        for ins in self.insights:
            rule = ins.get("rule_type", "") if isinstance(ins, dict) else getattr(ins, "rule_type", "")
            print(f"[TEMPORAL DEBUG] checking insight: rule={rule}, type={type(ins)}")
            if rule == "temporal_peaks":
                chart_data = ins.get("chart_data", {}) if isinstance(ins, dict) else getattr(ins, "chart_data", {})
                print(f"[TEMPORAL DEBUG] chart_data={chart_data}")
                peak = chart_data.get("peak_month", "") if chart_data else ""
                trough = chart_data.get("trough_month", "") if chart_data else ""
                gap = chart_data.get("pct_gap", 0) if chart_data else 0
                print(f"[TEMPORAL DEBUG] peak={peak}, trough={trough}, gap={gap}")
                if peak and trough:
                    return (
                        f"Revenue shows clear seasonality: {peak} is the peak month "
                        f"while {trough} is the trough — a {gap:.0f}% swing that demands "
                        f"proactive inventory and cash-flow planning."
                    )
        return ""

    def _find_temporal_finding_direct(self) -> str:
        """Directly compute peak/trough from the dataframe — no dependency on insight objects."""
        try:
            date_col = next(
                (c for c in self.df.columns if any(k in c.lower() for k in ["date", "time", "month"])),
                None
            )
            rev_col = next(
                (c for c in self.df.columns if any(k in c.lower() for k in ["sales", "amount", "revenue"])),
                None
            )
            if not date_col or not rev_col:
                return ""

            import pandas as pd
            pdf = self.df.to_pandas()
            pdf[date_col] = pd.to_datetime(pdf[date_col], errors="coerce")
            pdf = pdf.dropna(subset=[date_col])
            if len(pdf) < 30:
                return ""

            pdf["_month"] = pdf[date_col].dt.to_period("M")
            monthly = pdf.groupby("_month")[rev_col].sum()
            if len(monthly) < 2:
                return ""

            peak_month = monthly.idxmax().strftime("%B")
            trough_month = monthly.idxmin().strftime("%B")
            peak_val = float(monthly.max())
            trough_val = float(monthly.min())
            gap = ((peak_val - trough_val) / peak_val) * 100

            return (
                f"Revenue shows clear seasonality: {peak_month} is the peak month "
                f"while {trough_month} is the trough — a {gap:.0f}% swing that demands "
                f"proactive inventory and cash-flow planning."
            )
        except Exception as e:
            print(f"[TEMPORAL DIRECT] error: {e}")
            return ""

    def _find_top_segment_finding(self) -> str:
        """Surface the most impactful segment-level insight as a brief sentence."""
        for ins in self.insights:
            if isinstance(ins, dict):
                rule = ins.get("rule_type", "")
                qualified = ins.get("qualified_segments", [])
                excluded = ins.get("excluded_segments", [])
            else:
                rule = getattr(ins, "rule_type", "")
                qualified = getattr(ins, "qualified_segments", [])
                excluded = getattr(ins, "excluded_segments", [])

            if "revenue_by" in rule or "top_performers" in rule:
                if qualified:
                    if excluded:
                        return f"Segment analysis highlights {qualified[0]} as the leader and {excluded[0]} as the lagging segment."
                    return f"Segment analysis highlights {qualified[0]} as the standout performer."
        return ""


class RecommendationEngine:
    """Converts BusinessInsights into prioritized, distinct action items."""

    IMPACT_WEIGHT = {"High": 3, "Medium": 2, "Low": 1, "🔴 Critical": 4, "🟠 Important": 2, "🟢 Minor": 1}

    def __init__(self, domain: str):
        self.domain = domain

    def generate(self, insights: list, max_count: int = 5) -> list:
        """Returns up to max_count prioritized, deduplicated recommendations."""
        if not insights:
            return self._fallback_recommendations()

        # Score each insight
        scored = []
        for ins in insights:
            # Handle both objects and dicts
            if hasattr(ins, "impact"):
                impact = getattr(ins, "impact", "Medium")
                qualified_segments = getattr(ins, "qualified_segments", [])
                rule_type = getattr(ins, "rule_type", "")
                title = getattr(ins, "title", "")
                recommendation = getattr(ins, "recommendation", "")
                excluded_segments = getattr(ins, "excluded_segments", [])
            else:
                impact = ins.get("impact", "Medium")
                qualified_segments = ins.get("qualified_segments", [])
                rule_type = ins.get("rule_type", "")
                title = ins.get("title", "")
                recommendation = ins.get("recommendation", "")
                excluded_segments = ins.get("excluded_segments", [])

            score = self.IMPACT_WEIGHT.get(impact, 2)

            # Boost if has qualified segments (more actionable)
            if qualified_segments:
                score += 1
            # Penalize derived/correlation insights (less actionable)
            if "correlation" in rule_type.lower():
                score -= 2

            scored.append((score, {
                "impact": impact,
                "qualified_segments": qualified_segments,
                "rule_type": rule_type,
                "title": title,
                "recommendation": recommendation,
                "excluded_segments": excluded_segments
            }))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Build distinct recommendations
        recommendations = []
        seen_themes = set()

        for score, ins in scored:
            if len(recommendations) >= max_count:
                break

            theme = ins.get("rule_type", "general").split("_")[0]
            if theme in seen_themes:
                continue
            seen_themes.add(theme)

            rec_text = self._craft_action(ins)
            timeframe = self._infer_timeframe(ins)
            owner = self._infer_owner(ins)

            recommendations.append({
                "priority": len(recommendations) + 1,
                "action": rec_text,
                "timeframe": timeframe,
                "owner": owner,
                "linked_insight": ins.get("title", ""),
                "impact": ins.get("impact", "Medium"),
            })

        # Pad with fallbacks if too few
        if not recommendations:
            return self._fallback_recommendations()

        return recommendations

    def _craft_action(self, insight: dict) -> str:
        """Convert observation into action verb form."""
        rule_type = insight.get("rule_type", "")
        rec = insight.get("recommendation", "")

        # If recommendation is missing or just echoes title, build one
        title = insight.get("title", "")
        if not rec or rec.lower().startswith(title.lower()[:20]):
            qualified = insight.get("qualified_segments", [])
            excluded = insight.get("excluded_segments", [])

            if "revenue_by" in rule_type and qualified and excluded:
                return (
                    f"Run a 30-day operational audit on {excluded[0]} (lowest performer). "
                    f"Identify whether the gap with {qualified[0]} is demand-driven "
                    f"(deprioritize) or execution-driven (replicate top playbook)."
                )
            if "top_performers" in rule_type and qualified:
                return (
                    f"Concentrate Q3 marketing budget on {', '.join(qualified[:3])}. "
                    f"Simultaneously test 2 long-tail candidates for promotion lift."
                )
            if "skewed" in rule_type:
                return (
                    f"Replace mean with median in all dashboards for {qualified[0] if qualified else 'this metric'}. "
                    f"Carve out top 5% as a separate high-value cohort with dedicated retention strategy."
                )
            if "discount" in rule_type:
                return (
                    f"A/B test removing discounts on the highest-margin tier for 2 weeks. "
                    f"Measure volume impact. If volume holds within 10%, kill the discount permanently."
                )
            if "demographic" in rule_type and qualified and excluded:
                return (
                    f"Allocate 30-day paid-acquisition test budget specifically to {excluded[0]} segment. "
                    f"If conversion lags {qualified[0]}, accept the gap and double down on {qualified[0]}."
                )
            if "correlation" in rule_type:
                return (
                    f"Include this relationship as a leading indicator in your forecasting model. "
                    f"Set a monitoring alert if the correlation drops below 0.3 — it signals a structural shift "
                    f"that requires immediate investigation."
                )
            if "temporal" in rule_type:
                peak = insight.get("chart_data", {}) and insight.get("chart_data", {}).get("peak_month", "peak month")
                trough = insight.get("chart_data", {}) and insight.get("chart_data", {}).get("trough_month", "trough month")
                peak = peak if isinstance(peak, str) else "peak month"
                trough = trough if isinstance(trough, str) else "trough month"
                return (
                    f"Pre-position inventory and marketing budget ahead of {peak}. "
                    f"Diagnose the {trough} dip — run a post-mortem on promotions, "
                    f"stockouts, and demand signals from that period."
                )
            return f"Investigate the underlying drivers of: {title}"

        return rec

    def _infer_timeframe(self, insight: dict) -> str:
        impact = insight.get("impact", "Medium")
        if "Critical" in impact or "High" in impact:
            return "Next 14 days"
        if "Important" in impact or "Medium" in impact:
            return "Next 30 days"
        return "Next quarter"

    def _infer_owner(self, insight: dict) -> str:
        rule_type = insight.get("rule_type", "").lower()
        if "revenue" in rule_type or "top_perform" in rule_type:
            return "Growth / Sales lead"
        if "discount" in rule_type:
            return "Pricing / Revenue ops"
        if "demographic" in rule_type:
            return "Marketing lead"
        if "skew" in rule_type or "outlier" in rule_type:
            return "Data / Analytics lead"
        return "Strategy team"

    def _fallback_recommendations(self) -> list:
        return [{
            "priority": 1,
            "action": "Insufficient signal in this dataset to generate strategic recommendations. Re-run with a larger or more diverse sample.",
            "timeframe": "—",
            "owner": "Data team",
            "linked_insight": "",
            "impact": "Low",
        }]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _series_return_rate(s: pd.Series) -> float:
        try:
            positives = {"yes", "true", "1", "returned", "refund"}
            if s.dtype == bool:
                return s.sum() / max(len(s), 1)
            if s.dtype == object:
                return s.map(lambda v: str(v).strip().lower() in positives).sum() / max(len(s), 1)
            return (s > 0).sum() / max(len(s), 1)
        except Exception:
            return 0.0

    @staticmethod
    def _get_binary_flag_pd(s: pd.Series) -> pd.Series:
        positives = {"yes", "true", "1", "returned", "refund"}
        if s.dtype == bool:
            return s.astype(int)
        if s.dtype == object:
            return s.map(lambda v: 1 if str(v).strip().lower() in positives else 0)
        return (s > 0).astype(int)


# ============================================================
# 5. INSIGHT NARRATOR  (plain English wrapper — already baked into
#    BusinessRuleEngine descriptions above; exposed here for override)
# ============================================================

class InsightNarrator:
    """Post-process BusinessInsight list into final display-ready form."""

    def narrate(
        self, insights: list[BusinessInsight], profile: DataProfile
    ) -> list[dict]:
        out = []
        for ins in insights:
            # Enforce high-end decision intelligence structure
            # Step 4: Insight Compression & Conceptual Language Scrub
            parts = [f"**STRATEGIC OBSERVATION**: {ins.description}"]
            if ins.why_it_matters:
                parts.append(f"**WHY IT MATTERS**: {ins.why_it_matters}")
            if ins.evidence:
                parts.append(f"**SUPPORTING EVIDENCE**: {ins.evidence}")
            if ins.decision_implication:
                parts.append(f"**DECISION IMPLICATION**: {ins.decision_implication}")
            
            # Final language scrub for mechanical patterns (Step 4)
            # Remove "When X increases, Y increases" if any slipped through
            final_desc = "\n\n".join(parts)
            final_desc = final_desc.replace("When ", "Observation confirms ").replace(" increases", " scales").replace(" goes up", " trends higher")
            
            out.append({
                "title": ins.title,
                "description": final_desc,
                "impact": ins.impact,
                "recommendation": ins.recommendation,
                "decision_implication": ins.decision_implication,
                "is_unexpected": ins.is_unexpected,
                "chart_type": ins.chart_type,
                "chart_data": ins.chart_data,
                "rule_type": ins.rule_type,
            })
        return out


def is_chart_informative(values: list[float], min_variance_pct: float = 1.0) -> bool:
    """
    Returns False if all values are within min_variance_pct% of each other.
    Useful for suppressing flat-line charts that show no information.
    """
    if not values or len(values) < 2:
        return False
    
    import numpy as np
    arr = np.array(values, dtype=float)
    
    if np.all(arr == 0):
        return False
    
    mean = np.mean(arr)
    if mean == 0:
        return False
    
    # Coefficient of variation as percentage
    cv_pct = (np.std(arr) / abs(mean)) * 100
    return cv_pct >= min_variance_pct


# ============================================================
# 6. SMART CHART RECOMMENDER
# ============================================================

class SmartChartRecommender:
    """
    Recommend meaningful chart configurations, excluding ID columns and
    producing context-aware visualisations.
    """

    def recommend(
        self,
        df: pl.DataFrame,
        profile: DataProfile,
        insights: list[BusinessInsight],
        max_charts: int = 8,
        domain_id: str = "general"
    ) -> list[dict]:
        """Return a list of chart spec dicts ready for the existing Plotly renderer."""
        import plotly.express as px
        import plotly.graph_objects as go
        import json

        pdf = df.to_pandas()
        charts = []
        chart_ids_used: set[str] = set()

        # Re-detect domain when caller passed the default 'general'
        # but the dataset is clearly a known domain.
        if domain_id == "general":
            redetected = detect_domain(df)
            if redetected != "general":
                domain_id = redetected

        template = TEMPLATES.get(domain_id, TEMPLATES["general"])
        target_label = template.get("target_metric", "Value")

        # When domain is still 'general' (no template matched), the literal
        # "Key Performance Indicator" placeholder is meaningless. Use the
        # actual revenue/price column from the classifier profile instead.
        if domain_id == "general" and target_label == "Key Performance Indicator":
            actual_metric = (profile.revenue_col
                             or profile.price_col
                             or (profile.numericals[0] if profile.numericals else None))
            if actual_metric:
                target_label = actual_metric

        def add(chart_id: str, spec: dict) -> None:
            if chart_id not in chart_ids_used and len(charts) < max_charts:
                chart_ids_used.add(chart_id)
                charts.append(spec)

        cat  = profile.category_col
        geo_col = profile.geographic_col
        num_cols  = [c for c in profile.numericals if c not in profile.identifiers]
        date_col  = profile.date_col
        ret_col   = profile.return_col
        del_col   = profile.delivery_days_col
        # Revenue col = pre-aggregated column (Sales Amount, Revenue, etc.)
        # Price col   = unit price column (Unit Price, Price, etc.)
        # Rule: if revenue_col exists, always use it directly — never multiply by qty
        revenue_col_direct = profile.revenue_col   # e.g. "Sales Amount"
        price_col          = profile.price_col or profile.revenue_col
        qty_col            = profile.qty_col

        # The key guard: is our "price_col" already a revenue/sales/amount column?
        # If yes, multiplying by qty_col would give wrong inflated numbers.
        def _is_revenue_col(col_name: str) -> bool:
            if not col_name:
                return False
            cl = col_name.lower()
            return any(k in cl for k in ["sales", "amount", "revenue", "income"])

        # ── 1. Revenue by Category (horizontal bar) ────────────────────────
        if cat and price_col:
            try:
                # Guard: don't use an identifier column as price
                if price_col in profile.identifiers:
                    price_col = profile.revenue_col or next(
                        (c for c in profile.numericals if c not in profile.identifiers),
                        None,
                    )
                rev_col = "Revenue (₹)"
                pdf_tmp = pdf.copy()
                if _is_revenue_col(price_col):
                    # "Sales Amount" is already revenue — use directly
                    pdf_tmp[rev_col] = pdf[price_col].fillna(0)
                else:
                    # True unit price — multiply by qty to get revenue
                    pdf_tmp[rev_col] = (
                        pdf[price_col].fillna(0) * pdf[qty_col].fillna(0)
                        if qty_col else pdf[price_col].fillna(0)
                    )
                grp = (
                    pdf_tmp.groupby(cat)[rev_col].sum()
                    .reset_index()
                    .sort_values(rev_col, ascending=True)
                )

                # Zero-variance suppression
                if not is_chart_informative(grp[rev_col].tolist()):
                    print(f"[CHART SUPPRESSED] {target_label} by {cat} — all values flat")
                else:
                    # ✅ TIER 1 ENHANCEMENT: Add % contribution labels
                    total_rev = grp[rev_col].sum()
                    grp["_pct"] = (grp[rev_col] / total_rev * 100).round(1)
                    grp["_label"] = grp.apply(
                        lambda r: f"{r[rev_col]/1e6:.1f}M ({r['_pct']:.0f}%)", axis=1
                    )
                    
                    fig = px.bar(
                        grp, x=rev_col, y=cat, orientation="h",
                        title=f"{target_label} by {cat}",
                        color=rev_col, color_continuous_scale="Viridis",
                        text=grp["_label"]  # Use explicit labels instead of text_auto
                    )
                    fig.update_traces(textposition="inside", textfont_size=11)
                    
                    # ✅ TIER 1 ENHANCEMENT: Annotate top bar with % contribution
                    top_val = grp[rev_col].max()
                    top_cat = grp.loc[grp[rev_col].idxmax(), cat]
                    top_pct = (top_val / total_rev * 100)
                    fig.add_annotation(
                        x=top_val, y=top_cat,
                        text=f"Top: {top_pct:.0f}% of total",
                        showarrow=True, arrowhead=2,
                        font=dict(color="#ffffff", size=11),
                        bgcolor="#6366f1", borderpad=4,
                        xanchor="left", ax=20, ay=0
                    )
                    
                    fig.update_layout(template="plotly_dark",
                                      coloraxis_showscale=False, showlegend=False,
                                      xaxis_title=target_label)
                    add("revenue_by_cat", {
                        "chart_id": "revenue_by_cat",
                        "chart_type": "bar",
                        "title": f"{target_label} by {cat}",
                        "description": f"Total {target_label} breakdown across {cat} segments",
                        "plotly_json": json.loads(fig.update_layout(**CHART_LAYOUT_BASE).to_json()),
                        "columns_used": [cat, price_col] + ([qty_col] if qty_col else []),
                        "priority_score": 90,
                        "insight_reason": "Core business revenue metric",
                        "interest_level": "high"
                    })
                    
                    # ✅ TIER 1 ENHANCEMENT: Add Pareto Chart (80/20 analysis)
                    try:
                        grp_sorted = grp.sort_values(rev_col, ascending=False)
                        grp_sorted["cumulative_pct"] = (
                            grp_sorted[rev_col].cumsum() / grp_sorted[rev_col].sum() * 100
                        )
                        fig_pareto = go.Figure()
                        fig_pareto.add_trace(go.Bar(
                            x=grp_sorted[cat], y=grp_sorted[rev_col],
                            name="Revenue", marker_color="#6366f1",
                            text=[f"{v/1e6:.1f}M" for v in grp_sorted[rev_col]],
                            textposition="outside"
                        ))
                        fig_pareto.add_trace(go.Scatter(
                            x=grp_sorted[cat], y=grp_sorted["cumulative_pct"],
                            name="Cumulative %", yaxis="y2",
                            line=dict(color="#ef4444", width=2.5),
                            mode="lines+markers"
                        ))
                        fig_pareto.update_layout(
                            yaxis2=dict(
                                title="Cumulative %", overlaying="y",
                                side="right", range=[0, 110],
                                ticksuffix="%"
                            ),
                            template="plotly_dark",
                            legend=dict(orientation="h"),
                            title=f"Pareto: {cat} Revenue Contribution"
                        )
                        add("pareto_revenue", {
                            "chart_id": "pareto_revenue",
                            "chart_type": "pareto",
                            "title": f"Pareto: {cat} Revenue Contribution",
                            "description": "80/20 analysis — which categories drive 80% of revenue",
                            "plotly_json": json.loads(fig_pareto.update_layout(**CHART_LAYOUT_BASE).to_json()),
                            "columns_used": [cat, price_col],
                            "priority_score": 92,
                            "insight_reason": "Pareto principle applied to revenue concentration",
                            "interest_level": "high"
                        })
                    except Exception as e:
                        print(f"[PARETO CHART] Failed to generate: {e}")
            except Exception:
                pass

        # ── 2. Return Rate by Category (bar with reference line) ──────────
        if cat and ret_col:
            try:
                pdf_tmp = pdf.copy()
                def _rrate(s):
                    pos = {"yes","true","1","returned","refund"}
                    if s.dtype == object:
                        return s.map(lambda v: str(v).strip().lower() in pos).mean() * 100
                    return (s > 0).mean() * 100
                rates = pdf.groupby(cat)[ret_col].apply(_rrate).reset_index()
                rates.columns = [cat, "Return Rate (%)"]
                rates = rates.sort_values("Return Rate (%)", ascending=False)
                global_rate = _rrate(pdf[ret_col])
                
                # Zero-variance suppression
                if not is_chart_informative(rates["Return Rate (%)"].tolist()):
                    print(f"[CHART SUPPRESSED] Return Rate by {cat} — all values flat")
                else:
                    fig = px.bar(
                        rates, x=cat, y="Return Rate (%)",
                        title=get_smart_title(f"Return Rate by {cat}", cat, "Return Rate"),
                        color="Return Rate (%)", color_continuous_scale="RdYlGn_r",
                        text_auto=".1f"
                    )
                    fig.add_hline(y=global_rate, line_dash="dash",
                                  line_color="#94a3b8",
                                  annotation_text=f"Avg {global_rate:.1f}%")
                fig.update_layout(template="plotly_dark",
                                  coloraxis_showscale=False, showlegend=False)
                add("return_rate_by_cat", {
                    "chart_id": "return_rate_by_cat",
                    "chart_type": "bar",
                    "title": get_smart_title(f"Return Rate by {cat}", cat, "Return Rate"),
                    "description": "Categories above the dashed line have above-average return rates",
                    "plotly_json": json.loads(fig.update_layout(**CHART_LAYOUT_BASE).to_json()),
                    "columns_used": [cat, ret_col],
                    "priority_score": 88,
                    "insight_reason": "Return rate analysis by product category",
                    "interest_level": "high"
                })
            except Exception:
                pass

        # ── 3. Payment / Channel distribution (donut) ──────────────────────
        pay_col = next(
            (c for c in profile.categoricals
             if any(k in c.lower() for k in {"payment", "method", "channel"})),
            None
        )
        if pay_col:
            try:
                counts = pdf[pay_col].value_counts().reset_index()
                counts.columns = [pay_col, "count"]
                fig = px.pie(
                    counts, names=pay_col, values="count",
                    title=f"{pay_col} Distribution",
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(template="plotly_dark")
                fig.update_traces(textposition="inside", textinfo="percent+label")
                add("payment_dist", {
                    "chart_id": "payment_dist",
                    "chart_type": "pie",
                    "title": f"{pay_col} Distribution",
                    "description": "Share of transactions by payment or channel type",
                    "plotly_json": json.loads(fig.update_layout(**CHART_LAYOUT_BASE).to_json()),
                    "columns_used": [pay_col],
                    "priority_score": 80,
                    "insight_reason": "Channel / payment method diversity check",
                    "interest_level": "recommended"
                })
            except Exception:
                pass

        # ── 4. Delivery Days vs Returns (scatter) ─────────────────────────
        if del_col and ret_col:
            try:
                pdf_tmp = pdf[[del_col, ret_col]].dropna().copy()
                ret_flag = BusinessRuleEngine._get_binary_flag_pd(pdf_tmp[ret_col])
                pdf_tmp["Returned"] = ret_flag.map({0: "No", 1: "Yes"})
                fig = px.box(
                    pdf_tmp, x="Returned", y=del_col,
                    color="Returned",
                    title=f"{del_col} for Returned vs Not Returned Orders",
                    color_discrete_map={"Yes": "#ef4444", "No": "#10b981"}
                )
                fig.update_layout(template="plotly_dark", showlegend=False)
                add("delivery_vs_return", {
                    "chart_id": "delivery_vs_return",
                    "chart_type": "box",
                    "title": f"Delivery Time vs Returns",
                    "description": "Are longer delivery times linked to higher returns?",
                    "plotly_json": json.loads(fig.update_layout(**CHART_LAYOUT_BASE).to_json()),
                    "columns_used": [del_col, ret_col],
                    "priority_score": 85,
                    "insight_reason": "Operational risk: delivery time driving returns",
                    "interest_level": "high"
                })
            except Exception:
                pass

        # ── 5. Revenue over Time (line) ────────────────────────────────────
        if date_col and price_col:
            try:
                pdf_tmp = pdf.copy()
                # Added dayfirst=True to fix future dates bug
                pdf_tmp[date_col] = pd.to_datetime(pdf_tmp[date_col], errors="coerce", dayfirst=True)
                pdf_tmp = pdf_tmp.dropna(subset=[date_col])
                if qty_col:
                    pdf_tmp["__rev__"] = pdf_tmp[price_col].fillna(0) * pdf_tmp[qty_col].fillna(0)
                else:
                    pdf_tmp["__rev__"] = pdf_tmp[price_col].fillna(0)
                pdf_tmp["month"] = pdf_tmp[date_col].dt.to_period("M").astype(str)
                monthly = pdf_tmp.groupby("month")["__rev__"].sum().reset_index()
                monthly = monthly.sort_values("month")
                if len(monthly) >= 2:
                    t = get_smart_title("Monthly Revenue Trend", "Time", "Revenue")
                    fig = px.line(
                        monthly, x="month", y="__rev__",
                        title=t,
                        markers=True
                    )
                    fig.update_traces(line_color="#6366f1", line_width=2)
                    fig.update_layout(template="plotly_dark",
                                      xaxis_title="Month", yaxis_title="Revenue")
                    add("revenue_over_time", {
                        "chart_id": "revenue_over_time",
                        "chart_type": "line",
                        "title": t,
                        "description": "Revenue performance over time — identify growth or decline",
                        "plotly_json": json.loads(fig.update_layout(**CHART_LAYOUT_BASE).to_json()),
                        "columns_used": [date_col, price_col],
                        "priority_score": 87,
                        "insight_reason": "Time-series revenue trend analysis",
                        "interest_level": "high"
                    })
            except Exception:
                pass

        # ── 6. Geographic Revenue (bar) ────────────────────────────────────
        if geo_col and geo_col != cat and price_col:
            try:
                pdf_tmp = pdf.copy()
                # Use human readable name instead of __rev__
                rev_label = "Revenue"
                _pc2_lower = price_col.lower()
                _price2_is_revenue = any(k in _pc2_lower for k in ["sales", "amount", "revenue"])
                pdf_tmp[rev_label] = (
                    pdf[price_col].fillna(0) * pdf[qty_col].fillna(0)
                    if (qty_col and not _price2_is_revenue)
                    else pdf[price_col].fillna(0)
                )
                grp = (
                    pdf_tmp.groupby(geo_col)[rev_label].sum()
                    .reset_index()
                    .sort_values(rev_label, ascending=False)
                    .head(12)
                )
                
                # Zero-variance suppression
                if not is_chart_informative(grp[rev_label].tolist()):
                    print(f"[CHART SUPPRESSED] Revenue by {geo_col} — all values flat")
                else:
                    # Check if we should do a grouped bar (Feature 1)
                    # If cat exists, replace simple bar with grouped bar
                    if cat and cat != geo_col:
                        # Step 1: resolve revenue column
                        # Priority 1: explicit revenue_col from profile (most reliable)
                        # Priority 2: keyword-matched numeric column
                        # Priority 3: fallback to price_col
                        if revenue_col_direct:
                            revenue_col = revenue_col_direct
                        else:
                            price_candidates = [
                                c for c in num_cols
                                if any(k in c.lower() for k in ["sales", "amount", "revenue", "price", "profit"])
                            ]
                            revenue_col = price_candidates[0] if price_candidates else price_col
                        # Step 2: resolve region and category from profile.categoricals
                        all_cats = profile.categoricals
                        region_candidates = [
                            c for c in all_cats
                            if any(k in c.lower() for k in ["region", "area", "zone", "territory", "location", "city", "state", "country"])
                        ]
                        cat_candidates = [
                            c for c in all_cats
                            if any(k in c.lower() for k in ["category", "product", "type", "segment", "class"])
                            and c not in region_candidates
                        ]
                        region_col = region_candidates[0] if region_candidates else geo_col
                        cat_col    = cat_candidates[0]    if cat_candidates    else (
                            all_cats[1] if len(all_cats) > 1 else all_cats[0]
                        )
                        pdf_tmp = pdf.copy()
                        if _is_revenue_col(revenue_col):
                            pdf_tmp["_revenue"] = pdf[revenue_col].fillna(0)
                        else:
                            pdf_tmp["_revenue"] = (
                                pdf[revenue_col].fillna(0) * pdf[qty_col].fillna(0)
                                if qty_col else pdf[revenue_col].fillna(0)
                            )
                        grp_cat = (
                            pdf_tmp.groupby([region_col, cat_col])["_revenue"].sum()
                            .reset_index()
                        )
                        fig = px.bar(
                            grp_cat,
                            x=region_col,
                            y="_revenue",
                            color=cat_col,
                            barmode="group",
                            title=f"Which {cat_col} performs best in each {region_col}?",
                            text_auto=".2s",
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                        # Merge CHART_LAYOUT_BASE with geo-specific settings so
                        # tickformat=".2s" is never overwritten by the base yaxis dict.
                        geo_layout = {**CHART_LAYOUT_BASE}
                        geo_layout["yaxis"] = {
                            "gridcolor": "rgba(255,255,255,0.05)",
                            "tickformat": ".2s",
                        }
                        geo_layout["xaxis_title"] = region_col
                        geo_layout["yaxis_title"] = "Revenue"
                        geo_layout["legend_title"] = cat_col

                        fig.update_layout(template="plotly_dark")

                        # Suppress chart if all bars are nearly identical (uninformative)
                        geo_values = grp_cat["_revenue"].tolist()
                        if not is_chart_informative(geo_values, min_variance_pct=1.0):
                            print(f"[CHART SUPPRESSED] geo_cat_revenue — values too flat")
                        else:
                            add("geo_cat_revenue", {
                                "chart_id": "geo_cat_revenue",
                                "chart_type": "grouped_bar",
                                "title": f"Which {cat_col} performs best in each {region_col}?",
                                "description": f"Geographical revenue distribution across both {region_col} and {cat_col}",
                                "plotly_json": json.loads(fig.update_layout(**geo_layout).to_json()),
                                "columns_used": [region_col, cat_col, revenue_col],
                                "priority_score": 82,
                                "insight_reason": "Cross-category geographic performance analysis",
                                "interest_level": "high"
                            })
                    else:
                        fig = px.bar(
                            grp, x=geo_col, y=rev_label,
                            title=f"Which {geo_col} generates the most Revenue?",
                            color=rev_label, color_continuous_scale="Blues",
                            text_auto=".2s"
                        )
                        fig.update_layout(template="plotly_dark",
                                          coloraxis_showscale=False, showlegend=False)
                        add("geo_revenue", {
                            "chart_id": "geo_revenue",
                            "chart_type": "bar",
                            "title": f"Which {geo_col} generates the most Revenue?",
                            "description": f"Geographical revenue distribution across {geo_col}",
                        "plotly_json": json.loads(fig.update_layout(**CHART_LAYOUT_BASE).to_json()),
                        "columns_used": [geo_col, price_col],
                        "priority_score": 82,
                        "insight_reason": "Geographic performance analysis",
                        "interest_level": "recommended"
                    })
            except Exception:
                pass

        # ── 7. Top N Categorical count (bar) ────────────────────────────────
        # Shows record COUNT per category, NOT the metric.
        if cat:
            try:
                counts = pdf[cat].value_counts().reset_index().head(10)
                counts.columns = [cat, "count"]
                
                # Zero-variance suppression
                if not is_chart_informative(counts["count"].tolist()):
                    print(f"[CHART SUPPRESSED] Volume by {cat} — all values flat")
                else:
                    colors = ["#6B5CE7" if i == 0 else "#CBD5E1" for i in range(len(counts))]
                    fig = px.bar(
                        counts, x=cat, y="count",
                        title=f"Records per {cat}",
                        text_auto=True
                    )
                    fig.update_traces(marker_color=colors)
                    fig.update_layout(template="plotly_dark",
                                      coloraxis_showscale=False, showlegend=False,
                                      yaxis_title="Records")
                    add("count_by_cat", {
                        "chart_id": "count_by_cat",
                        "chart_type": "bar",
                        "title": f"Records per {cat}",
                        "description": f"Number of records in each {cat}",
                        "plotly_json": json.loads(fig.update_layout(**CHART_LAYOUT_BASE).to_json()),
                        "columns_used": [cat],
                        "priority_score": 70,
                        "insight_reason": "Volume distribution by category",
                        "interest_level": "recommended"
                    })
            except Exception:
                pass

        # ── 8. Price distribution (histogram) ─────────────────────────────
        if price_col:
            try:
                color_col = cat if cat and pdf[cat].nunique() <= 5 else None
                fig = px.histogram(
                    pdf.dropna(subset=[price_col]),
                    x=price_col,
                    color=color_col,
                    title=f"{price_col} Distribution",
                    nbins=30,
                    marginal="rug",
                    opacity=0.8
                )
                
                # ✅ TIER 1 ENHANCEMENT: Add median line annotation
                median_val = pdf[price_col].median()
                fig.add_vline(
                    x=median_val, line_dash="dash",
                    line_color="#ef4444", line_width=2,
                    annotation_text=f"Median: {median_val:,.0f}",
                    annotation_position="top right"
                )
                
                fig.update_layout(template="plotly_dark",
                                  barmode="overlay" if color_col else "relative")
                if not color_col:
                    fig.update_traces(marker_color="#6366f1")
                add("price_dist", {
                    "chart_id": "price_dist",
                    "chart_type": "histogram",
                    "title": f"{price_col} Distribution",
                    "description": f"Spread and shape of {price_col} values",
                    "plotly_json": json.loads(fig.update_layout(**CHART_LAYOUT_BASE).to_json()),
                    "columns_used": [price_col] + ([cat] if color_col else []),
                    "priority_score": 65,
                    "insight_reason": "Price/value distribution analysis",
                    "interest_level": "standard"
                })
            except Exception:
                pass

        # ── 9. Fallback generic charts if we still have space ─────────────
        if len(charts) < 4 and num_cols and profile.categoricals:
            self._add_fallback_charts(pdf, profile, num_cols, charts,
                                      chart_ids_used, max_charts)

        charts.sort(key=lambda c: c.get("priority_score", 50), reverse=True)
        return charts[:max_charts]

    def _add_fallback_charts(
        self, pdf, profile, num_cols, charts, chart_ids_used, max_charts
    ):
        import plotly.express as px, json
        cat = profile.category_col

        # Priority 1: revenue/sales/amount/price columns
        priority_nums = [
            c for c in num_cols
            if any(k in c.lower() for k in ["sales", "amount", "revenue", "price", "profit"])
        ]
        # Priority 2: any numeric column with meaningful scale (max >= 100)
        other_nums = [
            c for c in num_cols
            if c not in priority_nums
            and pdf[c].max() >= 100
        ]
        ordered_nums = (priority_nums + other_nums)[:2]

        for num in ordered_nums:
            if len(charts) >= max_charts:
                break
            if not cat or num == cat:
                continue
            # Skip low-scale columns (quantity, rating, count — max < 100)
            if pdf[num].max() < 100:
                print(f"[CHART SUPPRESSED] Fallback {num} — max={pdf[num].max():.1f} < 100")
                continue
            try:
                grp = (
                    pdf.groupby(cat)[num].median()
                    .reset_index()
                    .sort_values(num, ascending=False)
                )
                values = grp[num].tolist()
                if not is_chart_informative(values, min_variance_pct=5.0):
                    print(f"[CHART SUPPRESSED] Fallback {num} by {cat} — variance too low")
                    continue
                fig = px.bar(
                    grp,
                    x=num,
                    y=cat,
                    orientation="h",
                    title=f"Median {num} by {cat}",
                    text_auto=".1f",
                    color=num,
                    color_continuous_scale="Blues",
                )
                fig.update_layout(
                    template="plotly_dark",
                    coloraxis_showscale=False,
                    xaxis=dict(
                        range=[0, grp[num].max() * 1.15],
                        tickformat=".2s",
                    ),
                    yaxis=dict(autorange="reversed"),
                )
                cid = f"fallback_bar_{num}_{cat}"
                if cid not in chart_ids_used:
                    chart_ids_used.add(cid)
                    charts.append({
                        "chart_id": cid,
                        "chart_type": "bar",
                        "title": f"{num} by {cat}",
                        "description": f"Median {num} comparison (robust to outliers)",
                        "plotly_json": json.loads(
                            fig.update_layout(**CHART_LAYOUT_BASE).to_json()
                        ),
                        "columns_used": [cat, num],
                        "priority_score": 55,
                        "insight_reason": "Category vs numeric comparison",
                        "interest_level": "standard"
                    })
            except Exception:
                pass


# ============================================================
# 7. ANOMALY DETECTOR
# ============================================================

class AnomalyDetector:
    """IQR-based outlier detection on meaningful numerical columns."""

    def detect(
        self, df: pl.DataFrame, profile: DataProfile
    ) -> list[str]:
        warnings: list[str] = []

        # Small dataset
        if len(df) < 100:
            warnings.append(
                f"⚠️ Small dataset ({len(df)} rows): insights may not be statistically "
                f"significant. Aim for 500+ rows for reliable conclusions."
            )

        # Missing values
        high_missing = [
            col for col in df.columns
            if df[col].null_count() / max(len(df), 1) > 0.2
            and col not in profile.identifiers
        ]
        if high_missing:
            warnings.append(
                f"⚠️ High missing values (>20%) in: {', '.join(high_missing)}. "
                f"Consider imputing or excluding these columns."
            )

        # Outliers in key numerical columns (excluding IDs)
        for col in profile.numericals[:5]:
            try:
                data = df[col].drop_nulls()
                if len(data) < 20:
                    continue
                q1 = float(data.quantile(0.25))
                q3 = float(data.quantile(0.75))
                iqr = q3 - q1
                if iqr == 0:
                    continue
                outliers = data.filter(
                    (data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)
                )
                pct = len(outliers) / len(data) * 100
                if pct > 10:
                    warnings.append(
                        f"⚠️ {col} has {len(outliers)} outliers ({pct:.0f}% of values). "
                        f"Verify these are valid data points."
                    )
            except Exception:
                pass

        # Skewed distributions
        for col in profile.numericals[:3]:
            try:
                data = df[col].drop_nulls()
                if len(data) < 30:
                    continue
                mean   = float(data.mean())
                median = float(data.median())
                std    = float(data.std()) or 1.0
                if abs(mean - median) / std > 1.5:
                    direction = "right" if mean > median else "left"
                    warnings.append(
                        f"ℹ️ {col} is heavily {direction}-skewed "
                        f"(mean={mean:.1f}, median={median:.1f}). "
                        f"Use median for more representative insights."
                    )
            except Exception:
                pass

        return warnings


# ============================================================
# 8. EDGE CASE HANDLER
# ============================================================

class EdgeCaseHandler:
    """Detect edge cases and data quality issues."""

    def check(self, df: pl.DataFrame, profile: DataProfile) -> list[str]:
        return AnomalyDetector().detect(df, profile)


def _apply_smart_sampling(df: pl.DataFrame) -> pl.DataFrame:
    """Implement tiered sampling for large datasets to maintain performance."""
    rows = len(df)
    if rows < 10000:
        return df
    
    # Tiered Logic as requested
    if rows > 500000:
        sample_n = 50000
    elif rows > 100000:
        sample_n = 20000
    else:
        sample_n = 10000
        
    return df.sample(n=sample_n, seed=42)


SAMPLE_THRESHOLD = 10_000   # Updated threshold


def run_insight_engine(
    df: pl.DataFrame,
    max_insights: int = 10,
    max_charts: int = 8,
    progress_callback: object = None,  # callable(stage: str, pct: int)
) -> dict:
    """
    Full pipeline: classify → compute metrics → evaluate rules
    → narrate → recommend charts → detect anomalies.

    Returns a structured dict ready for API serialisation.

    `progress_callback(stage, pct)` is called at each stage so the
    background job tracker can push real-time progress to the frontend.
    """
    def _progress(stage: str, pct: int) -> None:
        if callable(progress_callback):
            try:
                progress_callback(stage, pct)
            except Exception:
                pass

    # ── Sampling for large datasets (FIX 4: Tiered Logic) ──────────
    original_row_count = len(df)
    sampled = False
    if original_row_count > 10000:
        df = _apply_smart_sampling(df)
        sampled = True

    _progress("classifying", 10)
    classifier = ColumnClassifier()
    profile    = classifier.classify(df)

    _progress("computing_metrics", 25)
    computer   = MetricComputer()
    metrics    = computer.compute(df, profile)

    _progress("evaluating_rules", 45)
    rule_eng   = BusinessRuleEngine()
    insights, rule_warnings = rule_eng.evaluate(df, profile, metrics)

    _progress("detecting_anomalies", 60)
    anomaly    = AnomalyDetector()
    warnings   = anomaly.detect(df, profile) + rule_warnings

    if sampled:
        warnings.insert(
            0,
            f"⚡ Large dataset: {original_row_count:,} rows sampled to "
            f"{len(df):,} for fast analysis. Metrics are statistically representative."
        )

    # Move Domain Detection BEFORE chart recommendation (Fix for UnboundLocalError)
    _progress("detecting_domain", 75)
    domain_engine = DomainDetector()
    domain_info = domain_engine.detect(df)
    domain_id = domain_info.get("id", "general")

    _progress("analyzing_drivers", 80)
    driver_engine = KeyDriverAnalyzer()
    driver_info = driver_engine.analyze(df, profile, domain_id=domain_id)

    # Step 4: Insight Synthesis & Compression (V2 Pipeline)
    synthesizer = DecisionIntelligenceSynthesizer()
    compressed_insights = synthesizer.synthesize(insights, driver_info, domain_id=domain_id)

    _progress("generating_charts", 85)
    chart_rec  = SmartChartRecommender()
    charts     = chart_rec.recommend(df, profile, compressed_insights, max_charts=max_charts, domain_id=domain_id)

    # Executive summary (Step 7: Strategic Brief)
    high_count = sum(1 for i in insights if "🔴" in str(i.impact))
    exec_summary = _build_exec_summary(df, profile, metrics, high_count, domain_info, driver_info, insights=compressed_insights, raw_insights=insights)

    _progress("done", 100)

    # Step 8: Narrate final state
    narrator = InsightNarrator()
    final_insight_dicts = narrator.narrate(compressed_insights, profile)

    # Step 9: Safe Mapping Layer (Step 1 - safe return layer)
    # Extract recommendations using the new RecommendationEngine
    rec_engine = RecommendationEngine(domain=domain_id)
    final_recs = rec_engine.generate(compressed_insights, max_count=5)

    result = {
        "domain": domain_info,
        "target": driver_info.get("target"),
        "key_drivers": driver_info.get("drivers", []),
        "profile": {
            "identifiers": profile.identifiers,
            "numericals": profile.numericals,
            "categoricals": profile.categoricals,
            "temporals": profile.temporals,
            "binaries": profile.binaries,
        },
        "computed_metrics": {k: {
            "name": v.name,
            "value": v.value,
            "formatted": v.formatted,
            "description": v.description
        } for k, v in metrics.items()},
        "strategic_brief": final_insight_dicts,
        "recommendations": final_recs,
        "executive_summary": exec_summary,
        "warnings": warnings
    }
    
    # Assertion Guard (Step 5)
    assert isinstance(result["strategic_brief"], list), "strategic_brief MUST be a list"
    print("DEBUG STRATEGIC BRIEF:", len(result["strategic_brief"]), "items found.")
    
    return result


# ============================================================
# HELPERS
# ============================================================

def _fmt_currency(val: float) -> str:
    """Format number as Indian Rupee with smart scaling."""
    abs_val = abs(val)
    sign = "" if val >= 0 else "-"
    if abs_val >= 1_00_00_000:      # 1 crore
        return f"{sign}₹{abs_val/1_00_00_000:.2f} Cr"
    if abs_val >= 1_00_000:          # 1 lakh
        return f"{sign}₹{abs_val/1_00_000:.2f} L"
    if abs_val >= 1_000:
        return f"{sign}₹{abs_val/1_000:.1f}K"
    return f"{sign}₹{abs_val:,.0f}"


def _build_exec_summary(df: pl.DataFrame, profile: DataProfile, metrics: dict, high_impact_count: int, domain_info: dict, driver_info: dict, insights: list = None, raw_insights: list = None) -> str:
    """Step 7: Generate High-End Executive Strategic Brief (3-5 lines)."""
    domain_id = domain_info.get("id", "general")

    # Pass raw_insights for temporal detection — compressed may have dropped it
    all_insights_for_temporal = (raw_insights or []) + (insights or [])
    builder = StrategicBriefBuilder(
        domain=domain_id,
        df=df,
        insights=all_insights_for_temporal,
        corr_matrix=driver_info.get("corr_matrix")
    )
    return builder.build()


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

# Keywords for column type inference (case-insensitive, snake_case or Title Case)
_NUMERIC_KW = {"amount", "quantity", "profit", "price", "revenue",
               "sales", "cost", "total", "count", "units", "value", "income"}
_DATE_KW    = {"date", "time", "day", "when", "created", "updated"}


def validate_dataframe(df: pl.DataFrame) -> dict:
    """
    Runs full data quality audit on uploaded DataFrame.
    Returns a structured report the frontend can render.
    Column matching is case-insensitive and supports both snake_case and Title Case.
    """
    import pandas as pd

    issues: list[dict] = []
    summary: dict = {"total_rows": len(df), "clean_rows": 0, "issue_count": 0}
    pdf = df.to_pandas()

    # ── 1. MISSING VALUES ─────────────────────────────────────────────────────
    for col in pdf.columns:
        count = int(pdf[col].isnull().sum())
        if count > 0:
            issues.append({
                "type":     "MISSING_VALUES",
                "severity": "medium",
                "column":   col,
                "count":    count,
                "message":  f"{count} missing values in '{col}'",
                "rows":     [int(r) for r in pdf[pdf[col].isnull()].index[:100]],
            })

    # ── 2. NON-NUMERIC IN NUMERIC-LIKE COLUMNS ────────────────────────────────
    for col in pdf.columns:
        if not any(kw in col.lower() for kw in _NUMERIC_KW):
            continue
        if pdf[col].dtype != object:
            continue
        bad_mask = pd.to_numeric(pdf[col], errors="coerce").isna() & pdf[col].notna()
        bad_rows = pdf[bad_mask]
        if len(bad_rows):
            issues.append({
                "type":     "NON_NUMERIC",
                "severity": "critical",
                "column":   col,
                "count":    len(bad_rows),
                "message":  f"{len(bad_rows)} non-numeric values in '{col}'",
                "rows":     [int(r) for r in bad_rows.index[:100]],
                "values":   [str(v) for v in bad_rows[col].tolist()[:20]],
            })

    # ── 3. UNPARSEABLE DATES ──────────────────────────────────────────────────
    for col in pdf.columns:
        if not any(kw in col.lower() for kw in _DATE_KW):
            continue
        if pdf[col].dtype in ("int64", "float64"):
            continue
        bad_dates = pdf[
            pd.to_datetime(pdf[col], errors="coerce").isna() & pdf[col].notna()
        ]
        if len(bad_dates):
            issues.append({
                "type":     "BAD_DATE",
                "severity": "critical",
                "column":   col,
                "count":    len(bad_dates),
                "message":  f"{len(bad_dates)} unparseable dates in '{col}'",
                "rows":     [int(r) for r in bad_dates.index[:100]],
                "values":   [str(v) for v in bad_dates[col].tolist()[:20]],
            })

    # ── 4. DUPLICATE ROWS ─────────────────────────────────────────────────────
    dup_count = int(pdf.duplicated().sum())
    if dup_count > 0:
        issues.append({
            "type":     "DUPLICATES",
            "severity": "medium",
            "column":   "ALL",
            "count":    dup_count,
            "message":  f"{dup_count} fully duplicate rows detected",
            "rows":     [int(r) for r in pdf[pdf.duplicated()].index[:100]],
        })

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    critical = sum(1 for i in issues if i["severity"] == "critical")
    medium   = sum(1 for i in issues if i["severity"] == "medium")

    summary["issue_count"] = len(issues)
    summary["critical"]    = critical
    summary["medium"]      = medium
    summary["clean_rows"]  = max(0, len(pdf.dropna()) - dup_count)
    summary["can_analyze"] = critical == 0

    return {"summary": summary, "issues": issues}


def auto_clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Auto-fix medium issues (nulls, duplicates).
    Returns cleaned DataFrame — critical issues must be fixed by user.
    """
    import pandas as pd

    pdf = df.to_pandas()
    pdf = pdf.dropna()
    pdf = pdf.drop_duplicates()

    for col in pdf.columns:
        if pdf[col].dtype == object and any(kw in col.lower() for kw in _NUMERIC_KW):
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
    pdf = pdf.dropna()

    for col in pdf.columns:
        if pdf[col].dtype == object and any(kw in col.lower() for kw in _DATE_KW):
            pdf[col] = pd.to_datetime(pdf[col], errors="coerce")
    pdf = pdf.dropna()

    return pl.from_pandas(pdf)

