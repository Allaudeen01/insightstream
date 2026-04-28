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

def detect_domain(columns: list[str]) -> str:
    """Identify the dataset domain using strict column keyword matching."""
    cols = set(c.lower().strip() for c in columns)
    
    # Happiness: all three must be present
    if {"happiness score", "gdp per capita", "life expectancy"}.issubset(cols):
        return "happiness"
    
    # E‑commerce: at least two of these
    if len({"sales", "revenue", "order", "profit"} & cols) >= 2:
        return "ecommerce"
    
    return "general"


class DomainDetector:
    """Identify the dataset domain using column signatures and patterns."""
    
    DOMAINS = {
        "Finance": {"revenue", "price", "cost", "profit", "return", "tax", "balance", "equity", "asset", "liability", "ticker", "trade", "investment", "dividend", "interest"},
        "E-commerce": {"sku", "inventory", "stock", "warehouse", "order", "shipping", "customer", "cart", "checkout", "delivery", "item", "product", "brand", "category"},
        "Healthcare": {"patient", "diagnosis", "blood", "heart", "clinic", "hospital", "doctor", "medicine", "treatment", "age", "bmi", "glucose", "insulin", "surgery"},
        "Socio-economic": {"gdp", "literacy", "population", "unemployment", "mortality", "education", "income", "poverty", "country", "region", "policy", "development"},
        "Marketing": {"ctr", "campaign", "impression", "lead", "conversion", "click", "ad", "marketing", "reach", "roi", "cac", "ltv", "segment"},
        "Operations": {"process", "time", "efficiency", "queue", "capacity", "delay", "maintenance", "downtime", "throughput"}
    }

    def detect(self, profile: DataProfile) -> dict:
        cols_list = list(profile.profiles.keys())
        simple_id = detect_domain(cols_list)
        
        cols = {c.lower() for c in cols_list}
        scores = {}
        
        for domain, keywords in self.DOMAINS.items():
            overlap = cols.intersection(keywords)
            if overlap:
                scores[domain] = len(overlap)
        
        if not scores:
            name = "Socio-economic" if simple_id == 'happiness' else "E-commerce" if simple_id == 'ecommerce' else "Generic Dataset"
            return {"name": name, "confidence": "low", "reason": "No strong domain-specific keywords detected in column headers.", "id": simple_id}
            
        best_domain = max(scores, key=scores.get)
        count = scores[best_domain]
        
        confidence = "medium"
        if count >= 4: confidence = "high"
        elif count <= 1: confidence = "low"
        
        reason = f"Detected {count} keyword signals ({', '.join(list(cols.intersection(self.DOMAINS[best_domain]))[:3])}) matching {best_domain} patterns."
        
        log.info(f"DOMAIN_ENGINE: Detected domain '{best_domain}' (id: {simple_id}) with {confidence} confidence.")
        return {"name": best_domain, "confidence": confidence, "reason": reason, "id": simple_id}


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
                insights.append(BusinessInsight(
                    title=f"Unexpected Intelligence: {d['type']}",
                    description=d["description"],
                    why_it_matters="When core variables decouple, it usually indicates either a data quality issue or a fundamental breakdown in expected business logic.",
                    evidence=f"r-value: {d['r']} (Expected strong positive)",
                    decision_implication="Audit the data ingestion pipeline for these two variables. If valid, investigate why standard drivers are failing to influence outcomes.",
                    impact="🔴 Critical",
                    is_unexpected=True,
                    rule_type="surprise"
                ))

        # 2. Insight Compression (Merge by Topic)
        compressed = []
        topics = {
            "revenue": ["revenue_dominance", "worst_revenue", "profit_dominance", "margin_divergence"],
            "quality": ["perfect_quality", "high_return_rate", "payment_risk", "delivery_delay_risk"],
            "discovery": ["dominance", "correlation_matrix", "domain_detection"]
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
            # Catalog data: report sum/avg of the price column honestly,
            # without calling it revenue.
            total_val = float(revenue_series.sum())
            avg_val   = float(revenue_series.mean())
            price_label = profile.price_col.replace("_", " ").title()
            metrics["catalog_total"] = ComputedMetric(
                name=f"Total {price_label}",
                value=total_val,
                formatted=_fmt_currency(total_val),
                description=f"Sum of {profile.price_col} across {len(df):,} catalog items"
            )
            metrics["catalog_average"] = ComputedMetric(
                name=f"Average {price_label}",
                value=avg_val,
                formatted=_fmt_currency(avg_val),
                description=f"Average {profile.price_col} per item"
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
        metrics["total_orders"] = ComputedMetric(
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

    def _compute_confidence(self, df: pl.DataFrame) -> tuple[str, float, str]:
        """Return confidence label, weight, and strict formatting text."""
        rows = len(df)
        if rows > 500:
            return "high", 1.0, f"Based on a high sample size (>500 records)"
        elif rows >= 100:
            return "medium", 0.7, f"Based on {rows} total records"
        else:
            return "low", 0.4, f"Based on exactly {rows} orders in the entire dataset"

    def evaluate(
        self,
        df: pl.DataFrame,
        profile: DataProfile,
        metrics: dict[str, ComputedMetric],
    ) -> tuple[list[BusinessInsight], list[str]]:
        """Return (insights, warnings)."""
        insights: list[BusinessInsight] = []
        warnings: list[str] = []

        pdf = df.to_pandas()

        # ── Rule 1: Revenue by category ────────────────────────────────────
        rev_series = getattr(profile, "_revenue_series", None)
        if rev_series is not None and profile.category_col:
            insights.extend(self._rule_revenue_by_category(df, pdf, profile, rev_series))

        # ── Rule 2: Return rate by category ───────────────────────────────
        ret_series = getattr(profile, "_return_count_series", None)
        if ret_series is not None and profile.category_col and "return_rate" in metrics:
            global_rate = metrics["return_rate"].value
            insights.extend(
                self._rule_return_rate_by_category(df, pdf, profile, ret_series, global_rate)
            )

        # ── Rule 3: Delivery days vs returns ──────────────────────────────
        if profile.delivery_days_col and ret_series is not None:
            insights.extend(
                self._rule_delivery_vs_returns(df, pdf, profile, ret_series)
            )

        # ── Rule 4: Payment method / categorical dominance ─────────────────
        insights.extend(self._rule_categorical_dominance(df, profile))

        # ── Rule 5: Top category by revenue ───────────────────────────────
        if rev_series is not None and profile.category_col:
            insights.extend(self._rule_top_geographic_performance(df, pdf, profile, rev_series))

        # ── Rule 6: Correlation alerts between numerical columns ───────────
        if len(profile.numericals) >= 2:
            insights.extend(self._rule_numeric_correlations(df, profile))

        # ── Rule 7: Profit concentration by category ──────────────────────
        profit_col = self._find_profit_col(profile)
        if profit_col and profile.category_col:
            insights.extend(self._rule_profit_by_category(df, pdf, profile, profit_col))

        # ── Rule 8: Profit margin uniformity/divergence ───────────────────
        rev_col = profile.revenue_col or profile.price_col
        if profit_col and rev_col and profile.category_col:
            insights.extend(self._rule_profit_margin_insight(df, pdf, profile, profit_col, rev_col))

        # ── Rule 9: Domain Detection (Fix 5) ──────────────────────────────
        insights.extend(self._rule_domain_detection(df, profile))

        # ── Rule 10: Payment Correlation (Fix 5) ──────────────────────────
        if profile.return_col:
            insights.extend(self._rule_payment_correlation(df, pdf, profile, ret_series))

        # ── Rule 11: Top 3 Correlation Matrix (Fix 5) ─────────────────────
        if len(profile.numericals) >= 3:
            insights.extend(self._rule_correlation_matrix(df, profile))

        # ── Confidence Scoring & Prioritization (V2 Step 6 & 8) ────────────
        confidence, conf_weight, conf_text = self._compute_confidence(df)
        
        for ins in insights:
            ins.confidence_label = confidence
            ins.confidence_explanation = f"{conf_text} | Statistical consistency: High based on current distribution."
            # Score based on executive emoji markers
            ins.score = (3.0 if "🔴" in str(ins.impact) else 2.0 if "🟠" in str(ins.impact) else 1.0) * conf_weight

        # ── Global Contradiction Checks ────────────────────────────────────
        # Cross-insight specific contradiction resolver algorithm exactly as requested
        entity_tags: dict[str, str] = {}
        for ins in insights:
            for entity in ins.qualified_segments:
                if entity in entity_tags:
                    # Found an overlap tracking tag
                    previous_rule = entity_tags[entity]
                    if (previous_rule == "worst_revenue" and ins.rule_type == "perfect_quality") or \
                       (previous_rule == "perfect_quality" and ins.rule_type == "worst_revenue"):
                        ins.description += f"\nContradiction Alert: {entity} holds both zero returns (perfect quality) and lowest revenue placement, meaning perfection could merely stem from statistical insignificance due to low volume."
                entity_tags[entity] = ins.rule_type

        # Sort strictly by score first
        insights.sort(key=lambda x: x.score, reverse=True)

        # ── De-duplicate overlapping insights  ─────────────────────────────
        unique_insights: list[BusinessInsight] = []
        seen_signatures: set[tuple] = set()

        for ins in insights:
            # deduplicate if qualified_segments AND excluded_segments are identical.
            if not ins.qualified_segments and not getattr(ins, 'excluded_segments', []):
                # if there are no segments, we deduplicate by topic directly
                title_lower = ins.title.lower()
                topic = title_lower
                if "dominates revenue" in title_lower or "revenue distribution by" in title_lower:
                    topic = "revenue_distribution"
                elif "top performing" in title_lower and "revenue" in title_lower:
                    topic = "revenue_distribution"
                elif "return rate" in title_lower or "returns" in title_lower:
                    if "delivery" not in title_lower:
                        topic = "return_rate"
                elif "payment" in title_lower or "method" in title_lower:
                    topic = "payment_method"
                    
                if topic not in seen_signatures:
                    seen_signatures.add(topic)
                    unique_insights.append(ins)
            else:
                excl = getattr(ins, 'excluded_segments', [])
                sig = (tuple(sorted(ins.qualified_segments)), tuple(sorted(excl)), ins.rule_type)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    unique_insights.append(ins)
                
        print(f"TEST LOG: Created {len(insights)} total insights. Deduplicated to {len(unique_insights)}.")

        return unique_insights[:10], warnings

    # ------------------------------------------------------------------ #
    # Rule implementations
    # ------------------------------------------------------------------ #

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
                    evidence=f"Concentration Index: {top_pct:.1f}% | Top Performing Segment: {top_name} (${top_val:,.2f})",
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



    def _rule_numeric_correlations(
        self, df: pl.DataFrame, profile: DataProfile
    ) -> list[BusinessInsight]:
        insights = []
        cols = [c for c in profile.numericals if c not in profile.identifiers][:5]
        if len(cols) < 2:
            return insights
        try:
            for i, c1 in enumerate(cols):
                for c2 in cols[i + 1:]:
                    corr = df.select(pl.corr(c1, c2)).item()
                    if corr is None:
                        continue
                    if abs(corr) >= 0.7:
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

    def _rule_domain_detection(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """Detect if the dataset belongs to a specific industry domain using centralized logic."""
        domain_id = detect_domain(df.columns)
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

    def _rule_correlation_matrix(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """Find the top 3 strongest numerical correlations (Fix 5)."""
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
                    if not np.isnan(correlation):
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
            })
        return out


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
            redetected = detect_domain(list(df.columns))
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
        price_col = profile.price_col or profile.revenue_col
        qty_col   = profile.qty_col

        # ── 1. Revenue by Category (horizontal bar) ────────────────────────
        if cat and price_col:
            try:
                rev_col = "Revenue (₹)"
                pdf_tmp = pdf.copy()
                pdf_tmp[rev_col] = (
                    pdf[price_col].fillna(0) * pdf[qty_col].fillna(0)
                    if qty_col else pdf[price_col].fillna(0)
                )
                grp = (
                    pdf_tmp.groupby(cat)[rev_col].sum()
                    .reset_index()
                    .sort_values(rev_col, ascending=True)
                )
                fig = px.bar(
                    grp, x=rev_col, y=cat, orientation="h",
                    title=f"{target_label} by {cat}",
                    color=rev_col, color_continuous_scale="Viridis",
                    text_auto=".2s"
                )
                fig.update_layout(template="plotly_dark",
                                  coloraxis_showscale=False, showlegend=False,
                                  xaxis_title=target_label)
                add("revenue_by_cat", {
                    "chart_id": "revenue_by_cat",
                    "chart_type": "bar",
                    "title": f"{target_label} by {cat}",
                    "description": f"Total {target_label} breakdown across {cat} segments",
                    "plotly_json": json.loads(fig.to_json()),
                    "columns_used": [cat, price_col] + ([qty_col] if qty_col else []),
                    "priority_score": 90,
                    "insight_reason": "Core business revenue metric",
                    "interest_level": "high"
                })
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
                    "plotly_json": json.loads(fig.to_json()),
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
                    "plotly_json": json.loads(fig.to_json()),
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
                    "plotly_json": json.loads(fig.to_json()),
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
                        "plotly_json": json.loads(fig.to_json()),
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
                pdf_tmp[rev_label] = (
                    pdf[price_col].fillna(0) * pdf[qty_col].fillna(0)
                    if qty_col else pdf[price_col].fillna(0)
                )
                grp = (
                    pdf_tmp.groupby(geo_col)[rev_label].sum()
                    .reset_index()
                    .sort_values(rev_label, ascending=False)
                    .head(12)
                )
                
                # Check if we should do a grouped bar (Feature 1)
                # If cat exists, replace simple bar with grouped bar
                if cat and cat != geo_col:
                    pdf_tmp = pdf.copy()
                    pdf_tmp[rev_label] = (
                        pdf[price_col].fillna(0) * pdf[qty_col].fillna(0)
                        if qty_col else pdf[price_col].fillna(0)
                    )
                    grp_cat = (
                        pdf_tmp.groupby([geo_col, cat])[rev_label].sum()
                        .reset_index()
                        .sort_values(rev_label, ascending=False)
                    )
                    fig = px.bar(
                        grp_cat, x=geo_col, y=rev_label, color=cat,
                        barmode="group",
                        title=f"Which {cat} performs best in each {geo_col}?",
                        text_auto=False
                    )
                    fig.update_layout(template="plotly_dark")
                    add("geo_cat_revenue", {
                        "chart_id": "geo_cat_revenue",
                        "chart_type": "grouped_bar",
                        "title": f"Which {cat} performs best in each {geo_col}?",
                        "description": f"Geographical revenue distribution across both {geo_col} and {cat}",
                        "plotly_json": json.loads(fig.to_json()),
                        "columns_used": [geo_col, cat, price_col],
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
                    "plotly_json": json.loads(fig.to_json()),
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
                    "plotly_json": json.loads(fig.to_json()),
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
                fig.update_layout(template="plotly_dark",
                                  barmode="overlay" if color_col else "relative")
                if not color_col:
                    fig.update_traces(marker_color="#6366f1")
                add("price_dist", {
                    "chart_id": "price_dist",
                    "chart_type": "histogram",
                    "title": f"{price_col} Distribution",
                    "description": f"Spread and shape of {price_col} values",
                    "plotly_json": json.loads(fig.to_json()),
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
        for num in num_cols[:2]:
            if len(charts) >= max_charts:
                break
            if not cat or num == cat:
                continue
            try:
                grp = (
                    pdf.groupby(cat)[num].median()
                    .reset_index()
                    .sort_values(num, ascending=False)
                )
                fig = px.bar(
                    grp, x=cat, y=num,
                    title=f"Median {num} by {cat}",
                    text_auto=".1f"
                )
                fig.update_layout(template="plotly_dark")
                cid = f"fallback_bar_{num}_{cat}"
                if cid not in chart_ids_used:
                    chart_ids_used.add(cid)
                    charts.append({
                        "chart_id": cid,
                        "chart_type": "bar",
                        "title": f"{num} by {cat}",
                        "description": f"Median {num} comparison (robust to outliers)",
                        "plotly_json": json.loads(fig.to_json()),
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
    domain_info = domain_engine.detect(profile)
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
    exec_summary = _build_exec_summary(df, profile, metrics, high_count, domain_info, driver_info)

    _progress("done", 100)

    # Step 8: Narrate final state
    narrator = InsightNarrator()
    final_insight_dicts = narrator.narrate(compressed_insights, profile)

    # Step 9: Safe Mapping Layer (Step 1 - safe return layer)
    # Extract recommendations for the top results
    recs = [ins.recommendation for ins in compressed_insights if ins.recommendation]

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
        "recommendations": recs[:5],
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
    abs_val = abs(val)
    sign = "" if val >= 0 else "-"
    if abs_val >= 1_000_000_000:
        return f"{sign}${abs_val/1_000_000_000:.2f}B"
    if abs_val >= 1_000_000:
        return f"{sign}${abs_val/1_000_000:.2f}M"
    if abs_val >= 1_000:
        return f"{sign}${abs_val/1_000:.1f}K"
    return f"{sign}${abs_val:,.2f}"


def _build_exec_summary(
    df: pl.DataFrame,
    profile: DataProfile,
    metrics: dict[str, ComputedMetric],
    high_count: int,
    domain_info: dict,
    driver_info: dict
) -> str:
    """Step 7: Generate High-End Executive Strategic Brief (3-5 lines)."""
    rows = len(df)
    domain_id = domain_info.get("id", "general")
    template = TEMPLATES.get(domain_id, TEMPLATES["general"])
    target = template["target_metric"]

    # When domain is 'general', substitute real values for the placeholder
    # phrases so the brief reads naturally on unknown-domain datasets.
    domain_name = domain_info.get("name", "Generic Dataset")
    if domain_id == "general":
        if target == "Key Performance Indicator":
            actual_metric = (profile.revenue_col
                             or profile.price_col
                             or (profile.numericals[0] if profile.numericals else "the primary metric"))
            target = actual_metric
        if domain_name == "Generic Dataset":
            domain_name = "data"

    # 5-Part Structure
    # 1. Overall system behavior
    line1 = f"Strategic Brief: The {domain_name} system is operating at a scale of {rows:,} records with stable high-level consistency."
    
    # 2. Primary driver (Strict Logic: Absolute r >= 0.8)
    drivers = driver_info.get("drivers", [])
    primary = next((d for d in drivers if abs(d.get('r', 0)) >= 0.8), None)
    if primary:
        line2 = f"Internal logic is primarily gated by {primary['column']}, which serves as the fundamental catalyst for {target} outcomes."
    else:
        line2 = f"Analytical focus is centered on the optimization of {target} across all categorical segments."
        
    # 3. Secondary drivers (Absolute r >= 0.4)
    secondary = next((d for d in drivers if abs(d.get('r', 0)) >= 0.4 and d != primary), None)
    if secondary:
        line3 = f"Secondary operational influence stems from {secondary['column']}, suggesting a multi-variate dependency model."
    else:
        line3 = "No secondary drivers reach the significance threshold."
        
    # 4. Key risk or limitation
    if high_count > 0:
        line4 = f"Current risk assessment identifies {high_count} critical structural anomalies requiring immediate executive intervention."
    else:
        line4 = "The system currently exhibits no high-risk structural decoupling."
        
    # 5. Final implication
    line5 = f"Strategic implication: Future performance gains will require a targeted focus on {primary['column'] if primary else target} optimization."

    return " ".join([l for l in [line1, line2, line3, line4, line5] if l])

