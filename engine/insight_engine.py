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
from dataclasses import dataclass, field
from typing import Optional

import polars as pl
import pandas as pd
import numpy as np

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
    description: str
    impact: str  # "high" | "medium" | "low"
    recommendation: str
    confidence: str = "medium"  # "high" | "medium" | "low"
    score: float = 0.0          # Used for prioritization
    chart_type: str = "none"
    chart_data: Optional[dict] = None
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
        """Assign semantic sub-roles to drive metric computation."""
        for col in profile.numericals:
            cl = col.lower()
            if any(k in cl for k in PRICE_KEYWORDS):
                if profile.price_col is None:
                    profile.price_col = col
            elif any(k in cl for k in QTY_KEYWORDS):
                if profile.qty_col is None:
                    profile.qty_col = col
            elif any(k in cl for k in REVENUE_KEYWORDS):
                if profile.revenue_col is None:
                    profile.revenue_col = col
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
# 3. METRIC COMPUTER
# ============================================================

class MetricComputer:
    """Compute derived business metrics from classified columns."""

    def compute(self, df: pl.DataFrame, profile: DataProfile) -> dict[str, ComputedMetric]:
        metrics: dict[str, ComputedMetric] = {}

        # ── Revenue ───────────────────────────────────────────────────────
        revenue_series = self._compute_revenue_series(df, profile)
        if revenue_series is not None:
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

    REVENUE_CONCENTRATION_THRESHOLD = 0.50   # >50% revenue from one category → risk
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

        # ── Confidence Scoring & Prioritization ────────────────────────────
        confidence, conf_weight, conf_text = self._compute_confidence(df)
        impact_scores = {"high": 3.0, "medium": 2.0, "low": 1.0}
        
        for ins in insights:
            ins.confidence = confidence
            ins.score = impact_scores.get(ins.impact, 1.0) * conf_weight
            
            # Format the output explicitly 
            ins.description = f"{ins.description}\nConfidence: {confidence.capitalize()} ({conf_text})."

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
                    title=f"{top_name} Dominates Revenue",
                    description=(
                        f"What is happening: Specific segments account for >{self.REVENUE_CONCENTRATION_THRESHOLD*100:.0f}% of total categorical revenue.\n"
                        f"Qualified segments & WHY: {qual_str} qualified by individually generating >{self.REVENUE_CONCENTRATION_THRESHOLD*100:.0f}% of total revenue.\n"
                        f"Excluded segments & WHY: Excluded segments like {excl_str} failed to cross the dependency threshold.\n"
                        f"Why it matters: This high concentration creates severe business risk if {top_name} experiences supply chain issues or demand shifts."
                    ),
                    impact="high",
                    recommendation=(
                        f"What action to take: Diversify revenue streams. Invest in growing other {cat} segments "
                        f"to reduce dependency on {top_name}. A target below 40% concentration is healthier."
                    ),
                    chart_type="bar",
                    chart_data={
                        "labels": grouped[cat].head(10).astype(str).tolist(),
                        "values": [round(v, 2) for v in grouped[rev_col_name].head(10).tolist()],
                        "title": f"Revenue by {cat}",
                        "y_label": "Revenue"
                    },
                    qualified_segments=[str(row[cat]) for _, row in high_concentration.iterrows()],
                    excluded_segments=[str(row[cat]) for _, row in non_qual.head(3).iterrows()] if not non_qual.empty else [],
                    rule_type="revenue_dominance"
                ))
            else:
                # Still show the top category insight
                bottom = grouped.iloc[-1]
                gap_pct = (top_val - bottom[rev_col_name]) / max(total_rev, 1) * 100
                
                qual_str = f"{top_name} ({_fmt_currency(top_val)}, {top_pct:.0f}%)"
                excl_str = f"{str(bottom[cat])} ({_fmt_currency(bottom[rev_col_name])}, {bottom[rev_col_name]/total_rev*100:.0f}%)"
                
                insights.append(BusinessInsight(
                    title=f"Revenue Distribution by {cat}",
                    description=(
                        f"What is happening: Revenue is distributed safely across {cat} segments, without critical dependency on a single source.\n"
                        f"Qualified segments & WHY: {qual_str} qualified as the leading segment by driving the highest volume.\n"
                        f"Excluded segments & WHY: {excl_str} is highlighted as the lowest performer, requiring assistance.\n"
                        f"Why it matters: Understanding the disparity between top and bottom performers helps allocate resources efficiently."
                    ),
                    impact="medium",
                    recommendation=(
                        f"What action to take: Protect market share and retain inventory for {top_name}. "
                        f"Investigate why {str(bottom[cat])} underperforms and consider strategic repositioning or discontinuation."
                    ),
                    chart_type="bar",
                    chart_data={
                        "labels": grouped[cat].astype(str).tolist(),
                        "values": [round(v, 2) for v in grouped[rev_col_name].tolist()],
                        "title": f"Revenue by {cat}",
                        "y_label": "Revenue"
                    },
                    qualified_segments=[str(bottom[cat])],
                    excluded_segments=[str(top_name)],
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
                    title=f"Perfect Quality: No Returns in Segment",
                    description=(
                        f"What is happening: Specific segments within {cat} have a confirmed mathematically perfect 0% return rate based on independent grouping.\n"
                        f"Qualified segments & WHY: {qual_str} qualified because their returned_count == 0.\n"
                        f"Excluded segments & WHY: Excluded segments like {excl_str} failed because their returned_count > 0."
                    ),
                    impact="high",
                    recommendation="What action to take: Replicate the quality assurance and delivery practices of the qualified segments across the entire catalog.",
                    chart_type="bar",
                    chart_data=chart_data,
                    qualified_segments=[str(row[cat]) for _, row in perfect_entities.iterrows()],
                    excluded_segments=[str(row[cat]) for _, row in non_perfect.head(3).iterrows()] if not non_perfect.empty else [],
                    rule_type="perfect_quality"
                ))

            # 2. High Return Rate logic
            high_entities = grouped[grouped["return_rate"] > (global_rate / 100.0) * self.HIGH_RETURN_RATE_MULTIPLIER]
            # Ensure return rate > 0
            high_entities = high_entities[high_entities["return_rate"] > 0].sort_values("return_rate", ascending=False)
            
            if not high_entities.empty:
                qual_str = ", ".join([f"{row[cat]} ({row['returned_count']}/{row['total_orders']} returns, {row['return_rate']*100:.1f}%)" for _, row in high_entities.head(3).iterrows()])
                if len(high_entities) > 3:
                    qual_str += " and others"
                    
                normal_entities = grouped[(grouped["return_rate"] <= (global_rate / 100.0) * self.HIGH_RETURN_RATE_MULTIPLIER) & (grouped["total_orders"] > 0)]
                excl_str = "None"
                if not normal_entities.empty:
                    excl_str = ", ".join([f"{row[cat]} ({row['returned_count']}/{row['total_orders']}, {row['return_rate']*100:.1f}%)" for _, row in normal_entities.head(3).iterrows()])

                insights.append(BusinessInsight(
                    title=f"High Return Rate Detected in {cat}",
                    description=(
                        f"What is happening: Certain segments severely underperform the global return average of {global_rate:.1f}%.\n"
                        f"Qualified segments & WHY: {qual_str} qualified because their return rates are > 1.5x the global average.\n"
                        f"Excluded segments & WHY: Excluded segments like {excl_str} maintained acceptable rates."
                    ),
                    impact="high",
                    recommendation="What action to take: Investigate product quality, description accuracy, and delivery conditions for the qualified segments.",
                    chart_type="bar",
                    chart_data=chart_data,
                    qualified_segments=[str(row[cat]) for _, row in high_entities.head(3).iterrows()],
                    excluded_segments=[str(row[cat]) for _, row in normal_entities.head(3).iterrows()] if not normal_entities.empty else [],
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
                    title="Delivery Delays Increase Returns",
                    description=(
                        f"What is happening: There is a {direction} correlation (r = {corr:.2f}) between "
                        f"longer delivery times ({del_col}) and higher return probabilities.\n"
                        f"Qualified segments & WHY: All {len(sub)} correlated rows qualified for the delivery delay risk.\n"
                        f"Excluded segments & WHY: Rows with missing {del_col} or {ret_col} were excluded.\n"
                        f"Why it matters: Delivery delays directly damage profitability via increased returns, likely because customers lose interest or buy elsewhere while waiting."
                    ),
                    impact="high",
                    recommendation=(
                        "What action to take: Prioritize and reduce delivery time. Negotiate tighter SLAs with logistics providers and offer express shipping for high-risk items."
                    ),
                    chart_type="scatter",
                    chart_data={"x_col": del_col, "y_col": ret_col}
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
                        desc = (
                            f"What is happening: {top_name} accounts for {pct*100:.0f}% of all {col} records.\n"
                            f"Qualified segments & WHY: {qual_str} qualified because it accounts for >{threshold*100:.0f}% of total volume.\n"
                            f"Excluded segments & WHY: Excluded segments like {excl_str} failed to cross the threshold.\n"
                            f"Why it matters: Heavy reliance on a single payment method creates systemic vulnerability and checkout friction for users preferring other options."
                        )
                        rec = f"What action to take: Optimize the checkout flow specifically for {top_name}. Simultaneously, introduce and promote alternative {col} options to reduce dependency risk."
                    else:
                        title = f"{top_name} Dominates {col}"
                        desc = (
                            f"What is happening: {top_name} makes up {pct*100:.0f}% of the volume in {col}.\n"
                            f"Qualified segments & WHY: {qual_str} qualified because it accounts for >{threshold*100:.0f}% of total volume.\n"
                            f"Excluded segments & WHY: Excluded segments like {excl_str} failed to cross the threshold.\n"
                            f"Why it matters: Over-concentration in one categorical attribute exposes the business to single points of failure."
                        )
                        rec = f"What action to take: Diversify offerings in {col} to broaden appeal and mitigate reliance on {top_name}."

                    insights.append(BusinessInsight(
                        title=title,
                        description=desc,
                        impact="medium",
                        recommendation=rec,
                        chart_type="pie",
                        chart_data={
                            "labels": counts[col].head(5).to_list(),
                            "values": counts["count"].head(5).to_list()
                        },
                        qualified_segments=[str(row[col]) for _, row in counts_filtered.iterrows()],
                        excluded_segments=[str(row[col]) for _, row in non_qual.head(3).iterrows()] if not non_qual.empty else [],
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
                title=f"Top-Performing {geo_col}: {top[geo_col]}",
                description=(
                    f"What is happening: Revenue generation is uneven across {geo_col} regions.\n"
                    f"Qualified segments & WHY: {qual_str} qualified as the leading segment by driving the highest volume.\n"
                    f"Excluded segments & WHY: {excl_str} is highlighted as the weakest performer, requiring assistance.\n"
                    f"Why it matters: Regional or categorical discrepancies in revenue generation highlight opportunities for localized strategies."
                ),
                impact="medium",
                recommendation=(
                    f"What action to take: Increase marketing and inventory allocation for {str(top[geo_col])}. "
                    f"Investigate barriers in {str(bottom[geo_col])} — consider localised campaigns, pricing adjustments, or logistics improvements."
                ),
                chart_type="bar",
                chart_data={
                    "labels": grouped[geo_col].tolist(),
                    "values": [round(v, 2) for v in grouped["Revenue (₹)"].tolist()],
                    "title": f"Revenue by {geo_col}",
                    "y_label": "Revenue (₹)"
                },
                qualified_segments=[str(bottom[geo_col])],
                excluded_segments=[str(top[geo_col])],
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
                        insights.append(BusinessInsight(
                            title=f"{c1} & {c2} Are Strongly Linked",
                            description=(
                                f"What is happening: When {c1} goes up, {c2} consistently {direction} "
                                f"(correlation = {corr:.2f}). "
                                f"Why it matters: This strong relationship indicates that changes in one metric can be used to reliably predict changes in the other."
                            ),
                            impact="medium",
                            recommendation=(
                                f"What action to take: Leverage the correlation between {c1} and {c2} to build predictive pricing or demand models."
                            )
                        ))
        except Exception:
            pass
        return insights

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
            out.append({
                "title": ins.title,
                "description": ins.description,
                "impact": ins.impact,
                "recommendation": ins.recommendation,
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
    ) -> list[dict]:
        """Return a list of chart spec dicts ready for the existing Plotly renderer."""
        import plotly.express as px
        import plotly.graph_objects as go
        import json

        pdf = df.to_pandas()
        charts = []
        chart_ids_used: set[str] = set()

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
                    title=f"Revenue by {cat}",
                    color=rev_col, color_continuous_scale="Viridis",
                    text_auto=".2s"
                )
                fig.update_layout(template="plotly_dark",
                                  coloraxis_showscale=False, showlegend=False)
                add("revenue_by_cat", {
                    "chart_id": "revenue_by_cat",
                    "chart_type": "bar",
                    "title": f"Revenue by {cat}",
                    "description": f"Total revenue breakdown across {cat} segments",
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
                    title=f"Return Rate by {cat}",
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
                    "title": f"Return Rate by {cat}",
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
                pdf_tmp[date_col] = pd.to_datetime(pdf_tmp[date_col], errors="coerce")
                pdf_tmp = pdf_tmp.dropna(subset=[date_col])
                if qty_col:
                    pdf_tmp["__rev__"] = pdf_tmp[price_col].fillna(0) * pdf_tmp[qty_col].fillna(0)
                else:
                    pdf_tmp["__rev__"] = pdf_tmp[price_col].fillna(0)
                pdf_tmp["month"] = pdf_tmp[date_col].dt.to_period("M").astype(str)
                monthly = pdf_tmp.groupby("month")["__rev__"].sum().reset_index()
                monthly = monthly.sort_values("month")
                if len(monthly) >= 2:
                    fig = px.line(
                        monthly, x="month", y="__rev__",
                        title="Monthly Revenue Trend",
                        markers=True
                    )
                    fig.update_traces(line_color="#6366f1", line_width=2)
                    fig.update_layout(template="plotly_dark",
                                      xaxis_title="Month", yaxis_title="Revenue")
                    add("revenue_over_time", {
                        "chart_id": "revenue_over_time",
                        "chart_type": "line",
                        "title": "Monthly Revenue Trend",
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
                pdf_tmp["__rev__"] = (
                    pdf[price_col].fillna(0) * pdf[qty_col].fillna(0)
                    if qty_col else pdf[price_col].fillna(0)
                )
                grp = (
                    pdf_tmp.groupby(geo_col)["__rev__"].sum()
                    .reset_index()
                    .sort_values("__rev__", ascending=False)
                    .head(12)
                )
                fig = px.bar(
                    grp, x=geo_col, y="__rev__",
                    title=f"Revenue by {geo_col}",
                    color="__rev__", color_continuous_scale="Blues",
                    text_auto=".2s"
                )
                fig.update_layout(template="plotly_dark",
                                  coloraxis_showscale=False, showlegend=False)
                add("geo_revenue", {
                    "chart_id": "geo_revenue",
                    "chart_type": "bar",
                    "title": f"Revenue by {geo_col}",
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
        if cat:
            try:
                counts = pdf[cat].value_counts().reset_index().head(10)
                counts.columns = [cat, "count"]
                fig = px.bar(
                    counts, x=cat, y="count",
                    title=f"Order Count by {cat}",
                    color="count", color_continuous_scale="Purples",
                    text_auto=True
                )
                fig.update_layout(template="plotly_dark",
                                  coloraxis_showscale=False, showlegend=False)
                add("count_by_cat", {
                    "chart_id": "count_by_cat",
                    "chart_type": "bar",
                    "title": f"Order Volume by {cat}",
                    "description": f"Number of orders per {cat} — volume ≠ revenue",
                    "plotly_json": json.loads(fig.to_json()),
                    "columns_used": [cat],
                    "priority_score": 70,
                    "insight_reason": "Order volume distribution by category",
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


SAMPLE_THRESHOLD = 50_000   # rows
SAMPLE_SIZE      = 20_000   # rows to keep


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

    # ── Sampling for large datasets ──────────────────────────────
    original_row_count = len(df)
    sampled = False
    if original_row_count > SAMPLE_THRESHOLD:
        df = df.sample(n=SAMPLE_SIZE, seed=42)
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
            f"{SAMPLE_SIZE:,} for fast analysis. Metrics are statistically representative."
        )

    _progress("narrating", 75)
    narrator   = InsightNarrator()
    insight_dicts = narrator.narrate(insights, profile)

    _progress("generating_charts", 85)
    chart_rec  = SmartChartRecommender()
    charts     = chart_rec.recommend(df, profile, insights, max_charts=max_charts)

    # Executive summary
    high_count = sum(1 for i in insights if i.impact == "high")
    exec_summary = _build_exec_summary(df, profile, metrics, high_count)

    _progress("done", 100)

    return {
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
        "insights": insight_dicts[:max_insights],
        "charts": charts,
        "warnings": warnings,
        "executive_summary": exec_summary,
        "recommendations": [i["recommendation"] for i in insight_dicts
                            if i.get("impact") in ("high", "medium")][:5],
    }


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
) -> str:
    parts = [
        f"Dataset contains {len(df):,} records across {len(df.columns)} columns."
    ]
    if "total_revenue" in metrics:
        parts.append(
            f"Total revenue: {metrics['total_revenue'].formatted}."
        )
    if "return_rate" in metrics:
        parts.append(
            f"Overall return rate: {metrics['return_rate'].formatted}."
        )
    if high_count > 0:
        parts.append(
            f"{high_count} high-impact finding{'s' if high_count > 1 else ''} "
            f"require immediate attention."
        )
    if profile.identifiers:
        parts.append(
            f"ID columns ({', '.join(profile.identifiers)}) excluded from analysis."
        )
    return " ".join(parts)
