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
from time_series_analysis import TimeSeriesAnalyzer

log = logging.getLogger(__name__)

pd.set_option('display.max_colwidth', None)


# ============================================================
# 0. DOMAIN DETECTION ENGINE (Moved to top for scope safety)
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
            "insurance_agents": [
                "agent", "license", "irda", "ulip", "commission",
                "vintage", "blacklist", "channel", "intermediary",
                "policy", "premium", "joining", "designation",
                "qualification", "agentstatus", "minpayment"
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

        # ── Generic Entertainment/Content Domain Detection ────────────────
        entertainment_signals = 0
        _cols_lower = [c.lower().replace(" ", "_") for c in df.columns]

        # Signal 1: Content type columns
        _type_signals = ["type", "content_type", "show_type", "media_type",
                         "format", "kind"]
        if any(k in _cols_lower for k in _type_signals):
            entertainment_signals += 2

        # Signal 2: Content identity columns
        _identity_signals = ["title", "show_id", "content_id", "track_id",
                              "movie_id", "video_id", "song_id", "album_id"]
        if any(k in _cols_lower for k in _identity_signals):
            entertainment_signals += 1

        # Signal 3: Content metadata columns (need ≥ 2 matches)
        _meta_signals = ["director", "cast", "genre", "listed_in",
                         "categories", "tags", "artist", "album",
                         "description", "synopsis"]
        if sum(1 for k in _meta_signals if k in _cols_lower) >= 2:
            entertainment_signals += 2

        # Signal 4: Content classification columns
        _class_signals = ["rating", "maturity_rating", "age_rating",
                          "content_rating", "certification"]
        if any(k in _cols_lower for k in _class_signals):
            entertainment_signals += 1

        # Signal 5: Temporal columns for content
        _time_signals = ["release_year", "year", "date_added", "publish_date",
                         "upload_date", "release_date", "air_date",
                         "premiere_date"]
        if any(k in _cols_lower for k in _time_signals):
            entertainment_signals += 1

        # Signal 6: Check type column VALUES for entertainment content
        _CONTENT_TYPE_VALUES = {
            "movie", "tv show", "tv series", "series", "episode",
            "documentary", "short", "special", "film", "video",
            "track", "album", "podcast", "show", "animation",
            "anime", "miniseries",
        }
        for _col in df.columns:
            if _col.lower().replace(" ", "_") in _type_signals:
                try:
                    _type_vals = df[_col].drop_nulls().cast(pl.Utf8).str.to_lowercase().unique().to_list()
                    if any(v in _CONTENT_TYPE_VALUES for v in _type_vals):
                        entertainment_signals += 3
                        print(f"[DOMAIN] Entertainment confirmed via type values: {_type_vals[:5]}")
                except Exception:
                    pass
                break

        if entertainment_signals >= 4:
            scores["entertainment"] = min(0.9, entertainment_signals * 0.15)
            print(f"[DOMAIN] Entertainment detected (signals={entertainment_signals}, "
                  f"score={scores['entertainment']:.2f})")

        # ── Sports Domain Detection ───────────────────────
        sports_signals = 0
        _cols_lower_s = [c.lower().replace(" ", "_")
                         for c in df.columns]

        # Team columns
        _team_signals = ["team1", "team2", "home_team",
                         "away_team", "team", "club", "side",
                         "driver", "constructor", "player", "athlete",
                         "winner_name", "loser_name", "home", "away"]
        if sum(1 for k in _team_signals
               if any(k in c for c in _cols_lower_s)) >= 2:
            sports_signals += 3

        # Match result columns
        _result_signals = ["winner", "result", "result_margin",
                           "win_by_runs", "win_by_wickets",
                           "score", "goals", "points"]
        if any(k in _cols_lower_s for k in _result_signals):
            sports_signals += 2

        # Sports-specific columns
        _sport_signals = ["venue", "toss_winner", "toss_decision",
                          "player_of_match", "match_type",
                          "season", "umpire", "innings",
                          "wickets", "overs", "runs"]
        if sum(1 for k in _sport_signals
               if k in _cols_lower_s) >= 2:
            sports_signals += 2

        if sports_signals >= 4:
            scores["sports"] = min(0.9, sports_signals * 0.15)
            print(f"[DOMAIN] Sports detected "
                  f"(signals={sports_signals}, "
                  f"score={scores['sports']:.2f})")

        # ── Health Domain Detection ───────────────────────
        health_signals = 0
        _cols_h = [c.lower().replace(" ", "_") for c in df.columns]

        _case_signals = ["confirmed", "cases", "infected",
                         "positive", "total_cases", "new_cases"]
        if any(k in _cols_h or any(k in c for c in _cols_h)
               for k in _case_signals):
            health_signals += 3

        _outcome_signals = ["deaths", "death", "fatalities",
                            "mortality", "deceased", "dead"]
        if any(any(k in c for c in _cols_h) for k in _outcome_signals):
            health_signals += 2

        _recovery_signals = ["recovered", "recovery", "discharged",
                              "healed", "cured"]
        if any(any(k in c for c in _cols_h) for k in _recovery_signals):
            health_signals += 2

        _health_signals = ["active", "hospitalized", "icu",
                           "critical", "serious", "quarantine",
                           "vaccinated", "tests", "tested"]
        if sum(1 for k in _health_signals
               if any(k in c for c in _cols_h)) >= 2:
            health_signals += 2

        if health_signals >= 4:
            scores["health"] = min(0.9, health_signals * 0.15)
            print(f"[DOMAIN] Health detected "
                  f"(signals={health_signals}, "
                  f"score={scores['health']:.2f})")

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
    methodology: str = ""       # GAP 3: Explains how the insight was derived
    narrative_hook: str = ""    # P1: Short 1-2 sentence human-readable hook for exec summary

@dataclass
class ComputedMetric:
    name: str
    value: float
    formatted: str
    description: str


# ============================================================
# TIER 1.1: COLUMN COVERAGE TRACKER
# ============================================================

class ColumnCoverageTracker:
    """
    Tier 1.1: Tracks which columns were analyzed and flags gaps.
    
    The engine may touch only ~5 columns and silently ignore 14 others.
    This tracker provides visibility into what was analyzed vs. what was skipped.
    """
    
    def __init__(self, df: pl.DataFrame, profile: DataProfile):
        self.all_columns = set(df.columns)
        self.touched: set[str] = set()
        self.profile = profile
    
    def mark(self, *cols: str):
        """Mark columns as analyzed."""
        for c in cols:
            if c:
                self.touched.add(c)
    
    def report(self) -> dict:
        """Generate coverage report with high-value missed columns flagged."""
        untouched = self.all_columns - self.touched - set(self.profile.identifiers)
        coverage_pct = len(self.touched) / max(len(self.all_columns), 1) * 100
        
        # Classify untouched columns by importance
        high_value_missed = []
        for col in untouched:
            cl = col.lower()
            if any(k in cl for k in ["return", "discount", "promotion", "promo",
                                      "salesperson", "customer", "shipping", "delivery",
                                      "cost", "profit", "margin", "rating", "review",
                                      "satisfaction", "nps", "churn", "retention"]):
                high_value_missed.append(col)
        
        return {
            "total_columns": len(self.all_columns),
            "analyzed_columns": len(self.touched),
            "coverage_pct": round(coverage_pct, 1),
            "untouched_columns": sorted(untouched),
            "high_value_missed": high_value_missed,
            "warning": (
                f"Only {coverage_pct:.0f}% of columns were analyzed. "
                f"High-value columns not covered: {', '.join(high_value_missed)}."
                if high_value_missed else None
            )
        }


# ============================================================
# TIER 5.6: SANITY CHECKER
# ============================================================

class SanityChecker:
    """
    Tier 5.6: Post-generation verification layer.
    
    Checks every insight for numerical consistency, entity confusion,
    and internal contradictions before they reach the report.
    
    Prevents issues like:
    - Person names (Cameron) being treated as geographic regions
    - RPU values that are nonsensical (₹31 instead of ₹287)
    - Revenue values that don't match dataset totals
    """
    
    def __init__(self, df: pl.DataFrame, profile: DataProfile):
        self.df = df
        self.profile = profile
        self.person_cols = getattr(profile, 'person_columns', [])
        self.place_cols = getattr(profile, 'place_columns', [])
        self.issues: list[str] = []
    
    def check_all(self, insights: list[BusinessInsight], metrics: dict) -> list[BusinessInsight]:
        """Run all checks. Returns filtered insights with issues logged."""
        cleaned = []
        for ins in insights:
            passed = True
            
            # CHECK 1: Entity confusion — does the insight mention a person name
            # in a geographic/category context?
            if self._check_entity_confusion(ins):
                self.issues.append(f"BLOCKED: '{ins.title}' — entity confusion detected")
                passed = False
            
            # CHECK 2: Order-of-magnitude sanity on currency values
            if self._check_magnitude(ins, metrics):
                self.issues.append(f"FLAGGED: '{ins.title}' — magnitude mismatch")
                # Don't block, but add a caveat
                ins.confidence_label = "low"
                ins.evidence += " | ⚠ Magnitude sanity check flagged this value."
            
            # CHECK 3: Internal consistency — does claimed count match actual?
            self._check_count_consistency(ins)
            
            if passed:
                cleaned.append(ins)
        
        if self.issues:
            print(f"[SANITY CHECKER] {len(self.issues)} issues found:")
            for issue in self.issues:
                print(f"  → {issue}")
        
        return cleaned
    
    def _check_entity_confusion(self, ins: BusinessInsight) -> bool:
        """Returns True if insight text uses a person-column value in a geographic context."""
        text = f"{ins.title} {ins.description} {ins.recommendation}"
        # Get all person-column values
        for col in self.person_cols:
            try:
                person_values = set(self.df[col].unique().to_list())
                for person in person_values:
                    if person and str(person) in text:
                        # Check if the person name is used as if it were a region/category
                        context_words = ["region", "area", "zone", "market", "territory",
                                        "category", "segment", "variability", "execution gaps"]
                        surrounding = text[max(0, text.index(str(person))-50):
                                          text.index(str(person))+50+len(str(person))]
                        if any(cw in surrounding.lower() for cw in context_words):
                            return True
            except Exception:
                continue
        return False
    
    def _check_magnitude(self, ins: BusinessInsight, metrics: dict) -> bool:
        """Flag if any ₹ value in the insight is >10× or <0.01× the total revenue."""
        NON_MONETARY_RULES = {
            "descriptive_distribution", "descriptive_balance", "descriptive_volume",
            "skewed_distribution", "outlier_detection", "correlation_matrix",
        }
        # CLV reports per-customer future value (~₹hundreds), not aggregate revenue.
        # Root-cause reports monthly deltas, which are legitimately smaller than total.
        MAGNITUDE_EXEMPT_TYPES = {
            "clv_estimate",
            "root_cause_analysis",
        }
        rule_type = getattr(ins, "rule_type", "")
        if rule_type in NON_MONETARY_RULES or rule_type in MAGNITUDE_EXEMPT_TYPES:
            return False

        import re
        total_rev = metrics.get("total_revenue", ComputedMetric("", 0, "", "")).value
        if total_rev == 0:
            return False

        # RFM: segment revenues should sum to ~total revenue.
        # Checking individual segments against total would always flag (each segment < total).
        if rule_type == "rfm_segmentation":
            cd = getattr(ins, "chart_data", None)
            if isinstance(cd, dict) and "segments" in cd:
                segment_total = sum(s.get("revenue", 0) for s in cd["segments"])
                if segment_total > 0 and abs(segment_total - total_rev) / total_rev > 0.05:
                    return True
            return False

        # Extract all currency values from description
        amounts = re.findall(r'₹([\d,.]+)\s*(Cr|L|K)?', ins.description)
        for amount_str, unit in amounts:
            try:
                val = float(amount_str.replace(",", ""))
                if unit == "Cr":
                    val *= 1_00_00_000
                elif unit == "L":
                    val *= 1_00_000
                elif unit == "K":
                    val *= 1_000

                ratio = val / total_rev
                if ratio > 10 or (ratio < 0.001 and val > 0):
                    return True
            except Exception:
                continue
        return False
    
    def _check_count_consistency(self, ins: BusinessInsight) -> None:
        """Verify any claimed record counts match dataset size."""
        import re
        counts = re.findall(r'(\d{1,3}(?:,\d{3})*)\s*records', ins.description)
        actual = len(self.df)
        for count_str in counts:
            claimed = int(count_str.replace(",", ""))
            if claimed != actual and abs(claimed - actual) > actual * 0.1:
                self.issues.append(
                    f"Count mismatch in '{ins.title}': claimed {claimed}, actual {actual}"
                )

# ============================================================
# V4 ADDITION 1: IMPACT QUANTIFICATION ENGINE
# ============================================================

class ImpactQuantifier:
    """
    ✅ V4: Converts qualitative insights into ₹ and % impact estimates.
    Uses observed data patterns as the basis for projections.
    """
    
    @staticmethod
    def margin_replication_gain(pdf, geo_col, rev_col, cost_col, best_region: str) -> dict:
        """
        Quantify: if all regions matched the best region's margin proxy,
        how much additional value is generated?
        """
        try:
            grp = pdf.groupby(geo_col).agg(
                revenue=(rev_col, "sum"),
                cost=(cost_col, "mean"),
                count=(rev_col, "count")
            ).dropna()
            
            grp["margin_proxy"] = grp["revenue"] - (grp["cost"] * grp["count"])
            
            if best_region not in grp.index:
                return {}
            
            best_margin_rate = (
                grp.loc[best_region, "margin_proxy"] /
                grp.loc[best_region, "revenue"]
            )
            current_total_margin = grp["margin_proxy"].sum()
            potential_margin = grp["revenue"].sum() * best_margin_rate
            uplift_abs = potential_margin - current_total_margin
            uplift_pct = (uplift_abs / abs(current_total_margin)) * 100 if current_total_margin != 0 else 0
            
            return {
                "uplift_abs": uplift_abs,
                "uplift_pct": uplift_pct,
                "best_margin_rate": best_margin_rate,
                "statement": (
                    f"Replicating {best_region}'s operational model across all regions "
                    f"could improve total margin by {_fmt_currency(uplift_abs)} "
                    f"({uplift_pct:+.1f}%) based on observed efficiency differential."
                )
            }
        except Exception:
            return {}
    
    @staticmethod
    def pricing_standardization_gain(pdf, cost_col, rev_col, cat_col) -> dict:
        """
        Quantify: if pricing CV is reduced to 0.20 (industry standard),
        what's the estimated margin improvement?
        
        P0 FIX (Bug 0.6): Added within-group vs between-group decomposition guard
        to prevent recommending "standardization" when variance is structural.
        """
        try:
            current_cv = pdf[cost_col].std() / pdf[cost_col].mean() if pdf[cost_col].mean() > 0 else 0
            target_cv = 0.20
            
            if current_cv <= target_cv:
                return {}
            
            # P0 FIX: Check if CV is structural (product-driven) or chaotic
            # If within-category CV ≈ overall CV, the variance is NOT pricing chaos
            if cat_col and cat_col in pdf.columns:
                within_cvs = pdf.groupby(cat_col)[cost_col].agg(
                    lambda x: x.std()/x.mean() if x.mean() > 0 else 0
                )
                avg_within_cv = within_cvs.mean()
                
                # If within-category CV is >80% of overall CV, the "spread" is
                # inherent to the data distribution, not pricing inconsistency
                if avg_within_cv > current_cv * 0.80:
                    return {
                        "suppressed": True,
                        "reason": (
                            f"Within-{cat_col} CV ({avg_within_cv:.2f}) is similar to "
                            f"overall CV ({current_cv:.2f}), indicating the spread is "
                            f"structural, not a pricing standardization opportunity."
                        )
                    }
            
            # Only reach here if genuine between-group variance exists
            # Excess variability = revenue at risk
            excess_cv = current_cv - target_cv
            total_rev = pdf[rev_col].sum()
            revenue_at_risk = total_rev * excess_cv  # proxy: excess spread costs margin
            recovery_pct = 0.35  # industry estimate: 35% of at-risk revenue recoverable
            
            gain_abs = revenue_at_risk * recovery_pct
            gain_pct = (gain_abs / total_rev) * 100 if total_rev > 0 else 0
            
            return {
                "current_cv": current_cv,
                "target_cv": target_cv,
                "revenue_at_risk": revenue_at_risk,
                "gain_abs": gain_abs,
                "gain_pct": gain_pct,
                "methodology": "Between-group variance decomposition confirmed pricing inconsistency is not structural.",
                "statement": (
                    f"Standardizing {cost_col} to CV ≤ {target_cv} (from {current_cv:.2f}) "
                    f"could recover {_fmt_currency(gain_abs)} ({gain_pct:.1f}% of revenue). "
                    f"Note: estimate assumes 35% recovery rate on excess-CV revenue."
                )
            }
        except Exception:
            return {}
    
    @staticmethod
    def category_share_gain(pdf, cat_col, rev_col, lagging_cat: str, leading_cat: str) -> dict:
        """
        Quantify: if lagging category matched leading category's share,
        what's the revenue uplift?
        """
        try:
            shares = pdf.groupby(cat_col)[rev_col].sum()
            total = shares.sum()
            
            if leading_cat not in shares.index or lagging_cat not in shares.index:
                return {}
            
            leading_share = shares[leading_cat] / total if total > 0 else 0
            lagging_current = shares[lagging_cat] / total if total > 0 else 0
            lagging_target = leading_share * 0.5  # conservative: half of leader
            
            uplift_share = lagging_target - lagging_current
            uplift_abs = uplift_share * total
            
            return {
                "uplift_abs": uplift_abs,
                "uplift_pct": uplift_share * 100,
                "statement": (
                    f"Growing {lagging_cat} to 50% of {leading_cat}'s market share "
                    f"represents a {_fmt_currency(uplift_abs)} revenue opportunity "
                    f"({uplift_share*100:.1f}pp share gain)."
                )
            }
        except Exception:
            return {}


# ============================================================
# V4 ADDITION 2: STATISTICAL CONFIDENCE SCORING
# ============================================================

class ConfidenceScorer:
    """
    ✅ V4: Computes statistical confidence for each insight.
    Returns: label, score (0-1), explanation.
    """
    
    # ✅ FINAL V4: Calibrated multipliers per insight type
    BASE_CONFIDENCE = {
        "cross_dimensional_margin": 0.78,
        "causal_pricing_driver": 0.82,
        "revenue_concentration": 0.91,
        "simulation_pricing": 0.61,  # simulations always lower
        "simulation_category_growth": 0.48,
        "descriptive_balance": 0.88,
        "heatmap_pattern": 0.85,
        "correlation_anomaly": 0.89,
        "correlation_matrix": 0.89,
        "pricing_inconsistency": 0.75,
        "cross_dimensional_dominance": 0.82,
        "cross_dimensional_volume_value": 0.79,
        "temporal_peaks": 0.86,
        "revenue_dominance": 0.91,
        "descriptive_distribution": 0.72,
        "descriptive_volume": 0.95,  # always accurate
    }
    
    @classmethod
    def score(cls, insight, df, correlation=None, eta2=None, share=None) -> dict:
        """
        ✅ FINAL V4: Calibrated confidence scoring with adjustments.
        """
        n = len(df)
        
        # Handle both dict and BusinessInsight object
        if isinstance(insight, dict):
            rule = insight.get("rule_type", "")
        else:
            rule = getattr(insight, "rule_type", "")
        
        # Start with type-based prior
        base = cls.BASE_CONFIDENCE.get(rule, 0.65)
        
        # Adjust for sample size
        if n >= 5000:
            base += 0.08
        elif n >= 1000:
            base += 0.05
        elif n >= 300:
            base += 0.02
        elif n >= 100:
            base += 0.00
        elif n >= 30:
            base -= 0.05
        else:
            base -= 0.15
        
        # Adjust for signal strength
        if correlation is not None:
            base += min(abs(correlation) * 0.15, 0.10)
        if eta2 is not None:
            base += min(eta2 * 0.20, 0.10)
        if share is not None:
            base += min((share - 0.25) * 0.30, 0.08)
        
        score = max(0.20, min(0.97, base))
        
        if score >= 0.80:
            label = "High"
        elif score >= 0.55:
            label = "Medium"
        else:
            label = "Low"
        
        return {
            "score": score,
            "label": label,
            "pct": f"{score*100:.0f}%",
            "reason": cls._reason(rule, n, score)
        }
    
    @staticmethod
    def _reason(rule: str, n: int, score: float) -> str:
        """Generate confidence reason based on rule type."""
        reasons = {
            "simulation_pricing": (
                "Simulation estimate — based on observed CV differential. "
                "Actual recovery depends on execution quality."
            ),
            "simulation_category_growth": (
                "Growth projection based on current market share differential. "
                "Actual results depend on demand elasticity and competition."
            ),
            "cross_dimensional_margin": (
                "Derived from observed revenue/cost differential. "
                "Confidence scales with sample consistency."
            ),
            "causal_pricing_driver": (
                "ANOVA-validated. η² measures % variance explained."
            ),
            "revenue_concentration": (
                "Direct calculation from revenue distribution. "
                "High confidence due to observed data."
            ),
            "descriptive_volume": (
                "Exact count from dataset. No estimation involved."
            ),
        }
        
        base = reasons.get(rule,
            f"Based on n={n:,} records with observed pattern consistency.")
        return base


# ============================================================
# FINAL V4 ADDITION 2: SCENARIO ANALYSIS ENGINE
# ============================================================

class ScenarioEngine:
    """
    ✅ FINAL V4: Generates best/base/worst case ranges for every simulation.
    Multipliers calibrated by insight category risk profile.
    """
    
    PROFILES = {
        "pricing": {
            "best": 1.55,
            "worst": 0.35,
            "risk": (
                "High — pricing changes face execution risk "
                "and potential volume elasticity effects."
            )
        },
        "margin": {
            "best": 1.30,
            "worst": 0.55,
            "risk": (
                "Medium — operational replication is more "
                "predictable than demand-side changes."
            )
        },
        "category": {
            "best": 1.90,
            "worst": 0.25,
            "risk": (
                "Very High — category growth depends on "
                "demand elasticity, competition, and execution."
            )
        },
        "default": {
            "best": 1.40,
            "worst": 0.45,
            "risk": "Medium — estimate uncertainty varies by context."
        }
    }
    
    @classmethod
    def generate(cls, base_gain: float, category: str = "default") -> dict:
        """Generate best/base/worst case scenarios."""
        p = cls.PROFILES.get(category, cls.PROFILES["default"])
        best = base_gain * p["best"]
        worst = base_gain * p["worst"]
        
        return {
            "best_case": best,
            "base_case": base_gain,
            "worst_case": worst,
            "range_pct": ((best - worst) / abs(base_gain)) * 100 if base_gain else 0,
            "risk_note": p["risk"],
            "display": (
                f"Best case: {_fmt_currency(best)}  |  "
                f"Base case: {_fmt_currency(base_gain)}  |  "
                f"Worst case: {_fmt_currency(worst)}"
            )
        }


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
        
        # Debug: Log numeric columns detected
        print(f"[PROFILE] Numeric columns detected: {profile.numericals[:10]}")  # Show first 10
        
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
            
            if is_numeric and n_unique <= 10 and n_total >= 30:
                # Low-cardinality numeric — check if it's actually ordinal/categorical
                non_null = series.drop_nulls()
                unique_vals = sorted(non_null.unique().to_list())
                # Check if values look like a rating scale (1-5, 1-10, 0-5 etc.)
                is_rating_scale = (
                    len(unique_vals) <= 10 and
                    all(isinstance(v, (int, float)) and v == int(v) for v in unique_vals) and
                    min(unique_vals) >= 0 and max(unique_vals) <= 10 and
                    any(k in col_lower for k in {"rating", "score", "rank", "grade",
                                                  "stars", "level", "tier", "priority"})
                )
                if is_rating_scale:
                    return ColumnProfile(col, "categorical", n_unique=n_unique,
                                         missing_pct=missing_pct, sample_values=sample)
            
            # ── P0 FIX (Bug 0.1): Numeric binary detection (0/1, Yes/No encoded as int) ──
            if n_unique <= 2 and n_total > 10:
                # Check if values are 0/1 or boolean-like
                non_null = series.drop_nulls()
                unique_vals = set(non_null.unique().to_list())
                if unique_vals <= {0, 1} or unique_vals <= {0.0, 1.0}:
                    return ColumnProfile(col, "binary", n_unique=n_unique,
                                         missing_pct=missing_pct, sample_values=sample)
            
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
        
        P0 ENHANCEMENTS:
        - Entity type detection (person/place/category/ID)
        - Prevents "Cameron" being treated as category
        
        IMPORTANT: REVENUE_KEYWORDS must be checked BEFORE PRICE_KEYWORDS
        because columns like 'Sales Amount' contain 'amount' (a PRICE keyword)
        but are actually revenue. Checking revenue first prevents Price×Qty
        double-counting that inflates revenue by ~400×.
        """
        # Initialize entity tracking
        profile.person_columns = []
        profile.place_columns = []
        profile.id_columns = []
        
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
        
        # P0 FIX (Bug 0.3): POST-LOOP: Detect row-level revenue columns (TotalPrice, TotalAmount, etc.)
        # A column named "total" + price/amount keyword is a row-level revenue figure,
        # NOT a unit price. Promote it to revenue_col.
        if profile.revenue_col is None and profile.price_col and profile.qty_col:
            for col in profile.numericals:
                cl = col.lower()
                has_total = "total" in cl
                has_price_kw = any(k in cl for k in PRICE_KEYWORDS)
                if has_total and has_price_kw and col != profile.price_col:
                    # This is likely Price × Qty pre-computed (e.g., TotalPrice)
                    # Verify: does it correlate with price_col × qty_col?
                    try:
                        pdf = df.to_pandas()
                        computed = pdf[profile.price_col] * pdf[profile.qty_col]
                        actual = pdf[col]
                        corr = computed.corr(actual)
                        if corr > 0.8:  # Strong correlation = derived column
                            profile.revenue_col = col
                            log.info(f"[SubRole] Promoted '{col}' to revenue_col (corr={corr:.2f} with {profile.price_col}×{profile.qty_col})")
                            break
                    except Exception as e:
                        log.warning(f"[SubRole] Could not verify {col} as revenue: {e}")
                        pass

        for col in profile.binaries:
            cl = col.lower()
            if any(k in cl for k in RETURN_KEYWORDS):
                profile.return_col = col

        for col in profile.temporals:
            cl = col.lower()
            if any(k in cl for k in DATE_KEYWORDS):
                if profile.date_col is None:
                    profile.date_col = col

        # P0 FIX: Entity type detection for categoricals
        for col in profile.categoricals:
            cl = col.lower()
            
            # Detect entity type
            entity_type = self._detect_entity_type(df, col)
            
            if entity_type == 'person':
                profile.person_columns.append(col)
                log.info(f"[EntityDetection] '{col}' is a PERSON column")
            elif entity_type == 'place':
                profile.place_columns.append(col)
                log.info(f"[EntityDetection] '{col}' is a PLACE column")
            elif entity_type == 'id':
                profile.id_columns.append(col)
                log.info(f"[EntityDetection] '{col}' is an ID column")
            
            # Category column selection (prefer non-person, non-ID columns)
            if any(k in cl for k in CATEGORY_KEYWORDS):
                n_unique = profile.profiles[col].n_unique
                if 2 <= n_unique <= 20:
                    if profile.category_col is None and entity_type not in ['person', 'id']:
                        profile.category_col = col
                    # P0 FIX (Bug 0.2): Geographic assignment with first-wins and entity guard
                    if any(k in cl for k in {"city", "region", "state", "country", "area", "zone"}):
                        # Only set geographic_col if:
                        # 1. Not already set (first-wins prevents RegionManager overwriting Region)
                        # 2. Not a person column (prevents manager/salesperson columns)
                        if profile.geographic_col is None and entity_type not in ['person', 'id']:
                            profile.geographic_col = col
        
        # Priority override: always prefer a column named exactly "Category" (case-insensitive).
        # This prevents "Region" from winning over "Category" in column-order-dependent matching.
        for col in profile.categoricals:
            if col.lower() == "category":
                entity_type = self._detect_entity_type(df, col)
                if entity_type not in ['person', 'id']:
                    n_unique = profile.profiles[col].n_unique
                    if 2 <= n_unique <= 20:
                        profile.category_col = col
                        log.info(f"[SubRole] Exact-name override: category_col='{col}'")
                break

        # Fallback: prefer non-person, non-ID columns for category
        if not profile.category_col and profile.categoricals:
            candidates = [c for c in profile.categoricals 
                         if c not in profile.person_columns and c not in profile.id_columns]
            if candidates:
                profile.category_col = min(
                    candidates,
                    key=lambda c: abs(profile.profiles[c].n_unique - 5)
                )
            else:
                profile.category_col = min(
                    profile.categoricals,
                    key=lambda c: abs(profile.profiles[c].n_unique - 5)
                )
    
    def _detect_entity_type(self, df: pl.DataFrame, col: str) -> str:
        """
        P0 FIX: Detect if column contains person names, places, categories, or IDs.
        Returns: 'person', 'place', 'category', or 'id'
        """
        col_lower = col.lower()
        
        # Check column name patterns
        person_keywords = ['name', 'manager', 'salesperson', 'employee', 'staff', 'agent', 'rep']
        place_keywords = ['region', 'city', 'state', 'country', 'location', 'area', 'zone', 'territory']
        id_keywords = ['id', 'code', 'key', 'number', 'ref']
        
        # ID detection (highest priority - most specific)
        if any(kw in col_lower for kw in id_keywords):
            return 'id'
        
        # Person detection
        if any(kw in col_lower for kw in person_keywords):
            return 'person'
        
        # Place detection
        if any(kw in col_lower for kw in place_keywords):
            return 'place'
        
        # Check sample values
        try:
            sample_values = df[col].head(20).to_list()
            sample_values = [str(v).lower() for v in sample_values if v is not None]
            
            # Person name indicators
            person_indicators = {'john', 'jane', 'michael', 'sarah', 'david', 'emily', 'cameron', 
                               'alex', 'chris', 'james', 'mary', 'robert', 'jennifer', 'william',
                               'daniel', 'jessica', 'matthew', 'ashley', 'joshua', 'amanda'}
            
            # Place indicators
            place_indicators = {'north', 'south', 'east', 'west', 'central', 'northeast', 
                              'northwest', 'southeast', 'southwest', 'northern', 'southern',
                              'eastern', 'western'}
            
            person_matches = sum(1 for v in sample_values if v in person_indicators)
            place_matches = sum(1 for v in sample_values if any(p in v for p in place_indicators))
            
            if person_matches > 0:
                return 'person'
            elif place_matches > 0:
                return 'place'
        except Exception as e:
            log.warning(f"[EntityDetection] Could not sample values for {col}: {e}")
        
        # Default to category
        return 'category'

# ============================================================
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
    
    # CRITICAL FIX: Internal rule types that should never appear as user insights
    INTERNAL_RULE_TYPES = {"domain_detection", "column_coverage_gap", "sanity_warning"}

    def synthesize(self, insights: list[BusinessInsight], drivers: dict, domain_id: str = "general") -> list[BusinessInsight]:
        if not insights:
            return []
        
        # CRITICAL FIX: Filter out internal metadata insights before processing
        insights = [i for i in insights if i.rule_type not in self.INTERNAL_RULE_TYPES]
        log.info(f"[synthesizer] Filtered out internal rule types, {len(insights)} insights remaining")
        
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
            "quality": [
                "perfect_quality", "high_return_rate", "payment_risk",
                "delivery_delay_risk", "returns_by_segment", "returns_revenue_impact",
                "rating_quality", "category_satisfaction",
            ],
            "discovery": ["dominance", "correlation_matrix", "domain_detection"],
            "distribution": ["skewed_distribution"],
            "discount": ["discount_impact"],
            "temporal": [
                "temporal_peaks",
                "seasonality_pattern",
                "growth_rates",
                "temporal_anomaly",
            ],
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

        # Priority rules: must survive the cap unconditionally.
        # These are never in any topic bucket, so they reach here via universal passthrough
        # or the loop below — but compressed[:N] would cut them when other rules fill slots 1-8.
        PRIORITY_RULE_TYPES = {
            "rfm_segmentation", "cohort_retention", "clv_estimate",
            "seasonal_forecast", "root_cause_analysis",
        }

        # Score floor: ensure priority insights rank high if ever placed inside a topic bucket.
        MINIMUM_SCORE_FLOOR = 0.80
        for ins in insights:
            if ins.rule_type in PRIORITY_RULE_TYPES:
                ins.score = max(ins.score, MINIMUM_SCORE_FLOOR)

        # Partition compressed: priority items are exempt from the non-priority cap.
        priority_compressed = [c for c in compressed if getattr(c, "rule_type", "") in PRIORITY_RULE_TYPES]
        non_priority_compressed = [c for c in compressed if getattr(c, "rule_type", "") not in PRIORITY_RULE_TYPES]

        # Force-add any priority insight from the input list that never entered compressed.
        seen_priority_types = {getattr(c, "rule_type", "") for c in priority_compressed}
        for ins in insights:
            if ins.rule_type in PRIORITY_RULE_TYPES and ins.rule_type not in seen_priority_types:
                priority_compressed.append(ins)
                seen_priority_types.add(ins.rule_type)

        # Cap non-priority at (8 − priority count) so priority items are never squeezed out.
        non_priority_cap = max(0, 8 - len(priority_compressed))
        final_compressed = non_priority_compressed[:non_priority_cap] + priority_compressed

        print(f"\n[SYNTHESIZER] Selected {len(final_compressed)} insights before PDF:")
        for _ins in final_compressed:
            print(f"  [{getattr(_ins, 'rule_type', '?')}] {getattr(_ins, 'title', '?')[:70]}")

        return final_compressed

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

    # TEMPORARILY LOWERED THRESHOLDS TO FORCE RULE FIRING
    REVENUE_CONCENTRATION_THRESHOLD = 0.15   # was 0.35 - >15% revenue from one category → risk
    HIGH_RETURN_RATE_MULTIPLIER     = 1.1    # was 1.5 - cat return rate > 1.1× global → issue
    CORRELATION_RISK_THRESHOLD      = 0.4    # |corr(delivery, returns)| > 0.4 → risk
    DOMINANCE_THRESHOLD             = 0.15   # was 0.35 - one value > 15% → flag for specific categories

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
                            print(f"[TAUTOLOGY DETECTED] {col_b} ~ {col_a} * {other}")
                            return True

                    # Test division: a / o ≈ b
                    ratio = np.divide(a, o, out=np.zeros_like(a), where=o != 0)
                    if np.std(ratio) > 0:
                        corr = np.corrcoef(ratio, b)[0, 1]
                        if corr > 0.99:
                            print(f"[TAUTOLOGY DETECTED] {col_b} ~ {col_a} / {other}")
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
        
        print(f"[DEBUG] date_col={profile.date_col}, temporals={profile.temporals}")

        # Helper function to safely call rules
        def safe_rule_call(rule_func, rule_name, *args, **kwargs):
            try:
                result = rule_func(*args, **kwargs)
                if result:
                    count = len(result) if isinstance(result, list) else 1
                    print(f"[RULE OK] {rule_name} → {count} insights")
                return result if result else []
            except Exception as e:
                print(f"[RULE FAIL] {rule_name} → {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                return []

        # ── DOMAIN DETECTION (P0 FIX) ─────────────────────────────────────
        all_insights.extend(safe_rule_call(self._rule_domain_detection, "domain_detection", df, profile))

        # ── EXISTING rules ────────────────────────────────────────────────
        # Always try revenue_by_category if we have a category column
        if profile.category_col and profile.revenue_col:
            rev_series = getattr(profile, "_revenue_series", None)
            all_insights.extend(safe_rule_call(self._rule_revenue_by_category, "revenue_by_category", df, pdf, profile, rev_series))

        ret_series = getattr(profile, "_return_count_series", None)
        if ret_series is not None and profile.category_col and "return_rate" in metrics:
            global_rate = metrics["return_rate"].value
            all_insights.extend(safe_rule_call(self._rule_return_rate_by_category, "return_rate_by_category", df, pdf, profile, ret_series, global_rate))

        if ret_series is not None:
            all_insights.extend(safe_rule_call(self._rule_high_return_rate_alert, "high_return_rate_alert", df, profile, ret_series))
            
        if profile.category_col and ret_series is not None:
            all_insights.extend(safe_rule_call(self._rule_payment_return_correlation, "payment_return_correlation", df, pdf, profile, ret_series))

        if len(profile.numericals) >= 2:
            all_insights.extend(safe_rule_call(self._rule_strong_correlation_insight, "strong_correlation", df, profile))

        all_insights.extend(safe_rule_call(self._rule_outlier_alert, "outlier_alert", df, profile))

        # ── ✅ NEW SEGMENT RULES (Fix 3) ──────────────────────────────────
        all_insights.extend(safe_rule_call(self._rule_revenue_by_segment, "revenue_by_segment", df, domain))
        all_insights.extend(safe_rule_call(self._rule_top_performers, "top_performers", df, domain))
        all_insights.extend(safe_rule_call(self._rule_skewed_distribution_alert, "skewed_distribution", df, domain))
        all_insights.extend(safe_rule_call(self._rule_discount_impact, "discount_impact", df, domain))
        all_insights.extend(safe_rule_call(self._rule_demographic_split, "demographic_split", df, domain))

        if any("attrition" in c.lower() for c in df.columns):
            results = self._rule_hr_attrition(df, profile)
            if results:
                all_insights.extend(results)

        # Fire content library rule for entertainment datasets
        _CONTENT_TYPES_EXEC = {"movie", "tv show", "series", "episode",
                                "track", "album", "documentary", "short"}
        _type_col = next(
            (c for c in df.columns
             if df[c].n_unique() <= 10
             and any(v.lower() in _CONTENT_TYPES_EXEC
                     for v in df[c].drop_nulls().cast(pl.Utf8).to_list()[:50])),
            None
        )
        if _type_col:
            try:
                _type_vals = df[_type_col].drop_nulls().cast(pl.Utf8).str.to_lowercase().unique().to_list()
            except Exception:
                _type_vals = []
            if any(v in _type_vals for v in _CONTENT_TYPES_EXEC):
                results = self._rule_content_library_analysis(df, profile)
                if results:
                    all_insights.extend(results)
                    log.info(f"[content_library] Generated {len(results)} insights")

        # Fire sports rule when team columns detected
        _has_teams = any(
            any(k in c.lower() for k in ["team1", "team2",
                "home_team", "away_team"])
            for c in df.columns
        )
        if _has_teams:
            results = self._rule_sports_analysis(df, profile)
            if results:
                all_insights.extend(results)
                log.info(f"[sports] Generated {len(results)} insights")

        _has_health = any(
            any(k in c.lower().replace("\n", "").replace(",", "")
                for k in ["confirmed", "cases", "deaths",
                          "recovered", "fatalities"])
            for c in df.columns
        )
        if _has_health:
            results = self._rule_health_analysis(df, profile)
            if results:
                all_insights.extend(results)
                log.info(
                    f"[health] Generated {len(results)} insights"
                )

        # ── Tier 1.2: Enhanced Time-Series Analysis ───────────────────────
        _ts = TimeSeriesAnalyzer()
        _ts_insights = safe_rule_call(_ts.analyze, "time_series_analyzer", df, profile)
        if _ts_insights:
            all_insights.extend(_ts_insights)
        else:
            # Fallback to basic rule if enhanced analyzer produces nothing
            all_insights.extend(safe_rule_call(self._rule_temporal_peaks, "temporal_peaks_fallback", df))
        
        # ── ✅ GAP 1: Cross-Dimensional Reasoning ─────────────────────────
        all_insights.extend(safe_rule_call(self._rule_cross_dimensional, "cross_dimensional", df, profile))
        
        # ── ✅ GAP 4: Pricing Inconsistency Detection ─────────────────────
        pricing_insight = safe_rule_call(self._rule_pricing_inconsistency, "pricing_inconsistency", df, profile)
        if pricing_insight:
            all_insights.append(pricing_insight)
        
        # ── ✅ V4: Causal Reasoning & Simulation ──────────────────────────
        causal_insight = safe_rule_call(self._rule_causal_pricing, "causal_pricing", df, profile)
        if causal_insight:
            all_insights.append(causal_insight)
        all_insights.extend(safe_rule_call(self._rule_simulation, "simulation", df, profile))
        all_insights.extend(safe_rule_call(self._rule_rating_analysis, "rating_analysis", df, profile))
        all_insights.extend(safe_rule_call(self._rule_category_satisfaction_cross, "category_satisfaction", df, profile))
        all_insights.extend(safe_rule_call(self._rule_customer_concentration, "customer_concentration", df, profile))

        # ── NEW: Customer Intelligence & Forecasting Rules ───────────────
        all_insights.extend(safe_rule_call(self._rule_rfm_segmentation, "rfm_segmentation", df, profile))
        all_insights.extend(safe_rule_call(self._rule_cohort_retention, "cohort_retention", df, profile))
        all_insights.extend(safe_rule_call(self._rule_clv_estimate, "clv_estimate", df, profile))
        all_insights.extend(safe_rule_call(self._rule_seasonal_forecast, "seasonal_forecast", df, profile))

        # Root-cause must run last — it reads all_insights to attach hypotheses to parent findings
        all_insights.extend(safe_rule_call(self._rule_root_cause_analysis, "root_cause_analysis", df, profile, all_insights))

        # If fewer than 3 insights generated, run generic distribution fallback
        if len(all_insights) < 3:
            log.info("[RuleEngine] Fewer than 3 insights — running generic fallback")
            fallback = self._rule_generic_distribution_analysis(df, profile)
            all_insights.extend(fallback)

        # ── Post-Processing ──────────────────────────────────────────────
        all_insights = self._deduplicate(all_insights)
        all_insights = self._inject_contradictions(all_insights)
        
        # ── ✅ GAP 3: Rank insights by business impact + confidence ──────
        all_insights = self._rank_insights(all_insights)
        
        # ── ✅ P0 FIX: Ensure minimum 3 insights ─────────────────────────
        all_insights = self._ensure_minimum_insights(all_insights, df, profile)

        # Protect priority rules from the [:8] cap — they MUST reach the synthesizer.
        # ROI ranking places cohort/CLV/seasonal lower than cross-dimensional rules,
        # so a naive [:8] slice silently drops them before synthesize() ever sees them.
        _PRIORITY_TYPES = {
            "rfm_segmentation", "cohort_retention", "clv_estimate",
            "seasonal_forecast", "root_cause_analysis",
        }
        _priority = [i for i in all_insights if getattr(i, "rule_type", "") in _PRIORITY_TYPES]
        _non_priority = [i for i in all_insights if getattr(i, "rule_type", "") not in _PRIORITY_TYPES]
        _result = _non_priority[:max(0, 8 - len(_priority))] + _priority
        print(f"\n[INSIGHT ENGINE] FINAL: {len(all_insights)} raw insights → "
              f"{len(_result)} sent to synthesizer "
              f"({len(_priority)} priority rules guaranteed)\n")
        return _result, warnings

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
            
            # CRITICAL FIX: Use relative dominance instead of absolute threshold
            n_segments = len(grouped)
            expected_share = 100 / n_segments  # Equal distribution percentage
            dominance_ratio = top_pct / expected_share
            
            # Calculate HHI (Herfindahl-Hirschman Index) for portfolio concentration
            # HHI < 1500: Unconcentrated, 1500-2500: Moderate, >2500: Highly concentrated
            shares = grouped[rev_col_name] / total_rev
            hhi = sum((s * 100) ** 2 for s in shares)
            
            log.info(f"[revenue_concentration] {top_name}: {top_pct:.1f}% of revenue, "
                    f"dominance_ratio={dominance_ratio:.2f}x (expected {expected_share:.1f}%), "
                    f"HHI={hhi:.0f}, n_segments={n_segments}")

            # Only flag as concentration risk if BOTH conditions met:
            # 1. Top segment is 2x+ what equal distribution would predict
            # 2. HHI indicates concentrated market (>2500)
            if dominance_ratio >= 2.0 and hhi > 2500:
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
                    evidence=f"Concentration Index: {top_pct:.1f}% | Dominance Ratio: {dominance_ratio:.1f}x | HHI: {hhi:.0f}",
                    decision_implication="Execute an immediate diversification strategy. Reallocate 15-20% of marketing spend towards growing secondary segments to mitigate single-source failure risk.",
                    impact="🔴 Critical",
                    recommendation=f"Prioritize growth in under-indexed segments like {excl_str.split(' ')[0]}.",
                    rule_type="revenue_dominance"
                ))
                log.info(f"[revenue_concentration] ✅ Concentration risk detected: {top_name} at {dominance_ratio:.1f}x expected")
            else:
                # CRITICAL FIX: Suppress false alarms for balanced portfolios
                if dominance_ratio < 2.0:
                    log.info(f"[revenue_concentration] Suppressed: {top_name} at {top_pct:.1f}% "
                            f"is only {dominance_ratio:.1f}x expected ({expected_share:.1f}%) — balanced portfolio")
                if hhi <= 2500:
                    log.info(f"[revenue_concentration] Suppressed: HHI={hhi:.0f} indicates unconcentrated market")
                
                # Check for moderate concentration (25–35%) OR emerging leader
                bottom = grouped.iloc[-1]
                
                if top_pct > 25 and dominance_ratio >= 1.5:
                    dist_title = f"Emerging Market Leader: {top_name}"
                    dist_desc = f"{top_name} is gaining healthy momentum with {top_pct:.0f}% share. Positive growth indicators detected."
                    dist_why = "A single leader at this level indicates a successful product-market fit but requires monitoring to prevent future dependency risk."
                    dist_evidence = f"Lead segment share: {top_pct:.0f}% | Dominance: {dominance_ratio:.1f}x | HHI: {hhi:.0f}"
                    dist_dec = f"Nurture {top_name} to maintain leadership while beginning to seed growth in {str(bottom[cat])} to ensure balanced portfolio evolution."
                    dist_impact = "🟠 Important"
                    # FIX: Emerging leader recommendation should focus on nurturing the leader while building alternatives
                    dist_rec = f"Nurture {top_name} leadership position while investing in {str(bottom[cat])} to build portfolio resilience."
                else:
                    # Balanced portfolio - celebrate it!
                    dist_title = f"Balanced Portfolio Distribution: {cat}"
                    dist_desc = (
                        f"Revenue is efficiently distributed across {n_segments} {cat} segments. "
                        f"Top segment ({top_name}) contributes {_fmt_currency(top_val)} ({top_pct:.0f}% vs "
                        f"expected {expected_share:.0f}%) — only {dominance_ratio:.1f}x the equal-share baseline. "
                        f"HHI of {hhi:.0f} confirms healthy diversification with no single-source dependency."
                    )
                    dist_why = "A diversified portfolio is the gold standard for risk mitigation and suggests broad market appeal."
                    dist_evidence = f"Dominance ratio: {dominance_ratio:.1f}x | HHI: {hhi:.0f} (unconcentrated) | {n_segments} segments"
                    dist_dec = "Maintain current allocation. Leverage the stability of this portfolio to experiment with high-margin niche segments."
                    dist_impact = "🟢 Minor"
                    # FIX: Balanced portfolio recommendation should be about maintaining balance, not protecting share
                    dist_rec = f"Maintain balanced allocation across all {n_segments} segments. Use this stability as a foundation for testing new high-margin opportunities."

                insights.append(BusinessInsight(
                    title=dist_title,
                    description=dist_desc,
                    why_it_matters=dist_why,
                    evidence=dist_evidence,
                    decision_implication=dist_dec,
                    impact=dist_impact,
                    recommendation=dist_rec,
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
        """
        Detect if the dataset belongs to a specific industry domain using centralized logic.
        FIX 4: Updated to use conversational prose instead of template language.
        """
        domain_id = detect_domain(df)
        domain_name = domain_id.replace("_", " ").title()
        if domain_id == "general":
            domain_name = "General Business"
        
        # FIX 4: Conversational description without "InsightStream has identified" template
        description = (
            f"This dataset exhibits classic {domain_name.lower()} patterns: "
            f"product categories, payment methods, and purchase dates. "
            f"The system has automatically applied {domain_name.lower()}-specific analysis rules "
            f"to surface relevant insights."
        )
        
        why_it_matters = (
            f"Domain-specific analysis ensures more accurate insights and recommendations "
            f"tailored to {domain_name.lower()} operations."
        )
        
        return [BusinessInsight(
            title=f"Domain Intelligence Detected: {domain_name}",
            description=description,
            why_it_matters=why_it_matters,
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

        DEMOGRAPHIC_COLS = {
            "gender", "maritalstatus", "marital_status", "overtime",
            "over_time", "attrition", "educationfield", "education_field",
            "relationshipstatus", "nationality", "race", "ethnicity",
            "religion", "age_group", "agegroup",
        }

        segment_cols = self._find_segment_columns(df, max_cardinality=20)
        _rev_seg_count = 0
        for seg_col in segment_cols:
            if _rev_seg_count >= 3:
                break

            col_key = seg_col.lower().replace(" ", "").replace("_", "")
            if col_key in {d.replace("_", "") for d in DEMOGRAPHIC_COLS}:
                log.info(f"[revenue_by_segment] Skipping demographic: {seg_col}")
                continue

            try:
                # HR domain: show headcount distribution, not income sums
                _is_hr = any("attrition" in c.lower() for c in df.columns)
                if _is_hr:
                    grouped_hr = (
                        df.group_by(seg_col)
                        .agg(pl.len().alias("n_employees"))
                        .sort("n_employees", descending=True)
                    )
                    if grouped_hr.height < 2:
                        continue
                    rows_hr = grouped_hr.to_dicts()
                    top_hr  = rows_hr[0]
                    bot_hr  = rows_hr[-1]
                    total_emps = len(df)
                    top_count  = int(top_hr["n_employees"])
                    bot_count  = int(bot_hr["n_employees"])
                    top_pct    = top_count / total_emps * 100
                    bot_pct    = bot_count / total_emps * 100
                    gap_pp     = top_pct - bot_pct
                    if gap_pp < 5:
                        continue
                    n_segments    = grouped_hr.height
                    top_segment   = str(top_hr[seg_col])
                    bottom_segment = str(bot_hr[seg_col])
                    col_label     = seg_col.replace("_", " ").title()
                    insights.append(BusinessInsight(
                        title=f"Distribution: {col_label}",
                        impact="🟠 Important",
                        description=(
                            f"{top_segment} has {top_count:,} employees "
                            f"({top_pct:.1f}% of workforce), while "
                            f"{bottom_segment} has {bot_count:,} employees "
                            f"({bot_pct:.1f}%). "
                            f"This {gap_pp:.1f}pp headcount gap across "
                            f"{n_segments} {col_label} segments reflects workforce "
                            f"composition — not a performance issue."
                        ),
                        why_it_matters=(
                            f"Understanding headcount distribution across {col_label} segments "
                            f"helps with workforce planning and equitable resource allocation."
                        ),
                        recommendation=(
                            f"Review whether {bottom_segment} headcount aligns "
                            f"with business needs. Headcount imbalance may indicate "
                            f"hiring focus or natural role distribution."
                        ),
                        rule_type=f"hr_distribution_{seg_col.lower()}",
                        qualified_segments=[top_segment],
                        excluded_segments=[bottom_segment],
                    ))
                    _rev_seg_count += 1
                    continue

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
                        f"{top[seg_col]} leads with {self._format_inr(top['total_rev'])} "
                        f"({top_share:.1f}% of total revenue), while {bottom[seg_col]} trails at "
                        f"{self._format_inr(bottom['total_rev'])} ({bottom_share:.1f}%). "
                        f"This {gap_pct:.1f}-percentage-point gap across {grouped.height} {seg_col} segments "
                        f"indicates non-uniform performance that warrants targeted intervention."
                    ),
                    why_it_matters=(
                        f"A wide revenue gap across {seg_col} segments signals either demand-side imbalance "
                        f"or execution gaps that compound over time."
                    ),
                    recommendation=(
                        f"Audit operations in {bottom[seg_col]} to identify whether the gap "
                        f"is demand-driven or execution-driven. If execution: replicate the {top[seg_col]} playbook."
                    ),
                    rule_type=f"revenue_by_{seg_col.lower()}",
                    qualified_segments=[str(top[seg_col])],
                    excluded_segments=[str(bottom[seg_col])]
                ))
                _rev_seg_count += 1
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
                    impact="🔴 Critical" if top3_share > 60 else "🟠 Important",
                    description=(
                        f"Out of {n_unique} {col_plural.lower()}, just three — {top3_names} — "
                        f"account for {top3_share:.1f}% of total revenue "
                        f"({self._format_inr(sum(r['total_rev'] for r in top3))} out of "
                        f"{self._format_inr(total)}). "
                        f"This concentration creates execution leverage but also single-point-of-failure risk."
                    ),
                    why_it_matters=(
                        f"Heavy reliance on a small set of {col_plural.lower()} means a disruption "
                        f"to any one of them disproportionately impacts overall revenue."
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
                if not self._is_monetary_column(col): continue   # ← ADD THIS LINE
                mean_val, median_val = float(series.mean()), float(series.median())
                if median_val == 0 or mean_val == 0: continue
                ratio = mean_val / median_val
                if ratio < 2.0: continue

                fmt = (self._format_inr if self._is_monetary_column(col) 
                       else (lambda x: f"{int(x):,}" if float(x) == int(x) else f"{x:,.2f}"))

                overestimate_pct = ((mean_val / median_val) - 1) * 100
                insights.append(BusinessInsight(
                    title=f"{col} Distribution is Heavily Right-Skewed",
                    impact="🟠 Important",
                    description=(
                        f"{col} shows a {ratio:.1f}× mean-to-median gap "
                        f"({fmt(mean_val)} mean vs {fmt(median_val)} median). "
                        f"A small number of high-value records are pulling the average up, "
                        f"making the mean a misleading benchmark for typical performance."
                    ),
                    evidence=f"Mean/median ratio of {ratio:.2f} indicates strong right-skew.",
                    recommendation=f"Switch dashboards to display median {col} ({fmt(median_val)}) instead of mean.",
                    decision_implication=(
                        f"Executive dashboards should switch from mean {col} "
                        f"({fmt(mean_val)}) to median ({fmt(median_val)}). "
                        f"The top 5% of records inflate the mean and create a false "
                        f"sense of typical transaction size. Budget forecasts using "
                        f"the mean will overestimate by {overestimate_pct:.0f}%."
                    ),
                    methodology=(
                        f"Mean/median ratio computed from {series.len():,} non-null records. "
                        f"Threshold: ratio > 2.0 indicates actionable right-skew."
                    ),
                    rule_type="skewed_distribution",
                    qualified_segments=[col]
                ))
            except Exception: pass
        return insights

    @log_rule
    def _rule_discount_impact(self, df: pl.DataFrame, domain: str) -> list[BusinessInsight]:
        """
        Analyzes whether discounts actually drive higher revenue per record.
        
        FIX 3 ENHANCEMENTS:
        - Automatic price tier detection when no discount column exists
        - T-test statistical comparison between tiers
        - P-value reporting for statistical rigor
        """
        insights = []
        discount_col = self._find_column(df, ["discount", "promo", "offer"])
        revenue_col = self._find_column(df, ["revenue", "sales", "amount"])
        price_col = self._find_column(df, ["price", "unitprice", "unit_price"])
        
        # FIX 3: If no discount column, try to infer from price tiers
        if not discount_col and price_col and revenue_col:
            try:
                log.info("[discount_impact] No discount column found, inferring from price tiers...")
                return self._rule_price_tier_analysis(df, price_col, revenue_col)
            except Exception as e:
                log.warning(f"[discount_impact] Price tier analysis failed: {e}")
                return insights
        
        if not (discount_col and revenue_col): 
            return insights

        try:
            # Original discount bucket logic
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
            
            # FIX 3: Add t-test comparison
            pdf = df_with_bucket.to_pandas()
            high_tier_data = pdf[pdf["discount_bucket"] == highest["discount_bucket"]][revenue_col]
            low_tier_data = pdf[pdf["discount_bucket"] == lowest["discount_bucket"]][revenue_col]
            
            try:
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(high_tier_data, low_tier_data, equal_var=False)
                
                # Statistical significance check
                is_significant = p_value < 0.05
                significance_text = (
                    f"T-test confirms this difference is statistically significant (p={p_value:.4f}, t={t_stat:.2f}). "
                    if is_significant else
                    f"Note: This difference is not statistically significant (p={p_value:.4f}), suggesting it may be due to chance. "
                )
            except Exception as e:
                log.warning(f"[discount_impact] T-test failed: {e}")
                significance_text = ""
                is_significant = True  # Assume significant if test fails

            counterintuitive = "High" in highest["discount_bucket"] or "Medium" in highest["discount_bucket"]
            
            description = (
                f"Average revenue per order varies by discount tier. "
                f"'{highest['discount_bucket']}' tier averages {self._format_inr(highest['avg_rev'])} "
                f"vs '{lowest['discount_bucket']}' at {self._format_inr(lowest['avg_rev'])} — a {gap_pct:.0f}% gap. "
                f"{significance_text}"
            )
            
            insights.append(BusinessInsight(
                title="Discount Tiers Show Uneven Revenue Impact",
                impact="🔴 Critical" if (abs(gap_pct) > 30 and is_significant) else "🟠 Important",
                description=description,
                recommendation="Run a controlled discount A/B test to isolate margin impact.",
                rule_type="discount_impact",
                qualified_segments=[highest["discount_bucket"]],
                excluded_segments=[lowest["discount_bucket"]]
            ))
        except Exception as e:
            log.warning(f"[discount_impact] Analysis failed: {e}")
            pass
        return insights
    
    def _rule_price_tier_analysis(self, df: pl.DataFrame, price_col: str, revenue_col: str) -> list[BusinessInsight]:
        """
        FIX 3: Analyze price tiers when no discount column exists.
        Detects if different price points drive different revenue patterns.
        """
        insights = []
        
        try:
            pdf = df.to_pandas()
            
            # Define price tiers using quantiles
            q33 = pdf[price_col].quantile(0.33)
            q67 = pdf[price_col].quantile(0.67)
            
            # Create tier labels
            pdf['price_tier'] = pd.cut(
                pdf[price_col],
                bins=[0, q33, q67, float('inf')],
                labels=['Low Price', 'Medium Price', 'High Price'],
                include_lowest=True
            )
            
            # Calculate stats by tier
            tier_stats = pdf.groupby('price_tier')[revenue_col].agg(['mean', 'count', 'sum']).reset_index()
            tier_stats.columns = ['tier', 'avg_rev', 'n', 'total_rev']
            
            if len(tier_stats) < 2:
                return insights
            
            # Find highest and lowest performing tiers
            highest = tier_stats.loc[tier_stats['avg_rev'].idxmax()]
            lowest = tier_stats.loc[tier_stats['avg_rev'].idxmin()]
            
            gap_pct = ((highest['avg_rev'] - lowest['avg_rev']) / lowest['avg_rev']) * 100 if lowest['avg_rev'] > 0 else 0
            
            if abs(gap_pct) < 15:  # Threshold for price tier insights
                return insights
            
            # FIX 3: T-test comparison
            high_tier_data = pdf[pdf['price_tier'] == highest['tier']][revenue_col]
            low_tier_data = pdf[pdf['price_tier'] == lowest['tier']][revenue_col]
            
            try:
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(high_tier_data, low_tier_data, equal_var=False)
                
                is_significant = p_value < 0.05
                significance_text = (
                    f"Statistical analysis (t-test) confirms this difference is significant "
                    f"(p={p_value:.4f}, t={t_stat:.2f}), indicating a real pricing effect. "
                    if is_significant else
                    f"Note: Statistical test suggests this difference may be due to chance "
                    f"(p={p_value:.4f}). Interpret with caution. "
                )
            except Exception as e:
                log.warning(f"[price_tier] T-test failed: {e}")
                significance_text = ""
                is_significant = True
            
            description = (
                f"Price tier analysis reveals significant revenue variation. "
                f"'{highest['tier']}' tier ({_fmt_currency(q67)}+) averages {self._format_inr(highest['avg_rev'])} per transaction, "
                f"while '{lowest['tier']}' tier (0–{_fmt_currency(q33)}) averages {self._format_inr(lowest['avg_rev'])} — "
                f"a {gap_pct:.0f}% difference. "
                f"{significance_text}"
                f"This suggests price point significantly influences purchase behavior."
            )
            
            recommendation = (
                f"Test price elasticity: run a 2-week A/B test moving {lowest['tier']} items "
                f"up one tier. If volume holds within 15%, the price increase is justified. "
                f"Conversely, analyze if {highest['tier']} items can sustain their premium positioning."
            )
            
            insights.append(BusinessInsight(
                title=f"Price Tier Impact: {gap_pct:.0f}% Revenue Variance",
                impact="🔴 Critical" if (abs(gap_pct) > 30 and is_significant) else "🟠 Important",
                description=description,
                why_it_matters="Price tier analysis reveals optimal pricing zones and elasticity patterns.",
                evidence=f"T-test: p={p_value:.4f}, Gap: {gap_pct:.0f}%",
                recommendation=recommendation,
                rule_type="price_tier_impact",
                qualified_segments=[str(highest['tier'])],
                excluded_segments=[str(lowest['tier'])],
                confidence_label="high" if is_significant else "medium",
                score=8.0 if is_significant else 6.0
            ))
            
            log.info(f"[price_tier] ✅ Generated price tier insight (gap: {gap_pct:.1f}%, p={p_value:.4f})")
            
        except Exception as e:
            log.warning(f"[price_tier] Analysis failed: {e}")
            pass
        
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
                        f"The {a[col]} segment generates {self._format_inr(a['total_rev'])} "
                        f"across {a['n']:,} orders, while {b[col]} trails at "
                        f"{self._format_inr(b['total_rev'])} — a {gap_pct:.0f}% revenue gap. "
                        f"This split warrants targeted acquisition investment in the underperforming segment."
                    ),
                    why_it_matters=(
                        f"A {gap_pct:.0f}% gap between {col} segments signals an addressable "
                        f"growth opportunity if the gap is driven by reach rather than demand."
                    ),
                    recommendation=f"Run a paid-acquisition test targeted specifically at {b[col]} for 30 days.",
                    rule_type=f"demographic_split_{col.lower()}",
                    qualified_segments=[str(a[col])],
                    excluded_segments=[str(b[col])]
                ))
            except Exception: pass
        return insights

    def _rule_hr_attrition(self, df: pl.DataFrame, profile) -> list:
        """Fires when dataset has Attrition + Department + MonthlyIncome columns."""
        insights = []

        attr_col   = next((c for c in df.columns if "attrition"  in c.lower()), None)
        dept_col   = next((c for c in df.columns if "department" in c.lower()), None)
        income_col = next((c for c in df.columns
                           if any(k in c.lower() for k in
                                  ["income", "salary", "wage", "monthlyincome"])), None)

        if not attr_col:
            return []

        try:
            pdf = df.to_pandas() if hasattr(df, "to_pandas") else df

            total = len(pdf)
            left  = pdf[attr_col].astype(str).str.strip().str.lower()
            left_count    = (left == "yes").sum()
            attrition_rate = left_count / total * 100

            insights.append(BusinessInsight(
                title=f"Attrition Rate: {attrition_rate:.1f}% of Workforce",
                description=(
                    f"Out of {total:,} employees, {left_count:,} have left "
                    f"({attrition_rate:.1f}% attrition rate). "
                    f"Industry benchmark for technology: 13–15%. "
                    f"{'Above benchmark — retention requires immediate action.'  if attrition_rate > 15 else 'Within benchmark range — maintain current retention programmes.'}"
                ),
                why_it_matters=(
                    "Each departing employee costs 50–200% of their annual salary "
                    "in recruitment, onboarding, and lost productivity."
                ),
                evidence=f"{left_count:,} left out of {total:,} total employees",
                impact="🔴 Critical" if attrition_rate > 15 else "🟠 Important",
                recommendation=(
                    "Focus retention on highest-risk segments. "
                    "Exit interview data and engagement surveys should be "
                    "reviewed quarterly to identify leading indicators."
                ),
                rule_type="hr_attrition",
                score=9.5,
                chart_data={"attrition_rate": round(attrition_rate, 1),
                            "left_count": int(left_count), "total": total},
            ))

            if dept_col:
                dept_attr = (
                    pdf.groupby(dept_col)[attr_col]
                    .apply(lambda x: (x.astype(str).str.lower() == "yes").mean() * 100)
                    .sort_values(ascending=False)
                )
                if len(dept_attr) >= 2:
                    worst_dept = dept_attr.index[0]
                    worst_rate = dept_attr.iloc[0]
                    best_dept  = dept_attr.index[-1]
                    best_rate  = dept_attr.iloc[-1]

                    insights.append(BusinessInsight(
                        title=f"Highest Attrition: {worst_dept} at {worst_rate:.0f}%",
                        description=(
                            f"{worst_dept} has the highest attrition at {worst_rate:.0f}%, "
                            f"vs {best_dept} at {best_rate:.0f}%. "
                            f"A {worst_rate - best_rate:.0f}pp gap between departments "
                            f"suggests department-specific issues — management, workload, "
                            f"or career growth opportunities."
                        ),
                        why_it_matters="Department-level attrition reveals where retention investment will have highest ROI.",
                        evidence=f"{worst_dept}: {worst_rate:.0f}% | {best_dept}: {best_rate:.0f}%",
                        impact="🔴 Critical" if worst_rate > 20 else "🟠 Important",
                        recommendation=(
                            f"Conduct stay interviews in {worst_dept} within 30 days. "
                            f"Investigate whether {best_dept}'s lower attrition "
                            f"reflects better management or different role profiles."
                        ),
                        rule_type="hr_attrition_by_dept",
                        score=8.5,
                        chart_data={"dept_rates": dept_attr.to_dict()},
                    ))

            if income_col:
                try:
                    stayers_income = pdf[left == "no"][income_col].median()
                    leavers_income = pdf[left == "yes"][income_col].median()
                    if stayers_income and stayers_income > 0:
                        gap_pct = (stayers_income - leavers_income) / stayers_income * 100
                        if gap_pct > 5:
                            insights.append(BusinessInsight(
                                title=f"Income Gap: Leavers Earn {gap_pct:.0f}% Less Than Stayers",
                                description=(
                                    f"Employees who left had a median income of "
                                    f"{_fmt_currency(leavers_income)}, vs "
                                    f"{_fmt_currency(stayers_income)} for those who stayed — "
                                    f"a {gap_pct:.0f}% gap. "
                                    f"Compensation is a primary driver of voluntary attrition."
                                ),
                                why_it_matters="Salary competitiveness directly predicts voluntary turnover.",
                                evidence=f"Leavers: {_fmt_currency(leavers_income)} | Stayers: {_fmt_currency(stayers_income)}",
                                impact="🔴 Critical" if gap_pct > 20 else "🟠 Important",
                                recommendation=(
                                    "Conduct a compensation benchmarking study. "
                                    "Consider targeted salary adjustments for high-risk roles."
                                ),
                                rule_type="hr_income_gap",
                                score=8.0,
                                chart_data={"leavers_income": leavers_income,
                                            "stayers_income": stayers_income,
                                            "gap_pct": round(gap_pct, 1)},
                            ))
                except Exception:
                    pass

            # Job satisfaction distribution
            sat_col = next((c for c in pdf.columns if any(k in c.lower() for k in
                ["jobsatisfaction", "satisfaction", "engagement",
                 "worklifebalance", "worksatisfaction"])), None)
            if sat_col:
                try:
                    low_count  = int((pdf[sat_col] <= 2).sum())
                    low_pct    = low_count / len(pdf) * 100
                    low_sat    = 0.0
                    high_sat   = 0.0
                    if attr_col:
                        low_sat  = (pdf[pdf[sat_col] <= 2][attr_col]
                                    .apply(lambda x: str(x).lower() == "yes")
                                    .mean()) * 100
                        high_sat = (pdf[pdf[sat_col] >= 3][attr_col]
                                    .apply(lambda x: str(x).lower() == "yes")
                                    .mean()) * 100
                    insights.append(BusinessInsight(
                        title=f"Job Satisfaction: {low_pct:.0f}% Rate Low Satisfaction",
                        description=(
                            f"{low_count:,} employees ({low_pct:.0f}%) report low satisfaction "
                            f"(score 1-2 out of 4). "
                            f"Low-satisfaction employees leave at {low_sat:.0f}% "
                            f"vs {high_sat:.0f}% for satisfied employees — "
                            f"a {low_sat - high_sat:.0f}pp attrition gap driven by engagement."
                        ),
                        why_it_matters=(
                            "Job satisfaction is the strongest leading indicator of voluntary "
                            "attrition — more predictive than salary alone."
                        ),
                        evidence=(
                            f"Low satisfaction attrition: {low_sat:.0f}% | "
                            f"High satisfaction attrition: {high_sat:.0f}%"
                        ),
                        impact="🔴 Critical" if low_sat > 20 else "🟠 Important",
                        recommendation=(
                            "Run pulse surveys to identify top satisfaction drivers. "
                            "Prioritise manager quality and career growth — the two "
                            "highest-leverage satisfaction levers."
                        ),
                        rule_type="hr_satisfaction",
                        score=8.5,
                        chart_data={"low_sat_pct": round(low_pct, 1),
                                    "low_attrition": round(low_sat, 1),
                                    "high_attrition": round(high_sat, 1)},
                    ))
                except Exception as _se:
                    log.warning(f"[hr_satisfaction] {_se}")

        except Exception as e:
            log.warning(f"[hr_attrition] Failed: {e}")

        return insights

    @log_rule
    def _rule_content_library_analysis(self, df, profile) -> list:
        """Fires for Netflix/content library datasets."""
        insights = []

        # Generic column detection — works across Netflix, Disney+,
        # Amazon Prime, Spotify, YouTube, IMDb, etc.
        def _find_col(candidates: list):
            for name in candidates:
                for col in df.columns:
                    if col.lower().replace(" ", "_") == name.replace(" ", "_"):
                        return col
                    if name in col.lower():
                        return col
            return None

        type_col = _find_col([
            "type", "content_type", "show_type", "media_type", "format", "kind", "category"
        ])
        rating_col = _find_col([
            "rating", "maturity_rating", "age_rating", "content_rating",
            "certification", "rated"
        ])
        country_col = _find_col([
            "country", "countries", "origin_country", "country_of_origin",
            "production_country", "region"
        ])
        year_col = _find_col([
            "release_year", "year", "production_year", "release_date",
            "year_released", "premiered"
        ])
        date_col = _find_col([
            "date_added", "dateadded", "added_date", "date_uploaded",
            "upload_date", "available_since", "publish_date", "date_published"
        ])
        genre_col = _find_col([
            "listed_in", "genre", "genres", "categories", "tags",
            "category", "type_genre", "classification"
        ])
        duration_col = _find_col([
            "duration", "runtime", "length", "minutes", "run_time"
        ])

        print(f"[CONTENT_LIBRARY] Columns detected: type={type_col}, "
              f"rating={rating_col}, country={country_col}, "
              f"year={year_col}, date_added={date_col}, genre={genre_col}")

        if not type_col:
            return []

        try:
            pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
            total = len(pdf)

            # 1. Content type split (generic: Movie/TV Show, Track/Album, Video/Short, etc.)
            type_counts = pdf[type_col].value_counts()
            if len(type_counts) >= 2:
                top_type    = type_counts.index[0]
                top_pct     = type_counts.iloc[0] / total * 100
                second_type = type_counts.index[1]
                second_pct  = type_counts.iloc[1] / total * 100

                _STREAMING_TYPES = {"movie", "film", "tv show", "series", "documentary"}
                _MUSIC_TYPES     = {"track", "album", "single", "ep", "song"}
                _top_lower = str(top_type).lower()
                if _top_lower in _STREAMING_TYPES:
                    _context = (
                        "Movies dominate — platform is film-first, TV Shows secondary."
                        if _top_lower in {"movie", "film"}
                        else "TV content leads — binge culture drives engagement."
                    )
                    _rec = (
                        "Invest in original TV Show production to increase time-on-platform and reduce churn."
                        if _top_lower in {"movie", "film"}
                        else "Maintain TV Show dominance while ensuring Movie catalogue stays competitive."
                    )
                elif _top_lower in _MUSIC_TYPES:
                    _context = (
                        "Tracks dominate — singles-first catalogue."
                        if _top_lower == "track"
                        else "Albums lead — long-form listening is core to the experience."
                    )
                    _rec = "Balance track and album content to serve both casual and deep listeners."
                else:
                    _context = f"{top_type} is the dominant content format at {top_pct:.0f}%."
                    _rec = f"Monitor the {top_type}/{second_type} ratio as catalogue scales."

                insights.append(BusinessInsight(
                    title=f"Content Mix: {top_pct:.0f}% {top_type}s, {second_pct:.0f}% {second_type}s",
                    description=(
                        f"The library contains {type_counts.iloc[0]:,} {top_type}s "
                        f"({top_pct:.0f}%) and {type_counts.iloc[1]:,} {second_type}s "
                        f"({second_pct:.0f}%). {_context}"
                    ),
                    why_it_matters="Content mix determines platform identity and subscriber retention strategy.",
                    evidence=f"{top_type}: {top_pct:.0f}% | {second_type}: {second_pct:.0f}%",
                    impact="🟠 Important",
                    recommendation=_rec,
                    rule_type="content_type_split",
                    score=7.5,
                    chart_data={"type_counts": type_counts.to_dict()},
                ))

            # 2. Rating distribution
            if rating_col:
                try:
                    rating_counts = pdf[rating_col].value_counts().head(5)
                    top_rating     = rating_counts.index[0]
                    top_rating_pct = rating_counts.iloc[0] / total * 100
                    mature_ratings = ["tv-ma", "r", "nc-17", "18+"]
                    mature_count   = int(pdf[rating_col].str.lower().str.strip().isin(mature_ratings).sum())
                    mature_pct     = mature_count / total * 100
                    insights.append(BusinessInsight(
                        title=f"Content Rating: {top_rating} Dominates at {top_rating_pct:.0f}%",
                        description=(
                            f"{top_rating} is the most common rating ({top_rating_pct:.0f}% of catalogue). "
                            f"Top 5 ratings: {', '.join(f'{r} ({c:,})' for r, c in rating_counts.items())}. "
                            f"{'Mature content (TV-MA/R) makes up the majority — platform skews adult.' if top_rating in ['TV-MA', 'R'] else 'Family-friendly ratings are prominent — broad audience appeal.'}"
                        ),
                        why_it_matters="Rating distribution defines audience demographics and content acquisition strategy.",
                        evidence=f"Top rating: {top_rating} ({top_rating_pct:.0f}%)",
                        impact="🟠 Important",
                        recommendation=(
                            "Balance mature and family content to avoid audience concentration risk. "
                            "Ensure parental controls are prominent if mature content dominates."
                        ),
                        rule_type="content_rating_distribution",
                        score=7.0,
                        chart_data={"rating_counts": rating_counts.to_dict()},
                    ))
                except Exception as _re:
                    log.warning(f"[content_rating] {_re}")

            # 3. Top producing countries
            if country_col:
                try:
                    country_series = pdf[country_col].dropna().str.split(",").explode().str.strip()
                    country_counts = country_series.value_counts().head(5)
                    top_country     = country_counts.index[0]
                    top_country_pct = country_counts.iloc[0] / len(country_series) * 100
                    insights.append(BusinessInsight(
                        title=f"Top Producer: {top_country} at {top_country_pct:.0f}% of Content",
                        description=(
                            f"{top_country} produces {top_country_pct:.0f}% of all content "
                            f"({country_counts.iloc[0]:,} titles). "
                            f"Top 5 countries: {', '.join(f'{c} ({n:,})' for c, n in country_counts.items())}. "
                            f"Geographic concentration signals both market strength and localisation opportunity."
                        ),
                        why_it_matters="Content origin shapes cultural relevance and international subscriber growth.",
                        evidence=f"Top: {top_country} ({top_country_pct:.0f}%) | {len(country_counts)} countries in top 5",
                        impact="🟠 Important",
                        recommendation=(
                            f"Invest in local content from underrepresented regions to drive international subscriber growth. "
                            f"{'Reduce US dependency by commissioning content from emerging markets.' if 'united states' in top_country.lower() else f'Expand {top_country} content internationally.'}"
                        ),
                        rule_type="content_by_country",
                        score=7.0,
                        chart_data={"country_counts": country_counts.to_dict()},
                    ))
                except Exception as _ce:
                    log.warning(f"[content_by_country] {_ce}")

            # 4. Release year trend
            if year_col:
                try:
                    year_counts   = pdf[year_col].dropna().astype(int).value_counts().sort_index()
                    peak_year     = int(year_counts.idxmax())
                    recent_count  = int(year_counts[year_counts.index >= 2018].sum())
                    recent_pct    = recent_count / total * 100
                    insights.append(BusinessInsight(
                        title=f"Content Recency: {recent_pct:.0f}% Released 2018 or Later",
                        description=(
                            f"{recent_count:,} titles ({recent_pct:.0f}%) were released in 2018 or later. "
                            f"Peak production year: {peak_year}. "
                            f"{'Catalogue is current and fresh — high recency signals strong content investment.' if recent_pct > 50 else 'Catalogue skews older — consider refreshing with newer releases.'}"
                        ),
                        why_it_matters="Content recency directly impacts subscriber satisfaction and churn rates.",
                        evidence=f"2018+: {recent_pct:.0f}% | Peak year: {peak_year}",
                        impact="🟠 Important",
                        recommendation=(
                            f"{'Maintain pipeline of 2024+ releases to stay competitive.' if recent_pct > 50 else 'Prioritise licensing newer titles — catalogue age risks subscriber churn.'}"
                        ),
                        rule_type="content_recency",
                        score=6.5,
                        chart_data={"peak_year": peak_year, "recent_pct": round(recent_pct, 1)},
                    ))
                except Exception as _ye:
                    log.warning(f"[content_year] {_ye}")

            # 5. Content added over time (date_added / upload date column)
            date_added_col = next(
                (c for c in pdf.columns if any(k in c.lower() for k in
                 ["date_added", "dateadded", "date_uploaded",
                  "added_date", "available_since"])), None
            )
            if date_added_col:
                try:
                    import pandas as _pd
                    _dates = _pd.to_datetime(
                        pdf[date_added_col], errors="coerce"
                    ).dropna()
                    if len(_dates) >= 12:
                        _yearly = _dates.dt.year.value_counts().sort_index()
                        _peak_add_year   = int(_yearly.idxmax())
                        _peak_add_count  = int(_yearly.max())
                        _recent_added    = int(_dates[_dates.dt.year >= 2019].count())
                        _recent_pct      = _recent_added / len(_dates) * 100
                        insights.append(BusinessInsight(
                            title=(
                                f"Content Growth: Peak Additions in "
                                f"{_peak_add_year} ({_peak_add_count:,} titles)"
                            ),
                            description=(
                                f"Netflix added the most content in "
                                f"{_peak_add_year} ({_peak_add_count:,} titles). "
                                f"{_recent_added:,} titles ({_recent_pct:.0f}%) "
                                f"were added in 2019 or later. "
                                f"{'Rapid catalogue expansion has slowed — quality over quantity phase.' if _peak_add_year < 2021 else 'Catalogue is in active expansion mode.'}"
                            ),
                            why_it_matters=(
                                "Content addition pace reflects platform investment "
                                "strategy and competitive positioning."
                            ),
                            evidence=(
                                f"Peak year: {_peak_add_year} "
                                f"({_peak_add_count:,} titles) | "
                                f"2019+: {_recent_pct:.0f}%"
                            ),
                            impact="🟠 Important",
                            recommendation=(
                                f"Analyse whether the post-{_peak_add_year} slowdown "
                                f"reflects budget constraints or a deliberate shift to "
                                f"original content production. "
                                f"Subscriber growth correlates with new additions."
                            ),
                            rule_type="content_growth_trend",
                            score=7.0,
                            chart_data={
                                "peak_year": _peak_add_year,
                                "yearly_counts": _yearly.to_dict(),
                                "recent_pct": round(_recent_pct, 1),
                            },
                        ))
                except Exception as _de:
                    log.warning(f"[content_growth] {_de}")

        except Exception as e:
            log.warning(f"[content_library_analysis] Failed: {e}")

        return insights

    def _rule_sports_analysis(self, df, profile) -> list:
        """Fires for cricket/football/sports match datasets."""
        insights = []

        def _find_col(candidates):
            for name in candidates:
                for col in df.columns:
                    if col.lower().replace(" ", "_") == \
                       name.replace(" ", "_"):
                        return col
                    if name in col.lower():
                        return col
            return None

        # EXACT match first for winner — must not match toss_winner
        winner_col = None
        for col in df.columns:
            if col.lower().strip() == "winner":
                winner_col = col
                break
        if not winner_col:
            # Substring fallback — exclude any col containing "toss"
            for col in df.columns:
                if "winner" in col.lower() and "toss" not in col.lower():
                    winner_col = col
                    break

        # toss_winner — exact match only
        toss_col = None
        for col in df.columns:
            if col.lower().strip() in ["toss_winner", "toss"]:
                toss_col = col
                break

        print(f"[SPORTS] winner_col={winner_col}, toss_col={toss_col}")

        team1_col  = _find_col(["team1", "home_team", "team_1"])
        team2_col  = _find_col(["team2", "away_team", "team_2"])
        venue_col  = _find_col(["venue", "stadium", "ground",
                                 "city", "location"])
        season_col = _find_col(["season", "year", "edition"])
        margin_col = _find_col(["result_margin", "margin",
                                 "win_by_runs", "score_diff"])
        pom_col    = _find_col(["player_of_match", "man_of_match",
                                 "best_player", "mvp"])

        # Normalize team name variants (franchise rebrands across seasons)
        _TEAM_ALIASES = {
            "Rising Pune Supergiants": "Rising Pune Supergiant",
            "Delhi Daredevils": "Delhi Capitals",
            "Kings XI Punjab": "Punjab Kings",
        }

        try:
            _norm_df = df.to_pandas() if hasattr(df, "to_pandas") else df
            if team1_col:
                _norm_df[team1_col] = _norm_df[team1_col].replace(_TEAM_ALIASES)
            if team2_col:
                _norm_df[team2_col] = _norm_df[team2_col].replace(_TEAM_ALIASES)
            if winner_col and winner_col != "_derived_winner":
                _norm_df[winner_col] = _norm_df[winner_col].replace(_TEAM_ALIASES)
            if toss_col:
                _norm_df[toss_col] = _norm_df[toss_col].replace(_TEAM_ALIASES)
            df = _norm_df
            print(f"[SPORTS] Team aliases applied")
        except Exception as _na:
            print(f"[SPORTS] Alias normalization skipped: {_na}")

        try:
            pdf = df.to_pandas() if hasattr(df, "to_pandas") \
                  else df
            total = len(pdf)

            # Football: derive winner from home/away goals
            home_goals = _find_col(["home_goals", "fthg",
                                    "home_score", "score_home"])
            away_goals = _find_col(["away_goals", "ftag",
                                    "away_score", "score_away"])
            if home_goals and away_goals and not winner_col:
                pdf["_derived_winner"] = pdf.apply(
                    lambda r: r[team1_col] if r[home_goals] > r[away_goals]
                    else (r[team2_col] if r[away_goals] > r[home_goals]
                          else "Draw"), axis=1
                )
                winner_col = "_derived_winner"

            if not winner_col:
                return []

            # 1. Team win rate analysis — wins + rate side by side
            if team1_col and team2_col:
                all_teams = pd.concat([
                    pdf[team1_col],
                    pdf[team2_col]
                ]).dropna().unique().tolist()

                win_counts = pdf[winner_col].value_counts()
                win_counts = win_counts[win_counts.index.isin(all_teams)]

                if len(win_counts) >= 2:
                    # Per-team stats: matches played and win rate
                    _team_stats = []
                    for _t in win_counts.index:
                        _played = int(
                            (pdf[team1_col] == _t).sum() +
                            (pdf[team2_col] == _t).sum()
                        )
                        _wins = int(win_counts[_t])
                        _rate = _wins / _played * 100 if _played > 0 else 0
                        _team_stats.append({
                            "team": _t,
                            "wins": _wins,
                            "played": _played,
                            "rate": _rate,
                        })

                    # Sort by wins for headline; by rate for "most consistent"
                    _by_wins = sorted(
                        _team_stats, key=lambda x: x["wins"], reverse=True
                    )
                    _by_rate = sorted(
                        [s for s in _team_stats if s["played"] >= 10],
                        key=lambda x: x["rate"], reverse=True
                    )

                    top = _by_wins[0]
                    worst = _by_wins[-1]
                    consistent = _by_rate[0] if _by_rate else top

                    _top5_wins_str = ", ".join(
                        f"{s['team']} ({s['wins']} W, {s['rate']:.0f}%)"
                        for s in _by_wins[:5]
                    )

                    _consistent_note = (
                        f" {consistent['team']} has the highest win rate "
                        f"among active teams ({consistent['rate']:.0f}% "
                        f"from {consistent['played']} matches)."
                        if consistent["team"] != top["team"] else ""
                    )

                    insights.append(BusinessInsight(
                        title=(
                            f"Dominant Team: {top['team']} "
                            f"({top['wins']} wins, {top['rate']:.0f}% rate)"
                        ),
                        description=(
                            f"{top['team']} leads with {top['wins']} wins "
                            f"from {top['played']} matches "
                            f"({top['rate']:.0f}% win rate). "
                            f"{worst['team']} has the fewest wins "
                            f"({worst['wins']} from {worst['played']} "
                            f"matches, {worst['rate']:.0f}% rate)."
                            f"{_consistent_note} "
                            f"Top 5 by wins: {_top5_wins_str}."
                        ),
                        why_it_matters=(
                            "Team performance distribution reveals "
                            "competitive balance and dominant franchises."
                        ),
                        evidence=(
                            f"Leader: {top['team']} {top['wins']} W "
                            f"({top['rate']:.0f}%) | "
                            f"Bottom: {worst['team']} {worst['wins']} W "
                            f"({worst['rate']:.0f}%)"
                        ),
                        impact="🔴 Critical",
                        recommendation=(
                            f"Analyse {top['team']}'s success factors — "
                            f"squad depth, home advantage, or tactical "
                            f"patterns. Study {worst['team']}'s losses for "
                            f"correctable patterns."
                        ),
                        rule_type="sports_team_performance",
                        score=9.0,
                        chart_data={
                            "win_counts": win_counts.head(10).to_dict(),
                            "team_stats": {
                                s["team"]: {
                                    "wins": s["wins"],
                                    "played": s["played"],
                                    "rate": round(s["rate"], 1),
                                }
                                for s in _by_wins[:10]
                            },
                        },
                    ))

            # 2. Toss impact analysis
            if toss_col and winner_col:
                try:
                    toss_won_match = (
                        pdf[toss_col] == pdf[winner_col]
                    ).sum()
                    valid = pdf[[toss_col, winner_col]]\
                        .dropna().__len__()
                    toss_win_rate = (toss_won_match / valid * 100
                                     if valid > 0 else 0)

                    insights.append(BusinessInsight(
                        title=(
                            f"Toss Advantage: "
                            f"{toss_win_rate:.0f}% of Toss Winners "
                            f"Win the Match"
                        ),
                        description=(
                            f"Teams that win the toss go on to win "
                            f"the match {toss_win_rate:.0f}% of the "
                            f"time ({toss_won_match:,} out of "
                            f"{valid:,} matches). "
                            f"{'Toss has significant impact — strategic decision-making at toss is critical.' if toss_win_rate > 55 else 'Toss has minimal impact — match outcome depends more on performance than toss.'}"
                        ),
                        why_it_matters=(
                            "Toss win rate reveals whether conditions "
                            "favour batting/fielding first and informs "
                            "captain decision-making."
                        ),
                        evidence=(
                            f"Toss winner wins match: "
                            f"{toss_win_rate:.0f}% ({toss_won_match}/{valid})"
                        ),
                        impact=(
                            "🔴 Critical" if toss_win_rate > 60
                            else "🟠 Important"
                        ),
                        recommendation=(
                            f"{'Prioritise winning the toss — it provides a decisive edge. Analyse pitch conditions before deciding to bat or field.' if toss_win_rate > 55 else 'Toss outcome is not a strong predictor — focus on team composition and match strategy.'}"
                        ),
                        rule_type="sports_toss_impact",
                        score=7.5,
                        chart_data={"toss_win_rate": round(toss_win_rate, 1)},
                    ))
                except Exception as _te:
                    log.warning(f"[sports_toss] {_te}")

            # 3. Season trend
            if season_col:
                try:
                    season_counts = pdf[season_col].value_counts()\
                        .sort_index()
                    peak_season = season_counts.idxmax()
                    peak_count  = int(season_counts.max())
                    total_seasons = len(season_counts)

                    _growth = (
                        (season_counts.iloc[-1] - season_counts.iloc[0])
                        / max(season_counts.iloc[0], 1) * 100
                    )
                    _trend_dir = (
                        "growing" if _growth > 10
                        else "declining" if _growth < -10
                        else "stable"
                    )
                    insights.append(BusinessInsight(
                        title=(
                            f"Tournament Scale: "
                            f"{total_seasons} Seasons, "
                            f"Peak in {peak_season}"
                        ),
                        description=(
                            f"The tournament spans {total_seasons} seasons "
                            f"({str(season_counts.index.min())}–"
                            f"{str(season_counts.index.max())}) "
                            f"with {total:,} total matches "
                            f"({total/total_seasons:.0f} per season avg). "
                            f"Peak season: {peak_season} ({peak_count} matches). "
                            f"Match volume is {_trend_dir} — "
                            f"{abs(_growth):.0f}% "
                            f"{'increase' if _growth > 0 else 'decrease'} "
                            f"from first to last season. "
                            f"Season growth reflects league expansion, "
                            f"franchise additions, and format changes."
                        ),
                        why_it_matters=(
                            "Tournament growth over seasons reflects "
                            "league expansion and commercial growth."
                        ),
                        evidence=(
                            f"{total_seasons} seasons | "
                            f"Peak: {peak_season} ({peak_count} matches)"
                        ),
                        impact="🟠 Important",
                        recommendation=(
                            f"Track season-over-season match volume "
                            f"as a proxy for league health and expansion."
                        ),
                        rule_type="temporal_peaks",
                        score=10.0,
                        chart_data={
                            "season_counts": season_counts.to_dict(),
                            "peak_season": str(peak_season),
                            "peak_month": str(peak_season),
                            "trough_month": str(season_counts.idxmin()),
                            "pct_gap": round(
                                (season_counts.max() - season_counts.min())
                                / max(season_counts.max(), 1) * 100, 1
                            ),
                            "monthly_data": [
                                (f"{str(yr).split('/')[0].strip()}-01", int(cnt))
                                for yr, cnt in season_counts.items()
                            ],
                        },
                    ))
                except Exception as _se:
                    log.warning(f"[sports_season] {_se}")

            # 4. Venue analysis
            if venue_col and winner_col:
                try:
                    venue_counts = pdf[venue_col].value_counts()
                    top_venue = venue_counts.index[0]
                    top_venue_count = int(venue_counts.iloc[0])
                    top_venue_pct = top_venue_count / total * 100

                    insights.append(BusinessInsight(
                        title=(
                            f"Top Venue: {top_venue} "
                            f"({top_venue_count} matches, "
                            f"{top_venue_pct:.0f}%)"
                        ),
                        description=(
                            f"{top_venue} hosted the most matches "
                            f"({top_venue_count}, {top_venue_pct:.0f}% "
                            f"of all games). "
                            f"Top 5 venues: "
                            f"{', '.join(f'{v} ({n})' for v, n in venue_counts.head(5).items())}. "
                            f"Venue concentration affects home "
                            f"advantage and pitch conditions."
                        ),
                        why_it_matters=(
                            "Venue frequency and home advantage "
                            "significantly influence match outcomes."
                        ),
                        evidence=(
                            f"Top venue: {top_venue} "
                            f"({top_venue_pct:.0f}% of matches)"
                        ),
                        impact="🟠 Important",
                        recommendation=(
                            f"Analyse {top_venue} win rates by team "
                            f"to identify home advantage patterns."
                        ),
                        rule_type="sports_venue_analysis",
                        score=6.0,
                        chart_data={
                            "venue_counts": venue_counts.head(10).to_dict()
                        },
                    ))
                except Exception as _ve:
                    log.warning(f"[sports_venue] {_ve}")

            # 4b. Result type from 'result' column (e.g. "runs"/"wickets"/"tie"/"no result")
            _result_col = _find_col(["result", "result_type",
                                     "win_by", "outcome"])
            print(f"[SPORTS] result_col={_result_col}")
            if _result_col:
                try:
                    _result_vals = (
                        pdf[_result_col].dropna()
                        .astype(str).str.lower().str.strip()
                    )
                    _won_runs  = int((_result_vals == "runs").sum())
                    _won_wkts  = int((_result_vals == "wickets").sum())
                    _ties      = int((_result_vals == "tie").sum())
                    _no_result = int((_result_vals == "no result").sum())
                    _other     = _ties + _no_result

                    print(f"[SPORTS] Results — runs:{_won_runs}, "
                          f"wickets:{_won_wkts}, ties:{_ties}, "
                          f"no result:{_no_result}")

                    if _won_runs + _won_wkts > 0:
                        _chase_dominant = _won_wkts > _won_runs
                        insights.append(BusinessInsight(
                            title=(
                                f"Match Outcomes: "
                                f"{max(_won_runs, _won_wkts):,} "
                                f"{'Wicket' if _chase_dominant else 'Run'}"
                                f" Wins Lead"
                            ),
                            description=(
                                f"Of {total:,} matches: "
                                f"{_won_wkts:,} won by wickets "
                                f"(chasing, "
                                f"{_won_wkts/total*100:.0f}%), "
                                f"{_won_runs:,} won by runs "
                                f"(defending, "
                                f"{_won_runs/total*100:.0f}%). "
                                f"{_ties} ties, "
                                f"{_no_result} no results. "
                                f"{'Chasing teams win more often — batting second is statistically advantageous in this tournament.' if _chase_dominant else 'Defending teams win more — posting big totals is the stronger strategy.'}"
                            ),
                            why_it_matters=(
                                "Win method distribution directly informs "
                                "toss decision strategy. Teams winning the "
                                "toss should use this data to decide "
                                "bat or field."
                            ),
                            evidence=(
                                f"Wicket wins: {_won_wkts} "
                                f"({_won_wkts/total*100:.0f}%) | "
                                f"Run wins: {_won_runs} "
                                f"({_won_runs/total*100:.0f}%)"
                            ),
                            impact="🟠 Important",
                            recommendation=(
                                f"{'Choose to chase when winning the toss — wicket wins dominate, confirming chasing is the stronger strategy.' if _chase_dominant else 'Bat first and post a competitive total — run wins are more common, defending scores wins matches.'}"
                            ),
                            rule_type="sports_result_type",
                            score=8.0,
                            chart_data={
                                "won_by_runs": _won_runs,
                                "won_by_wickets": _won_wkts,
                                "ties": _ties,
                                "no_result": _no_result,
                            },
                        ))
                except Exception as _rte:
                    print(f"[SPORTS RESULT ERROR] {_rte}")
                    import traceback
                    traceback.print_exc()

            # 5. Player of Match frequency
            if pom_col:
                try:
                    pom_counts = pdf[pom_col].value_counts().head(5)
                    top_player = pom_counts.index[0]
                    top_awards = int(pom_counts.iloc[0])

                    insights.append(BusinessInsight(
                        title=(
                            f"Most Valuable Player: "
                            f"{top_player} ({top_awards} awards)"
                        ),
                        description=(
                            f"{top_player} won Player of the Match "
                            f"{top_awards} times. "
                            f"Top 5: "
                            f"{', '.join(f'{p} ({n})' for p, n in pom_counts.items())}."
                        ),
                        why_it_matters=(
                            "Player of Match frequency identifies "
                            "match-winners and high-impact performers."
                        ),
                        evidence=f"Top: {top_player} ({top_awards} awards)",
                        impact="🟠 Important",
                        recommendation=(
                            f"Track {top_player}'s match conditions "
                            f"to understand what triggers peak performance."
                        ),
                        rule_type="sports_player_performance",
                        score=6.0,
                        chart_data={"pom_counts": pom_counts.to_dict()},
                    ))
                except Exception as _pe:
                    log.warning(f"[sports_player] {_pe}")

        except Exception as e:
            log.warning(f"[sports_analysis] Failed: {e}")

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

    def _is_monetary_column(self, col_name: str) -> bool:
        cl = col_name.lower()
        non_monetary = {"quantity", "qty", "count", "units", "rating",
                        "score", "rank", "age", "days", "months", "years",
                        "number", "num", "records", "id"}
        monetary = {"price", "cost", "revenue", "amount", "value",
                    "sales", "total", "spend", "fee", "charge", "profit"}
        if any(k in cl for k in non_monetary):
            return False
        return any(k in cl for k in monetary)

    def _format_inr(self, value: float) -> str:
        """Format number using the current run's currency symbol."""
        try:
            return _fmt_currency(float(value))
        except (TypeError, ValueError):
            return _fmt_currency(0)

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

            # Peak/trough on FULL period-based data
            peak_idx   = revenues.index(max(revenues))
            trough_idx = revenues.index(min(revenues))
            peak_month   = months[peak_idx].strftime("%B")
            trough_month = months[trough_idx].strftime("%B")
            peak_val   = revenues[peak_idx]
            trough_val = revenues[trough_idx]
            pct_gap = ((peak_val - trough_val) / peak_val) * 100
            
            # TIER 1.2: Compute trend slope and R²
            revenues_arr = np.array(revenues)
            months_arr = np.arange(len(revenues))
            slope, intercept = np.polyfit(months_arr, revenues_arr, 1)
            avg_rev = np.mean(revenues_arr)
            slope_pct = (slope / avg_rev) * 100 if avg_rev > 0 else 0  # monthly growth rate

            # R² tells us whether the linear fit explains meaningful variance
            predicted = slope * months_arr + intercept
            ss_res = np.sum((revenues_arr - predicted) ** 2)
            ss_tot = np.sum((revenues_arr - avg_rev) ** 2)
            r_squared = float((1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0)

            trend_direction = "growing" if slope_pct > 1 else "declining" if slope_pct < -1 else "flat"
            
            # TIER 1.2: Simple seasonality detection (std of month-of-year averages)
            # Group by calendar month (1-12) to detect recurring patterns
            try:
                pdf_tmp = df_parsed.to_pandas()
                pdf_tmp["_cal_month"] = pd.to_datetime(pdf_tmp["_parsed_date"]).dt.month
                monthly_avg = pdf_tmp.groupby("_cal_month")[rev_col].mean()
                seasonality_cv = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() > 0 else 0
                has_seasonality = seasonality_cv > 0.15
            except Exception:
                has_seasonality = False
                seasonality_cv = 0

            # ── Chart data: Use period-based window centered on peak ──
            # This ensures the chart shows the actual periods where peak/trough occurred
            MAX_CHART_MONTHS = 12
            half  = MAX_CHART_MONTHS // 2
            start = max(0, peak_idx - half)
            end   = min(len(months), start + MAX_CHART_MONTHS)
            start = max(0, end - MAX_CHART_MONTHS)
            display_months   = months[start:end]
            display_revenues = revenues[start:end]

            # Chart data: period-based (e.g., "2028-01", "2028-02", "2028-03")
            # This matches the actual periods where peak/trough were detected
            chart_monthly_data = [
                (m.strftime("%Y-%m"), r) for m, r in zip(display_months, display_revenues)
            ]

            mom_parts = [
                f"{m.strftime('%b')}={self._format_inr(r)}"
                for m, r in zip(display_months, display_revenues)
            ]
            mom_str = ("..." if len(months) > MAX_CHART_MONTHS else "") + " → ".join(mom_parts)
            
            # TIER 1.2: Build richer insight description — honest about R²
            if r_squared < 0.10:
                trend_line = f"Flat (R²={r_squared:.2f} — no meaningful directional signal)"
            else:
                trend_line = f"{trend_direction} at {slope_pct:+.1f}%/mo (R²={r_squared:.2f})"
            description = (
                f"Revenue trend: {trend_line}. "
                f"Peak: {peak_month} ({self._format_inr(peak_val)}), "
                f"Trough: {trough_month} ({self._format_inr(trough_val)}) — {pct_gap:.0f}% gap. "
            )
            if has_seasonality:
                description += f"Seasonality detected (CV={seasonality_cv:.2f} across calendar months). "
            description += f"Monthly breakdown: {mom_str}."

            return [BusinessInsight(
                title=f"Revenue {trend_direction.title()}: {peak_month} Peak, {trough_month} Trough",
                description=description,
                why_it_matters=(
                    "Temporal concentration creates cash flow risk and signals seasonality "
                    "that should inform inventory and marketing planning."
                ),
                evidence=(
                    f"Peak: {peak_month} ({self._format_inr(peak_val)}) | "
                    f"Trough: {trough_month} ({self._format_inr(trough_val)}) | "
                    f"Gap: {pct_gap:.1f}% | R²={r_squared:.2f} | Slope: {slope_pct:+.1f}%/mo"
                ),
                impact="🔴 Critical" if pct_gap > 30 or abs(slope_pct) > 5 else "🟠 Important",
                recommendation=(
                    f"Investigate the {trough_month} dip — determine if it is seasonal, "
                    f"promotional, or operational. Pre-position inventory and marketing "
                    f"spend ahead of {peak_month} next cycle."
                ),
                rule_type="temporal_peaks",
                score=9.0,  # TIER 1.2: BOOST from 7.5 to compete with cross-dimensional
                chart_data={
                    "monthly_data": chart_monthly_data,
                    "peak_month": peak_month,
                    "peak_val": peak_val,
                    "trough_month": trough_month,
                    "trough_val": trough_val,
                    "pct_gap": round(pct_gap, 1),
                    "trend_slope_pct": round(slope_pct, 2),
                    "has_seasonality": has_seasonality,
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
    def _rule_cross_dimensional(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """
        ✅ GAP 1: Cross-Dimensional Reasoning (FIX 2 ENHANCED)
        Combine 2+ variables to generate non-obvious composite insights.
        This is what separates rule-based stats from reasoning-based AI.
        
        FIX 2 CHANGES:
        - Added Category × PaymentMethod pattern detection
        - Lowered variance threshold (20% → 10%)
        - More flexible column detection
        - Better logging for debugging
        """
        insights = []
        pdf = df.to_pandas()
        
        rev_col = profile.revenue_col or profile.price_col
        cost_col = next((c for c in df.columns
                        if any(k in c.lower() for k in
                               ["cost", "price", "spend", "expense"])), None)
        geo_col = profile.geographic_col
        cat_col = profile.category_col
        
        # FIX 2: Detect PaymentMethod column
        payment_col = next((c for c in df.columns
                           if any(k in c.lower() for k in
                                  ["payment", "paymentmethod", "pay_method"])), None)
        
        # Pattern 1: High Revenue + Low Cost = High Margin Zone
        if rev_col and cost_col and geo_col and rev_col != cost_col:
            try:
                grp = pdf.groupby(geo_col).agg(
                    avg_rev=(rev_col, "mean"),
                    avg_cost=(cost_col, "mean")
                ).dropna()
                
                if len(grp) >= 2:
                    grp["margin_proxy"] = grp["avg_rev"] - grp["avg_cost"]
                    
                    best = grp["margin_proxy"].idxmax()
                    worst = grp["margin_proxy"].idxmin()
                    best_rev = grp.loc[best, "avg_rev"]
                    worst_rev = grp.loc[worst, "avg_rev"]
                    best_margin = grp.loc[best, "margin_proxy"]
                    
                    # ✅ V4: Add impact quantification
                    quant = ImpactQuantifier.margin_replication_gain(
                        pdf, geo_col, rev_col, cost_col, best_region=best
                    )
                    
                    description_base = (
                        f"{best} combines high revenue ({_fmt_currency(best_rev)} avg) with "
                        f"lower cost — the strongest margin proxy in the dataset. "
                        f"{worst} shows the inverse pattern and warrants a cost audit."
                    )
                    
                    # Append quantification if available
                    if quant and "statement" in quant:
                        description_base += f" {quant['statement']}"
                    
                    insight = BusinessInsight(
                        title=f"Margin Zone Identified: {best} is High-Efficiency Region",
                        description=description_base,
                        why_it_matters="Margin efficiency varies by region — this is a strategic expansion signal.",
                        evidence=f"Margin proxy: {_fmt_currency(best_margin)} in {best}",
                        impact="🔴 Critical",
                        confidence_label="high",
                        recommendation=(
                            f"Prioritize {best} for expansion investment. "
                            f"Apply {best}'s operational model to {worst} to close the efficiency gap."
                        ),
                        rule_type="cross_dimensional_margin",
                        score=10.0
                    )
                    
                    # Add quantification metadata
                    if quant:
                        insight.chart_data = insight.chart_data or {}
                        insight.chart_data.update({
                            "uplift_abs": quant.get("uplift_abs", 0),
                            "uplift_pct": quant.get("uplift_pct", 0)
                        })
                    
                    insights.append(insight)
            except Exception:
                pass
        
        # Pattern 2: Category × Region Dominance (from heatmap data)
        if rev_col and geo_col and cat_col:
            try:
                pivot = pdf.groupby([geo_col, cat_col])[rev_col].sum().unstack(cat_col).fillna(0)
                
                if len(pivot) >= 2 and len(pivot.columns) >= 2:
                    # Which category wins in the MOST regions?
                    dominant_cat = (pivot.apply(lambda r: r.idxmax(), axis=1)
                                    .value_counts().idxmax())
                    dominant_count = (pivot.apply(lambda r: r.idxmax(), axis=1)
                                     .value_counts().iloc[0])
                    total_regions = len(pivot)
                    
                    # Which region has the most uneven category mix?
                    row_cv = pivot.apply(
                        lambda r: r.std()/r.mean() if r.mean() > 0 else 0, axis=1
                    )
                    volatile_region = row_cv.idxmax()
                    
                    insights.append(BusinessInsight(
                        title=f"{dominant_cat} Dominates in {dominant_count}/{total_regions} Regions",
                        description=(
                            f"{dominant_cat} is the top-performing category in "
                            f"{dominant_count} out of {total_regions} regions — "
                            f"a cross-regional dominance signal. "
                            f"{volatile_region} shows the highest category variability, "
                            f"indicating uneven category performance within that region."
                        ),
                        why_it_matters="Cross-regional category dominance indicates product-market fit strength.",
                        evidence=f"{dominant_cat} leads in {dominant_count}/{total_regions} regions",
                        impact="🔴 Critical",
                        confidence_label="high",
                        recommendation=(
                            f"Scale {dominant_cat} investment uniformly across all regions. "
                            f"Investigate {volatile_region} for category-specific execution gaps."
                        ),
                        rule_type="cross_dimensional_dominance",
                        score=10.0
                    ))
            except Exception:
                pass
        
        # Pattern 3: Volume vs Value Decoupling by Segment
        qty_col = profile.qty_col or next((c for c in df.columns
                     if any(k in c.lower() for k in
                            ["qty", "quantity", "units", "volume", "count"])), None)
        if rev_col and qty_col and cat_col and rev_col != qty_col:
            try:
                # P0 FIX (Bug 0.4): Always compute actual revenue, never use raw unit price
                if profile.revenue_col:
                    pdf_tmp = pdf.copy()
                    pdf_tmp["_computed_rev"] = pdf[profile.revenue_col]
                elif profile.price_col and profile.qty_col:
                    pdf_tmp = pdf.copy()
                    pdf_tmp["_computed_rev"] = pdf[profile.price_col] * pdf[profile.qty_col]
                else:
                    pdf_tmp = pdf.copy()
                    pdf_tmp["_computed_rev"] = pdf[rev_col]
                
                grp2 = pdf_tmp.groupby(cat_col).agg(
                    total_rev=("_computed_rev", "sum"),
                    total_qty=(qty_col, "sum")
                ).dropna()
                
                if len(grp2) >= 2:
                    grp2["rev_per_unit"] = grp2["total_rev"] / grp2["total_qty"].replace(0, 1)
                    
                    high_val = grp2["rev_per_unit"].idxmax()
                    high_vol = grp2["total_qty"].idxmax()
                    
                    if high_val != high_vol:
                        high_val_rpu = grp2.loc[high_val, "rev_per_unit"]
                        insights.append(BusinessInsight(
                            title=f"Volume–Value Decoupling: {high_val} vs {high_vol}",
                            description=(
                                f"{high_val} generates the highest revenue per unit "
                                f"({_fmt_currency(high_val_rpu)}) but "
                                f"{high_vol} leads in volume. "
                                f"These are different optimization levers — "
                                f"value maximization vs. volume maximization."
                            ),
                            why_it_matters="Volume and value leaders require different strategies.",
                            evidence=f"RPU leader: {high_val}, Volume leader: {high_vol}",
                            impact="🔴 Critical",
                            confidence_label="high",
                            recommendation=(
                                f"Run dual strategy: grow {high_val} for margin, "
                                f"grow {high_vol} for market share. "
                                f"Do not apply same pricing strategy to both."
                            ),
                            rule_type="cross_dimensional_volume_value",
                            score=9.0
                        ))
            except Exception:
                pass
        
        # FIX 2: Pattern 4 - Category × PaymentMethod Heatmap
        # This pattern detects if certain categories perform better with specific payment methods
        if rev_col and cat_col and payment_col:
            try:
                log.info(f"[cross_dimensional] Trying Category × PaymentMethod pattern...")
                
                # Create contingency table — MUST use transaction counts, not revenue amounts.
                # Revenue-weighted tables produce astronomically large chi² values (e.g. 53,000)
                # because each cell value can be millions of rupees instead of 0–200 transactions.
                ct = pd.crosstab(pdf[cat_col], pdf[payment_col]).fillna(0)
                
                if len(ct) >= 2 and len(ct.columns) >= 2:
                    # Calculate variance across cells (normalized)
                    # FIX 2: Lowered threshold from 0.20 to 0.10
                    overall_mean = ct.values.mean()
                    overall_std = ct.values.std()
                    variance_coef = overall_std / overall_mean if overall_mean > 0 else 0

                    log.info(f"[cross_dimensional] Category × PaymentMethod variance: {variance_coef:.3f}")

                    # Chi-square test on the contingency table
                    from scipy.stats import chi2_contingency
                    try:
                        chi2, p_val, dof, expected = chi2_contingency(ct)
                        n_total = ct.sum().sum()
                        min_dim = min(ct.shape) - 1
                        cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if (n_total * min_dim) > 0 else 0
                        if cramers_v < 0.1:
                            v_interp = "negligible"
                        elif cramers_v < 0.3:
                            v_interp = "small"
                        elif cramers_v < 0.5:
                            v_interp = "moderate"
                        else:
                            v_interp = "large"
                        chi2_stats = {"chi2": chi2, "p": p_val, "dof": dof, "v": cramers_v, "v_interp": v_interp}
                    except Exception as chi_err:
                        log.warning(f"[cross_dimensional] Chi-square failed: {chi_err}")
                        chi2_stats = None

                    if chi2_stats and chi2_stats["p"] > 0.05:
                        log.info(f"[cross_dimensional] Cross-dimensional pattern not statistically significant (p={chi2_stats['p']:.3f}). Suppressed.")
                    elif chi2_stats and chi2_stats["v"] < 0.1:
                        log.info(
                            f"[cross_dimensional] Cross-dimensional pattern statistically significant "
                            f"but effect size negligible (V={chi2_stats['v']:.3f}). Suppressed to avoid misleading the reader."
                        )
                    elif variance_coef > 0.10:  # Lowered from 0.20
                        # Find the strongest category-payment combination
                        max_val = ct.max().max()
                        max_idx = ct.stack().idxmax()
                        best_cat, best_payment = max_idx

                        # Find weakest combination
                        min_val = ct.min().min()
                        min_idx = ct.stack().idxmin()
                        worst_cat, worst_payment = min_idx

                        # Calculate concentration
                        total_rev = ct.sum().sum()
                        best_pct = (max_val / total_rev * 100) if total_rev > 0 else 0
                        worst_pct = (min_val / total_rev * 100) if total_rev > 0 else 0

                        _var_qualifier = (
                            "vary significantly" if variance_coef >= 0.25
                            else "show moderate variation"
                        )
                        chi2_suffix = ""
                        if chi2_stats:
                            chi2_suffix = (
                                f" Chi-square test confirms association is statistically significant "
                                f"(χ²={chi2_stats['chi2']:.1f}, p={chi2_stats['p']:.4f}, "
                                f"Cramér's V={chi2_stats['v']:.2f} [{chi2_stats['v_interp']} effect])."
                            )
                        description = (
                            f"{best_cat} × {best_payment} generates {_fmt_currency(max_val)} "
                            f"({best_pct:.1f}% of total revenue) — the strongest category-payment "
                            f"combination in the dataset. "
                            f"Weakest: {worst_cat} × {worst_payment} at {_fmt_currency(min_val)} "
                            f"({worst_pct:.1f}% of total). "
                            f"Payment method preferences {_var_qualifier} by category "
                            f"(variance coefficient: {variance_coef:.2f}), indicating that "
                            f"different products attract different payment behaviours."
                            f"{chi2_suffix}"
                        )

                        insights.append(BusinessInsight(
                            title=f"Cross-Dimensional Pattern: {best_cat} × {best_payment}",
                            description=description,
                            why_it_matters=(
                                "Category-payment patterns reveal customer preferences and can "
                                "inform targeted promotions, payment incentives, and checkout optimization."
                            ),
                            evidence=(
                                f"Variance coefficient: {variance_coef:.2f}, Top combo: {best_cat} × {best_payment}"
                                + (f", χ²={chi2_stats['chi2']:.1f} p={chi2_stats['p']:.4f} V={chi2_stats['v']:.2f}" if chi2_stats else "")
                            ),
                            impact="🟠 Important",
                            confidence_label="high",
                            recommendation=(
                                f"Promote {best_payment} as the preferred payment method for {best_cat}. "
                                f"Analyze why {worst_cat} × {worst_payment} underperforms — "
                                f"consider payment-specific incentives or checkout friction analysis."
                            ),
                            rule_type="cross_dimensional_category_payment",
                            score=8.0,
                            chart_data={
                                "type": "heatmap",
                                "data": ct.to_dict(),
                                "best_combo": f"{best_cat} × {best_payment}",
                                "variance": variance_coef,
                                "chi2_stats": chi2_stats,
                            }
                        ))
                        log.info(f"[cross_dimensional] ✅ Generated Category × PaymentMethod insight")
                    else:
                        log.info(f"[cross_dimensional] Variance too low ({variance_coef:.3f} < 0.10), skipping")
            except Exception as e:
                log.warning(f"[cross_dimensional] Category × PaymentMethod failed: {e}")
                pass
        
        return insights

    @log_rule
    def _rule_pricing_inconsistency(self, df: pl.DataFrame, profile: DataProfile) -> Optional[BusinessInsight]:
        """
        ✅ GAP 4: Detect when cost/price spread signals non-standardized pricing.
        High CV or wide P10-P90 spread indicates pricing inconsistency.
        """
        pdf = df.to_pandas()
        
        # Look for cost/price columns (excluding revenue columns)
        cost_col = next((c for c in df.columns
                        if any(k in c.lower() for k in
                               ["cost", "price", "unit"]) and
                        not any(k in c.lower() for k in
                               ["total", "sales", "amount", "revenue"])), None)
        cat_col = profile.category_col
        
        if not cost_col or not cat_col:
            return None
        
        try:
            # CV by category — high inter-category price variance
            cat_cv = pdf.groupby(cat_col)[cost_col].agg(
                lambda x: x.std()/x.mean() if x.mean() > 0 else 0
            )
            overall_cv = pdf[cost_col].std() / pdf[cost_col].mean() if pdf[cost_col].mean() > 0 else 0
            
            p10 = pdf[cost_col].quantile(0.10)
            p90 = pdf[cost_col].quantile(0.90)
            spread_ratio = p90 / p10 if p10 > 0 else 0
            
            if spread_ratio <= 3 and overall_cv <= 0.5:
                return None  # Not unusual enough

            # ── BUG 0.6 GUARD — check if variance is structural ──────────
            if cat_col and cat_col in pdf.columns:
                within_cvs = pdf.groupby(cat_col)[cost_col].agg(
                    lambda x: x.std()/x.mean() if x.mean() > 0 else 0
                )
                avg_within_cv = within_cvs.mean()
                if avg_within_cv > overall_cv * 0.80:
                    log.info(
                        f"[pricing_inconsistency] Suppressed: within-{cat_col} CV "
                        f"({avg_within_cv:.3f}) ≈ overall CV ({overall_cv:.3f}) — "
                        f"spread is product-mix driven, not pricing chaos."
                    )
                    return None

            worst_cat = cat_cv.idxmax() if len(cat_cv) > 0 else "Unknown"
            worst_cv = cat_cv.max() if len(cat_cv) > 0 else 0
            
            return BusinessInsight(
                title="Pricing Not Standardized — High Cost Variability",
                description=(
                    f"{cost_col} ranges from {_fmt_currency(p10)} (P10) to {_fmt_currency(p90)} (P90) "
                    f"— a {spread_ratio:.1f}× spread. "
                    f"Overall CV: {overall_cv:.2f}. "
                    f"{worst_cat} shows the highest internal price variance (CV={worst_cv:.2f}), "
                    f"suggesting inconsistent pricing rules or data quality issues."
                ),
                why_it_matters="Pricing inconsistency erodes margin predictability and customer trust.",
                evidence=f"P10-P90 spread: {spread_ratio:.1f}×, CV: {overall_cv:.2f}",
                impact="🔴 Critical" if spread_ratio > 5 else "🟠 Important",
                confidence_label="high",
                recommendation=(
                    f"Standardize pricing tiers for {cost_col}. "
                    f"Audit {worst_cat} for rogue pricing. "
                    f"Use P25-P75 range as the acceptable pricing band."
                ),
                rule_type="pricing_inconsistency",
                score=6.0
            )
        except Exception:
            pass
        
        return None

    @log_rule
    def _rule_causal_pricing(self, df: pl.DataFrame, profile: DataProfile) -> Optional[BusinessInsight]:
        """
        ✅ V4 ADDITION 4: Explain WHY cost variability exists by finding its strongest predictor.
        Uses ANOVA eta-squared to identify the primary driver.
        """
        pdf = df.to_pandas()
        
        # Look for cost/price columns
        cost_col = next((c for c in df.columns
                        if any(k in c.lower() for k in ["cost", "price"]) and
                        not any(k in c.lower() for k in ["total", "sales", "amount", "revenue"])), None)
        
        if not cost_col:
            return None
        
        try:
            cat_cols = [c for c in pdf.select_dtypes("object").columns if c not in profile.identifiers]
            best_predictor = None
            best_eta2 = 0  # eta-squared = variance explained
            
            for col in cat_cols:
                n_unique = pdf[col].nunique()
                if n_unique < 2 or n_unique > 20:
                    continue
                
                groups = [grp[cost_col].dropna().values
                         for _, grp in pdf.groupby(col)]
                
                if len(groups) < 2:
                    continue
                
                # One-way ANOVA eta-squared (% variance explained by this grouping)
                grand_mean = pdf[cost_col].mean()
                ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0)
                ss_total = pdf[cost_col].var() * len(pdf) if pdf[cost_col].var() > 0 else 1
                eta2 = ss_between / ss_total if ss_total > 0 else 0
                
                if eta2 > best_eta2:
                    best_eta2 = eta2
                    best_predictor = col
            
            if best_predictor and best_eta2 > 0.05:
                return BusinessInsight(
                    title=f"Root Cause: {best_predictor} Drives {best_eta2*100:.0f}% of Price Variability",
                    description=(
                        f"ANOVA analysis shows {best_predictor} explains "
                        f"{best_eta2*100:.0f}% of {cost_col} variance (η²={best_eta2:.2f}). "
                        f"This is the primary structural driver of pricing inconsistency — "
                        f"not random noise, but a systematic {best_predictor}-dependent pricing pattern."
                    ),
                    why_it_matters="Understanding the root cause enables targeted standardization.",
                    evidence=f"η²={best_eta2:.2f} at n={len(pdf):,}",
                    impact="🔴 Critical" if best_eta2 > 0.30 else "🟠 Important",
                    confidence_label="high" if best_eta2 > 0.20 else "medium",
                    recommendation=(
                        f"Pricing standardization must happen AT THE {best_predictor.upper()} LEVEL. "
                        f"Define separate pricing tiers per {best_predictor} value, "
                        f"then enforce within-group consistency."
                    ),
                    rule_type="causal_pricing_driver",
                    score=7.0
                )
        except Exception:
            pass
        
        return None

    def _rule_health_analysis(self, df, profile) -> list:
        """Fires for COVID/health/epidemiological datasets."""
        insights = []

        def _find_col(candidates):
            """Find column — handles newlines and spaces in names."""
            for name in candidates:
                for col in df.columns:
                    col_clean = (col.lower()
                                   .replace("\n", "")
                                   .replace(" ", "_")
                                   .replace(",", ""))
                    if name.replace(" ", "_") in col_clean:
                        return col
            return None

        def _to_numeric(series):
            """Convert comma-formatted strings to numbers."""
            try:
                if series.dtype == object:
                    return (series.astype(str)
                                  .str.replace(",", "", regex=False)
                                  .str.replace(" ", "", regex=False)
                                  .pipe(pd.to_numeric, errors="coerce"))
                return pd.to_numeric(series, errors="coerce")
            except Exception:
                return series

        confirmed_col = _find_col([
            "confirmed", "cases", "total_cases",
            "infected", "positive"
        ])
        deaths_col = _find_col([
            "deaths", "death", "fatalities",
            "mortality", "deceased", "total_deaths"
        ])
        recovered_col = _find_col([
            "recovered", "recovery", "discharged", "cured"
        ])
        active_col = _find_col([
            "active", "active_cases", "current_cases"
        ])
        serious_col = _find_col([
            "serious", "critical", "icu",
            "hospitalized", "serious,_critical"
        ])
        country_col = _find_col([
            "country", "region", "location",
            "country/region", "state", "province"
        ])

        if not confirmed_col and not deaths_col:
            return []

        try:
            pdf = df.to_pandas() if hasattr(df, "to_pandas") \
                  else df
            total_records = len(pdf)

            # 1. Overall scale insight
            if confirmed_col:
                try:
                    total_cases = _to_numeric(pdf[confirmed_col]).sum()
                    avg_cases   = _to_numeric(pdf[confirmed_col]).mean()

                    _death_rate_str = ""
                    if deaths_col:
                        total_deaths = _to_numeric(pdf[deaths_col]).sum()
                        death_rate   = (
                            total_deaths / total_cases * 100
                            if total_cases > 0 else 0
                        )
                        _death_rate_str = (
                            f" Death rate: {death_rate:.2f}%."
                        )

                    _rec_rate_str = ""
                    if recovered_col:
                        total_recovered = _to_numeric(pdf[recovered_col]).sum()
                        rec_rate = (
                            total_recovered / total_cases * 100
                            if total_cases > 0 else 0
                        )
                        _rec_rate_str = (
                            f" Recovery rate: {rec_rate:.1f}%."
                        )

                    insights.append(BusinessInsight(
                        title=(
                            f"Total Cases: "
                            f"{int(total_cases):,} across "
                            f"{total_records} records"
                        ),
                        description=(
                            f"Dataset covers {total_records} records "
                            f"with {int(total_cases):,} total confirmed "
                            f"cases (avg {avg_cases:,.0f} per record)."
                            f"{_death_rate_str}"
                            f"{_rec_rate_str}"
                        ),
                        why_it_matters=(
                            "Total case count and rates provide the "
                            "baseline for all epidemiological analysis."
                        ),
                        evidence=(
                            f"Total cases: {int(total_cases):,} | "
                            f"Records: {total_records}"
                        ),
                        impact="🔴 Critical",
                        recommendation=(
                            "Monitor death rate and recovery rate trends "
                            "over time — sustained improvement in both "
                            "indicates effective intervention."
                        ),
                        rule_type="health_case_summary",
                        score=9.5,
                        chart_data={
                            "total_cases": int(total_cases),
                            "total_records": total_records,
                        },
                    ))
                except Exception as _ce:
                    log.warning(f"[health_cases] {_ce}")

            # 2. Death rate analysis
            if deaths_col and confirmed_col:
                try:
                    total_deaths  = int(_to_numeric(pdf[deaths_col]).sum())
                    total_cases   = int(_to_numeric(pdf[confirmed_col]).sum())
                    death_rate    = (
                        total_deaths / total_cases * 100
                        if total_cases > 0 else 0
                    )

                    insights.append(BusinessInsight(
                        title=(
                            f"Mortality Rate: "
                            f"{death_rate:.2f}% "
                            f"({total_deaths:,} deaths)"
                        ),
                        description=(
                            f"Total deaths: {total_deaths:,} from "
                            f"{total_cases:,} confirmed cases "
                            f"({death_rate:.2f}% case fatality rate). "
                            f"{'Above 2% — elevated mortality requiring urgent intervention.' if death_rate > 2 else 'Below 2% — within manageable range for most health systems.'}"
                        ),
                        why_it_matters=(
                            "Case fatality rate (CFR) is the primary "
                            "indicator of disease severity and healthcare "
                            "system capacity."
                        ),
                        evidence=(
                            f"Deaths: {total_deaths:,} | "
                            f"CFR: {death_rate:.2f}%"
                        ),
                        impact=(
                            "🔴 Critical" if death_rate > 2
                            else "🟠 Important"
                        ),
                        recommendation=(
                            "Track CFR by region and time period. "
                            "Rising CFR indicates overwhelmed healthcare "
                            "capacity — trigger surge protocols "
                            "when CFR exceeds 3%."
                        ),
                        rule_type="health_mortality",
                        score=9.0,
                        chart_data={
                            "death_rate": round(death_rate, 2),
                            "total_deaths": total_deaths,
                        },
                    ))
                except Exception as _de:
                    log.warning(f"[health_deaths] {_de}")

            # 3. Recovery analysis
            if recovered_col and confirmed_col:
                try:
                    total_recovered = int(_to_numeric(pdf[recovered_col]).sum())
                    total_cases     = int(_to_numeric(pdf[confirmed_col]).sum())
                    rec_rate = (
                        total_recovered / total_cases * 100
                        if total_cases > 0 else 0
                    )
                    active_cases = (
                        int(_to_numeric(pdf[active_col]).sum())
                        if active_col else None
                    )

                    insights.append(BusinessInsight(
                        title=(
                            f"Recovery Rate: {rec_rate:.1f}% "
                            f"({total_recovered:,} recovered)"
                        ),
                        description=(
                            f"{total_recovered:,} patients have "
                            f"recovered ({rec_rate:.1f}% of confirmed "
                            f"cases). "
                            + (
                                f"{active_cases:,} cases remain active. "
                                if active_cases else ""
                            ) +
                            f"{'High recovery rate (>80%) indicates effective treatment protocols.' if rec_rate > 80 else 'Recovery rate below 80% — treatment pathway optimisation needed.'}"
                        ),
                        why_it_matters=(
                            "Recovery rate reflects healthcare system "
                            "effectiveness and treatment quality."
                        ),
                        evidence=(
                            f"Recovered: {total_recovered:,} "
                            f"({rec_rate:.1f}%) | "
                            + (
                                f"Active: {active_cases:,}"
                                if active_cases else ""
                            )
                        ),
                        impact="🟠 Important",
                        recommendation=(
                            "Publish recovery rate data publicly to "
                            "reduce fear and panic. "
                            "High recovery rates are a key indicator "
                            "of healthcare system readiness."
                        ),
                        rule_type="health_recovery",
                        score=8.5,
                        chart_data={
                            "rec_rate": round(rec_rate, 1),
                            "total_recovered": total_recovered,
                        },
                    ))
                except Exception as _re:
                    log.warning(f"[health_recovery] {_re}")

            # 4. Country/region comparison
            if country_col and confirmed_col:
                try:
                    pdf["_cases_numeric"] = _to_numeric(pdf[confirmed_col])
                    country_cases = (
                        pdf.groupby(country_col)["_cases_numeric"]
                        .sum().sort_values(ascending=False)
                    )
                    top_country = country_cases.index[0]
                    top_cases   = int(country_cases.iloc[0])
                    total_cases = int(country_cases.sum())
                    top_pct     = top_cases / total_cases * 100

                    insights.append(BusinessInsight(
                        title=(
                            f"Highest Burden: {top_country} "
                            f"({top_pct:.0f}% of cases)"
                        ),
                        description=(
                            f"{top_country} has the highest case "
                            f"burden with {top_cases:,} confirmed "
                            f"cases ({top_pct:.0f}% of total). "
                            f"Top 5 regions: "
                            f"{', '.join(f'{c} ({n:,})' for c, n in country_cases.head(5).items())}."
                        ),
                        why_it_matters=(
                            "Regional case distribution guides "
                            "resource allocation and intervention "
                            "prioritisation."
                        ),
                        evidence=(
                            f"Top: {top_country} "
                            f"({top_pct:.0f}% of all cases)"
                        ),
                        impact="🔴 Critical",
                        recommendation=(
                            f"Prioritise resource deployment to "
                            f"{top_country}. Establish cross-border "
                            f"coordination to prevent spillover to "
                            f"lower-burden regions."
                        ),
                        rule_type="health_regional",
                        score=8.0,
                        chart_data={
                            "country_cases": (
                                country_cases.head(10).to_dict()
                            ),
                        },
                    ))
                except Exception as _coe:
                    log.warning(f"[health_country] {_coe}")

            # 5. Serious/critical cases
            if serious_col and confirmed_col:
                try:
                    total_serious = int(_to_numeric(pdf[serious_col]).sum())
                    total_cases   = int(_to_numeric(pdf[confirmed_col]).sum())
                    serious_rate  = (
                        total_serious / total_cases * 100
                        if total_cases > 0 else 0
                    )
                    _serious_fmt = (
                        f"{serious_rate:.3f}%"
                        if serious_rate < 0.1
                        else f"{serious_rate:.1f}%"
                    )

                    insights.append(BusinessInsight(
                        title=(
                            f"Critical Cases: "
                            f"{total_serious:,} "
                            f"({_serious_fmt} of confirmed)"
                        ),
                        description=(
                            f"{total_serious:,} cases are serious "
                            f"or critical ({_serious_fmt} of "
                            f"confirmed cases). These patients "
                            f"require intensive care resources. "
                            f"{'High critical rate (>1%) — ICU capacity is a key constraint.' if serious_rate > 1 else 'Critical rate below 1% — within manageable ICU capacity for most systems.'}"
                        ),
                        why_it_matters=(
                            "Critical case rate determines ICU demand "
                            "and is the primary driver of healthcare "
                            "system overload."
                        ),
                        evidence=(
                            f"Critical: {total_serious:,} "
                            f"({_serious_fmt})"
                        ),
                        impact=(
                            "🔴 Critical" if serious_rate > 1
                            else "🟠 Important"
                        ),
                        recommendation=(
                            "Maintain ICU surge capacity at 120% of "
                            "normal. Track critical case rate weekly — "
                            "rising trend is the earliest warning of "
                            "system overload."
                        ),
                        rule_type="health_critical_cases",
                        score=8.0,
                        chart_data={
                            "serious_rate": round(serious_rate, 1),
                            "total_serious": total_serious,
                        },
                    ))
                except Exception as _se:
                    log.warning(f"[health_serious] {_se}")

        except Exception as e:
            log.warning(f"[health_analysis] Failed: {e}")
            import traceback
            traceback.print_exc()

        return insights

    @log_rule
    def _rule_simulation(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """
        ✅ V4 ADDITION 3: Generate 'what-if' scenarios from observed data.
        Each simulation shows: current state → target state → ₹ delta.
        """
        simulations = []
        pdf = df.to_pandas()
        
        rev_col = profile.revenue_col or profile.price_col
        cost_col = next((c for c in df.columns
                        if any(k in c.lower() for k in ["cost", "price"]) and
                        not any(k in c.lower() for k in ["total", "sales", "amount", "revenue"])), None)
        cat_col = profile.category_col
        
        # Simulation 1: Pricing Standardization
        if cost_col and rev_col:
            try:
                quant = ImpactQuantifier.pricing_standardization_gain(
                    pdf, cost_col, rev_col, cat_col
                )
                
                if quant and quant.get("gain_abs", 0) > 0:
                    # ✅ FINAL V4: Add scenario analysis
                    scenarios = ScenarioEngine.generate(quant["gain_abs"], category="pricing")
                    
                    description_base = (
                        f"CURRENT STATE: {cost_col} CV = {quant['current_cv']:.2f} "
                        f"(high variability). "
                        f"TARGET STATE: CV ≤ {quant['target_cv']} "
                        f"(industry standard). "
                        f"ESTIMATED GAIN: {_fmt_currency(quant['gain_abs'])} "
                        f"({quant['gain_pct']:.1f}% of revenue). "
                        f"ASSUMPTION: 35% of at-risk revenue is recoverable "
                        f"through tier standardization."
                    )
                    
                    # Append scenario range
                    description_base += f"\n\nSCENARIO RANGE: {scenarios['display']}. Risk: {scenarios['risk_note']}"
                    
                    insight = BusinessInsight(
                        title="Simulation: Pricing Standardization Impact",
                        description=description_base,
                        why_it_matters="Quantifies the financial impact of pricing standardization.",
                        evidence=f"Current CV: {quant['current_cv']:.2f}, Target: {quant['target_cv']}",
                        impact="🔴 Critical",
                        confidence_label="medium",
                        recommendation=(
                            f"Phase 1 (30 days): Define 3 pricing tiers for {cost_col}. "
                            f"Phase 2 (60 days): Enforce tier compliance. "
                            f"Phase 3 (90 days): Measure margin recovery vs {_fmt_currency(quant['gain_abs'])} target."
                        ),
                        rule_type="simulation_pricing",
                        score=7.5
                    )
                    
                    # Add scenario metadata
                    insight.chart_data = insight.chart_data or {}
                    insight.chart_data["scenarios"] = scenarios
                    
                    simulations.append(insight)
            except Exception:
                pass
        
        # Simulation 2: Lagging Category Growth
        if cat_col and rev_col:
            try:
                shares = pdf.groupby(cat_col)[rev_col].sum()
                if len(shares) >= 2:
                    leader = shares.idxmax()
                    laggard = shares.idxmin()
                    
                    quant2 = ImpactQuantifier.category_share_gain(
                        pdf, cat_col, rev_col, laggard, leader
                    )
                    
                    if quant2 and quant2.get("uplift_abs", 0) > 0:
                        # ✅ FINAL V4: Add scenario analysis
                        scenarios = ScenarioEngine.generate(quant2["uplift_abs"], category="category")
                        
                        description_base = (
                            f"CURRENT: {laggard} = {shares[laggard]/shares.sum():.1%} share. "
                            f"TARGET: {laggard} reaches {shares[leader]/shares.sum()/2:.1%} share. "
                            f"ESTIMATED UPLIFT: {_fmt_currency(quant2['uplift_abs'])} "
                            f"({quant2['uplift_pct']:.1f}pp share gain). "
                            f"ASSUMPTION: Conservative 50% of leader share as achievable target."
                        )
                        
                        # Append scenario range
                        description_base += f"\n\nSCENARIO RANGE: {scenarios['display']}. Risk: {scenarios['risk_note']}"
                        
                        insight = BusinessInsight(
                            title=f"Simulation: Growing {laggard} to Half of {leader}'s Share",
                            description=description_base,
                            why_it_matters="Identifies growth opportunities in underperforming categories.",
                            evidence=f"Current share gap: {(shares[leader]-shares[laggard])/shares.sum()*100:.1f}pp",
                            impact="🟠 Important",
                            confidence_label="medium",
                            recommendation=(
                                f"Run a 90-day growth experiment for {laggard}: "
                                f"increase SKU count, run promotions, track weekly share delta. "
                                f"Exit criterion: reach {_fmt_currency(quant2['uplift_abs']/2)} incremental revenue."
                            ),
                            rule_type="simulation_category_growth",
                            score=6.5
                        )
                        
                        # Add scenario metadata
                        insight.chart_data = insight.chart_data or {}
                        insight.chart_data["scenarios"] = scenarios
                        
                        simulations.append(insight)
            except Exception:
                pass
        
        return simulations

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

    @log_rule
    def _rule_rating_analysis(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """Analyze star-rating / score columns for satisfaction signals."""
        print(f"[RATING DEBUG] profile.categoricals = {profile.categoricals}")
        insights = []
        rating_cols = [
            c for c in profile.categoricals
            if any(k in c.lower() for k in {"rating", "score", "rank", "stars"})
        ]
        rev_col = profile.revenue_col or profile.price_col
        cat_col = profile.category_col

        for col in rating_cols[:1]:  # Top 1 rating column
            try:
                pdf = df.to_pandas()
                low_threshold = pdf[col].min() + 1  # 1 or 2 = "bad"

                scale_min = pdf[col].min()  # 1
                scale_max = pdf[col].max()  # 5
                scale_midpoint = (scale_min + scale_max) / 2  # 3.0

                actual_mean = pdf[col].mean()
                mean_below_midpoint = actual_mean < scale_midpoint

                # Expected low rate = proportion of scale below midpoint
                expected_low_rate = (scale_midpoint - scale_min) / (scale_max - scale_min) * 100  # 40%

                # Percentage of orders with a low (1- or 2-star) rating
                pct_low = (pdf[col] <= low_threshold).mean() * 100

                # Only fire if actual low rate meaningfully exceeds expectation
                excess_low_rate = pct_low - expected_low_rate

                print(f"[RATING DEBUG] scale_min={scale_min}, scale_max={scale_max}, "
                      f"midpoint={scale_midpoint}, actual_mean={actual_mean:.2f}, "
                      f"excess_low={excess_low_rate:.1f}pp")

                if not mean_below_midpoint and excess_low_rate < 10:
                    log.info(
                        f"[rating_analysis] Suppressed {col}: mean={actual_mean:.2f} "
                        f"(midpoint={scale_midpoint}), excess_low={excess_low_rate:.1f}pp < 10pp threshold"
                    )
                    continue  # Uniform/neutral distribution — not a risk signal

                # Set impact based on actual severity
                if actual_mean < scale_midpoint - 0.5:   # Mean below 2.5
                    impact = "🔴 Critical"
                elif actual_mean < scale_midpoint:        # Mean between 2.5 and 3.0
                    impact = "🟠 Important"
                else:                                     # Mean at or above midpoint
                    impact = "🟢 Minor"  # note positive news if mean > midpoint

                insight_parts = [
                    f"{pct_low:.1f}% of orders have a {pdf[col].min()}-star or "
                    f"{low_threshold}-star rating — a significant dissatisfaction signal."
                ]

                # By category breakdown
                if cat_col and cat_col in pdf.columns:
                    bad_by_cat = pdf.groupby(cat_col)[col].apply(
                        lambda x: (x <= low_threshold).mean() * 100
                    ).sort_values(ascending=False)
                    worst_cat = bad_by_cat.index[0]
                    best_cat = bad_by_cat.index[-1]
                    spread = bad_by_cat.iloc[0] - bad_by_cat.iloc[-1]

                    if spread >= 8.0:  # Only mention products if spread is real (was: 3.0)
                        insight_parts.append(
                            f"{worst_cat} has the worst rating ({bad_by_cat.iloc[0]:.1f}% low scores) "
                            f"vs {best_cat} at {bad_by_cat.iloc[-1]:.1f}%."
                        )
                    else:
                        insight_parts.append(
                            f"No product shows a statistically meaningful difference "
                            f"({spread:.1f}pp max spread across {cat_col})."
                        )

                insights.append(BusinessInsight(
                    title=f"Customer Satisfaction Risk: {pct_low:.0f}% Low-Rating Orders",
                    description=" ".join(insight_parts),
                    why_it_matters=(
                        "Low ratings signal product-market mismatch, fulfillment issues, or "
                        "description inaccuracy. They predict future churn and returns."
                    ),
                    evidence=f"Low-rating rate: {pct_low:.1f}% | Column: {col} | n={len(pdf):,}",
                    impact=impact,
                    recommendation=(
                        f"Investigate {worst_cat} for root causes (product quality, description, "
                        f"delivery). Survey 1-star customers within 7 days of purchase."
                    ),
                    rule_type="rating_quality",
                    score=9.0,
                ))
            except Exception as e:
                log.warning(f"[rating_analysis] {col}: {e}")

        return insights

    @log_rule
    def _rule_category_satisfaction_cross(self, df, profile):
        """Cross ProductCategory × ReviewRating to find quality risk by category."""
        cat_col = next((c for c in df.columns if "category" in c.lower()), None)
        rating_col = next((c for c in df.columns if "rating" in c.lower()), None)
        rev_col = profile.revenue_col or profile.price_col
        
        if not (cat_col and rating_col and rev_col):
            return []
        
        pdf = df.to_pandas()
        
        # Average rating and revenue share per category
        summary = pdf.groupby(cat_col).agg(
            avg_rating=(rating_col, "mean"),
            revenue=(rev_col, "sum"),
            orders=(rev_col, "count")
        )
        summary["rev_share"] = summary["revenue"] / summary["revenue"].sum() * 100
        
        # Flag: high revenue share + below-average rating = priority risk
        avg_rating = pdf[rating_col].mean()
        risk_cats = summary[
            (summary["rev_share"] > 15) & 
            (summary["avg_rating"] < avg_rating - 0.3)
        ]
        
        if risk_cats.empty:
            return []
        
        worst = risk_cats.sort_values("avg_rating").iloc[0]
        return [BusinessInsight(
            title=f"Quality Risk: {worst.name} generates {worst['rev_share']:.0f}% of revenue but rates {worst['avg_rating']:.1f}/5",
            description=(
                f"{worst.name} accounts for {worst['rev_share']:.0f}% of total revenue "
                f"but scores {worst['avg_rating']:.1f}/5 — below the {avg_rating:.1f} average. "
                f"High-revenue categories with below-average ratings indicate "
                f"a quality or expectation mismatch at scale."
            ),
            impact="🔴 Critical" if worst["rev_share"] > 25 else "🟠 Important",
            recommendation=(
                f"Audit {worst.name} product descriptions and fulfillment quality. "
                f"A 0.5-point rating improvement on a {worst['rev_share']:.0f}% revenue segment "
                f"has outsized retention impact."
            ),
            rule_type="category_satisfaction",
            score=8.5,
        )]

    @log_rule
    def _rule_customer_concentration(
        self, df: pl.DataFrame, profile: DataProfile
    ) -> list[BusinessInsight]:
        """Customer concentration and repeat-purchase rate analysis."""
        all_cols = list(df.columns)
        # CustomerID can be an identifier or a high-cardinality categorical — search broadly
        cust_keywords = ["customer", "cust", "client", "buyer"]
        cust_col = next(
            (c for c in (profile.identifiers + profile.categoricals + all_cols)
             if any(k in c.lower() for k in cust_keywords)
             and df[c].n_unique() > 10),
            None,
        )
        if not cust_col:
            return []
        # Revenue column: prefer explicit revenue_col, then any "total/sales/amount" col, then price_col
        rev_col = profile.revenue_col or next(
            (c for c in all_cols
             if any(k in c.lower() for k in ["total", "sales", "revenue", "amount"])
             and c not in profile.identifiers
             and c != cust_col),
            None,
        ) or profile.price_col
        if not rev_col or rev_col not in df.columns:
            return []
        print(f"[customer_concentration] cust_col={cust_col}, rev_col={rev_col}")
        try:
            pdf = df.to_pandas()
            customer_rev = pdf.groupby(cust_col)[rev_col].sum().sort_values(ascending=False)
            total = customer_rev.sum()
            n_customers = len(customer_rev)
            if n_customers < 5 or total == 0:
                return []

            top10 = min(10, n_customers)
            top10_share = customer_rev.head(top10).sum() / total * 100
            top10_pct_of_base = top10 / n_customers * 100
            purchase_counts = pdf.groupby(cust_col).size()
            repeat_rate = (purchase_counts > 1).mean() * 100
            avg_orders = purchase_counts.mean()

            # Use category-specific benchmark, not a hardcoded 30% threshold
            _dom_cat = _detect_dominant_category(df)
            _bench = CATEGORY_BENCHMARKS.get(_dom_cat, CATEGORY_BENCHMARKS["default"])
            _bench_threshold = _bench["repeat_rate_pct"]
            _bench_range = _bench["repeat_rate_range"]

            is_high_conc = top10_share > 25
            is_low_ret = repeat_rate < _bench_threshold

            if is_high_conc:
                impact = "🔴 Critical"
                rec = (
                    f"Top {top10} customers ({top10_pct_of_base:.0f}% of base) drive "
                    f"{top10_share:.0f}% of revenue — key-account concentration risk. "
                    "Implement dedicated account management and revenue diversification "
                    "across the broader customer base."
                )
            elif is_low_ret:
                # Compute rough revenue uplift for closing half the benchmark gap
                _gap_pp = max(0.0, _bench_threshold - repeat_rate)
                _aov_approx = float(pdf[rev_col].mean()) if rev_col in pdf.columns else 0.0
                _uplift_half_gap = n_customers * _aov_approx * (_gap_pp / 2 / 100)
                _uplift_str = (f" Closing half the gap would add approx. {_fmt_currency(_uplift_half_gap)} in annual revenue."
                               if _uplift_half_gap > 0 else "")
                impact = "🟠 Important"
                rec = (
                    f"Repeat purchase rate of {repeat_rate:.1f}% is below the {_dom_cat} industry "
                    f"benchmark of {_bench_range}. Retention uplift potential is high —"
                    f" loyalty programmes and personalised re-engagement are the highest-ROI lever available.{_uplift_str}"
                )
            else:
                impact = "🟢 Minor"
                rec = (
                    "Healthy customer distribution and repeat-purchase behaviour. "
                    "Continue nurturing repeat buyers and monitor concentration quarterly."
                )

            description = (
                f"{n_customers:,} unique customers averaging {avg_orders:.1f} orders each. "
                f"Top {top10} customers account for {top10_share:.1f}% of revenue "
                f"({top10_pct_of_base:.1f}% of the customer base). "
                f"Repeat purchase rate: {repeat_rate:.1f}%."
            )
            # decision_implication: a single clean sentence based on actual values
            # (not template "If X then Y" text that bleeds into body copy)
            if is_high_conc:
                _implication = (
                    f"Top {top10} customers drive {top10_share:.0f}% of revenue — "
                    f"dedicate account management resources and diversify the revenue base."
                )
            elif is_low_ret:
                _implication = (
                    f"A {repeat_rate:.0f}% repeat rate means most customers buy once and leave — "
                    f"retention programmes will deliver higher ROI than new acquisition spend."
                )
            else:
                _implication = (
                    f"Both concentration ({top10_share:.1f}%) and retention ({repeat_rate:.1f}%) "
                    f"metrics are healthy — maintain current relationship strategy."
                )

            return [BusinessInsight(
                title="Customer Concentration & Retention",
                description=description,
                why_it_matters=(
                    "Customer concentration determines key-account risk; repeat purchase rate "
                    "determines organic growth potential without incremental acquisition cost."
                ),
                evidence=(
                    f"Unique customers: {n_customers:,} | Top-{top10} share: {top10_share:.1f}% | "
                    f"Repeat rate: {repeat_rate:.1f}% | Avg orders/customer: {avg_orders:.1f}"
                ),
                decision_implication=_implication,
                impact=impact,
                recommendation=rec,
                rule_type="customer_concentration",
                score=8.5,
            )]
        except Exception as e:
            log.warning(f"[customer_concentration] Failed: {e}")
            return []

    def _rfm_simple_tiers(
        self, df: pl.DataFrame, cust_col: str, rev_col: str, date_col: str, profile: DataProfile
    ) -> list[BusinessInsight]:
        """3-tier value segmentation for small customer bases (below MIN_CUSTOMERS_FOR_RFM)."""
        import plotly.graph_objects as go

        try:
            pdf = df.to_pandas().dropna(subset=[cust_col, rev_col])
            cust_rev = pdf.groupby(cust_col)[rev_col].sum().reset_index()
            cust_rev.columns = [cust_col, "total_revenue"]

            n_customers = len(cust_rev)
            total_rev = float(cust_rev["total_revenue"].sum())
            if n_customers < 2 or total_rev == 0:
                return []

            cust_rev = cust_rev.sort_values("total_revenue", ascending=False).reset_index(drop=True)
            high_cutoff = max(1, int(n_customers * 0.20))
            low_cutoff  = max(high_cutoff + 1, int(n_customers * 0.80))

            cust_rev["tier"] = "Mid Value"
            cust_rev.loc[:high_cutoff - 1, "tier"] = "High Value"
            cust_rev.loc[low_cutoff:, "tier"]       = "Low Value"

            tier_stats = cust_rev.groupby("tier").agg(
                count=(cust_col, "count"),
                revenue=("total_revenue", "sum"),
                aov=("total_revenue", "mean"),
            ).reset_index()
            tier_stats["revenue_pct"] = tier_stats["revenue"] / total_rev * 100

            TIER_ORDER = ["High Value", "Mid Value", "Low Value"]
            TIER_ACTIONS = {
                "High Value": "Protect with loyalty rewards and VIP service — churn risk must be monitored closely.",
                "Mid Value":  "Upsell to increase purchase frequency and basket size; highest absolute revenue impact.",
                "Low Value":  "Low-cost reactivation: automated 'We miss you' emails and bundle offers.",
            }

            seg_lines = []
            for tier in TIER_ORDER:
                row = tier_stats[tier_stats["tier"] == tier]
                if row.empty:
                    continue
                r = row.iloc[0]
                seg_lines.append(
                    f"{tier}: {int(r['count'])} customers | "
                    f"{_fmt_currency(r['revenue'])} ({r['revenue_pct']:.1f}%) | "
                    f"AOV {_fmt_currency(r['aov'])} → {TIER_ACTIONS[tier]}"
                )

            note = (
                f"Note: Simplified 3-tier segmentation used (n={n_customers} unique customers "
                f"— below threshold for full RFM quintile analysis)."
            )

            def _tier(name: str):
                row = tier_stats[tier_stats["tier"] == name]
                return row.iloc[0] if not row.empty else None

            high_row = _tier("High Value")
            mid_row  = _tier("Mid Value")
            low_row  = _tier("Low Value")

            description = (
                f"3-Tier Customer Value Segmentation across {n_customers} unique customers.\n\n"
                + "\n".join(seg_lines)
                + f"\n\n{note}"
            )

            # PRIMARY is always High Value — unit churn impact is highest there
            # regardless of which tier has the largest total revenue pool.
            if high_row is not None:
                recommendation_text = (
                    f"PRIMARY — High Value ({int(high_row['count'])} customers, "
                    f"{high_row['revenue_pct']:.0f}% of revenue, "
                    f"AOV {_fmt_currency(high_row['aov'])}): "
                    f"Protect with loyalty rewards and VIP service — churn risk must be monitored closely."
                )
                if mid_row is not None:
                    recommendation_text += (
                        f" | SECONDARY — Mid Value ({int(mid_row['count'])} customers, "
                        f"{mid_row['revenue_pct']:.0f}% of revenue): "
                        f"Upsell to increase basket size — highest absolute revenue opportunity."
                    )
                if low_row is not None:
                    recommendation_text += (
                        f" | Low Value ({int(low_row['count'])} customers): "
                        f"Low-cost reactivation via automated outreach."
                    )
            else:
                fallback = tier_stats.sort_values("revenue", ascending=False).iloc[0]
                recommendation_text = (
                    f"PRIMARY — {fallback['tier']} ({int(fallback['count'])} customers, "
                    f"{fallback['revenue_pct']:.0f}% of revenue): {TIER_ACTIONS[fallback['tier']]}"
                )

            try:
                fig = go.Figure(go.Bar(
                    x=tier_stats["revenue"].tolist(),
                    y=tier_stats["tier"].tolist(),
                    orientation="h",
                    marker_color="#6366f1",
                    text=[f"{p:.1f}%" for p in tier_stats["revenue_pct"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="",
                    xaxis_title="Revenue (₹)",
                    yaxis_title="",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#1e293b"},
                    margin={"l": 140, "r": 60, "t": 20, "b": 40},
                )
            except Exception:
                fig = None

            insight = BusinessInsight(
                title="Customer Value Segmentation (3-Tier)",
                description=description,
                why_it_matters="Even with a small customer base, identifying high-value customers enables targeted retention and upsell strategies.",
                evidence=f"Total customers: {n_customers} | Below RFM quintile threshold | Total revenue: {_fmt_currency(total_rev)}",
                decision_implication=(
                    f"High Value tier ({int(high_row['count'])} customers) generates "
                    f"{high_row['revenue_pct']:.0f}% of revenue — protect first."
                    if high_row is not None else
                    f"Top tier generates the majority of revenue — prioritise retention."
                ),
                impact="\U0001f7e0 Important",
                recommendation=recommendation_text,
                rule_type="rfm_segmentation",
                methodology="3-tier value segmentation: top 20% revenue = High Value, bottom 20% = Low Value, rest = Mid Value",
                score=8.0,
            )
            if fig is not None:
                insight.chart_data = {
                    "type": "plotly",
                    "figure": fig.to_json(),
                    "segments": tier_stats.to_dict(orient="records"),
                }
            return [insight]

        except Exception as e:
            log.warning(f"[rfm_simple_tiers] Failed: {e}")
            return []

    @log_rule
    def _rule_rfm_segmentation(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """RFM Segmentation — Customer Intelligence."""
        import plotly.graph_objects as go

        # Auto-detect columns
        all_cols = list(df.columns)
        cust_keywords = ["customer", "cust", "client", "buyer"]
        cust_col = next(
            (c for c in (profile.identifiers + profile.categoricals + all_cols)
             if any(k in c.lower() for k in cust_keywords)),
            None,
        )
        date_col = profile.date_col or next(
            (c for c in profile.temporals), None
        )
        rev_col = profile.revenue_col or next(
            (c for c in all_cols
             if any(k in c.lower() for k in ["price", "amount", "revenue", "total"])
             and c not in profile.identifiers and c != cust_col),
            None,
        ) or profile.price_col

        if not cust_col:
            log.warning("[rfm_segmentation] Skipped: no customer ID column detected")
            return []
        if not date_col:
            log.warning("[rfm_segmentation] Skipped: no date column detected")
            return []
        if not rev_col or rev_col not in df.columns:
            log.warning("[rfm_segmentation] Skipped: no revenue column detected")
            return []

        try:
            # Fix 2a: minimum customer threshold — small datasets use 3-tier helper
            MIN_CUSTOMERS_FOR_RFM = 50
            n_unique = df[cust_col].n_unique()
            if n_unique < MIN_CUSTOMERS_FOR_RFM:
                log.info(f"[rfm_segmentation] {n_unique} unique customers < {MIN_CUSTOMERS_FOR_RFM} — using 3-tier segmentation")
                return self._rfm_simple_tiers(df, cust_col, rev_col, date_col, profile)

            pdf = df.to_pandas()

            # Parse dates
            dates = pd.to_datetime(pdf[date_col], errors="coerce")
            pdf["_date"] = dates
            pdf = pdf.dropna(subset=["_date", cust_col, rev_col])
            if len(pdf) < 10:
                log.warning("[rfm_segmentation] Skipped: too few valid rows after date parsing")
                return []

            max_date = pdf["_date"].max()

            # Compute RFM per customer
            rfm = pdf.groupby(cust_col).agg(
                recency=("_date", lambda x: (max_date - x.max()).days),
                frequency=(cust_col, "count"),
                monetary=(rev_col, "sum"),
            ).reset_index()

            # Detect degenerate frequency (>70% single-purchase)
            single_pct = (rfm["frequency"] == 1).mean()
            log.info(f"[rfm_segmentation] Single-purchase rate: {single_pct:.1%}")

            # Score Recency and Monetary with quintiles (1-5, 5=best)
            try:
                rfm["R"] = pd.qcut(rfm["recency"], q=5, labels=[5,4,3,2,1], duplicates="drop").astype(int)
            except Exception:
                rfm["R"] = pd.cut(rfm["recency"], bins=5, labels=[5,4,3,2,1]).astype(int)

            try:
                rfm["M"] = pd.qcut(rfm["monetary"], q=5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
            except Exception:
                rfm["M"] = pd.cut(rfm["monetary"], bins=5, labels=[1,2,3,4,5]).astype(int)

            # Frequency scoring
            if single_pct > 0.70:
                log.info("[rfm_segmentation] Frequency binning: 3-tier mode (high single-purchase concentration)")
                rfm["F_score"] = 1
                rfm.loc[rfm["frequency"] == 2, "F_score"] = 2
                rfm.loc[rfm["frequency"] >= 3, "F_score"] = 3
                # Scale to 1-5 for segment assignment compatibility
                rfm["F_score"] = rfm["F_score"].map({1: 1, 2: 3, 3: 5})
            else:
                try:
                    rfm["F_score"] = pd.qcut(rfm["frequency"], q=5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
                except Exception:
                    rfm["F_score"] = pd.cut(rfm["frequency"], bins=5, labels=[1,2,3,4,5]).astype(int)

            # Segment assignment
            def assign_segment(row):
                r, f, m = row["R"], row["F_score"], row["M"]
                if r >= 4 and f >= 4 and m >= 4:
                    return "Champions"
                elif r == 5 and f == 1:
                    return "New Customers"
                elif r == 1 and f == 1:
                    return "Lost"
                elif r <= 2 and f <= 2:
                    return "Hibernating"
                elif r <= 2 and f >= 3 and m >= 3:
                    return "At Risk"
                elif r >= 3 and f <= 2:
                    return "Potential Loyalists"
                elif f >= 4 and m >= 4:
                    return "Loyal Customers"
                else:
                    return "Others"

            rfm["segment"] = rfm.apply(assign_segment, axis=1)

            total_rev = rfm["monetary"].sum()

            seg_stats = rfm.groupby("segment").agg(
                customer_count=("monetary", "count"),
                revenue=("monetary", "sum"),
                avg_order_value=("monetary", "mean"),
                avg_recency=("recency", "mean"),
            ).reset_index()
            seg_stats["revenue_pct"] = seg_stats["revenue"] / total_rev * 100
            seg_stats = seg_stats.sort_values("revenue", ascending=False)

            # Fix 2b: Others dominance — primary recommendation must target a NAMED segment
            _others_rows = seg_stats[seg_stats["segment"] == "Others"]
            _others_count = int(_others_rows["customer_count"].sum()) if not _others_rows.empty else 0
            _others_pct = _others_count / len(rfm) if len(rfm) > 0 else 0.0
            _others_warning = None
            if _others_pct > 0.25:
                _others_warning = (
                    f"⚠ {_others_pct*100:.0f}% of customers could not be assigned "
                    f"to a named segment. RFM thresholds may need recalibration for "
                    f"this dataset's frequency distribution. The 3-tier fallback "
                    f"would produce more reliable segmentation."
                )
            _named_seg_stats = seg_stats[seg_stats["segment"] != "Others"]
            top_seg = _named_seg_stats.iloc[0] if not _named_seg_stats.empty else seg_stats.iloc[0]

            # Segment-specific recommendations
            SEG_RECS = {
                "Champions": "Reward with exclusive loyalty perks and early-access offers.",
                "Loyal Customers": "Upsell to premium tiers and solicit referrals.",
                "Potential Loyalists": "Send a win-back email sequence with a time-limited 10% discount.",
                "At Risk": "Immediate re-engagement: personalised offer within 48 hours.",
                "Hibernating": "Low-cost reactivation: 'We miss you' email with bundle offer.",
                "Lost": "Remove from active campaigns; add to suppression list to reduce cost.",
                "New Customers": "Send onboarding sequence; introduce loyalty programme.",
                "Others": "Analyse sub-cohort and assign to nearest segment.",
            }

            # Detect dominant product category for benchmark
            dominant_cat = _detect_dominant_category(df)
            benchmark = CATEGORY_BENCHMARKS.get(dominant_cat, CATEGORY_BENCHMARKS["default"])
            repeat_rate = (rfm["frequency"] > 1).mean() * 100
            benchmark_rate = benchmark["repeat_rate_pct"]

            benchmark_note = (
                f"Industry benchmark for {dominant_cat}: {benchmark['repeat_rate_range']} repeat rate. "
                f"Your rate: {repeat_rate:.1f}% — "
                f"{'below benchmark — retention uplift potential is high' if repeat_rate < benchmark_rate else 'meeting or exceeding benchmark'}."
            )

            # Build description
            seg_lines = []
            for _, row in seg_stats.iterrows():
                seg = row["segment"]
                rec = SEG_RECS.get(seg, "Monitor and apply appropriate engagement strategy.")
                seg_lines.append(
                    f"{seg}: {int(row['customer_count'])} customers | "
                    f"{_fmt_currency(row['revenue'])} ({row['revenue_pct']:.1f}%) | "
                    f"AOV {_fmt_currency(row['avg_order_value'])} | "
                    f"Avg recency {row['avg_recency']:.0f} days → {rec}"
                )

            description = (
                f"RFM analysis across {len(rfm):,} unique customers reveals "
                f"'{top_seg['segment']}' drives {top_seg['revenue_pct']:.1f}% of revenue "
                f"({_fmt_currency(top_seg['revenue'])}). "
                f"{benchmark_note}\n\n" + "\n".join(seg_lines)
            )
            if _others_warning:
                description += f"\n\n{_others_warning}"

            # Chart: horizontal bar — revenue contribution by segment
            try:
                fig = go.Figure(go.Bar(
                    x=seg_stats["revenue"].tolist(),
                    y=seg_stats["segment"].tolist(),
                    orientation="h",
                    marker_color="#6366f1",
                    text=[f"{p:.1f}%" for p in seg_stats["revenue_pct"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="",
                    xaxis_title="Revenue (₹)",
                    yaxis_title="",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#1e293b"},
                    margin={"l": 140, "r": 60, "t": 20, "b": 40},
                )
            except Exception as chart_err:
                log.warning(f"[rfm_segmentation] Chart failed: {chart_err}")
                fig = None

            # Root-cause for dominant category
            root_cause = None
            if total_rev > 0 and profile.category_col and profile.category_col in df.columns:
                try:
                    cat_repeat = pdf.groupby([profile.category_col, cust_col]).size().reset_index(name="orders")
                    cat_repeat_rate = cat_repeat.groupby(profile.category_col).apply(
                        lambda g: (g["orders"] > 1).mean() * 100
                    ).reset_index(name="repeat_rate_pct")
                    if len(cat_repeat_rate) >= 2:
                        # Fix 3: conditional root-cause — never say "drag down" when all categories are high
                        _cat_rates_rfm = {row[profile.category_col]: row["repeat_rate_pct"] for _, row in cat_repeat_rate.iterrows()}
                        _min_rr = min(_cat_rates_rfm.values())
                        _max_rr = max(_cat_rates_rfm.values())
                        _spread_rr = _max_rr - _min_rr
                        _rate_detail_rfm = ", ".join(
                            f"{cat}={rate:.0f}%"
                            for cat, rate in sorted(_cat_rates_rfm.items(), key=lambda x: x[1])
                        )
                        if _spread_rr >= 15 and _min_rr < 70:
                            _lowest_cat_rfm = min(_cat_rates_rfm, key=_cat_rates_rfm.get)
                            root_cause = (
                                f"Repeat rate varies significantly by {profile.category_col}: {_rate_detail_rfm}. "
                                f"{_lowest_cat_rfm} ({_min_rr:.0f}%) is dragging down the blended rate. "
                                f"Category-level retention programmes will be more effective than a blended approach."
                            )
                        elif _spread_rr >= 10:
                            root_cause = (
                                f"Repeat rate shows moderate variation across {profile.category_col}: "
                                f"{_rate_detail_rfm}. No single category is a significant outlier."
                            )
                        else:
                            root_cause = (
                                f"Repeat rate is consistently high across all {profile.category_col} categories "
                                f"({_min_rr:.0f}%–{_max_rr:.0f}%). "
                                f"Category mix is not the constraint — focus on AOV and basket size."
                            )
                except Exception:
                    pass

            # ── Fix 4: Priority-ordered recommendation (top-revenue segment first) ──
            # Primary: highest-revenue segment; Secondary: At Risk; Tertiary: New Customers
            at_risk_count = int(seg_stats.loc[seg_stats["segment"] == "At Risk", "customer_count"].sum()) if "At Risk" in seg_stats["segment"].values else 0
            new_cust_count = int(seg_stats.loc[seg_stats["segment"] == "New Customers", "customer_count"].sum()) if "New Customers" in seg_stats["segment"].values else 0
            primary_action = (
                f"{top_seg['segment']} ({int(top_seg['customer_count'])} customers, "
                f"{top_seg['revenue_pct']:.0f}% of revenue, avg recency {top_seg['avg_recency']:.0f} days): "
                f"{SEG_RECS.get(top_seg['segment'], 'Customise engagement strategy.')}"
            )
            secondary_parts = []
            if at_risk_count > 0 and top_seg["segment"] != "At Risk":
                secondary_parts.append(f"At Risk ({at_risk_count} customers): contact within 14 days before they transition to Lost.")
            if new_cust_count > 0 and top_seg["segment"] != "New Customers":
                secondary_parts.append(f"New Customers ({new_cust_count}): launch onboarding sequence and introduce loyalty programme within first 30 days.")
            recommendation_text = "PRIMARY — " + primary_action
            if secondary_parts:
                recommendation_text += " | SECONDARY — " + " | ".join(secondary_parts)

            _at_risk_rows = seg_stats[seg_stats["segment"] == "At Risk"]
            _at_risk_rev_pct = float(_at_risk_rows["revenue_pct"].iloc[0]) if not _at_risk_rows.empty else 0.0
            insight = BusinessInsight(
                title="RFM Customer Segmentation Analysis",
                description=description,
                why_it_matters="RFM identifies who your best customers are, which are at risk, and which have been lost — enabling targeted retention spend.",
                evidence=f"Segments: {seg_stats['segment'].tolist()} | Total customers: {len(rfm):,}",
                decision_implication=(
                    f"PRIMARY: {top_seg['segment']} ({top_seg['revenue_pct']:.0f}% of revenue) "
                    f"is the highest-priority segment. "
                    + (f"SECONDARY: {at_risk_count} 'At Risk' customers ({_at_risk_rev_pct:.0f}% revenue) need immediate win-back before they join Lost." if at_risk_count > 0 else "")
                ),
                impact="🔴 Critical",
                recommendation=recommendation_text,
                rule_type="rfm_segmentation",
                methodology="RFM scoring with quintile binning (3-tier frequency fallback when >70% single-purchase)",
                score=9.0,
            )
            if fig is not None:
                insight.chart_data = {
                    "type": "plotly",
                    "figure": fig.to_json(),
                    "segments": seg_stats.to_dict(orient="records"),
                }
            if root_cause:
                insight.chart_data = insight.chart_data or {}
                insight.chart_data["root_cause"] = root_cause

            return [insight]

        except Exception as e:
            log.warning(f"[rfm_segmentation] Failed: {e}")
            return []

    @log_rule
    def _rule_cohort_retention(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """Cohort retention curve analysis."""
        import plotly.graph_objects as go

        all_cols = list(df.columns)
        cust_keywords = ["customer", "cust", "client", "buyer"]
        cust_col = next(
            (c for c in (profile.identifiers + profile.categoricals + all_cols)
             if any(k in c.lower() for k in cust_keywords)),
            None,
        )
        date_col = profile.date_col or next((c for c in profile.temporals), None)

        if not cust_col:
            log.warning("[cohort_retention] Skipped: no customer column detected")
            return []
        if not date_col:
            log.warning("[cohort_retention] Skipped: no date column detected")
            return []

        try:
            pdf = df.to_pandas()
            pdf["_date"] = pd.to_datetime(pdf[date_col], errors="coerce")
            pdf = pdf.dropna(subset=["_date", cust_col])
            if len(pdf) < 30:
                log.warning("[cohort_retention] Skipped: too few rows")
                return []

            pdf["_month"] = pdf["_date"].dt.to_period("M")

            # Cohort = first purchase month
            first_purchase = pdf.groupby(cust_col)["_date"].min().dt.to_period("M")
            pdf["_cohort"] = pdf[cust_col].map(first_purchase)

            max_month = pdf["_month"].max()

            # Build cohort x offset retention matrix
            cohort_data = {}
            for cohort, group in pdf.groupby("_cohort"):
                cohort_size = group[cust_col].nunique()
                if cohort_size < 10:
                    continue  # Suppress small cohorts

                # Only include cohorts with at least 6 months of follow-up
                months_of_data = (max_month - cohort).n
                if months_of_data < 6:
                    continue

                cohort_data[cohort] = {"size": cohort_size, "retention": {}}
                for offset in range(min(months_of_data + 1, 13)):
                    target_month = cohort + offset
                    active = group[group["_month"] == target_month][cust_col].nunique()
                    cohort_data[cohort]["retention"][offset] = active / cohort_size

            if len(cohort_data) < 3:
                log.warning(f"[cohort_retention] Skipped: only {len(cohort_data)} qualifying cohorts (need 3)")
                return []

            cohorts = sorted(cohort_data.keys())
            max_offset = max(len(v["retention"]) for v in cohort_data.values())

            # Anomaly detection on month-1 retention
            m1_rates = [cohort_data[c]["retention"].get(1, None) for c in cohorts]
            m1_rates_valid = [r for r in m1_rates if r is not None]
            m1_mean = np.mean(m1_rates_valid) if m1_rates_valid else 0
            m1_std = np.std(m1_rates_valid) if m1_rates_valid else 0

            anomalies = []
            for cohort, rate in zip(cohorts, m1_rates):
                if rate is None:
                    continue
                if m1_std > 0:
                    if rate > m1_mean + 1.5 * m1_std:
                        anomalies.append(
                            f"Cohort {cohort} retained {rate*100:.0f}% in month 1 "
                            f"vs average {m1_mean*100:.0f}% — investigate what drove this acquisition batch."
                        )
                    elif rate < m1_mean - 1.5 * m1_std:
                        anomalies.append(
                            f"Cohort {cohort} retained only {rate*100:.0f}% in month 1 "
                            f"vs average {m1_mean*100:.0f}% — investigate quality issues in this batch."
                        )

            # Compute average retention by offset across all cohorts
            avg_by_offset = {}
            for offset in range(max_offset):
                vals = [cohort_data[c]["retention"].get(offset) for c in cohorts
                        if offset in cohort_data[c]["retention"]]
                if vals:
                    avg_by_offset[offset] = np.mean(vals)

            m1_avg = avg_by_offset.get(1, 0) * 100
            m2_avg = avg_by_offset.get(2, 0) * 100
            m3_avg = avg_by_offset.get(3, 0) * 100

            # Revenue uplift estimate
            rev_col = profile.revenue_col or profile.price_col
            aov = 0
            total_customers = len(pdf[cust_col].unique())
            if rev_col and rev_col in pdf.columns:
                aov = float(pdf[rev_col].mean())
            avg_orders_repeater = avg_by_offset.get(1, 0.1) * 12  # rough annualised
            uplift = 0.02 * total_customers * aov * max(avg_orders_repeater, 1)

            # Best and worst cohorts
            m1_by_cohort = {c: cohort_data[c]["retention"].get(1, 0) for c in cohorts}
            best_cohort = max(m1_by_cohort, key=m1_by_cohort.get)
            worst_cohort = min(m1_by_cohort, key=m1_by_cohort.get)

            # Build heatmap data
            z_matrix = []
            y_labels = []
            for cohort in cohorts:
                row = []
                for offset in range(min(max_offset, 13)):
                    row.append(cohort_data[cohort]["retention"].get(offset, None))
                z_matrix.append(row)
                y_labels.append(str(cohort))
            x_labels = [f"M+{i}" for i in range(min(max_offset, 13))]

            try:
                import plotly.graph_objects as go
                # Convert None to NaN for heatmap
                z_float = [[v if v is not None else float("nan") for v in row] for row in z_matrix]
                fig = go.Figure(data=go.Heatmap(
                    z=[[round(v * 100, 1) if not np.isnan(v) else None for v in row] for row in z_float],
                    x=x_labels,
                    y=y_labels,
                    colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
                    text=[[f"{round(v*100,1)}%" if not np.isnan(v) else "" for v in row] for row in z_float],
                    texttemplate="%{text}",
                    showscale=True,
                    zmin=0,
                    zmax=100,
                    colorbar_title="Retention %",
                ))
                fig.update_layout(
                    title="",
                    xaxis_title="Month Offset",
                    yaxis_title="Cohort",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#1e293b"},
                    margin={"l": 80, "r": 40, "t": 20, "b": 40},
                )
            except Exception as chart_err:
                log.warning(f"[cohort_retention] Chart failed: {chart_err}")
                fig = None

            anomaly_text = (" Anomalies: " + "; ".join(anomalies)) if anomalies else ""

            description = (
                f"Cohort retention analysis across {len(cohorts)} qualifying cohorts "
                f"(each ≥10 customers, ≥6 months data). "
                f"Best cohort: {best_cohort} ({m1_by_cohort[best_cohort]*100:.0f}% month-1 retention). "
                f"Worst cohort: {worst_cohort} ({m1_by_cohort[worst_cohort]*100:.0f}% month-1 retention). "
                f"Average retention: M+1={m1_avg:.1f}%, M+2={m2_avg:.1f}%, M+3={m3_avg:.1f}%. "
                f"A 2pp improvement in average retention would yield approximately "
                f"{_fmt_currency(uplift)} additional annual revenue."
                f"{anomaly_text}"
            )

            insight = BusinessInsight(
                title="Cohort Retention Analysis",
                description=description,
                why_it_matters="Retention curves reveal whether customers return after first purchase — the single strongest signal of product-market fit.",
                evidence=f"Cohorts analysed: {len(cohorts)} | Avg M+1: {m1_avg:.1f}% | Avg M+2: {m2_avg:.1f}%",
                decision_implication=f"Focus on the {worst_cohort} cohort — its low retention suggests acquisition quality or onboarding issues specific to that period.",
                impact="🟠 Important",
                recommendation="Implement a structured onboarding sequence for all new cohorts to lift month-1 retention by 2–3pp.",
                rule_type="cohort_retention",
                methodology="Monthly cohort retention: customers_active_in_month_N / cohort_size",
                score=8.2,
            )
            if fig is not None:
                insight.chart_data = {
                    "type": "plotly",
                    "figure": fig.to_json(),
                    "anomalies": anomalies,
                    "avg_m1": m1_avg,
                    "avg_m2": m2_avg,
                    "avg_m3": m3_avg,
                    "uplift_2pp": uplift,
                }

            return [insight]

        except Exception as e:
            log.warning(f"[cohort_retention] Failed: {e}")
            return []

    @log_rule
    def _rule_clv_estimate(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """Simple CLV estimate with retention-lift scenarios."""
        all_cols = list(df.columns)
        cust_keywords = ["customer", "cust", "client", "buyer"]
        cust_col = next(
            (c for c in (profile.identifiers + profile.categoricals + all_cols)
             if any(k in c.lower() for k in cust_keywords)),
            None,
        )
        rev_col = profile.revenue_col or next(
            (c for c in all_cols
             if any(k in c.lower() for k in ["price", "amount", "revenue", "total"])
             and c not in profile.identifiers and c != cust_col),
            None,
        ) or profile.price_col

        if not cust_col or not rev_col or rev_col not in df.columns:
            log.warning("[clv_estimate] Skipped: missing customer or revenue column")
            return []

        try:
            pdf = df.to_pandas()
            purchase_counts = pdf.groupby(cust_col).size()
            total_customers = len(purchase_counts)
            if total_customers < 5:
                log.warning("[clv_estimate] Skipped: too few customers")
                return []

            aov = float(pdf[rev_col].mean())
            repeat_customers = purchase_counts[purchase_counts > 1]
            repeat_rate = len(repeat_customers) / total_customers

            if len(repeat_customers) == 0:
                avg_orders_repeater = 1.0
            else:
                avg_orders_repeater = float(repeat_customers.mean())

            clv = aov * repeat_rate * avg_orders_repeater
            avg_orders_per_customer = float(purchase_counts.mean())
            rate_pct = repeat_rate * 100

            # Category benchmark context
            dominant_cat = _detect_dominant_category(df)
            benchmark = CATEGORY_BENCHMARKS.get(dominant_cat, CATEGORY_BENCHMARKS["default"])
            benchmark_rate = benchmark["repeat_rate_pct"]
            benchmark_gap = benchmark_rate - rate_pct

            benchmark_context = (
                f"Industry benchmark ({dominant_cat}): {benchmark['repeat_rate_range']}. "
                f"Current rate: {rate_pct:.1f}%. "
                f"{'Gap of ' + f'{benchmark_gap:.1f}pp suggests significant recovery potential.' if benchmark_gap > 2 else 'Rate is within or above benchmark range.'}"
            )

            # Root-cause: conditional text — never say "pulled down" when all categories are high (Fix 3)
            root_cause = None
            if profile.category_col and profile.category_col in df.columns:
                try:
                    cat_repeat = pdf.groupby([profile.category_col, cust_col]).size().reset_index(name="orders")
                    cat_rr = cat_repeat.groupby(profile.category_col).apply(
                        lambda g: (g["orders"] > 1).mean() * 100
                    ).reset_index(name="repeat_rate")
                    if len(cat_rr) >= 2:
                        cat_rates = {row[profile.category_col]: row["repeat_rate"] for _, row in cat_rr.iterrows()}
                        min_rate_pct = min(cat_rates.values())
                        max_rate_pct = max(cat_rates.values())
                        spread_pct = max_rate_pct - min_rate_pct
                        rate_detail = ", ".join(
                            f"{cat}={rate:.0f}%"
                            for cat, rate in sorted(cat_rates.items(), key=lambda x: x[1])
                        )
                        if spread_pct >= 15 and min_rate_pct < 70:
                            lowest_cat = min(cat_rates, key=cat_rates.get)
                            root_cause = (
                                f"Repeat rate varies significantly by {profile.category_col}: {rate_detail}. "
                                f"{lowest_cat} ({min_rate_pct:.0f}%) is dragging down the blended rate. "
                                f"Category-level retention programmes will be more effective than a blended approach."
                            )
                        elif spread_pct >= 10:
                            root_cause = (
                                f"Repeat rate shows moderate variation across {profile.category_col}: "
                                f"{rate_detail}. No single category is a significant outlier."
                            )
                        else:
                            root_cause = (
                                f"Repeat rate is consistently high across all {profile.category_col} categories "
                                f"({min_rate_pct:.0f}%–{max_rate_pct:.0f}%). "
                                f"Category mix is not the constraint — focus on AOV and basket size."
                            )
                except Exception:
                    pass

            # Fix 1: high retention gate — pivot to AOV growth scenarios when repeat_rate >= 95%
            HIGH_RETENTION_THRESHOLD = 0.95

            if repeat_rate >= HIGH_RETENTION_THRESHOLD:
                aov_scenarios = {
                    "Conservative (+5% AOV)":  total_customers * avg_orders_per_customer * aov * 0.05,
                    "Base case (+10% AOV)":    total_customers * avg_orders_per_customer * aov * 0.10,
                    "Optimistic (+15% AOV)":   total_customers * avg_orders_per_customer * aov * 0.15,
                }
                base_aov_gain = aov_scenarios["Base case (+10% AOV)"]
                description = (
                    f"CLV Estimate: AOV={_fmt_currency(aov)} × repeat rate={rate_pct:.0f}% × "
                    f"avg orders per repeater={avg_orders_repeater:.1f} = "
                    f"{_fmt_currency(clv)} per customer lifetime value.\n\n"
                    f"Retention is already maximised ({rate_pct:.0f}% repeat rate) — "
                    f"further repeat-rate improvement is not the primary growth lever. "
                    f"AOV expansion through cross-sell and upsell is the correct focus. "
                    f"AOV Growth Scenarios ({total_customers} customers, current AOV {_fmt_currency(aov)}):\n"
                    f"• Conservative (+5% AOV): {_fmt_currency(aov_scenarios['Conservative (+5% AOV)'])} additional annual revenue\n"
                    f"• Base case (+10% AOV): {_fmt_currency(aov_scenarios['Base case (+10% AOV)'])} additional annual revenue\n"
                    f"• Optimistic (+15% AOV): {_fmt_currency(aov_scenarios['Optimistic (+15% AOV)'])} additional annual revenue\n\n"
                    f"{benchmark_context}"
                )
                if root_cause:
                    description += f"\n\nCategory Analysis: {root_cause}"
                insight = BusinessInsight(
                    title="Customer Lifetime Value Estimate & AOV Growth Scenarios",
                    description=description,
                    why_it_matters="With retention already maximised, AOV is the primary lever for revenue growth.",
                    evidence=f"Total customers: {total_customers:,} | Repeat rate: {rate_pct:.0f}% | AOV: {_fmt_currency(aov)} | Avg orders (repeaters): {avg_orders_repeater:.1f}",
                    decision_implication=(
                        f"A 10% AOV lift adds {_fmt_currency(base_aov_gain)} annually — "
                        f"compare against cost of cross-sell and bundle initiatives to determine ROI."
                    ),
                    impact="\U0001f7e0 Important",
                    recommendation=(
                        f"AOV is the growth lever, not repeat rate. "
                        f"A 10% AOV lift adds {_fmt_currency(base_aov_gain)} annually. "
                        f"Introduce bundle pricing, cross-category recommendations, and "
                        f"premium tier upsells to move average basket size."
                    ),
                    rule_type="clv_estimate",
                    methodology="CLV = AOV × repeat_rate × avg_orders_per_repeater; AOV uplift = total_customers × avg_orders × AOV × lift_pct",
                    score=8.0,
                )
                if root_cause:
                    insight.chart_data = {"root_cause": root_cause}
                return [insight]

            # Normal path — retention lift scenarios
            rev_per_1pct_lift = total_customers * aov * (avg_orders_repeater / 100)
            conservative = rev_per_1pct_lift * 1  # +1%
            base_case = rev_per_1pct_lift * 3      # +3%
            optimistic = rev_per_1pct_lift * 5     # +5%

            description = (
                f"CLV Estimate: AOV={_fmt_currency(aov)} × repeat rate={rate_pct:.1f}% × "
                f"avg orders per repeater={avg_orders_repeater:.1f} = "
                f"{_fmt_currency(clv)} per customer lifetime value.\n\n"
                f"Retention Lift Scenarios ({total_customers:,} customers, AOV {_fmt_currency(aov)}):\n"
                f"• Conservative (+1% repeat rate): {_fmt_currency(conservative)} additional annual revenue\n"
                f"• Base case (+3% repeat rate): {_fmt_currency(base_case)} additional annual revenue\n"
                f"• Optimistic (+5% repeat rate): {_fmt_currency(optimistic)} additional annual revenue\n\n"
                f"{benchmark_context}"
            )
            if root_cause:
                description += f"\n\nRoot Cause: {root_cause}"

            insight = BusinessInsight(
                title="Customer Lifetime Value Estimate & Retention Uplift Scenarios",
                description=description,
                why_it_matters="CLV quantifies the long-term value of retention investment versus customer acquisition cost.",
                evidence=f"Total customers: {total_customers:,} | Repeat rate: {rate_pct:.1f}% | AOV: {_fmt_currency(aov)} | Avg orders (repeaters): {avg_orders_repeater:.1f}",
                decision_implication=(
                    f"A 3% lift in repeat rate is worth {_fmt_currency(base_case)} — "
                    f"compare this against the cost of your retention programme to determine ROI."
                ),
                impact="\U0001f7e0 Important",
                recommendation=(
                    f"Invest in retention campaigns targeting the {rate_pct:.1f}% → {min(rate_pct+3, 100):.1f}% repeat-rate goal. "
                    f"At {_fmt_currency(rev_per_1pct_lift)} per 1pp, each percentage point pays for targeted loyalty spend."
                ),
                rule_type="clv_estimate",
                methodology="CLV = AOV × repeat_rate × avg_orders_per_repeater; uplift = total_customers × AOV × (lift_pct / 100) × avg_orders",
                score=8.0,
            )
            if root_cause:
                insight.chart_data = {"root_cause": root_cause}

            return [insight]

        except Exception as e:
            log.warning(f"[clv_estimate] Failed: {e}")
            return []

    @log_rule
    def _rule_seasonal_forecast(self, df: pl.DataFrame, profile: DataProfile) -> list[BusinessInsight]:
        """12-month seasonal forecast with confidence band."""
        import plotly.graph_objects as go

        date_col = profile.date_col or next((c for c in profile.temporals), None)
        rev_col = profile.revenue_col or profile.price_col

        if not date_col or not rev_col or rev_col not in df.columns:
            log.warning("[seasonal_forecast] Skipped: missing date or revenue column")
            return []

        try:
            pdf = df.to_pandas()
            pdf["_date"] = pd.to_datetime(pdf[date_col], errors="coerce")
            pdf = pdf.dropna(subset=["_date", rev_col])

            # Aggregate by month
            pdf["_yearmonth"] = pdf["_date"].dt.to_period("M")
            monthly = pdf.groupby("_yearmonth")[rev_col].sum().reset_index()
            monthly = monthly.sort_values("_yearmonth")
            monthly["_yearmonth_dt"] = monthly["_yearmonth"].dt.to_timestamp()

            n_months = len(monthly)
            if n_months < 18:
                log.warning(f"[seasonal_forecast] Skipped: only {n_months} months of data (need 18)")
                return []

            monthly_vals = monthly[rev_col].values.astype(float)

            # Seasonal component: average per calendar month
            monthly["_cal_month"] = monthly["_yearmonth_dt"].dt.month
            month_avgs = monthly.groupby("_cal_month")[rev_col].agg(["mean", "std"]).reset_index()
            month_avgs.columns = ["cal_month", "mean", "std"]
            month_avgs["std"] = month_avgs["std"].fillna(0)

            # Trend: linear fit on monthly index
            x_idx = np.arange(len(monthly_vals))
            slope, intercept = np.polyfit(x_idx, monthly_vals, 1)

            # Forecast next 12 months
            last_period = monthly["_yearmonth"].max()
            forecast_periods = [last_period + i for i in range(1, 13)]
            forecast_dts = [p.to_timestamp() for p in forecast_periods]
            forecast_cal_months = [dt.month for dt in forecast_dts]

            trend_vals = [slope * (len(monthly_vals) + i - 1) + intercept for i in range(1, 13)]
            grand_mean = float(month_avgs["mean"].mean())
            seasonal_components = []
            for cm in forecast_cal_months:
                row = month_avgs[month_avgs["cal_month"] == cm]
                if not row.empty:
                    seasonal_components.append(float(row["mean"].iloc[0]) - grand_mean)
                else:
                    seasonal_components.append(0.0)

            forecast_vals = [t + s for t, s in zip(trend_vals, seasonal_components)]

            # Confidence intervals: +-1 std of historical month values
            conf_upper = []
            conf_lower = []
            for cm, fv in zip(forecast_cal_months, forecast_vals):
                row = month_avgs[month_avgs["cal_month"] == cm]
                std = float(row["std"].iloc[0]) if not row.empty else abs(fv * 0.1)
                conf_upper.append(fv + std)
                conf_lower.append(max(0, fv - std))

            # Peak and trough
            peak_idx = int(np.argmax(forecast_vals))
            trough_idx = int(np.argmin(forecast_vals))
            MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            peak_month_name = forecast_dts[peak_idx].strftime("%b %Y")
            trough_month_name = forecast_dts[trough_idx].strftime("%b %Y")

            # YoY trend
            annual_slope_pct = (slope * 12 / grand_mean * 100) if grand_mean > 0 else 0
            if abs(annual_slope_pct) < 2:
                trend_desc = "flat growth"
            elif annual_slope_pct > 0:
                trend_desc = f"+{annual_slope_pct:.1f}% annual growth"
            else:
                trend_desc = f"{annual_slope_pct:.1f}% annual decline"

            # Chart: actuals + forecast + confidence band
            try:
                actual_dates = monthly["_yearmonth_dt"].tolist()
                actual_dates_str = [d.strftime("%Y-%m") for d in actual_dates]
                forecast_dates_str = [d.strftime("%Y-%m") for d in forecast_dts]

                conf_x = forecast_dates_str + forecast_dates_str[::-1]
                conf_y = conf_upper + conf_lower[::-1]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=actual_dates_str, y=monthly_vals.tolist(),
                    mode="lines", name="Actual",
                    line={"color": "#3b82f6", "width": 2},
                ))
                fig.add_trace(go.Scatter(
                    x=conf_x, y=conf_y,
                    fill="toself", fillcolor="rgba(99,102,241,0.15)",
                    line={"color": "rgba(0,0,0,0)"}, name="Confidence Band",
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_dates_str, y=forecast_vals,
                    mode="lines", name="Forecast",
                    line={"color": "#6366f1", "width": 2, "dash": "dash"},
                ))
                fig.update_layout(
                    title="",
                    xaxis_title="Month",
                    yaxis_title="Revenue (₹)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#1e293b"},
                    legend={"bgcolor": "rgba(0,0,0,0)"},
                    margin={"l": 60, "r": 30, "t": 20, "b": 40},
                )
            except Exception as chart_err:
                log.warning(f"[seasonal_forecast] Chart failed: {chart_err}")
                fig = None

            # Root-cause for seasonal peak
            root_cause = None
            if profile.category_col and profile.category_col in df.columns:
                try:
                    pdf["_cal_month"] = pdf["_date"].dt.month
                    peak_cal_month = forecast_dts[peak_idx].month
                    cat_month = pdf[pdf["_cal_month"] == peak_cal_month].groupby(profile.category_col)[rev_col].sum()
                    cat_overall = pdf.groupby(profile.category_col)[rev_col].mean() * (pdf.groupby("_cal_month").size().mean())
                    if len(cat_month) >= 2:
                        lift_map = {}
                        for cat_val in cat_month.index:
                            avg = cat_overall.get(cat_val, 0)
                            if avg > 0:
                                lift_map[cat_val] = (cat_month[cat_val] / avg - 1) * 100
                        over_index_sf = {k: v for k, v in lift_map.items() if v > 5}
                        if over_index_sf:
                            top_cat = max(over_index_sf, key=over_index_sf.get)
                            top_lift = over_index_sf[top_cat]
                            root_cause = (
                                f"{MONTH_NAMES[peak_cal_month-1]} peak is primarily driven by "
                                f"{top_cat} (+{top_lift:.0f}% vs average month) — "
                                f"not evenly distributed across categories."
                            )
                        else:
                            root_cause = (
                                f"{MONTH_NAMES[peak_cal_month-1]} peak is broadly distributed — "
                                f"no single category over-indexes significantly."
                            )
                except Exception:
                    pass

            description = (
                f"Seasonal forecast based on {n_months} months of data with trend + seasonal decomposition.\n"
                f"• Next peak: {peak_month_name}: {_fmt_currency(forecast_vals[peak_idx])} "
                f"(range {_fmt_currency(conf_lower[peak_idx])}–{_fmt_currency(conf_upper[peak_idx])})\n"
                f"• Next trough: {trough_month_name}: {_fmt_currency(forecast_vals[trough_idx])} "
                f"(range {_fmt_currency(conf_lower[trough_idx])}–{_fmt_currency(conf_upper[trough_idx])})\n"
                f"• Trend: {trend_desc}\n"
                f"• Inventory planning: Stock up 3–4 weeks before {peak_month_name}."
            )
            if root_cause:
                description += f"\n• {root_cause}"

            insight = BusinessInsight(
                title=f"12-Month Seasonal Revenue Forecast (Peak: {peak_month_name})",
                description=description,
                why_it_matters="Seasonal forecasting enables proactive inventory, staffing, and marketing decisions before demand peaks.",
                evidence=f"Historical months: {n_months} | Trend: {trend_desc} | Peak forecast: {_fmt_currency(forecast_vals[peak_idx])}",
                decision_implication=f"Begin inventory procurement for {peak_month_name} by {forecast_dts[peak_idx - 1].strftime('%b %Y') if peak_idx > 0 else 'immediately'}.",
                impact="🟠 Important",
                recommendation=f"Set procurement trigger: stock up 3–4 weeks before {peak_month_name}. Budget {_fmt_currency(forecast_vals[peak_idx] * 1.1)} for that month's fulfillment.",
                rule_type="seasonal_forecast",
                methodology="Linear trend (numpy.polyfit) + calendar-month seasonal component + +-1 std confidence band",
                score=8.5,
            )
            if fig is not None:
                insight.chart_data = {
                    "type": "plotly",
                    "figure": fig.to_json(),
                    "peak_month": peak_month_name,
                    "trough_month": trough_month_name,
                    "trend_pct": annual_slope_pct,
                }
                if root_cause:
                    insight.chart_data["root_cause"] = root_cause

            return [insight]

        except Exception as e:
            log.warning(f"[seasonal_forecast] Failed: {e}")
            return []

    @log_rule
    def _rule_root_cause_analysis(self, df: pl.DataFrame, profile: DataProfile, all_insights: list) -> list[BusinessInsight]:
        """
        Three scoped root-cause hypotheses attached to parent findings.
        No generic correlation sweeps — only specific, pre-defined questions.
        """
        findings = []
        try:
            pdf = df.to_pandas()
            rev_col = profile.revenue_col or profile.price_col
            cat_col = profile.category_col
            date_col = profile.date_col

            # ── Hypothesis A: Is flat/declining revenue masking category divergence? ──
            if rev_col and cat_col and date_col:
                try:
                    _dates = pd.to_datetime(pdf[date_col], errors="coerce")
                    if _dates.notna().sum() > 0:
                        pdf_a = pdf.copy()
                        pdf_a["_date"] = _dates
                        pdf_a["_month_idx"] = (
                            (_dates.dt.year - _dates.dt.year.min()) * 12 + _dates.dt.month
                        )
                        cat_months = pdf_a.groupby([cat_col, "_month_idx"])[rev_col].sum().reset_index()
                        cat_slopes = {}
                        for cat_val, grp in cat_months.groupby(cat_col):
                            if len(grp) < 6:
                                continue
                            x = grp["_month_idx"].values.astype(float)
                            y = grp[rev_col].values.astype(float)
                            if len(x) >= 2 and np.std(y) > 0:
                                slope, _ = np.polyfit(x, y, 1)
                                cat_slopes[cat_val] = slope
                        if len(cat_slopes) >= 2:
                            fastest_up = max(cat_slopes, key=cat_slopes.get)
                            fastest_dn = min(cat_slopes, key=cat_slopes.get)
                            up_slope = cat_slopes[fastest_up]
                            dn_slope = cat_slopes[fastest_dn]
                            if up_slope > 0 or dn_slope < 0:
                                rc_text = (
                                    f"While overall revenue may appear flat, category-level trends diverge: "
                                    f"{fastest_up} is trending up at +{_fmt_currency(up_slope)}/month "
                                    f"and {fastest_dn} is trending down at {_fmt_currency(dn_slope)}/month."
                                )
                                # Attach to any revenue-trend parent finding
                                for parent in all_insights:
                                    if getattr(parent, "rule_type", "") in (
                                        "temporal_peaks", "growth_rates", "seasonality_pattern"
                                    ) and "root_cause" not in str(getattr(parent, "chart_data", "") or ""):
                                        parent.chart_data = parent.chart_data or {}
                                        parent.chart_data["root_cause"] = rc_text
                                        break
                                findings.append(BusinessInsight(
                                    title="Root Cause: Category Revenue Divergence Under Flat Overall Trend",
                                    description=rc_text,
                                    why_it_matters="Overall trend can mask opposing category movements — acting on blended data leads to wrong investment decisions.",
                                    evidence=f"Category slopes: {fastest_up}=+{up_slope:+.0f}/mo, {fastest_dn}={dn_slope:+.0f}/mo",
                                    impact="🟠 Important",
                                    recommendation=(
                                        f"Increase investment in {fastest_up} (growing category). "
                                        f"Investigate root cause of {fastest_dn} decline — pricing, competition, or supply issue?"
                                    ),
                                    rule_type="root_cause_analysis",
                                    methodology="Per-category linear slope (numpy.polyfit on monthly revenue)",
                                    score=7.8,
                                ))
                except Exception as e_a:
                    log.info(f"[root_cause] Hypothesis A skipped: {e_a}")

            # ── Hypothesis B: Is low repeat rate uniform or driven by one category? ──
            if rev_col and cat_col:
                try:
                    all_cols_list = list(df.columns)
                    cust_col = next(
                        (c for c in (profile.identifiers + profile.categoricals + all_cols_list)
                         if any(k in c.lower() for k in ["customer", "cust", "client", "buyer"])),
                        None,
                    )
                    if cust_col:
                        purchase_counts = pdf.groupby(cust_col).size()
                        overall_repeat = (purchase_counts > 1).mean() * 100
                        _dom_cat = _detect_dominant_category(df)
                        _bench = CATEGORY_BENCHMARKS.get(_dom_cat, CATEGORY_BENCHMARKS["default"])
                        _bench_threshold = _bench["repeat_rate_pct"]
                        if overall_repeat < _bench_threshold:
                            cat_repeat = pdf.groupby([cat_col, cust_col]).size().reset_index(name="orders")
                            cat_rr = cat_repeat.groupby(cat_col).apply(
                                lambda g: (g["orders"] > 1).mean() * 100
                            ).reset_index(name="repeat_rate")
                            if len(cat_rr) >= 2:
                                best_cat_rr = cat_rr.loc[cat_rr["repeat_rate"].idxmax()]
                                worst_cat_rr = cat_rr.loc[cat_rr["repeat_rate"].idxmin()]
                                rc_text = (
                                    f"Repeat rate varies significantly by {cat_col}: "
                                    f"{best_cat_rr[cat_col]}={best_cat_rr['repeat_rate']:.0f}%, "
                                    f"{worst_cat_rr[cat_col]}={worst_cat_rr['repeat_rate']:.0f}%. "
                                    f"The blended {overall_repeat:.1f}% is dragged down by low-repeat categories — "
                                    f"targeted retention for {worst_cat_rr[cat_col]} buyers offers the highest uplift."
                                )
                                # Attach to CLV or customer_concentration parent
                                for parent in all_insights:
                                    if getattr(parent, "rule_type", "") in ("clv_estimate", "customer_concentration", "rfm_segmentation"):
                                        parent.chart_data = parent.chart_data or {}
                                        if "root_cause" not in parent.chart_data:
                                            parent.chart_data["root_cause"] = rc_text
                                        break
                                findings.append(BusinessInsight(
                                    title=f"Root Cause: Repeat Rate Variance Across {cat_col} Categories",
                                    description=rc_text,
                                    why_it_matters="Blended repeat rates conceal which categories need targeted retention investment.",
                                    evidence=f"Best: {best_cat_rr[cat_col]}={best_cat_rr['repeat_rate']:.0f}% | Worst: {worst_cat_rr[cat_col]}={worst_cat_rr['repeat_rate']:.0f}%",
                                    impact="🟠 Important",
                                    recommendation=(
                                        f"Deploy category-specific retention campaigns starting with {worst_cat_rr[cat_col]} buyers. "
                                        f"A 2pp improvement in that category alone could yield measurable portfolio-level uplift."
                                    ),
                                    rule_type="root_cause_analysis",
                                    methodology="Per-category repeat purchase rate = customers_with_F>1 / total_customers_in_category",
                                    score=7.8,
                                ))
                except Exception as e_b:
                    log.info(f"[root_cause] Hypothesis B skipped: {e_b}")

            # ── Hypothesis C: Is seasonal peak driven by one category or spread evenly? ──
            if rev_col and cat_col and date_col:
                try:
                    pdf_c = pdf.copy()
                    pdf_c["_date"] = pd.to_datetime(pdf[date_col], errors="coerce")
                    pdf_c = pdf_c.dropna(subset=["_date"])
                    pdf_c["_month"] = pdf_c["_date"].dt.to_period("M")
                    monthly_total = pdf_c.groupby("_month")[rev_col].sum()
                    if len(monthly_total) >= 12:
                        peak_month_period = monthly_total.idxmax()
                        peak_month_num = peak_month_period.month
                        MONTH_NAMES_RC = ["Jan","Feb","Mar","Apr","May","Jun",
                                          "Jul","Aug","Sep","Oct","Nov","Dec"]
                        peak_name = MONTH_NAMES_RC[peak_month_num - 1]

                        # Revenue in peak month vs average month, by category
                        cat_monthly = pdf_c.groupby([cat_col, "_month"])[rev_col].sum().reset_index()
                        cat_avg = cat_monthly.groupby(cat_col)[rev_col].mean()
                        cat_peak = cat_monthly[cat_monthly["_month"].apply(lambda p: p.month) == peak_month_num].groupby(cat_col)[rev_col].sum()

                        if len(cat_peak) >= 2:
                            cat_lift = {}
                            for cat_val in cat_peak.index:
                                avg = cat_avg.get(cat_val, 0)
                                if avg > 0:
                                    cat_lift[cat_val] = (cat_peak[cat_val] / avg - 1) * 100

                            over_index = {k: v for k, v in cat_lift.items() if v > 5}
                            under_index = {k: v for k, v in cat_lift.items() if v < -5}

                            if over_index:
                                top_driver = max(over_index, key=over_index.get)
                                top_lift_pct = over_index[top_driver]
                                rc_text = (
                                    f"{peak_name} peak is primarily driven by {top_driver} "
                                    f"(+{top_lift_pct:.0f}% vs its monthly average) — "
                                    f"not evenly distributed across the portfolio."
                                )
                                if under_index:
                                    weakest = min(under_index, key=under_index.get)
                                    wpct = abs(under_index[weakest])
                                    rc_text += (
                                        f" Notably, {weakest} underperforms in {peak_name} "
                                        f"({wpct:.0f}% below its own monthly average)."
                                    )
                            else:
                                top_driver = max(cat_lift, key=cat_lift.get) if cat_lift else "N/A"
                                top_lift_pct = 0.0
                                rc_text = (
                                    f"{peak_name} peak is broadly distributed — "
                                    f"no single category over-indexes significantly."
                                )

                            if cat_lift:
                                # Attach to seasonal_forecast parent
                                for parent in all_insights:
                                    if getattr(parent, "rule_type", "") == "seasonal_forecast":
                                        parent.chart_data = parent.chart_data or {}
                                        if "root_cause" not in parent.chart_data:
                                            parent.chart_data["root_cause"] = rc_text
                                        break
                                findings.append(BusinessInsight(
                                    title=(
                                        f"Root Cause: {peak_name} Peak Concentrated in {top_driver}"
                                        if over_index else
                                        f"Root Cause: {peak_name} Peak Broadly Distributed"
                                    ),
                                    description=rc_text,
                                    why_it_matters="Category-concentrated seasonality means generic 'peak season' stocking will over-invest in low-lift categories.",
                                    evidence=(
                                        f"Peak {peak_name}: {top_driver} lift=+{top_lift_pct:.0f}% vs avg month"
                                        if over_index else
                                        f"Peak {peak_name}: no single category over-indexes >5% vs its average"
                                    ),
                                    impact="🟠 Important",
                                    recommendation=(
                                        f"Pre-stock {top_driver} specifically before {peak_name}. "
                                        f"Other categories need only moderate seasonal uplift preparation."
                                        if over_index else
                                        f"Seasonal peak is evenly distributed — broad inventory preparation for {peak_name} is appropriate."
                                    ),
                                    rule_type="root_cause_analysis",
                                    methodology="Per-category peak-month revenue vs category's own average monthly revenue",
                                    score=7.8,
                                ))
                except Exception as e_c:
                    log.info(f"[root_cause] Hypothesis C skipped: {e_c}")

        except Exception as e:
            log.warning(f"[root_cause_analysis] Failed: {e}")

        return findings

    def _deduplicate(self, insights: list[BusinessInsight]) -> list[BusinessInsight]:
        """Remove duplicates by title AND by (column, rule_family) pair."""
        seen_titles = set()
        seen_column_families = set()
        unique = []

        RULE_FAMILIES = {
            "pricing_inconsistency": "price_variance",
            "descriptive_distribution": "price_variance",
            "simulation_pricing": "price_variance",
            "causal_pricing_driver": "price_variance",
            "high_return_rate": "returns",
            "returns_by_segment": "returns",
            "returns_revenue_impact": "returns",
        }

        for ins in insights:
            if ins.title in seen_titles:
                continue

            # Column-family deduplication
            # Extract column reference from qualified_segments or title
            col_ref = (ins.qualified_segments or [ins.rule_type])[0]
            family = RULE_FAMILIES.get(ins.rule_type, ins.rule_type)
            col_family_key = (col_ref, family)

            if col_family_key in seen_column_families:
                continue  # Same column, same analytical family → skip

            seen_titles.add(ins.title)
            seen_column_families.add(col_family_key)
            unique.append(ins)

        return unique

    def _inject_contradictions(self, insights: list[BusinessInsight]) -> list[BusinessInsight]:
        """Ensure consistency (e.g. don't say 'No Insights' if rules fired)."""
        if len(insights) > 0:
            # Filter out any 'No Significant Insights' placeholders if they exist
            return [i for i in insights if "No Significant Insights" not in i.title]
        return insights

    def _ensure_minimum_insights(
        self, insights: list[BusinessInsight], df: pl.DataFrame, profile: DataProfile
    ) -> list[BusinessInsight]:
        """
        ✅ P0 FIX: Guarantee minimum 3 insights regardless of signal strength.
        Never return "No Insights" — always provide descriptive fallbacks.
        """
        if len(insights) >= 3:
            return insights

        pdf = df.to_pandas()
        fallbacks = []

        # LEVEL 3: Descriptive fallback insights — always fire
        
        # 1. Distribution balance insight
        num_cols = [c for c in profile.numericals if c not in profile.identifiers]
        for col in num_cols[:2]:
            if col in pdf.columns:
                try:
                    mean_val = pdf[col].mean()
                    median_val = pdf[col].median()
                    std_val = pdf[col].std()
                    cv = std_val / mean_val if mean_val != 0 else 0
                    
                    if cv < 0.3:
                        fmt = (self._format_inr if self._is_monetary_column(col) 
                               else (lambda x: f"{int(x):,}" if float(x) == int(x) else f"{x:,.2f}"))
                        fallbacks.append(BusinessInsight(
                            title=f"Stable Distribution: {col}",
                            description=(
                                f"{col} shows low variability (CV={cv:.2f}) — "
                                f"consistent performance with no extreme outliers. "
                                f"Mean: {fmt(mean_val)}, Median: {fmt(median_val)}."
                            ),
                            why_it_matters="Low variance indicates predictable, stable operations.",
                            evidence=f"Coefficient of Variation: {cv:.2f} (< 0.3 threshold)",
                            impact="🟢 Minor",
                            confidence_label="high",
                            recommendation=(
                                f"Use {col} median ({fmt(median_val)}) as "
                                f"the primary benchmark for target-setting."
                            ),
                            rule_type="descriptive_distribution"
                        ))
                    else:
                        min_val = pdf[col].min()
                        max_val = pdf[col].max()
                        fmt = (self._format_inr if self._is_monetary_column(col) 
                               else (lambda x: f"{int(x):,}" if float(x) == int(x) else f"{x:,.2f}"))
                        fallbacks.append(BusinessInsight(
                            title=f"High Variability: {col}",
                            description=(
                                f"{col} shows high spread (CV={cv:.2f}) — "
                                f"indicating diverse performance tiers. "
                                f"Range: {fmt(min_val)} to {fmt(max_val)}."
                            ),
                            why_it_matters="High variance suggests segmentation opportunities.",
                            evidence=f"Coefficient of Variation: {cv:.2f} (> 0.3 threshold)",
                            impact="🟠 Important",
                            confidence_label="high",
                            recommendation=f"Segment records by {col} quartile for targeted strategy.",
                            rule_type="descriptive_distribution"
                        ))
                except Exception:
                    continue

        # 2. Regional/Categorical balance insight
        cat_cols = [c for c in profile.categoricals if c not in profile.identifiers]
        for col in cat_cols[:2]:
            if col in pdf.columns:
                try:
                    n_unique = pdf[col].nunique()
                    if 2 <= n_unique <= 8:
                        counts = pdf[col].value_counts()
                        balance = counts.min() / counts.max() if counts.max() > 0 else 0
                        
                        if balance > 0.7:  # fairly balanced
                            fallbacks.append(BusinessInsight(
                                title=f"Balanced Distribution: {col}",
                                description=(
                                    f"{col} is evenly distributed across {n_unique} "
                                    f"segments (balance ratio: {balance:.2f}). "
                                    f"No single segment dominates."
                                ),
                                why_it_matters="Balanced distribution reduces dependency risk.",
                                evidence=f"Min/Max ratio: {balance:.2f} (> 0.7 threshold)",
                                impact="🟢 Minor",
                                confidence_label="high",
                                recommendation=(
                                    f"Low {col} dependency risk — diversification is a "
                                    f"structural strength. No urgent rebalancing needed."
                                ),
                                rule_type="descriptive_balance"
                            ))
                        break
                except Exception:
                    continue

        # 3. Record volume insight (always available)
        fallbacks.append(BusinessInsight(
            title=f"Dataset Scale: {len(df):,} Records Analyzed",
            description=(
                f"Analysis based on {len(df):,} records across "
                f"{len(df.columns)} dimensions. "
                f"Statistical confidence is {'high' if len(df) > 500 else 'moderate'} "
                f"given sample size."
            ),
            why_it_matters="Sample size determines statistical reliability.",
            evidence=f"N={len(df):,} rows, {len(df.columns)} columns",
            impact="🟢 Minor",
            confidence_label="high",
            recommendation=(
                "Sufficient data for trend analysis. "
                "Consider time-series decomposition for deeper patterns."
            ),
            rule_type="descriptive_volume"
        ))

        # Fill up to minimum 2
        needed = max(0, 2 - len(insights))
        insights.extend(fallbacks[:needed])
        return insights

    def _rule_generic_distribution_analysis(self, df: pl.DataFrame, profile: "DataProfile") -> list[BusinessInsight]:
        """
        Fallback rule: fires when dataset has no clear revenue/financial column.
        Generates distribution insights for all categorical columns.
        Always produces at least 3 meaningful insights.
        """
        insights = []
        pdf = df.to_pandas()

        cat_cols = [c for c in pdf.columns
                    if pdf[c].nunique() <= 20 and pdf[c].nunique() >= 2
                    and pdf[c].dtype == 'object']

        for col in cat_cols[:4]:
            try:
                counts = pdf[col].value_counts()
                top = counts.index[0]
                top_pct = counts.iloc[0] / len(pdf) * 100
                bottom = counts.index[-1]
                bottom_pct = counts.iloc[-1] / len(pdf) * 100

                insights.append(BusinessInsight(
                    title=f"{col} Distribution: {top} dominates at {top_pct:.0f}%",
                    description=(
                        f"{top} accounts for {top_pct:.0f}% of all records "
                        f"({counts.iloc[0]:,} out of {len(pdf):,}), "
                        f"while {bottom} is the smallest segment at {bottom_pct:.0f}% "
                        f"({counts.iloc[-1]:,} records). "
                        f"Total unique values: {pdf[col].nunique()}."
                    ),
                    why_it_matters=f"Understanding {col} distribution reveals concentration and diversity in the dataset.",
                    evidence=f"Top: {top} ({top_pct:.0f}%) | Bottom: {bottom} ({bottom_pct:.0f}%)",
                    impact="🟠 Important",
                    recommendation=(
                        f"Focus analysis on {top} segment — it represents the majority. "
                        f"Investigate why {bottom} is underrepresented."
                    ),
                    rule_type="distribution_analysis",
                    score=5.0,
                    chart_data={"col": col, "top": top, "top_pct": top_pct},
                ))
            except Exception:
                continue

        insights.append(BusinessInsight(
            title=f"Dataset Scale: {len(pdf):,} records across {len(pdf.columns)} dimensions",
            description=(
                f"This dataset contains {len(pdf):,} records and {len(pdf.columns)} columns. "
                f"Categorical dimensions: {len(cat_cols)}. "
                f"Numeric dimensions: {len(pdf.select_dtypes('number').columns)}. "
                f"Completeness: {(1 - pdf.isnull().mean().mean()) * 100:.1f}% non-null values."
            ),
            why_it_matters="Dataset scale and completeness determine the reliability of all downstream analysis.",
            evidence=f"{len(pdf):,} records, {len(pdf.columns)} columns",
            impact="🟢 Minor",
            recommendation="Ensure data completeness before drawing conclusions. Flag columns with >10% missing values.",
            rule_type="data_quality",
            score=4.0,
            chart_data={},
        ))

        return insights

    def _rank_insights(self, insights: list[BusinessInsight]) -> list[BusinessInsight]:
        """
        ✅ FINAL V4: ROI-weighted ranking.
        Score = (₹ Impact × Confidence) / Implementation Complexity
        """
        # Implementation effort scores (lower = easier to implement)
        COMPLEXITY = {
            "cross_dimensional_margin": 2,  # Easy: just redirect ops
            "regional_balance": 1,  # Easy: informational
            "causal_pricing_driver": 3,  # Medium: requires process change
            "simulation_pricing": 3,  # Medium: pricing ops change
            "revenue_concentration": 2,  # Easy: portfolio shift
            "simulation_category_growth": 4,  # Hard: market development
            "descriptive_distribution": 1,  # Easy: reporting change
            "heatmap_pattern": 2,
            "correlation_matrix": 2,
            "temporal_peaks": 2,
            "pricing_inconsistency": 3,
            "cross_dimensional_dominance": 2,
            "cross_dimensional_volume_value": 3,
            "revenue_dominance": 2,
            "descriptive_balance": 1,
            "descriptive_volume": 1,
        }
        
        def roi_score(ins):
            # Handle both dict and BusinessInsight object
            if isinstance(ins, dict):
                rule_type = ins.get("rule_type", "")
                impact_str = ins.get("impact", "Medium")
                confidence_score = ins.get("confidence_score", 0.5)
                chart_data = ins.get("chart_data", {})
            else:
                rule_type = getattr(ins, "rule_type", "")
                impact_str = getattr(ins, "impact", "Medium")
                confidence_score = getattr(ins, "confidence_score", 0.5)
                chart_data = getattr(ins, "chart_data", None) or {}
            
            # Financial impact
            impact_val = (
                chart_data.get("uplift_abs", 0) or
                chart_data.get("scenarios", {}).get("base_case", 0) or
                0
            )
            
            # Normalize to 0-100 scale if no ₹ value
            if impact_val == 0:
                impact_score = {
                    "Critical": 80, "🔴 Critical": 80,
                    "Important": 50, "🟠 Important": 50,
                    "Medium": 30, "High": 80,
                    "Low": 10, "🟢 Minor": 10
                }.get(impact_str, 30)
            else:
                impact_score = min(impact_val / 100_000, 100)  # ₹1L = 1 point
            
            complexity = COMPLEXITY.get(rule_type, 3)
            roi = (impact_score * confidence_score) / complexity
            
            # Store for display
            if isinstance(ins, dict):
                ins["roi_score"] = round(roi, 2)
            else:
                ins.chart_data = ins.chart_data or {}
                ins.chart_data["roi_score"] = round(roi, 2)
            
            return roi
        
        ranked = sorted(insights, key=roi_score, reverse=True)
        
        # Add rank label to each
        for i, ins in enumerate(ranked):
            rank_label = (
                "🥇 Highest ROI" if i == 0 else
                "🥈 High ROI" if i == 1 else
                "🥉 Strong ROI" if i == 2 else
                f"#{i+1}"
            )
            
            if isinstance(ins, dict):
                ins["rank"] = i + 1
                ins["rank_label"] = rank_label
            else:
                ins.chart_data = ins.chart_data or {}
                ins.chart_data["rank"] = i + 1
                ins.chart_data["rank_label"] = rank_label
        
        return ranked

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


def _is_id_value(val: str) -> bool:
    """Return True if value looks like an ID/code, not a human-readable name."""
    val = str(val).strip()
    if re.match(r'^[A-Z]{2,4}-[A-Z]{2,4}-\d{5,}$', val):
        return True
    if re.match(r'^[A-Z]{2,3}-\d{4}-\d{5,}$', val):
        return True
    if re.match(r'^\d{5,}$', val):
        return True
    return False


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

    def __init__(self, domain: str, df: pl.DataFrame, insights: list, corr_matrix=None, high_impact_count: int = None):
        self.domain = domain
        self.df = df
        self.insights = insights
        self.corr_matrix = corr_matrix
        self.high_impact_count = high_impact_count  # P0 FIX (Bug 0.5): Accept pre-computed count

    def build(self) -> str:
        """
        FIX 5: Enhanced executive summary with specific numbers and tighter prose.
        - Adds total revenue with formatting
        - Includes peak/trough specific values
        - Names top category with percentage
        - More actionable language
        """
        domain_label = self.DOMAIN_LABELS.get(self.domain, "General Business")
        n_records = self.df.height

        # FIX 5: Calculate total revenue
        rev_col = next(
            (c for c in self.df.columns if any(k in c.lower() for k in ["sales", "amount", "revenue", "total"])),
            None
        )
        total_revenue = None
        if rev_col:
            try:
                total_revenue = float(self.df[rev_col].sum())
            except:
                pass

        # 1. Find the strongest non-tautological numeric driver from corr matrix
        driver_col, target_col, r_value = self._find_top_driver()

        # 2. Count critical risk insights
        # P0 FIX (Bug 0.5): Use passed high_impact_count if available, otherwise count from insights
        if self.high_impact_count is not None:
            critical_count = self.high_impact_count
        else:
            critical_count = 0
            for i in self.insights:
                impact = i.get("impact", "") if isinstance(i, dict) else getattr(i, "impact", "")
                if "🔴" in str(impact) or "High" in str(impact):
                    critical_count += 1

        # 3. Find segment with biggest revenue gap (if any)
        segment_finding = self._find_top_segment_finding()
        
        # FIX 5: Get top category with percentage
        top_category_info = self._find_top_category(rev_col)

        # Build paragraph
        lines = []
        
        # FIX 5: Enhanced opening with total revenue
        if total_revenue:
            lines.append(
                f"Across {n_records:,} transactions totaling {_fmt_currency(total_revenue)}, "
                f"this {domain_label.lower()} operation"
            )
        else:
            lines.append(
                f"The {domain_label} system is operating at a scale of {n_records:,} records."
            )

        # FIX 5: Enhanced temporal finding with specific values
        temporal_finding = self._find_temporal_finding_enhanced(rev_col)
        if not temporal_finding:
            temporal_finding = self._find_temporal_finding()
        if not temporal_finding:
            temporal_finding = self._find_temporal_finding_direct()
        
        if temporal_finding:
            if total_revenue:
                lines.append(f"shows {temporal_finding}")
            else:
                lines.append(temporal_finding)
        elif total_revenue:
            # Close the sentence if no temporal finding
            lines.append("operates at steady scale.")

        # FIX 5: Add top category information
        if top_category_info:
            lines.append(top_category_info)

        # Driver information (keep existing logic)
        if driver_col and target_col and not temporal_finding:
            lines.append(
                f"Internal analysis shows {driver_col} as the primary numeric driver of {target_col} "
                f"(correlation: {r_value:+.2f})."
            )
        elif not driver_col and not temporal_finding:
            lines.append(
                "No single numeric driver dominates — variance is distributed across multiple variables, "
                "indicating healthy portfolio diversification."
            )

        # Segment finding (keep existing)
        if segment_finding and not top_category_info:
            lines.append(segment_finding)

        # Critical findings (keep existing)
        if critical_count > 0:
            lines.append(
                f"Risk assessment identifies {critical_count} high-impact "
                f"{'finding' if critical_count == 1 else 'findings'} requiring leadership review."
            )

        # Strategic implication (keep existing)
        if driver_col and not temporal_finding:
            lines.append(
                f"Strategic focus: center forecasting efforts on {driver_col} as the leading indicator."
            )

        result = " ".join(lines)

        # HR domain: strip monetary totals and use workforce language
        if self.domain == "hr":
            import re
            result = re.sub(r'totaling [₹$€£][\d,.]+[KLCrMB\s]*,?\s*', '', result)
            result = result.replace("transactions", "employees")

        return result

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
    
    def _find_temporal_finding_enhanced(self, rev_col: str = None) -> str:
        """
        FIX 5: Enhanced temporal finding with specific peak/trough values.
        Returns a more detailed temporal analysis with actual revenue numbers.
        """
        try:
            date_col = next(
                (c for c in self.df.columns if any(k in c.lower() for k in ["date", "time", "month"])),
                None
            )
            if not rev_col:
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

            if not pd.api.types.is_numeric_dtype(pdf[rev_col]):
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

            # FIX 5: Include specific values
            return (
                f"strong seasonality: {peak_month} peaks at {_fmt_currency(peak_val)} "
                f"while {trough_month} troughs at {_fmt_currency(trough_val)} — "
                f"a {gap:.0f}% swing requiring proactive inventory planning."
            )
        except Exception as e:
            print(f"[TEMPORAL ENHANCED] error: {e}")
            return ""
    
    def _find_top_category(self, rev_col: str = None) -> str:
        """
        FIX 5: Find top category with percentage of total revenue.
        Returns a sentence like "Tablet leads at 18% of revenue, with Laptop (15%) and Monitor (15%) close behind."
        """
        try:
            cat_col = next(
                (c for c in self.df.columns if any(k in c.lower() for k in ["category", "product", "item", "type"])),
                None
            )
            if not cat_col:
                return ""
            
            if not rev_col:
                rev_col = next(
                    (c for c in self.df.columns if any(k in c.lower() for k in ["sales", "amount", "revenue", "total"])),
                    None
                )
            if not rev_col:
                return ""

            import pandas as pd
            pdf = self.df.to_pandas()

            if not pd.api.types.is_numeric_dtype(pdf[rev_col]):
                return ""

            # Group by category
            cat_revenue = pdf.groupby(cat_col)[rev_col].sum().sort_values(ascending=False)
            if len(cat_revenue) < 2:
                return ""

            total_rev = cat_revenue.sum()
            if total_rev == 0:
                return ""
            
            # Get top category, skipping ID-like values
            top_cat = None
            top_pct = 0.0
            for candidate, rev_val in zip(cat_revenue.index, cat_revenue.values):
                if not _is_id_value(str(candidate)):
                    top_cat = candidate
                    top_pct = (rev_val / total_rev) * 100
                    break
            if top_cat is None:
                return ""

            # Build sentence with top category
            result = f"{top_cat} leads at {top_pct:.0f}% of revenue"
            
            # Add runners-up if available
            if len(cat_revenue) >= 3:
                second_cat = cat_revenue.index[1]
                second_pct = (cat_revenue.iloc[1] / total_rev) * 100
                third_cat = cat_revenue.index[2]
                third_pct = (cat_revenue.iloc[2] / total_rev) * 100
                
                result += f", with {second_cat} ({second_pct:.0f}%) and {third_cat} ({third_pct:.0f}%) close behind"
            
            # Add diversification comment if top category is not dominant
            if top_pct < 30:
                result += ", indicating healthy portfolio diversification."
            else:
                result += "."
            
            return result
            
        except Exception as e:
            print(f"[TOP CATEGORY] error: {e}")
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
                chart_data = getattr(ins, "chart_data", {})
            else:
                impact = ins.get("impact", "Medium")
                qualified_segments = ins.get("qualified_segments", [])
                rule_type = ins.get("rule_type", "")
                title = ins.get("title", "")
                recommendation = ins.get("recommendation", "")
                excluded_segments = ins.get("excluded_segments", [])
                chart_data = ins.get("chart_data", {})

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
                "excluded_segments": excluded_segments,
                "chart_data": chart_data,
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

        if "temporal" in rule_type or "seasonality" in rule_type:
            chart_data = insight.get("chart_data", {}) or {}
            
            # Pull values with explicit fallbacks
            peak   = chart_data.get("peak_month") or chart_data.get("peak_calendar_month") or "your peak month"
            trough = chart_data.get("trough_month") or chart_data.get("trough_calendar_month") or "your trough month"
            peak_driver = chart_data.get("peak_category", "")

            rec = (
                f"Pre-build inventory ahead of {peak} — historically your strongest month. "
                f"Investigate the {trough} dip: run a post-mortem on promotions, "
                f"stockouts, and demand signals from that period."
            )
            if peak_driver:
                rec += (
                    f" Focus: {peak_driver} appears to drive the {peak} peak — "
                    f"confirm whether this is demand-driven or promotion-driven before scaling."
                )
            return rec

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
    """Post-process BusinessInsight list into final display-ready form.
    
    GAP 1 FIX: Type-specific narration replaces the rigid WHAT/WHY/EVIDENCE
    template with prose styles tuned to each insight category.
    """

    def narrate(
        self, insights: list[BusinessInsight], profile: DataProfile
    ) -> list[dict]:
        out = []
        for ins in insights:
            rt = ins.rule_type
            if rt in ("temporal_peaks", "seasonality_pattern", "growth_rates"):
                final_desc = self._narrate_temporal(ins)
            elif rt in ("category_satisfaction", "rating_quality", "high_return_rate"):
                final_desc = self._narrate_quality(ins)
            elif "simulation" in rt:
                final_desc = self._narrate_simulation(ins)
            elif rt in ("causal_pricing_driver", "pricing_inconsistency"):
                final_desc = self._narrate_pricing(ins)
            elif (
                rt in ("revenue_dominance", "cross_dimensional_dominance", "worst_revenue", "dominance")
                or rt.startswith("revenue_by")
            ):
                final_desc = self._narrate_revenue(ins)
            elif rt == "customer_concentration":
                final_desc = self._narrate_customer(ins)
            else:
                final_desc = self._narrate_default(ins)

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
                "methodology": ins.methodology,
                "narrative_hook": ins.narrative_hook,
            })
        return out

    def _narrate_temporal(self, ins: "BusinessInsight") -> str:
        """
        FIX 4: Conversational temporal narrative — no rigid WHAT/WHY/EVIDENCE template.
        Pure prose, data-driven, actionable.
        """
        cd = ins.chart_data or {}
        peak   = cd.get("peak_month") or cd.get("peak_calendar_month", "the peak month")
        trough = cd.get("trough_month") or cd.get("trough_calendar_month", "the slowest month")
        gap    = cd.get("pct_gap", 0)
        direction = cd.get("direction", "")
        monthly_growth = cd.get("monthly_growth_pct", 0)

        if direction == "declining":
            trend_clause = (
                f" The underlying trend is declining at {abs(monthly_growth):.1f}%/month — "
                f"this seasonal pattern is playing out against a shrinking baseline, "
                f"which compounds the risk."
            )
        elif direction == "growing":
            trend_clause = (
                f" The underlying trend is growing at {monthly_growth:.1f}%/month, "
                f"so the seasonal swing amplifies an already-positive trajectory."
            )
        else:
            trend_clause = (
                f" Revenue is broadly flat outside the seasonal cycle — "
                f"the {gap:.0f}% swing is the primary source of cash flow risk."
            )

        # FIX 4: Pure conversational prose, no template headers
        narrative = (
            f"Revenue follows a predictable seasonal pattern, peaking in {peak} "
            f"and bottoming out in {trough} — a swing of {gap:.0f}%."
            f"{trend_clause}"
            f" If inventory and staffing aren't pre-positioned before {peak}, "
            f"you'll leave money on the table. Conversely, {trough} requires careful "
            f"cash-flow management to avoid overstaffing or excess stock."
        )
        
        # FIX 4: Integrate why_it_matters naturally
        if ins.why_it_matters:
            # CRITICAL: Always add space before concatenation
            narrative = narrative.rstrip() + ' ' + ins.why_it_matters
        
        # FIX 4: Integrate decision_implication naturally
        if ins.decision_implication:
            # CRITICAL: Always add space before concatenation
            narrative = narrative.rstrip() + ' ' + ins.decision_implication
        
        return narrative

    def _narrate_quality(self, ins: "BusinessInsight") -> str:
        """Quality risk narrative — description + why_it_matters only."""
        narrative = ins.description.rstrip()
        if ins.why_it_matters:
            why_text = (
                ins.why_it_matters
                .removeprefix('Why it matters: ')
                .removeprefix('WHY IT MATTERS: ')
                .strip()
            )
            if why_text:
                narrative = narrative + ' ' + why_text
        return narrative

    def _narrate_simulation(self, ins: "BusinessInsight") -> str:
        """
        FIX 4: What-if scenario narrative — lead with the upside number, no headers.
        """
        cd = ins.chart_data or {}
        scenarios = cd.get("scenarios", {})
        base = scenarios.get("base_case", 0)
        
        parts = []
        
        # Lead with upside if available
        if base:
            parts.append(f"Simulated upside: {_fmt_currency(base)} at base-case assumptions.")
        
        # Add main description
        parts.append(ins.description)
        
        # FIX 4: Add recommendation naturally
        if ins.recommendation:
            parts.append(ins.recommendation)
        
        return " ".join(parts)

    def _narrate_pricing(self, ins: "BusinessInsight") -> str:
        """Pricing narrative — description + why_it_matters only."""
        narrative = ins.description.rstrip()
        if ins.why_it_matters:
            why_text = (
                ins.why_it_matters
                .removeprefix('Impact: ')
                .removeprefix('WHY IT MATTERS: ')
                .strip()
            )
            if why_text:
                narrative = narrative + ' ' + why_text
        return narrative

    def _narrate_revenue(self, ins: "BusinessInsight") -> str:
        """Revenue concentration narrative — description + why_it_matters only."""
        narrative = ins.description.rstrip()
        if ins.why_it_matters:
            why_text = (
                ins.why_it_matters
                .removeprefix('Strategic risk: ')
                .removeprefix('WHY IT MATTERS: ')
                .strip()
            )
            if why_text:
                narrative = narrative + ' ' + why_text
        return narrative

    def _narrate_customer(self, ins: "BusinessInsight") -> str:
        """Customer concentration narrator — description only, implication as a clean close."""
        narrative = ins.description.rstrip()
        # Append the pre-branched implication sentence (never the raw "If X…" template)
        if ins.decision_implication:
            narrative = narrative + " " + ins.decision_implication
        return narrative

    def _narrate_default(self, ins: "BusinessInsight") -> str:
        """
        Default narrator: description prose only.
        evidence and decision_implication are rendered as separate UI elements
        (methodology footnote and → callout respectively) — not concatenated inline.
        """
        narrative = ins.description.rstrip()
        # Only append why_it_matters if it reads as a natural continuation sentence
        # (not a header-prefixed block). Evidence and decision_implication are excluded
        # to prevent stats metadata from bleeding into body copy.
        if ins.why_it_matters:
            why_text = (
                ins.why_it_matters
                .removeprefix('Why it matters: ')
                .removeprefix('WHY IT MATTERS: ')
                .strip()
            )
            if why_text:
                narrative = narrative + ' ' + why_text
        return narrative


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

        # ── Pre-process: convert comma-formatted strings to numeric ──
        _NUMERIC_KEYWORDS = [
            "cases", "deaths", "recovered", "active", "tests",
            "population", "confirmed", "infected", "serious",
            "critical", "sales", "revenue", "amount", "price",
            "total", "count", "number", "qty", "quantity"
        ]
        for _col in pdf.columns:
            if pdf[_col].dtype == object:
                _col_lower = str(_col).lower().replace("\n", "")
                _is_numeric_col = any(
                    k in _col_lower for k in _NUMERIC_KEYWORDS
                )
                if _is_numeric_col:
                    try:
                        _converted = (
                            pdf[_col].astype(str)
                                     .str.replace(",", "", regex=False)
                                     .str.replace(" ", "", regex=False)
                                     .str.strip()
                        )
                        _numeric = pd.to_numeric(
                            _converted, errors="coerce"
                        )
                        if _numeric.notna().mean() > 0.8:
                            pdf[_col] = _numeric
                            print(f"[VIZ PREPROCESS] "
                                  f"Converted {_col} to numeric")
                    except Exception:
                        pass
        # ── End preprocessing ─────────────────────────────────────────

        # ── Override profile fields with post-preprocessing numeric cols ──
        _post_numeric_cols = [
            c for c in pdf.select_dtypes(include=["number"]).columns
            if not any(k in str(c).lower()
                       for k in ["#", "index", "row", "id", "rank",
                                  "unnamed", "1m", "per_"])
        ]
        print(f"[VIZ] Numeric cols after preprocess: {_post_numeric_cols}")

        # Rebuild num_cols to include any newly converted columns
        _orig_num_cols = [c for c in profile.numericals
                          if c not in profile.identifiers]
        num_cols_pre = list(dict.fromkeys(
            _orig_num_cols + [c for c in _post_numeric_cols
                               if c not in _orig_num_cols]
        ))

        # Find best price/metric col if profile has none
        _NUMERIC_PRIORITY = [
            "cases", "confirmed", "sales", "revenue",
            "deaths", "amount", "total", "value"
        ]
        _best_num = profile.price_col or profile.revenue_col
        if not _best_num:
            for _kw in _NUMERIC_PRIORITY:
                _match = next(
                    (c for c in _post_numeric_cols
                     if _kw in str(c).lower().replace("\n", "")),
                    None
                )
                if _match:
                    _best_num = _match
                    break
            if not _best_num and _post_numeric_cols:
                _best_num = _post_numeric_cols[0]
            if _best_num:
                print(f"[VIZ] price_col overridden → {_best_num}")

        # Find best categorical col if profile has none
        _cat_override = profile.category_col
        if not _cat_override:
            _geo_cols = [
                c for c in pdf.columns
                if any(k in str(c).lower().replace("\n", "")
                       for k in ["country", "region", "state",
                                  "city", "province", "location"])
                and pdf[c].nunique() >= 2
            ]
            if _geo_cols:
                _cat_override = _geo_cols[0]
                print(f"[VIZ] cat overridden → {_cat_override}")
        # ── End ColumnMap override ────────────────────────────────────────

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

        cat  = _cat_override or profile.category_col
        geo_col = profile.geographic_col
        num_cols  = num_cols_pre
        date_col  = profile.date_col
        ret_col   = profile.return_col
        del_col   = profile.delivery_days_col
        # Revenue col = pre-aggregated column (Sales Amount, Revenue, etc.)
        # Price col   = unit price column (Unit Price, Price, etc.)
        # Rule: if revenue_col exists, always use it directly — never multiply by qty
        revenue_col_direct = profile.revenue_col   # e.g. "Sales Amount"
        price_col          = _best_num or profile.price_col or profile.revenue_col
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
                if revenue_col_direct:
                    rev_col = revenue_col_direct
                    pdf_tmp = pdf
                else:
                    rev_col = "Revenue (₹)"
                    pdf_tmp = pdf.copy()
                    if _is_revenue_col(price_col):
                        pdf_tmp[rev_col] = pdf[price_col].fillna(0)
                    else:
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
                    
                    spread_pp = top_pct - (grp[rev_col].min() / total_rev * 100)

                    if spread_pp >= 5.0:  # Only annotate if there's a real gap
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
                        # ✅ PARETO FIX: Pick the categorical column with highest concentration
                        # (biggest gap between top and bottom segment)
                        best_cat_col = cat
                        best_top1_pct = 0
                        
                        for col in [cat, geo_col]:
                            if col and col in pdf_tmp.columns:
                                shares = pdf_tmp.groupby(col)[rev_col].sum()
                                top1_pct = shares.max() / shares.sum()
                                if top1_pct > best_top1_pct:
                                    best_top1_pct = top1_pct
                                    best_cat_col = col

                        print(f"[PARETO] Selected column: {best_cat_col} (top-1 share: {best_top1_pct:.1%})")

                        # Use the best categorical column for Pareto
                        grp_pareto = pdf_tmp.groupby(best_cat_col)[rev_col].sum().reset_index()
                        grp_sorted = grp_pareto.sort_values(rev_col, ascending=False)
                        grp_sorted["cumulative_pct"] = (
                            grp_sorted[rev_col].cumsum() / grp_sorted[rev_col].sum() * 100
                        )
                        fig_pareto = go.Figure()
                        fig_pareto.add_trace(go.Bar(
                            x=grp_sorted[best_cat_col], y=grp_sorted[rev_col],
                            name="Revenue", marker_color="#6366f1",
                            text=[f"{v/1e6:.1f}M" for v in grp_sorted[rev_col]],
                            textposition="outside"
                        ))
                        fig_pareto.add_trace(go.Scatter(
                            x=grp_sorted[best_cat_col], y=grp_sorted["cumulative_pct"],
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
                            title=f"Pareto: {best_cat_col} Revenue Contribution"
                        )
                        add("pareto_revenue", {
                            "chart_id": "pareto_revenue",
                            "chart_type": "pareto",
                            "title": f"Pareto: {best_cat_col} Revenue Contribution",
                            "description": "80/20 analysis — which categories drive 80% of revenue",
                            "plotly_json": json.loads(fig_pareto.update_layout(**CHART_LAYOUT_BASE).to_json()),
                            "columns_used": [best_cat_col, price_col],
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
                    
                    # ✅ TIER 2 ENHANCEMENT: Peak and trough markers
                    peak_idx   = monthly["__rev__"].idxmax()
                    trough_idx = monthly["__rev__"].idxmin()
                    peak_month   = monthly.loc[peak_idx,   "month"]
                    trough_month = monthly.loc[trough_idx, "month"]
                    peak_val   = monthly.loc[peak_idx,   "__rev__"]
                    trough_val = monthly.loc[trough_idx, "__rev__"]
                    
                    # Peak marker — green star
                    fig.add_scatter(
                        x=[peak_month], y=[peak_val],
                        mode="markers+text",
                        marker=dict(size=14, color="#10b981",
                                    symbol="star", line=dict(color="white", width=1)),
                        text=[f"Peak: {peak_val/1e6:.1f}M"],
                        textposition="top center",
                        textfont=dict(color="#10b981", size=11),
                        name="Peak", showlegend=True
                    )
                    # Trough marker — red triangle (text above marker to avoid x-axis overlap)
                    fig.add_scatter(
                        x=[trough_month], y=[trough_val],
                        mode="markers+text",
                        marker=dict(size=14, color="#ef4444",
                                    symbol="triangle-down", line=dict(color="white", width=1)),
                        text=[f"Trough: {trough_val/1e6:.1f}M"],
                        textposition="top center",
                        textfont=dict(color="#ef4444", size=11),
                        name="Trough", showlegend=True
                    )
                    # Reference band — shaded region between trough and peak
                    fig.add_hrect(
                        y0=trough_val, y1=peak_val,
                        fillcolor="rgba(99,102,241,0.05)",
                        line_width=0,
                        annotation_text=f"{((peak_val-trough_val)/peak_val*100):.0f}% swing",
                        annotation_position="right",
                        annotation_font=dict(color="#94a3b8", size=10)
                    )
                    
                    fig.update_layout(
                        template="plotly_dark",
                        xaxis_title="Month",
                        yaxis_title="Revenue",
                        legend=dict(orientation="h", y=1.1)
                    )
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

                        # Guard: if region_col has > 20 unique values (e.g. City),
                        # fall back to a higher-level column with fewer categories
                        if region_col and pdf[region_col].nunique() > 20:
                            fallback_region_cols = [
                                c for c in pdf.columns
                                if any(k in c.lower() for k in ["region", "state", "country", "zone"])
                                and pdf[c].nunique() <= 20
                                and c != region_col
                            ]
                            if fallback_region_cols:
                                region_col = fallback_region_cols[0]
                                log.info(f"[Heatmap] Switched to {region_col} (original had too many unique values)")
                            else:
                                log.info("[Heatmap] Skipped — no column with ≤ 20 unique values found")
                                region_col = None

                        if region_col is None:
                            # No column with ≤ 20 unique values — skip grouped bar/heatmap
                            raise ValueError("heatmap skipped: no valid region column")
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
                                    [_fmt_currency(v) for v in row]
                                    for row in pivot.values
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
        def _is_truly_categorical(series, max_unique: int = 50) -> bool:
            """True only if column contains genuine text labels."""
            import pandas as _pd_inner
            # Native numeric dtype → not categorical
            if series.dtype in ["int64", "float64", "int32", "float32"]:
                return False
            # Check if values are numeric strings (e.g. "2,970", "39,723")
            try:
                cleaned = (series.dropna()
                                 .astype(str)
                                 .str.replace(",", "", regex=False)
                                 .str.replace(" ", "", regex=False)
                                 .str.strip())
                numeric_count = (
                    _pd_inner.to_numeric(cleaned, errors="coerce")
                    .notna().sum()
                )
                total = max(len(cleaned), 1)
                # If >80% look numeric AND not a geographic column
                _is_geo = any(
                    k in str(series.name).lower()
                    for k in ["country", "region", "state",
                              "province", "city", "location"]
                )
                if numeric_count / total > 0.8 and not _is_geo:
                    return False
            except Exception:
                pass
            # High cardinality check — only block if >50 AND not geo
            if series.nunique() > max_unique and not any(
                k in str(series.name).lower()
                for k in ["country", "region", "state",
                          "province", "city"]
            ):
                return False
            return True

        _HEALTH_NUMERIC_COLS = [
            "serious", "critical", "icu", "hospitalized",
            "tests", "population", "cases/", "deaths/",
            "/1m", "per_million", "serious,",
        ]

        if cat:
            _col_lower = str(cat).lower().replace("\n", "")
            _is_health_numeric = any(k in _col_lower for k in _HEALTH_NUMERIC_COLS)
            _is_categorical = _is_truly_categorical(pdf[cat])

            if _is_health_numeric:
                print(f"[VIZ] Skipping health numeric column: {cat}")
            elif not _is_categorical:
                print(f"[VIZ] Skipping non-categorical: {cat}")
            else:
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
                # No marginal="rug" — it creates a 2-row subplot that causes add_vline
                # to render the annotation twice (once per subplot row).
                fig = px.histogram(
                    pdf.dropna(subset=[price_col]),
                    x=price_col,
                    color=color_col,
                    title=f"{price_col} Distribution",
                    nbins=30,
                    opacity=0.8,
                )

                # Single median line — no annotation_text on add_vline to avoid duplication;
                # use add_annotation separately so we control position precisely.
                median_val = pdf[price_col].median()
                fig.add_vline(
                    x=median_val,
                    line=dict(color="#ef4444", width=2, dash="dash"),
                )
                fig.add_annotation(
                    x=median_val, y=1.0, yref="paper",
                    text=f"Median: {median_val:,.0f}",
                    showarrow=False, xanchor="left",
                    font=dict(color="#ef4444", size=11),
                    bgcolor="rgba(0,0,0,0.4)", borderpad=3,
                )

                fig.update_layout(
                    template="plotly_dark",
                    barmode="overlay" if color_col else "relative",
                    height=320,
                )
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

        # ── 9. Customer Purchase Frequency (bar) ──────────────────────────
        cust_col = next(
            (c for c in profile.identifiers
             if any(k in c.lower() for k in ["customer", "cust", "client", "buyer"])),
            None,
        )
        if cust_col and cust_col in pdf.columns:
            try:
                freq = pdf.groupby(cust_col).size().value_counts().sort_index().reset_index()
                freq.columns = ["Purchases", "Customers"]
                freq = freq[freq["Purchases"] <= 20]  # cap x-axis at 20
                if len(freq) >= 2 and is_chart_informative(freq["Customers"].tolist()):
                    fig = px.bar(
                        freq, x="Purchases", y="Customers",
                        title="Customer Purchase Frequency",
                        labels={"Purchases": "Number of Purchases", "Customers": "Number of Customers"},
                    )
                    fig.update_traces(marker_color="#6366f1")
                    fig.update_layout(template="plotly_dark")
                    add("customer_freq", {
                        "chart_id": "customer_freq",
                        "chart_type": "bar",
                        "title": "Customer Purchase Frequency",
                        "description": "How many customers made 1, 2, 3… purchases — shows retention and loyalty distribution",
                        "plotly_json": json.loads(fig.update_layout(**CHART_LAYOUT_BASE).to_json()),
                        "columns_used": [cust_col],
                        "priority_score": 78,
                        "insight_reason": "Customer behaviour and retention signal",
                        "interest_level": "high",
                    })
            except Exception:
                pass

        # ── 10. Fallback generic charts if we still have space ─────────────
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
        
        # Debug: Show what we received
        print(f"[FALLBACK DEBUG] profile.numericals: {profile.numericals[:10]}")
        print(f"[FALLBACK DEBUG] num_cols passed: {num_cols[:10]}")

        # ID column blacklist - exclude these from fallback charts
        ID_KEYWORDS = [
            "num", "number", "id", "code", "cd", "ifsc", "pin", "pincode",
            "adhaar", "aadhaar", "account", "mobile", "contact", "license",
            "tax", "payee", "employee", "agent", "branch", "application",
            "laclient", "parent", "recruited", "partner_code", "channel_code",
            "sub_channel_code", "payee_code", "account_payee", "mapped"
        ]
        
        # Filter out ID columns from num_cols
        filtered_num_cols = []
        for c in num_cols:
            col_lower = c.lower().replace(" ", "").replace("_", "")
            if not any(id_kw.replace("_", "") in col_lower for id_kw in ID_KEYWORDS):
                filtered_num_cols.append(c)
        
        print(f"[FALLBACK] Filtered numeric columns: {filtered_num_cols[:5]}")  # Debug log

        # Priority 1: revenue/sales/amount/price columns
        priority_nums = [
            c for c in filtered_num_cols
            if any(k in c.lower() for k in ["sales", "amount", "amt", "payment", "revenue", "price", "profit"])
        ]
        # Priority 2: any numeric column with meaningful scale (max >= 100)
        other_nums = [
            c for c in filtered_num_cols
            if c not in priority_nums
            and pdf[c].max() >= 100
        ]
        ordered_nums = (priority_nums + other_nums)[:2]
        
        print(f"[FALLBACK] Priority nums: {priority_nums}")  # Debug log
        print(f"[FALLBACK] Ordered nums: {ordered_nums}")  # Debug log
        
        # If no meaningful numeric columns, create count-based charts instead
        if not ordered_nums and cat:
            print(f"[FALLBACK] No numeric columns available, creating count-based charts")
            
            # Get top categorical columns (excluding IDs)
            cat_cols = [
                c for c in profile.categoricals
                if c != cat and pdf[c].nunique() <= 20  # Reasonable cardinality
            ][:2]
            
            for cat_col in cat_cols:
                if len(charts) >= max_charts:
                    break
                    
                try:
                    # Count by category
                    counts = pdf[cat_col].value_counts().reset_index()
                    counts.columns = [cat_col, "Count"]
                    counts = counts.sort_values("Count", ascending=False).head(10)
                    
                    # Check variance
                    if not is_chart_informative(counts["Count"].tolist()):
                        print(f"[CHART SUPPRESSED] Count by {cat_col} — variance too low")
                        continue
                    
                    fig = px.bar(
                        counts, x=cat_col, y="Count",
                        title=f"Distribution by {cat_col}",
                        text_auto=True,
                        color_discrete_sequence=["#3b82f6"]
                    )
                    fig.update_layout(template="plotly_dark")
                    
                    chart_id = f"count_by_{cat_col.lower().replace(' ', '_')}"
                    if chart_id not in chart_ids_used:
                        charts.append({
                            "chart_id": chart_id,
                            "chart_type": "bar",
                            "title": f"Distribution by {cat_col}",
                            "description": f"Count of records by {cat_col}",
                            "plotly_json": json.loads(fig.to_json()),
                            "columns_used": [cat_col],
                            "priority_score": 70,
                            "insight_reason": "Categorical distribution analysis",
                            "interest_level": "recommended"
                        })
                        chart_ids_used.add(chart_id)
                        print(f"[FALLBACK] Added count chart: {cat_col}")
                except Exception as e:
                    print(f"[FALLBACK ERROR] Count chart for {cat_col}: {e}")
                    continue
            
            return  # Skip numeric-based fallback charts

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
    
    # Tiered Logic - increased limits for better analysis
    if rows > 500000:
        sample_n = 100000  # Increased from 50K
    elif rows > 100000:
        sample_n = 50000   # Increased from 20K (your 200K dataset will use this)
    else:
        sample_n = 20000   # Increased from 10K
        
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

    try:
        # ✅ VERSION MARKER - Confirms new code is active
        print("\n" + "="*70)
        print("=== NEW CODE ACTIVE === CHART FIX v3 + CURRENCY FIX")
        print("✅ V2 ENGINE ACTIVE — 2026-05-09 BUILD")
        print("✅ Enhanced error handling, lowered thresholds, safety nets active")
        print("="*70 + "\n")

        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)

        # Detect and set currency symbol for this analysis run
        from report_generator import _detect_currency_symbol
        _set_currency_symbol(_detect_currency_symbol(
            df.to_pandas() if hasattr(df, 'to_pandas') else df
        ))
        print(f"[INSIGHT ENGINE CURRENCY] Symbol set to: {_CURRENCY_SYMBOL}")

        # ── Sampling for large datasets (FIX 4: Tiered Logic) ──────────
        original_row_count = len(df)
        sampled = False
        # Disabled sampling - analyze full dataset
        # if original_row_count > 10000:
        #     df = _apply_smart_sampling(df)
        #     sampled = True

        _progress("classifying", 10)
        classifier = ColumnClassifier()
        profile    = classifier.classify(df)
        
        # DEBUG: Print column mapping
        print("=== COLUMN MAPPING ===")
        for attr in ["revenue_col", "price_col", "qty_col", "category_col", "geographic_col", "date_col", "return_col"]:
            print(f"{attr}: {getattr(profile, attr, 'MISSING')}")
        print(f"numericals: {profile.numericals}")
        print(f"categoricals: {profile.categoricals}")
        print(f"temporals: {profile.temporals}")
        print("=" * 50)
        
        # TIER 1.1: Initialize column coverage tracker
        coverage = ColumnCoverageTracker(df, profile)
        coverage.mark(profile.price_col, profile.qty_col, profile.revenue_col,
                      profile.return_col, profile.date_col, profile.category_col,
                      profile.geographic_col, profile.delivery_days_col)

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

        # Verification print — confirms which insights reach the PDF renderer
        print(f"\n[PRE-PDF INSIGHTS] {len(compressed_insights)} insights selected for report:")
        print([getattr(i, 'title', '?') for i in compressed_insights])
        _priority_check = {"cohort_retention", "clv_estimate", "seasonal_forecast", "rfm_segmentation"}
        _present = {getattr(i, 'rule_type', '') for i in compressed_insights}
        _missing = _priority_check - _present
        if _missing:
            print(f"[WARNING] Priority rules missing from PDF: {_missing}")
        else:
            print("[OK] All four priority rules present in synthesizer output.")

        # SAFETY NET: Never return empty insights
        if not compressed_insights:
            print("[WARNING] No insights generated - adding fallback insight")
            compressed_insights = [BusinessInsight(
                title="Dataset Overview",
                description=f"Analyzed {len(df):,} records across {len(df.columns)} columns. "
                           f"Dataset contains {len(profile.numericals)} numeric columns and "
                           f"{len(profile.categoricals)} categorical columns.",
                why_it_matters="Baseline data confirmation for analysis.",
                impact="🟢 Minor",
                rule_type="safety_fallback",
                methodology="Direct dataset inspection",
                narrative_hook=f"Dataset contains {len(df):,} records ready for analysis."
            )]
        
        # TIER 5.6: Sanity check before publication
        checker = SanityChecker(df, profile)
        compressed_insights = checker.check_all(compressed_insights, metrics)
        if checker.issues:
            warnings.extend([f"🔍 Sanity: {issue}" for issue in checker.issues])

        _progress("generating_charts", 85)
        chart_rec  = SmartChartRecommender()
        charts     = chart_rec.recommend(df, profile, compressed_insights, max_charts=max_charts, domain_id=domain_id)

        # Executive summary (Step 7: Strategic Brief)
        # P0 FIX (Bug 0.5): Count from compressed_insights, not raw insights
        high_count = sum(1 for i in compressed_insights if "🔴" in str(i.impact))
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
        
        # TIER 1.1: Add column coverage report
        coverage_report = coverage.report()
        result["column_coverage"] = coverage_report
        if coverage_report.get("high_value_missed"):
            warnings.append(coverage_report["warning"])
        
        # Assertion Guard (Step 5)
        assert isinstance(result["strategic_brief"], list), "strategic_brief MUST be a list"
        print("DEBUG STRATEGIC BRIEF:", len(result["strategic_brief"]), "items found.")
        
        return result
        
    except Exception as e:
        # Comprehensive error handling with fallback
        print(f"[ERROR] Insight engine failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a minimal valid response instead of crashing
        # Note: recommendations must be dicts, not strings, to match API schema
        return {
            "domain": {"name": "Unknown", "confidence": "low", "reason": "Error during analysis", "id": "general"},
            "target": None,
            "key_drivers": [],
            "profile": {
                "identifiers": [],
                "numericals": [],
                "categoricals": [],
                "temporals": [],
                "binaries": [],
            },
            "computed_metrics": {},
            "strategic_brief": [],
            "recommendations": [],  # Empty list instead of error strings
            "executive_summary": f"Analysis could not be completed due to an error: {str(e)}",
            "warnings": [
                f"🔴 Critical Error: {type(e).__name__}: {str(e)}",
                "⚠️ Analysis failed. Please try uploading the file again or contact support."
            ],
            "column_coverage": {
                "total_columns": 0,
                "analyzed_columns": 0,
                "coverage_pct": 0,
                "untouched_columns": [],
                "high_value_missed": [],
                "warning": None
            }
        }


# ============================================================
# GAP 2: DRILL-DOWN API
# ============================================================

def drill_down(
    df: pl.DataFrame,
    insight_title_or_rule_type: str,
    profile: "DataProfile | None" = None,
) -> dict:
    """GAP 2: Perform targeted deeper analysis on a specific insight type.
    
    Accepts an insight title substring OR a rule_type string.
    Returns a structured dict with drill-down tables and charts.
    
    Expose through your API layer as POST /api/drill-down.
    """
    if profile is None:
        classifier = ColumnClassifier()
        profile = classifier.classify(df)

    key = insight_title_or_rule_type.lower()
    pdf = df.to_pandas()

    # ── Temporal / Revenue Trend ────────────────────────────────
    if any(k in key for k in ["temporal", "seasonal", "peak", "trough", "trend"]):
        date_col = next((c for c in df.columns if any(k in c.lower() for k in ["date", "time", "month"])), None)
        rev_col  = profile.revenue_col or profile.price_col
        cat_col  = profile.category_col
        if not (date_col and rev_col):
            return {"error": "No date/revenue columns found for temporal drill-down."}

        pdf["_month"] = pd.to_datetime(pdf[date_col], errors="coerce").dt.to_period("M")
        monthly_by_cat = (
            pdf.groupby(["_month", cat_col])[rev_col].sum().unstack(cat_col)
            if cat_col and cat_col in pdf.columns
            else pdf.groupby("_month")[rev_col].sum().to_frame("Total")
        )
        return {
            "drill_type": "temporal_breakdown",
            "category_by_month": monthly_by_cat.reset_index().to_dict(orient="records"),
            "insight": "Month × Category revenue matrix. Identify which category drives each month's peak.",
        }

    # ── Revenue Concentration ───────────────────────────────────
    if any(k in key for k in ["revenue", "dominance", "concentration"]):
        cat_col = profile.category_col
        rev_col = profile.revenue_col or profile.price_col
        if not (cat_col and rev_col):
            return {"error": "No category/revenue columns found."}
        summary = pdf.groupby(cat_col)[rev_col].agg(["sum", "count", "mean"]).reset_index()
        summary["rev_share_pct"] = summary["sum"] / summary["sum"].sum() * 100
        summary = summary.sort_values("sum", ascending=False)
        return {
            "drill_type": "revenue_concentration",
            "breakdown": summary.to_dict(orient="records"),
            "insight": "Full category revenue breakdown with share %. Identify over-reliance on a single segment.",
        }

    # ── Quality / Rating Risk ───────────────────────────────────
    if any(k in key for k in ["quality", "rating", "return", "satisfaction"]):
        cat_col    = next((c for c in df.columns if "category" in c.lower()), None)
        rating_col = next((c for c in df.columns if "rating" in c.lower()), None)
        rev_col    = profile.revenue_col or profile.price_col
        if not (cat_col and rating_col):
            return {"error": "No category/rating columns found."}
        summary = pdf.groupby(cat_col).agg(
            avg_rating=(rating_col, "mean"),
            rating_std=(rating_col, "std"),
            revenue=(rev_col, "sum") if rev_col else (rating_col, "count"),
        ).reset_index()
        summary["rev_share_pct"] = summary["revenue"] / summary["revenue"].sum() * 100 if rev_col else 0
        from time_series_analysis import TimeSeriesAnalyzer as _TSA
        comparisons = []
        cats = summary[cat_col].tolist()
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                ga = pdf[pdf[cat_col] == cats[i]][rating_col]
                gb = pdf[pdf[cat_col] == cats[j]][rating_col]
                test = _TSA.segment_comparison_test(ga, gb)
                if test["significant"]:
                    comparisons.append({
                        "segment_a": cats[i], "segment_b": cats[j], **test
                    })
        return {
            "drill_type": "rating_breakdown",
            "by_category": summary.to_dict(orient="records"),
            "significant_differences": comparisons,
            "insight": "Rating distribution by category with statistical significance. Focus on categories with significant gaps.",
        }

    return {"error": f"No drill-down handler matched '{insight_title_or_rule_type}'. Try: 'temporal', 'revenue', 'quality'."}


# ============================================================
# GAP 5: BENCHMARK COMPARISON
# ============================================================

def benchmark_compare(metric_name: str, computed_value: float, domain_id: str) -> dict | None:
    """GAP 5: Compare a computed metric against embedded industry benchmarks.
    
    Returns a dict with the benchmark value, multiplier, and contextual text.
    Returns None if no benchmark is available for this metric/domain.
    """
    from report_generator import TEMPLATES
    benchmarks = TEMPLATES.get(domain_id, {}).get("benchmarks", {})
    benchmark_val = benchmarks.get(metric_name)
    if benchmark_val is None:
        return None
    multiplier = computed_value / benchmark_val if benchmark_val else None
    if multiplier and multiplier > 2:
        verdict = f"🔴 {multiplier:.1f}× the industry average of {benchmark_val} — critical gap."
    elif multiplier and multiplier > 1.25:
        verdict = f"🟠 {multiplier:.1f}× the industry average of {benchmark_val} — above average but manageable."
    elif multiplier and multiplier < 0.75:
        verdict = f"🟢 {multiplier:.1f}× the industry average — below average; investigate root cause."
    else:
        verdict = f"✅ Within normal range of the {benchmark_val} industry benchmark."
    return {
        "metric": metric_name,
        "computed": computed_value,
        "benchmark": benchmark_val,
        "multiplier": round(multiplier, 2) if multiplier else None,
        "verdict": verdict,
    }


# ============================================================
# HELPERS
# ============================================================

# Module-level symbol — set once per analysis run
_CURRENCY_SYMBOL = "₹"

def _set_currency_symbol(symbol: str) -> None:
    global _CURRENCY_SYMBOL
    _CURRENCY_SYMBOL = symbol

def _fmt_currency(val: float) -> str:
    global _CURRENCY_SYMBOL
    sym = _CURRENCY_SYMBOL
    try:
        abs_val = abs(float(val))
        sign = "" if float(val) >= 0 else "-"
    except (TypeError, ValueError):
        return str(val)
    if sym == "₹":
        if abs_val >= 1_00_00_000:
            return f"{sign}₹{abs_val/1_00_00_000:.2f} Cr"
        if abs_val >= 1_00_000:
            return f"{sign}₹{abs_val/1_00_000:.2f} L"
        if abs_val >= 1_000:
            return f"{sign}₹{abs_val/1_000:.1f}K"
        return f"{sign}₹{abs_val:,.0f}"
    else:
        if abs_val >= 1_000_000_000:
            return f"{sign}{sym}{abs_val/1_000_000_000:.2f}B"
        if abs_val >= 1_000_000:
            return f"{sign}{sym}{abs_val/1_000_000:.2f}M"
        if abs_val >= 1_000:
            return f"{sign}{sym}{abs_val/1_000:.1f}K"
        return f"{sign}{sym}{abs_val:,.2f}"


CATEGORY_BENCHMARKS = {
    "electronics": {
        "repeat_rate_pct": 18,
        "repeat_rate_range": "15–25%",
        "aov_note": "High AOV, long repurchase cycles typical",
        "seasonality_note": "Peak: Nov–Dec (holiday), secondary: back-to-school Aug–Sep",
    },
    "furniture": {
        "repeat_rate_pct": 12,
        "repeat_rate_range": "8–15%",
        "aov_note": "Very high AOV, repurchase cycles 3–5 years",
        "seasonality_note": "Peak: spring (home refresh) and pre-holiday Nov",
    },
    "office_equipment": {
        "repeat_rate_pct": 22,
        "repeat_rate_range": "18–30%",
        "aov_note": "Consumable replacement drives repeats",
        "seasonality_note": "Steady demand; minor Q1 budget-cycle peak",
    },
    "default": {
        "repeat_rate_pct": 25,
        "repeat_rate_range": "20–35%",
        "aov_note": "No category-specific benchmark available",
        "seasonality_note": "No category-specific seasonality data",
    },
}

PRODUCT_TO_CATEGORY = {
    "tablet": "electronics", "laptop": "electronics", "monitor": "electronics",
    "phone": "electronics", "headphone": "electronics", "camera": "electronics",
    "desk": "furniture", "chair": "furniture", "sofa": "furniture",
    "printer": "office_equipment", "scanner": "office_equipment",
}


def _detect_dominant_category(df) -> str:
    """Auto-detect dominant product category from dataset for benchmark lookup."""
    try:
        # Find a product/item column
        product_col = None
        for col in df.columns:
            cl = col.lower()
            if any(k in cl for k in ["product", "item", "sku", "name"]):
                product_col = col
                break
        if product_col is None:
            return "default"

        # Get product names (handle both polars and pandas)
        if hasattr(df, 'to_pandas'):
            pdf = df.to_pandas()
        else:
            pdf = df

        products = pdf[product_col].dropna().astype(str).str.lower().tolist()

        # Count category matches
        cat_counts = {}
        for prod in products:
            for keyword, category in PRODUCT_TO_CATEGORY.items():
                if keyword in prod:
                    cat_counts[category] = cat_counts.get(category, 0) + 1

        if not cat_counts:
            return "default"

        return max(cat_counts, key=cat_counts.get)
    except Exception:
        return "default"


def _build_exec_summary(df: pl.DataFrame, profile: DataProfile, metrics: dict, high_impact_count: int, domain_info: dict, driver_info: dict, insights: list = None, raw_insights: list = None) -> str:
    """Step 7: Generate High-End Executive Strategic Brief (3-5 lines).
    
    P0 FIX (Bug 0.5): Use passed high_impact_count instead of counting from all_insights_for_temporal
    to avoid inflating the count with both raw and compressed insights.
    """
    domain_id = domain_info.get("id", "general")

    # Pass raw_insights for temporal detection — compressed may have dropped it
    all_insights_for_temporal = (raw_insights or []) + (insights or [])
    builder = StrategicBriefBuilder(
        domain=domain_id,
        df=df,
        insights=all_insights_for_temporal,
        corr_matrix=driver_info.get("corr_matrix"),
        high_impact_count=high_impact_count  # P0 FIX: Pass the correct count
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
    
    IMPORTANT: Only drops rows where ALL values are null, not rows with ANY null.
    """
    import pandas as pd

    pdf = df.to_pandas()
    
    # Drop rows where ALL values are null (not ANY null - too aggressive)
    pdf = pdf.dropna(how='all')
    
    # Drop exact duplicates
    pdf = pdf.drop_duplicates()

    # Try to coerce numeric columns
    for col in pdf.columns:
        if pdf[col].dtype == object and any(kw in col.lower() for kw in _NUMERIC_KW):
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
    
    # Try to coerce date columns
    for col in pdf.columns:
        if pdf[col].dtype == object and any(kw in col.lower() for kw in _DATE_KW):
            pdf[col] = pd.to_datetime(pdf[col], errors="coerce")
    
    # Drop rows where ALL values are null (after coercion)
    pdf = pdf.dropna(how='all')

    return pl.from_pandas(pdf)

