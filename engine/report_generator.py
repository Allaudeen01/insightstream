"""
report_generator.py  —  InsightStream Analytics
────────────────────────────────────────────────────────────────────────────────
ROOT-CAUSE FIX:
  The previous version used exact lowercase key matches ("sales", "category")
  against df.columns.  Real-world CSVs have names like "Sales Amount",
  "Product Category", "Region" — none of which matched, so generate_all()
  silently returned {}, the PDF had no charts, and embed_chart_safely showed
  its fallback text (which looked blank).

  This version uses _fuzzy_col() — a contains-based resolver that finds
  "Sales Amount" when you ask for "sales", "Product Category" when you ask
  for "category", etc.  Every resolution decision is logged so you can
  diagnose any remaining mismatches instantly.
"""

from __future__ import annotations

import contextlib
import logging
import re
import os
import tempfile
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # ← MUST precede all other matplotlib imports

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Image as RLImage, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
class C:
    FIG_W, FIG_H   = 10, 5
    DPI            = 150
    FACECOLOR      = "white"
    SNS_STYLE      = "whitegrid"
    PALETTE        = "Blues_d"

    PAGE_W, PAGE_H = A4
    MARGIN         = 0.75 * inch
    SAFE_IMG_W     = 480
    SAFE_IMG_H     = 280

    BRAND_DARK   = "#1A1A2E"
    BRAND_LIGHT  = "#F0F4FF"
    BRAND_ACCENT = "#EBF5FB"
    TEXT_DARK    = "#333333"
    TEXT_GREY    = "#555555"
    TEXT_MUTED   = "#999999"
    RULE_GREY    = "#CCCCCC"
    RULE_LIGHT   = "#EEEEEE"
    PURPLE       = "#4B0082"

    # Keyword priority lists for fuzzy column matching
    NUMERIC_KEYWORDS  = ["sales", "revenue", "profit", "amount", "value", "total", "price", "income"]
    NUMERIC2_KEYWORDS = ["quantity", "qty", "units", "count", "volume", "orders"]
    CATEGORY_KEYWORDS = ["category", "type", "segment", "department", "group"]
    REGION_KEYWORDS   = ["region", "area", "zone", "territory", "location", "city", "country", "state"]
    DATE_KEYWORDS     = ["date", "time", "month", "year", "period", "day", "week"]
    LABEL_KEYWORDS    = ["product", "item", "name", "sku", "title", "label"]

TEMPLATE_CONFIGS = {
    "modern": {
        "brand_dark": "#1A1A2E",
        "brand_light": "#F0F4FF",
        "purple": "#4B0082",
        "font_main": "Helvetica",
        "font_bold": "Helvetica-Bold"
    },
    "executive": {
        "brand_dark": "#000000",
        "brand_light": "#F5F5F5",
        "purple": "#333333",
        "font_main": "Times-Roman",
        "font_bold": "Times-Bold"
    },
    "creative": {
        "brand_dark": "#8E44AD",
        "brand_light": "#F5EEF8",
        "purple": "#9B59B6",
        "font_main": "Courier",
        "font_bold": "Courier-Bold"
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN TEMPLATES (Modernized logic)
# ══════════════════════════════════════════════════════════════════════════════

TEMPLATES = {
    "happiness": {
        "report_title": "Strategic Happiness & Socio-economic Report",
        "target_metric": "Happiness Score",
        "high_correlation_threshold": 0.80,
        "secondary_threshold": 0.40,
        "regional_insight_threshold": 0.15,
        "correlation_primary_label": "fundamental catalyst",
        "regional_chart_title": "Regional Happiness Variance Analysis",
        "executive_summary_header": "Happiness Index Strategic Overview"
    },
    "ecommerce": {
        "report_title": "Strategic Commerce & Revenue Report",
        "target_metric": "Sales Amount",
        "high_correlation_threshold": 0.70,
        "secondary_threshold": 0.35,
        "regional_insight_threshold": 0.10,
        "correlation_primary_label": "revenue driver",
        "regional_chart_title": "Geographical Revenue Distribution",
        "executive_summary_header": "Commerce Performance Executive Summary"
    },
    "general": {
        "report_title": "Strategic Data Analysis Report",
        "target_metric": "Key Performance Indicator",
        "high_correlation_threshold": 0.60,
        "secondary_threshold": 0.30,
        "regional_insight_threshold": 0.10,
        "correlation_primary_label": "primary metric driver",
        "regional_chart_title": "Regional Metric Performance",
        "executive_summary_header": "Executive Data Insights"
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# FUZZY COLUMN RESOLVER  ←  the core fix
# ══════════════════════════════════════════════════════════════════════════════
def _fuzzy_col(df: pd.DataFrame, keywords: list[str],
               exclude: Optional[list[str]] = None) -> Optional[str]:
    """
    Return the first column whose lowercase name *contains* any keyword.

    "Sales Amount"    → matches keyword "sales"    ✓
    "Product Category"→ matches keyword "category" ✓
    "Region"          → matches keyword "region"   ✓
    """
    excl = {c.lower() for c in (exclude or [])}
    for col in df.columns:
        cl = col.lower()
        if cl in excl:
            continue
        for kw in keywords:
            if kw in cl:
                return col
    return None


def _fuzzy_numeric(df: pd.DataFrame, keywords: list[str],
                   exclude: Optional[list[str]] = None) -> Optional[str]:
    """Like _fuzzy_col but restricted to numeric-dtype columns."""
    numeric = set(df.select_dtypes("number").columns)
    excl    = {c.lower() for c in (exclude or [])}
    for col in df.columns:
        if col not in numeric or col.lower() in excl:
            continue
        for kw in keywords:
            if kw in col.lower():
                return col
    # fallback: first unused numeric column
    for col in df.select_dtypes("number").columns:
        if col.lower() not in excl:
            return col
    return None


class ColumnMap:
    """
    Resolves which actual DataFrame columns to use for each chart role.
    Logs every decision — check server output if a chart is still missing.
    """
    def __init__(self, df: pd.DataFrame):
        claimed: list[str] = []

        self.numeric = _fuzzy_numeric(df, C.NUMERIC_KEYWORDS)
        if self.numeric: claimed.append(self.numeric.lower())

        self.numeric2 = _fuzzy_numeric(df, C.NUMERIC2_KEYWORDS, exclude=claimed)
        if self.numeric2: claimed.append(self.numeric2.lower())

        self.category = _fuzzy_col(df, C.CATEGORY_KEYWORDS, exclude=claimed)
        if self.category: claimed.append(self.category.lower())

        self.region = _fuzzy_col(df, C.REGION_KEYWORDS, exclude=claimed)
        if self.region: claimed.append(self.region.lower())

        self.date = _fuzzy_col(df, C.DATE_KEYWORDS, exclude=claimed)
        if self.date: claimed.append(self.date.lower())

        self.label = _fuzzy_col(df, C.LABEL_KEYWORDS, exclude=claimed)

        log.info(
            "ColumnMap → numeric=%r  numeric2=%r  category=%r  region=%r  date=%r  label=%r",
            self.numeric, self.numeric2, self.category, self.region, self.date, self.label,
        )


def get_region_column(df: pd.DataFrame) -> Optional[str]:
    """Identify the column containing regional/geographic data."""
    for col in df.columns:
        if col.strip().lower() == 'region':
            return col
    # fallback: first categorical column with low cardinality (<= 15 unique values)
    for col in df.select_dtypes('object'):
        if df[col].nunique() <= 15:
            return col
    return None


def generate_markdown_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    """
    Convert a pandas DataFrame to a markdown table, truncating if needed.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        return "No data available."
    
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
        
    total = len(df)
    truncated = total > max_rows
    if truncated:
        df = df.head(max_rows)
    
    # Build markdown table
    headers = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---" for _ in df.columns]) + "|"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    
    table_md = headers + "\n" + sep + "\n" + "\n".join(rows)
    if truncated:
        table_md += f"\n\n*(+{total - max_rows} more rows)*"
    return table_md


# ══════════════════════════════════════════════════════════════════════════════
# CHART GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
class ChartGenerator:
    """
    Generates PNG charts into an isolated session temp directory.

    Guarantees
    ──────────
    • matplotlib.use('Agg')     — no GUI thread required
    • _safe_fig context mgr     — plt.close() always called even on exceptions
    • _verify()                 — only confirmed non-empty files enter charts{}
    • All methods return None   — never raise on chart failure
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or uuid.uuid4().hex
        self.output_dir = Path(tempfile.gettempdir()) / f"insightstream_{self.session_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("ChartGenerator → %s", self.output_dir)
        self.titles = {}

    def detect_chart_targets(self, df: pd.DataFrame):
        """Identify key columns and generate dynamic titles."""
        cm = ColumnMap(df)
        cat_col = cm.category or "Category"
        num_col = cm.numeric or "Value"
        
        self.titles = {
            'distribution': f"Distribution of {num_col}",
            'category_sales': f"{num_col} by {cat_col}",
            'region_sales': f"Total {num_col} by {cm.region or 'Region'}",
            'correlation': "Feature Correlation Heatmap",
            'distribution_title': f"Which {cat_col} has the most records?",
        }
        
        return {
            'category_col': cat_col,
            'numerical_col': num_col,
            'titles': self.titles,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def _safe_fig(self, filename: str):
        """Yield (fig, ax). Save on success. Always call plt.close(fig)."""
        path = self.output_dir / filename
        fig, ax = plt.subplots(figsize=(C.FIG_W, C.FIG_H))
        try:
            yield fig, ax
            fig.savefig(str(path), dpi=C.DPI, bbox_inches="tight", facecolor=C.FACECOLOR)
            log.info("  ✓ %s", filename)
        except Exception as exc:
            log.warning("  ✗ %s — %s", filename, exc)
        finally:
            plt.close(fig)           # ← guaranteed, even on exception

    def _verify(self, filename: str) -> Optional[str]:
        p = self.output_dir / filename
        if p.exists() and p.stat().st_size > 0:
            return str(p)
        log.warning("  ✗ verify failed — %s", filename)
        return None

    def bar_chart(self, df: pd.DataFrame, cat_col: str, val_col: str, 
                  title: str = "Bar Chart", filename: str = "bar_chart.png") -> Optional[str]:
        """Generic bar chart generator used by domain-specific reports."""
        if cat_col not in df.columns or val_col not in df.columns:
            return None
            
        data = df.groupby(cat_col)[val_col].median().sort_values(ascending=False).head(12)
        if data.empty: return None
        
        sns.set_style(C.SNS_STYLE)
        with self._safe_fig(filename) as (fig, ax):
            sns.barplot(x=data.index.astype(str), y=data.values, palette=C.PALETTE, ax=ax)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel(cat_col)
            ax.set_ylabel(f"Median {val_col}")
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()
        return self._verify(filename)

    # ── charts ────────────────────────────────────────────────────────────────

    def _chart_category(self, df: pd.DataFrame, cm: ColumnMap) -> Optional[str]:
        fname = "chart_category.png"
        if not cm.category or not cm.numeric:
            log.warning("  ✗ category chart skipped (category=%r, numeric=%r)", cm.category, cm.numeric)
            return None

        data = (df.groupby(cm.category)[cm.numeric]
                  .sum().sort_values(ascending=False).head(10))
        if data.empty:
            return None

        sns.set_style(C.SNS_STYLE)
        with self._safe_fig(fname) as (fig, ax):
            bp = sns.barplot(x=data.index.astype(str), y=data.values,
                             palette=C.PALETTE, ax=ax)
            ax.set_title(f"Total {cm.numeric} by {cm.category}",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel(cm.category); ax.set_ylabel(cm.numeric)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 1000 else f"{v:,.0f}"))
            plt.xticks(rotation=30, ha="right")
            for bar in bp.patches:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                        f"{h:,.0f}", ha="center", va="bottom", fontsize=8)
        return self._verify(fname)

    def _chart_region(self, df: pd.DataFrame, cm: ColumnMap) -> Optional[str]:
        fname = "chart_region.png"
        if not cm.region or not cm.numeric:
            log.warning("  ✗ region chart skipped (region=%r, numeric=%r)", cm.region, cm.numeric)
            return None

        sns.set_style(C.SNS_STYLE)
        with self._safe_fig(fname) as (fig, ax):
            if cm.category and cm.category != cm.region:
                pivot = (df.groupby([cm.region, cm.category])[cm.numeric]
                           .sum().unstack(cm.category).fillna(0))
                pivot.plot(kind="bar", ax=ax, colormap="tab10", width=0.75)
                ax.set_title(f"{cm.numeric} by {cm.region} & {cm.category}",
                             fontsize=13, fontweight="bold")
                ax.legend(title=cm.category, bbox_to_anchor=(1.01, 1),
                          loc="upper left", fontsize=8)
            else:
                data = df.groupby(cm.region)[cm.numeric].sum().sort_values(ascending=False)
                sns.barplot(x=data.index.astype(str), y=data.values, palette=C.PALETTE, ax=ax)
                ax.set_title(f"Total {cm.numeric} by {cm.region}",
                             fontsize=13, fontweight="bold")
            ax.set_xlabel(cm.region); ax.set_ylabel(cm.numeric)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 1000 else f"{v:,.0f}"))
            plt.xticks(rotation=20, ha="right")
            fig.tight_layout()
        return self._verify(fname)

    def _chart_distribution(self, df: pd.DataFrame, cm: ColumnMap) -> Optional[str]:
        fname = "chart_distribution.png"
        if not cm.numeric:
            return None
        col_data = df[cm.numeric].dropna()
        if len(col_data) < 5 or col_data.nunique() < 3:
            log.warning("  ✗ distribution skipped — insufficient variance")
            return None
        sns.set_style(C.SNS_STYLE)
        with self._safe_fig(fname) as (fig, ax):
            sns.histplot(col_data, bins=30, kde=True, color=C.PURPLE,
                         alpha=0.55, ax=ax, line_kws={"linewidth": 2})
            ax.axvline(col_data.mean(),   color="red",    linestyle="--",
                       linewidth=1.5, label=f"Mean: {col_data.mean():,.0f}")
            ax.axvline(col_data.median(), color="orange", linestyle="--",
                       linewidth=1.5, label=f"Median: {col_data.median():,.0f}")
            ax.set_title(f"Distribution of {cm.numeric}", fontsize=13, fontweight="bold")
            ax.set_xlabel(cm.numeric); ax.set_ylabel("Frequency")
            ax.legend(fontsize=9)
        return self._verify(fname)

    def _chart_correlation(self, df: pd.DataFrame) -> Optional[str]:
        fname = "chart_correlation.png"
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if len(numeric_cols) < 2:
            log.warning("  ✗ correlation skipped — fewer than 2 numeric columns")
            return None
        corr = df[numeric_cols].corr()
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        sns.set_style("white")
        with self._safe_fig(fname) as (fig, ax):
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                        cmap="coolwarm", center=0, linewidths=0.5,
                        ax=ax, annot_kws={"size": 9},
                        cbar_kws={"shrink": 0.8})
            ax.set_title("Numeric Column Correlations", fontsize=13, fontweight="bold")
            plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
        return self._verify(fname)

    def _chart_order_count(self, df: pd.DataFrame, cm: ColumnMap) -> Optional[str]:
        fname = "chart_order_count.png"
        if not cm.category:
            return None
        counts = df[cm.category].value_counts().sort_values()
        palette = ["#A9A9D0"] * len(counts)
        palette[-1] = C.PURPLE
        sns.set_style(C.SNS_STYLE)
        with self._safe_fig(fname) as (fig, ax):
            ax.barh(counts.index.astype(str), counts.values, color=palette[::-1])
            ax.set_title(f"Order Volume by {cm.category}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Order Count")
            for i, (idx, v) in enumerate(counts.items()):
                ax.text(v + counts.max() * 0.01,
                        len(counts) - 1 - i,
                        f"{int(v)}", va="center", fontsize=9)
            ax.set_xlim(0, counts.max() * 1.15)
        return self._verify(fname)

    # ── public ────────────────────────────────────────────────────────────────

    def generate_all(self, df: pd.DataFrame | pl.DataFrame) -> tuple[dict[str, str], ColumnMap]:
        """
        Run all generators. Returns (charts_dict, ColumnMap).
        Only verified paths enter the dict — failed charts are silently omitted.
        """
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
            
        cm = ColumnMap(df)
        generators = {
            "category":    lambda: self._chart_category(df, cm),
            "region":      lambda: self._chart_region(df, cm),
            "distribution":lambda: self._chart_distribution(df, cm),
            "correlation": lambda: self._chart_correlation(df),
            "order_count": lambda: self._chart_order_count(df, cm),
        }
        charts: dict[str, str] = {}
        for key, fn in generators.items():
            result = fn()
            if result:
                charts[key] = result

        log.info("Charts ready: %d/%d — %s", len(charts), len(generators), list(charts.keys()))
        return charts, cm


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
class PDFReportGenerator:

    def __init__(self):
        self.config = TEMPLATE_CONFIGS["modern"]
        self._setup_styles()

    def _setup_styles(self):
        base = getSampleStyleSheet()
        cfg = self.config
        self.S = {
            "Title": ParagraphStyle("RTitle", parent=base["Title"],
                fontSize=20, textColor=colors.HexColor(cfg["purple"]),
                spaceAfter=4, alignment=TA_CENTER, fontName=cfg["font_bold"]),
            "Subtitle": ParagraphStyle("RSub", parent=base["Normal"],
                fontSize=10, textColor=colors.HexColor(C.TEXT_GREY),
                alignment=TA_CENTER, spaceAfter=20, fontName=cfg["font_main"]),
            "Section": ParagraphStyle("RSec", parent=base["Heading2"],
                fontSize=14, textColor=colors.HexColor(cfg["brand_dark"]),
                spaceBefore=14, spaceAfter=8, fontName=cfg["font_bold"]),
            "ChartTitle": ParagraphStyle("RChartTitle", parent=base["Normal"],
                fontSize=11, textColor=colors.HexColor(C.TEXT_DARK),
                fontName=cfg["font_bold"], spaceAfter=6),
            "Insight": ParagraphStyle("RIns", parent=base["Normal"],
                fontSize=9, textColor=colors.HexColor(C.TEXT_DARK),
                backColor=colors.HexColor(cfg["brand_light"]),
                borderPad=8, leftIndent=10, rightIndent=10,
                leading=12, fontName=cfg["font_main"]),
            "Fallback": ParagraphStyle("RFall", parent=base["Normal"],
                fontSize=8, textColor=colors.HexColor(C.TEXT_MUTED),
                fontName=cfg["font_main"]),
            "Normal": base["Normal"],
        }

    def _clean_insights_dict(self, insights: dict) -> dict:
        """Sanitise insight strings — strip raw data dumps before they reach PDF."""
        import re
        cleaned = {}
        for key, val in insights.items():
            s = str(val)
            # Replace space-separated number sequences
            if re.search(r'\b\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+', s):
                s = "[Strategic insight unavailable \u2013 raw data suppressed]"
            # Replace comma-separated number lists
            elif re.search(r'\d+(\.\d+)?,\s*\d+(\.\d+)?,\s*\d+(\.\d+)?,\s*\d+', s):
                s = "[Strategic insight unavailable \u2013 raw data suppressed]"
            # High digit density → data dump
            elif len(s) > 50 and sum(1 for c in s if c in '0123456789.,') / max(len(s), 1) > 0.6:
                s = "[Strategic insight unavailable \u2013 raw data suppressed]"
            elif len(s) > 400:
                s = s[:400] + "\u2026"
            cleaned[key] = s
        return cleaned

    def _is_safe_element(self, el) -> bool:
        """Return False if the element is a Paragraph containing a raw data dump."""
        if not isinstance(el, Paragraph):
            return True

        text = el.getPlainText().strip()
        if len(text) < 80:                     # short texts are fine
            return True

        # Remove common formatting characters that don't change the "data-ness"
        cleaned = text.replace(",", " ").replace(".", " ")   # integers like "1." → "1 "

        # Count numeric tokens (sequences of digits, possibly with a decimal point)
        tokens = cleaned.split()                              # split on whitespace
        numeric_tokens = [t for t in tokens if re.sub(r'[^\d.]', '', t).replace('.', '').isdigit()]

        ratio = len(numeric_tokens) / max(len(tokens), 1)

        # If > 70% of tokens are numbers and the string is long, it's a dump
        if ratio > 0.7:
            log.warning("BLOCKED unsafe element: %s...", text[:80])
            return False
        return True

    def embed_chart_safely(self, elements: list, chart_path: Optional[str],
                           title: str, insight: str) -> None:
        """Triple-guard chart embedding — never raises, never crashes the PDF build."""
        elements.append(Paragraph(title, self.S["ChartTitle"]))

        if not chart_path:
            elements.append(Paragraph(
                "⚠ Chart skipped — required column not found in dataset.",
                self.S["Fallback"]))
            elements.append(Spacer(1, 12)); return

        if not os.path.exists(chart_path):
            elements.append(Paragraph(
                f"⚠ Chart file missing: {chart_path}", self.S["Fallback"]))
            elements.append(Spacer(1, 12)); return

        if os.path.getsize(chart_path) == 0:
            elements.append(Paragraph(
                "⚠ Chart file is empty (render error).", self.S["Fallback"]))
            elements.append(Spacer(1, 12)); return

        try:
            img = RLImage(chart_path, width=C.SAFE_IMG_W, height=C.SAFE_IMG_H)
            elements.append(img)
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(f"📊  {insight}", self.S["Insight"]))
        except Exception as exc:
            log.error("ReportLab failed loading %s: %s", chart_path, exc)
            elements.append(Paragraph(f"⚠ Render error: {exc}", self.S["Fallback"]))

        elements.append(Spacer(1, 22))

    @staticmethod
    def _normalize_kpis(kpis):
        """Coerce arbitrary KPI payload shapes into a flat {label: display_string}."""
        if not isinstance(kpis, dict):
            return {}
        out = {}
        for raw_key, raw_val in kpis.items():
            if isinstance(raw_val, dict):
                display = raw_val.get("formatted")
                if display is None:
                    v = raw_val.get("value")
                    display = "—" if v is None else (
                        f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
                    )
                label = raw_val.get("name") or str(raw_key).replace("_", " ").title()
            elif isinstance(raw_val, (list, tuple, set)):
                display = f"{len(raw_val)} item(s)"
                label = str(raw_key).replace("_", " ").title()
            else:
                display = str(raw_val)
                label = str(raw_key).replace("_", " ").title() if raw_key.islower() else str(raw_key)
            if display.startswith("{") and display.endswith("}"):
                display = "—"
            out[label] = display
        return out

    def _kpi_table(self, kpis: dict) -> Table:
        kpis = self._normalize_kpis(kpis)
        if not kpis:
            kpis = {"Status": "No KPI data"}
        col_w = (C.PAGE_W - 2 * C.MARGIN) / max(len(kpis), 1)
        cfg = self.config
        hdr = [Paragraph(k, ParagraphStyle("kh", fontName=cfg["font_bold"],
               fontSize=8, textColor=colors.white, alignment=TA_CENTER))
               for k in kpis]
        val = [Paragraph(str(v), ParagraphStyle("kv", fontName=cfg["font_bold"],
               fontSize=15, textColor=colors.HexColor(cfg["brand_dark"]),
               alignment=TA_CENTER)) for v in kpis.values()]
        tbl = Table([hdr, val], colWidths=[col_w] * len(kpis))
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor(cfg["purple"])),
            ("BACKGROUND",    (0, 1), (-1, 1),  colors.HexColor(cfg["brand_light"])),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor(C.RULE_GREY)),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return tbl

    def build(self, df: pd.DataFrame, charts: dict[str, str],
              insights: dict[str, str], output_path: str,
              cm: Optional[ColumnMap] = None,
              title: str = "InsightStream Analytics Report",
              template: str = "modern",
              domain_template: Optional[dict] = None) -> str:

        self.config = TEMPLATE_CONFIGS.get(template, TEMPLATE_CONFIGS["modern"])
        # Refresh styles with new config
        self._setup_styles()

        # ── Sanitise insights before they reach any Paragraph ────────────
        insights = self._clean_insights_dict(insights)

        doc = SimpleDocTemplate(output_path, pagesize=A4,
            leftMargin=C.MARGIN, rightMargin=C.MARGIN,
            topMargin=C.MARGIN,  bottomMargin=C.MARGIN)
        elements: list = []
        
        # VERSION BANNER FOR VERIFICATION
        elements.append(Paragraph("BUILD METHOD VERSION: 2025-04-25-clean", self.S["Normal"]))
        elements.append(Spacer(1, 10))

        # 1. Domain Detection & Asset Prep
        target_metric = domain_template.get("target_metric", "Value") if domain_template else "Value"
        final_title = domain_template.get("report_title", title) if domain_template else title
        
        cg = ChartGenerator()
        
        # Region Detection Logic (Fix for missing region detail)
        region_col = cm.region if cm and cm.region else get_region_column(df)
        
        # Override or supplement charts if we have a region column and a target metric
        if region_col and target_metric in df.columns:
            region_chart = cg.bar_chart(
                df, region_col, target_metric,
                title=f"Median {target_metric} by {region_col}",
                filename="region_target_median.png"
            )
            if region_chart:
                charts["region_target"] = region_chart
                
            # Prepare region stats for the markdown table
            region_stats_df = df.groupby(region_col)[target_metric].median().reset_index()
            region_stats_df.columns = [region_col, f"Median {target_metric}"]
            md_table = generate_markdown_table(region_stats_df)
        else:
            md_table = ""

        # 2. Header
        elements.append(Paragraph(final_title, self.S["Title"]))
        elements.append(Paragraph(
            f"Authored by InsightStream AI  •  {date.today().strftime('%m/%d/%Y')}",
            self.S["Subtitle"]))
        elements.append(HRFlowable(width="100%", thickness=1,
                                   color=colors.HexColor(C.RULE_GREY)))
        elements.append(Spacer(1, 14))

        # 3. KPIs
        kpis = self._derive_kpis(df, cm)
        if kpis:
            elements.append(Paragraph("Key Metrics", self.S["Section"]))
            elements.append(self._kpi_table(kpis))
            elements.append(Spacer(1, 20))

        # 4. Regional Table (New)
        if md_table:
            elements.append(Paragraph(f"Regional {target_metric} Distribution", self.S["Section"]))
            # Convert MD table to a ReportLab Table
            lines = md_table.split("\n")
            table_data = [line.strip("|").split("|") for line in lines if "---" not in line]
            table_data = [[cell.strip() for cell in row] for row in table_data]
            
            if table_data:
                t = Table(table_data, hAlign='LEFT')
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor(self.config["brand_dark"])),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), self.config["font_bold"]),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor(self.config["brand_light"])),
                    ('GRID', (0,0), (-1,-1), 1, colors.HexColor(C.RULE_GREY))
                ]))
                elements.append(t)
                elements.append(Spacer(1, 20))

        # 5. Charts
        elements.append(Paragraph("Visual Analysis", self.S["Section"]))
        elements.append(HRFlowable(width="100%", thickness=0.5,
                                   color=colors.HexColor(C.RULE_LIGHT)))
        elements.append(Spacer(1, 10))

        chart_list = [
            ("region_target", f"Strategic Distribution: {target_metric}"),
            ("category",      f"Total {target_metric} by Category"),
            ("distribution",  f"Distribution Analysis: {target_metric}"),
            ("correlation",   "Inter-Variable Correlation Matrix"),
        ]
        
        for key, chart_title in chart_list:
            if key in charts:
                self.embed_chart_safely(
                    elements,
                    charts.get(key),
                    chart_title,
                    insights.get(key, "Strategic AI narrative pending for this segment.")
                )

        # ── Final safety pass: strip any raw-numeric Paragraph elements ──
        elements = [el for el in elements if self._is_safe_element(el)]

        doc.build(elements)
        log.info("PDF written → %s  (%.1f KB)",
                 output_path, os.path.getsize(output_path) / 1024)
        return output_path


    @staticmethod
    def _derive_kpis(df: pd.DataFrame, cm: Optional[ColumnMap]) -> dict:
        kpis: dict = {}
        if cm and cm.numeric and cm.numeric in df.columns:
            total = df[cm.numeric].sum()
            kpis[f"Total {cm.numeric}"] = f"${total:,.0f}" if total > 100 else f"{total:,.2f}"
            kpis[f"Avg {cm.numeric}"]   = f"${df[cm.numeric].mean():,.0f}"
        if cm and cm.numeric2 and cm.numeric2 in df.columns:
            kpis[f"Total {cm.numeric2}"] = f"{df[cm.numeric2].sum():,.0f}"
        if cm and cm.category and cm.category in df.columns:
            kpis[f"{cm.category}s"] = df[cm.category].nunique()
        kpis["Records"] = f"{len(df):,}"
        return kpis


class UnifiedReportGenerator(PDFReportGenerator):
    """
    Assembles a professional PDF from pre-rendered assets (e.g. Plotly images from frontend).
    Ensures 1:1 visual parity with the dashboard by embedding exact frontend state.
    """

    def _decode_image(self, base64_str: str, session_id: str) -> Optional[str]:
        """Decode base64 image to a temporary file for ReportLab."""
        import base64
        try:
            if not base64_str: return None
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            
            data = base64.b64decode(base64_str)
            temp_dir = Path(tempfile.gettempdir()) / f"insightstream_export_{session_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            fname = f"chart_{uuid.uuid4().hex[:8]}.png"
            fpath = temp_dir / fname
            with open(fpath, "wb") as f:
                f.write(data)
            return str(fpath)
        except Exception as exc:
            log.error("Base64 decode failed: %s", exc)
            return None

    def build_from_assets(self, 
                          output_path: str,
                          charts: list[dict], # [{id, title, image_base64, error, insight}]
                          kpis: dict,
                          ai_summary: str = "",
                          insights: list[str] = [],
                          text_blocks: list[dict] = [],
                          title: str = "Executive Intelligence Report",
                          project_name: str = "InsightStream",
                          template: str = "modern",
                          session_id: str = "default",
                          df: Optional[pd.DataFrame | pl.DataFrame] = None,
                          domain_id: str = "general") -> str:
        """Construct a structured multi-page PDF with domain-aware visuals and narratives."""
        if df is not None and hasattr(df, "to_pandas"):
            df = df.to_pandas()
            
        self.config = TEMPLATE_CONFIGS.get(template, TEMPLATE_CONFIGS["modern"])
        self._setup_styles()
        
        domain_template = TEMPLATES.get(domain_id, TEMPLATES["general"])
        target_metric = domain_template["target_metric"]
        final_title = domain_template.get("report_title", title)

        doc = SimpleDocTemplate(output_path, pagesize=A4,
            leftMargin=C.MARGIN, rightMargin=C.MARGIN,
            topMargin=C.MARGIN,  bottomMargin=C.MARGIN)
        elements: list = []

        # 1. PAGE 1: TITLE PAGE
        elements.append(Spacer(1, 2 * inch))
        elements.append(Paragraph(project_name.upper(), self.S["Section"]))
        elements.append(Paragraph(final_title, self.S["Title"]))
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(
            f"Official Strategic Analysis  •  {date.today().strftime('%B %d, %Y')}",
            self.S["Subtitle"]))
        elements.append(PageBreak())

        # 2. PAGE 2: EXECUTIVE SUMMARY & KPIs
        elements.append(Paragraph(domain_template.get("executive_summary_header", "Executive Overview"), self.S["Section"]))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(C.RULE_GREY)))
        elements.append(Spacer(1, 20))

        if kpis:
            elements.append(Paragraph("Key Performance Indicators", self.S["ChartTitle"]))
            elements.append(self._kpi_table(kpis))
            elements.append(Spacer(1, 30))

        if ai_summary:
            elements.append(Paragraph("AI Intelligence Brief", self.S["ChartTitle"]))
            elements.append(Paragraph(ai_summary, self.S["Insight"]))
            elements.append(Spacer(1, 20))

        # 3. PAGE 3: REGIONAL ANALYSIS (Rendered if DF provided)
        if df is not None:
            region_col = get_region_column(df)
            if region_col and target_metric in df.columns:
                elements.append(PageBreak())
                elements.append(Paragraph(domain_template.get("regional_chart_title", "Regional Breakdown"), self.S["Section"]))
                elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(C.RULE_GREY)))
                elements.append(Spacer(1, 20))
                
                # Generate actual chart image
                cg = ChartGenerator()
                chart_path = cg.bar_chart(
                    df, region_col, target_metric,
                    title=domain_template.get("regional_chart_title"),
                    filename=f"reg_{session_id}.png"
                )
                if chart_path:
                    self.embed_chart_safely(elements, chart_path, 
                                            f"Strategic Distribution: {target_metric}", 
                                            f"Analysis of {target_metric} variance across identified regional clusters.")
                
                # Add Markdown Table
                region_stats_df = df.groupby(region_col)[target_metric].median().reset_index()
                region_stats_df.columns = [region_col, f"Median {target_metric}"]
                if pd.api.types.is_numeric_dtype(region_stats_df[f"Median {target_metric}"]):
                    region_stats_df[f"Median {target_metric}"] = region_stats_df[f"Median {target_metric}"].round(2)
                md_table = generate_markdown_table(region_stats_df)
                if md_table:
                    elements.append(Spacer(1, 20))
                    elements.append(Paragraph(f"Regional {target_metric} Statistics", self.S["ChartTitle"]))
                    lines = md_table.split("\n")
                    table_data = [line.strip("|").split("|") for line in lines if "---" not in line]
                    table_data = [[cell.strip() for cell in row] for row in table_data]
                    if table_data:
                        t = Table(table_data, hAlign='LEFT')
                        t.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), colors.HexColor(self.config["brand_dark"])),
                            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                            ('GRID', (0,0), (-1,-1), 1, colors.HexColor(C.RULE_GREY))
                        ]))
                        elements.append(t)

        elements.append(PageBreak())

        # 4. PAGE 4: STRATEGIC FINDINGS & NOTES
        if insights or text_blocks:
            elements.append(Paragraph("Strategic Findings & Key Results", self.S["Section"]))
            elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(C.RULE_GREY)))
            elements.append(Spacer(1, 20))

            if insights:
                for ins in insights:
                    # Clean up any potential raw data strings in insights
                    clean_ins = str(ins).split('[')[0].split('{')[0] if '[' in str(ins) or '{' in str(ins) else str(ins)
                    elements.append(Paragraph(f"• {clean_ins}", self.S["Insight"]))
                    elements.append(Spacer(1, 8))
                elements.append(Spacer(1, 20))

            if text_blocks:
                elements.append(Paragraph("Expert Annotations", self.S["ChartTitle"]))
                for block in text_blocks:
                    content = block.get("content", "")
                    if content:
                        elements.append(Paragraph(content, self.S["Insight"]))
                        elements.append(Spacer(1, 12))
            
            elements.append(PageBreak())

        # 5. PAGE 5+: VISUAL ANALYSIS (Frontend Captures)
        elements.append(Paragraph("Detailed Dashboard Visualizations", self.S["Section"]))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(C.RULE_GREY)))
        elements.append(Spacer(1, 20))

        valid_charts = 0
        general_placeholder = TEMPLATES["general"]["target_metric"]
        should_substitute = target_metric != general_placeholder

        for i, chart in enumerate(charts):
            img_path = self._decode_image(chart.get("image_base64", ""), session_id)
            err = chart.get("error")
            chart_title = chart.get("title", "Visualization Segment")
            chart_insight = chart.get("insight", "Segmented data analysis.")
            if should_substitute:
                chart_title = chart_title.replace(general_placeholder, target_metric)
                chart_insight = chart_insight.replace(general_placeholder, target_metric)
            if err:
                if "{" not in str(err) and "[" not in str(err):
                    elements.append(Paragraph(f"⚠ {chart_title}: {err}", self.S["Fallback"]))
                    elements.append(Spacer(1, 20))
            else:
                self.embed_chart_safely(elements, img_path, chart_title, chart_insight)
                valid_charts += 1
            if (valid_charts > 0 and valid_charts % 2 == 0):
                elements.append(PageBreak())

        # ── Final safety pass: strip any raw-numeric Paragraph elements ──
        elements = [el for el in elements if self._is_safe_element(el)]

        doc.build(elements)
        log.info("Multi-page Unified PDF written → %s", output_path)
        return output_path


# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════════════════
def cleanup_temp_files(*paths: str) -> None:
    """Delete temp dirs/files AFTER FileResponse is transmitted (BackgroundTasks)."""
    import shutil
    for path in paths:
        if not path: continue
        try:
            if   os.path.isdir(path):  shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path): os.remove(path)
            log.info("Cleaned → %s", path)
        except Exception as exc:
            log.warning("Cleanup failed %s: %s", path, exc)
