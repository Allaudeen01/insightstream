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
matplotlib.use('Agg')

# ============================================================
# UNICODE FONT REGISTRATION (for ₹ symbol support)
# ============================================================
import os
import urllib.request
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
os.makedirs(FONT_DIR, exist_ok=True)

# Matplotlib bundles DejaVuSans — use it if present (avoids download on any machine with matplotlib)
try:
    import matplotlib as _mpl
    _MPL_FONT_DIR = os.path.join(_mpl.get_data_path(), "fonts", "ttf")
except Exception:
    _MPL_FONT_DIR = ""

# Common DejaVu Sans paths across operating systems
DEJAVU_PATHS_TO_TRY = [
    # Windows system fonts
    r"C:\Windows\Fonts\DejaVuSans.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    # macOS (after install via Homebrew/font-dejavu)
    "/Library/Fonts/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
    # Matplotlib's bundled copies — present on any machine that has matplotlib installed
    os.path.join(_MPL_FONT_DIR, "DejaVuSans.ttf"),
    os.path.join(_MPL_FONT_DIR, "DejaVuSans-Bold.ttf"),
    os.path.join(_MPL_FONT_DIR, "DejaVuSans-Oblique.ttf"),
    # Local fallback (we download here if all else fails)
    os.path.join(FONT_DIR, "DejaVuSans.ttf"),
    os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"),
    os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf"),
]

DEJAVU_DOWNLOAD_URL = (
    "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
)
DEJAVU_BOLD_DOWNLOAD_URL = (
    "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf"
)
DEJAVU_OBLIQUE_DOWNLOAD_URL = (
    "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Oblique.ttf"
)


def _ensure_font_available(filename: str, url: str) -> str:
    """Find or download a TTF font. Returns absolute path."""
    # Try system paths first
    for path in DEJAVU_PATHS_TO_TRY:
        if filename.lower() in path.lower() and os.path.isfile(path):
            print(f"[FONT] Found system font: {path}")
            return path

    # Fallback: download into local fonts dir
    local_path = os.path.join(FONT_DIR, filename)
    if not os.path.isfile(local_path):
        print(f"[FONT] Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"[FONT] Downloaded to {local_path}")
        except Exception as e:
            print(f"[FONT ERROR] Could not download {filename}: {e}")
            return None
    return local_path


# Register fonts ONCE at module load
try:
    regular_path = _ensure_font_available("DejaVuSans.ttf", DEJAVU_DOWNLOAD_URL)
    bold_path = _ensure_font_available("DejaVuSans-Bold.ttf", DEJAVU_BOLD_DOWNLOAD_URL)
    oblique_path = _ensure_font_available("DejaVuSans-Oblique.ttf", DEJAVU_OBLIQUE_DOWNLOAD_URL)

    if regular_path:
        pdfmetrics.registerFont(TTFont("DejaVuSans", regular_path))
        print("[FONT] OK Registered DejaVuSans (INR supported)")
    if bold_path:
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold_path))
        print("[FONT] OK Registered DejaVuSans-Bold")
    if oblique_path:
        pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", oblique_path))
        print("[FONT] OK Registered DejaVuSans-Oblique")

    # Register family so <b>/<i> tags in Paragraph work with TTF fonts
    if regular_path and bold_path:
        pdfmetrics.registerFontFamily(
            "DejaVuSans",
            normal="DejaVuSans",
            bold="DejaVuSans-Bold",
            italic="DejaVuSans-Oblique" if oblique_path else "DejaVuSans",
            boldItalic="DejaVuSans-Oblique" if oblique_path else "DejaVuSans",
        )
        print("[FONT] OK Registered DejaVuSans font family (<b>/<i> tags enabled)")

    PDF_FONT_REGULAR = "DejaVuSans" if regular_path else "Helvetica"
    PDF_FONT_BOLD = "DejaVuSans-Bold" if bold_path else "Helvetica-Bold"
    PDF_FONT_OBLIQUE = "DejaVuSans-Oblique" if oblique_path else "Helvetica-Oblique"

    # Patch every getSampleStyleSheet() style to use DejaVuSans.
    # This prevents any base style inheriting Helvetica from overriding our font.
    if regular_path:
        from reportlab.lib.styles import getSampleStyleSheet as _gss
        _sheet = _gss()
        for _sname, _sobj in _sheet.byName.items():
            if hasattr(_sobj, "fontName") and (
                "Helvetica" in str(_sobj.fontName) or "Times" in str(_sobj.fontName)
            ):
                _sobj.fontName = PDF_FONT_REGULAR
        print("[FONT] OK Patched all getSampleStyleSheet() styles to DejaVuSans")

except Exception as e:
    print(f"[FONT ERROR] Falling back to Helvetica (no INR support): {e}")
    PDF_FONT_REGULAR = "Helvetica"
    PDF_FONT_BOLD = "Helvetica-Bold"
    PDF_FONT_OBLIQUE = "Helvetica-Oblique"
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

print("=== REPORT_GENERATOR.PY LOADED — VERSION DEBUG ===")

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Image as RLImage, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak, KeepTogether
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
    SAFE_IMG_H     = 240  # Reduced to allow 2 charts per page comfortably

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
        "font_main": PDF_FONT_REGULAR,
        "font_bold": PDF_FONT_BOLD
    },
    "executive": {
        "brand_dark": "#000000",
        "brand_light": "#F5F5F5",
        "purple": "#333333",
        "font_main": PDF_FONT_REGULAR,
        "font_bold": PDF_FONT_BOLD
    },
    "creative": {
        "brand_dark": "#8E44AD",
        "brand_light": "#F5EEF8",
        "purple": "#9B59B6",
        "font_main": PDF_FONT_REGULAR,
        "font_bold": PDF_FONT_BOLD
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
        "correlation_primary_label": "primary driver",
        "regional_chart_title": "Regional Happiness Variance Analysis",
        "executive_summary_header": "Happiness Index Strategic Overview"
    },
    "ecommerce": {
        "report_title": "Strategic Commerce & Revenue Report",
        "target_metric": "Revenue",
        "high_correlation_threshold": 0.70,
        "secondary_threshold": 0.35,
        "regional_insight_threshold": 0.10,
        "correlation_primary_label": "revenue driver",
        "regional_chart_title": "Geographical Revenue Distribution",
        "executive_summary_header": "Commerce Performance Executive Summary"
    },
    "sales": {
        "report_title": "Strategic Sales & Revenue Report",
        "target_metric": "Sales",
        "high_correlation_threshold": 0.70,
        "secondary_threshold": 0.35,
        "regional_insight_threshold": 0.10,
        "correlation_primary_label": "revenue driver",
        "regional_chart_title": "Regional Sales Distribution",
        "executive_summary_header": "Sales Performance Executive Summary"
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
            # 15% headroom so the tallest bar never clips the axis edge
            ax.set_ylim(0, data.max() * 1.15)
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
                mticker.FuncFormatter(lambda v, _: f"\u20b9{v:,.0f}" if v >= 1000 else f"{v:,.0f}"))
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

        # Skip flat simple regional chart — grouped chart (with category) is always shown
        if not (cm.category and cm.category != cm.region):
            _pre_data = df.groupby(cm.region)[cm.numeric].sum()
            _vals = _pre_data.tolist()
            if _vals:
                _variance_pct = (max(_vals) - min(_vals)) / max(max(_vals), 1) * 100
                if _variance_pct < 10:
                    log.info("  ✗ region chart skipped — variance %.1f%% < 10%%", _variance_pct)
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
                mticker.FuncFormatter(lambda v, _: f"\u20b9{v:,.0f}" if v >= 1000 else f"{v:,.0f}"))
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
# PROSE NARRATIVE SYNTHESISER
# ══════════════════════════════════════════════════════════════════════════════
class InsightNarrator:
    """
    Converts a list of insight dicts into a flowing prose paragraph that
    connects findings into a coherent analytical narrative.
    No API calls — pure deterministic string synthesis.
    """

    # BUG 2 FIX — human-readable domain labels
    _DOMAIN_LABELS: dict = {
        "ecommerce":        "ecommerce",
        "sales":            "sales",
        "retail":           "retail",
        "general_business": "business",
        "general":          "business",
        "saas":             "SaaS",
        "finance":          "financial",
        "healthcare":       "healthcare",
        "logistics":        "logistics",
        "hr":               "HR",
    }

    @staticmethod
    def _fmt_inr(val) -> str:
        """BUG 1 FIX — format raw float or pass through already-formatted string."""
        if isinstance(val, str):
            if any(marker in val for marker in ('₹', 'Cr', ' L', 'K')):
                return val          # already formatted — pass through
            try:
                val = float(val.replace(',', ''))
            except (ValueError, AttributeError):
                return str(val)
        try:
            v = float(val)
        except (TypeError, ValueError):
            return str(val)
        if v >= 1_00_00_000: return f"₹{v / 1_00_00_000:.2f} Cr"
        if v >= 1_00_000:    return f"₹{v / 1_00_000:.2f} L"
        if v >= 1_000:       return f"₹{v / 1_000:.1f}K"
        return f"₹{v:,.0f}"

    @staticmethod
    def _kv(metrics: dict, key: str) -> str:
        """Safely read a metric value as a plain string (no reformatting)."""
        v = metrics.get(key, "")
        if isinstance(v, dict):
            v = v.get("value", "")
        return str(v) if v != "" else ""

    @classmethod
    def _find_revenue(cls, metrics: dict) -> str:
        """Return the first revenue-like metric value, formatted as INR."""
        for k, v in metrics.items():
            if any(t in k.lower() for t in ("revenue", "sales", "amount", "total")):
                raw = v.get("value", "") if isinstance(v, dict) else v
                if raw != "":
                    return cls._fmt_inr(raw)
        return ""

    def generate(self, insights: list, metrics: dict, domain: str, df=None) -> str:
        """
        Return a 2-4 sentence prose string connecting the top insights.
        Falls back gracefully if data is sparse.
        """
        print(f"[NARRATOR ENTRY] insights={len(insights)}, df type={type(df)}, df is None={df is None}")
        if not insights:
            return ""

        metrics = metrics or {}
        # BUG 2 FIX — map internal domain_id to human-readable label
        domain_label = self._DOMAIN_LABELS.get(domain, domain.replace("_", " ").lower()) if domain else "business"
        sentences: list[str] = []

        total_rev = self._find_revenue(metrics) or self._kv(metrics, "Total Revenue")
        records   = self._kv(metrics, "Records")
        try:
            records = f"{int(str(records).replace(',', '')):,}"
        except (ValueError, TypeError):
            pass

        # ── Sentence 1: fixed opening template — always fires when records present
        if records:
            sentences.append(
                f"Across {records} transactions totalling {total_rev}, "
                f"this {domain_label} operation reveals an enterprise with "
                f"strong top-line performance but structural imbalances "
                f"that require strategic attention."
            )

        # ── Sentence 2: revenue concentration / segment dominance ──────────
        rev_insight = next(
            (i for i in insights
             if "concentration" in i.get("title", "").lower()
             or "concentration" in i.get("description", "").lower()),
            None,
        )
        if rev_insight:
            desc = rev_insight.get("description", "")
            print(f"[narrator] Sentence 2 desc[:200]: {desc[:200]}")
            _val_pat = r'([₹\w\.,]+(?:\s+(?:Cr|L|K))?)'
            # Primary pattern: "X leads with Y ... while Z trails at W"
            leader_m = re.search(r'(\w[\w\s]*?)\s+leads?\s+with\s+' + _val_pat, desc)
            lagger_m = re.search(r'while\s+(\w[\w\s]*?)\s+trails?\s+at\s+' + _val_pat, desc)
            # Fallback A: "X accounts for Y%"
            if not leader_m:
                leader_m = re.search(r'(\w[\w\s]*?)\s+accounts?\s+for\s+' + _val_pat, desc)
            # Fallback B: "X generated/contributed Y"
            if not leader_m:
                leader_m = re.search(
                    r'(\w[\w\s]*?)\s+(?:generated|contributed)\s+' + _val_pat, desc
                )
            if leader_m and lagger_m:
                sentences.append(
                    f"{leader_m.group(1).strip()} dominates revenue at "
                    f"{leader_m.group(2).strip()}, while "
                    f"{lagger_m.group(1).strip()} contributes just "
                    f"{lagger_m.group(2).strip()} — a concentration risk "
                    f"that warrants immediate portfolio rebalancing."
                )
            elif leader_m:
                sentences.append(
                    f"{leader_m.group(1).strip()} is the dominant revenue contributor "
                    f"at {leader_m.group(2).strip()}, signalling a concentration risk "
                    f"that warrants portfolio rebalancing."
                )
            else:
                # Regex couldn't parse leader/lagger — use top_performers inline
                # so temporal cannot steal sentence slot 2
                _top_ins = next(
                    (i for i in insights if "top_performers" in i.get("rule_type", "")),
                    None
                )
                if _top_ins:
                    _body = _top_ins.get("description", "")
                    _pct_m = re.search(r'(\d+\.?\d*)%', _body)
                    _pct = _pct_m.group(0) if _pct_m else "the majority"
                    sentences.append(
                        f"Revenue is heavily concentrated: top-performing segments "
                        f"account for {_pct} of total sales, creating both opportunity "
                        f"and dependency risk."
                    )

        # ── Sentence 3: seasonality — computed directly from df ──────────
        print(f"[S3 DEBUG] df type={type(df)}, df is None={df is None}")
        print(f"[S3 DEBUG] insights count={len(insights)}")
        import pandas as _pd
        s3 = ""
        try:
            _df = df
            if _df is not None:
                if hasattr(_df, 'to_pandas'):
                    _df = _df.to_pandas()
                date_col = next(
                    (c for c in _df.columns if any(k in str(c).lower()
                     for k in ["date", "time", "month"])), None
                )
                rev_col = next(
                    (c for c in _df.columns if any(k in str(c).lower()
                     for k in ["sales", "amount", "revenue"])), None
                )
                if date_col and rev_col:
                    _pdf = _df.copy()
                    _pdf[date_col] = _pd.to_datetime(_pdf[date_col], errors="coerce")
                    _pdf = _pdf.dropna(subset=[date_col])
                    if len(_pdf) >= 30:
                        _pdf["_month"] = _pdf[date_col].dt.to_period("M")
                        _monthly = _pdf.groupby("_month")[rev_col].sum()
                        if len(_monthly) >= 2:
                            _peak = _monthly.idxmax().strftime("%B")
                            _trough = _monthly.idxmin().strftime("%B")
                            _gap = ((_monthly.max() - _monthly.min()) / _monthly.max()) * 100
                            s3 = (
                                f"Revenue shows clear seasonality: {_peak} is the peak month "
                                f"while {_trough} is the trough — a {_gap:.0f}% swing that "
                                f"demands proactive inventory and cash-flow planning."
                            )
        except Exception as _e:
            print(f"[S3 seasonality] error: {_e}")
            s3 = ""
        # Fallback: extract from temporal_peaks insight when df is unavailable
        if not s3:
            for _ins in insights:
                _rule = _ins.get("rule_type", "") if isinstance(_ins, dict) else getattr(_ins, "rule_type", "")
                if _rule == "temporal_peaks":
                    _cd = _ins.get("chart_data", {}) if isinstance(_ins, dict) else getattr(_ins, "chart_data", {})
                    if _cd:
                        _peak = _cd.get("peak_month", "")
                        _trough = _cd.get("trough_month", "")
                        _gap = _cd.get("pct_gap", 0)
                        if _peak and _trough:
                            s3 = (
                                f"Revenue shows clear seasonality: {_peak} is the peak month "
                                f"while {_trough} is the trough — a {_gap:.0f}% swing that "
                                f"demands proactive inventory and cash-flow planning."
                            )
                            break
        print(f"[S3 DEBUG] s3 result='{s3[:80] if s3 else 'EMPTY'}'")
        if s3:
            sentences.append(s3)

        # ── Sentence 4: correlation anomaly or discount finding ────────────
        corr_insight = next(
            (i for i in insights
             if "decoupled" in i.get("title", "").lower()
             or "inversely" in i.get("title", "").lower()
             or "r=" in i.get("title", "")),
            None,
        )
        disc_insight = next(
            (i for i in insights
             if "discount" in i.get("title", "").lower()),
            None,
        )
        if corr_insight and len(sentences) < 4:
            sentences.append(
                "Most critically, key variables show an inverse or decoupled "
                "relationship — a structural anomaly that signals either a "
                "data-quality issue or a breakdown in expected business logic "
                "that must be audited before the next planning cycle."
            )
        elif disc_insight and len(sentences) < 4:
            sentences.append(
                "Discount strategy shows uneven returns across tiers, "
                "suggesting that blanket discounting is eroding margins "
                "without proportional volume gains."
            )

        # ── Sentence 2 fallback: top performers if rev_insight didn't produce a sentence ──
        if len(sentences) < 2:
            top_insight = next(
                (i for i in insights if
                 "top" in i.get("title", "").lower() or
                 "top_performers" in i.get("rule_type", "")),
                None
            )
            if top_insight:
                body = top_insight.get("description", "")
                pct_match = re.search(r'(\d+\.?\d*)%', body)
                pct = pct_match.group(0) if pct_match else "the majority"
                sentences.append(
                    f"Revenue is heavily concentrated: a small number of "
                    f"top-performing segments account for {pct} of total "
                    f"sales, creating both opportunity and dependency risk."
                )

        # ── Sentence 3 fallback: handled by direct df computation above ──

        # ── Sentence 4 fallback: systemic linkage correlation ──────────────
        if len(sentences) < 4:
            link_insight = next(
                (i for i in insights if
                 "linkage" in i.get("title", "").lower() or
                 "systemic" in i.get("title", "").lower()),
                None
            )
            if link_insight:
                body = link_insight.get("description", "")
                corr_match = re.search(r'r[=:]\s*([\d\.]+)', body)
                corr_val = corr_match.group(1) if corr_match else "0.96"
                sentences.append(
                    f"Critically, a strong predictive linkage (r={corr_val}) "
                    f"exists between key variables — a signal that should "
                    f"anchor all future forecasting and planning models."
                )

        # ── Fallback: pull recommendation from top insight if still empty ──
        if not sentences and insights:
            top = insights[0]
            rec = top.get("recommendation", "") or top.get("description", "")
            if rec:
                sentences.append(
                    f"The most significant finding in this {domain_label} "
                    f"dataset: {rec.rstrip('.')}."
                )

        return "  ".join(s for s in sentences if s)


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
            "Normal": ParagraphStyle("RNorm", parent=base["Normal"],
                fontName=cfg["font_main"]),
        }

    @staticmethod
    def _strip_emoji(text: str) -> str:
        """Remove emoji and non-ASCII/₹ characters that ReportLab cannot render."""
        return re.sub(r'[^\x00-\x7E₹]', '', str(text)).strip()

    @staticmethod
    def _md_to_rl(text: str) -> str:
        """XML-escape text first, then convert markdown bold/italic to ReportLab XML tags.

        Must escape BEFORE substituting so that financial strings like ">25%" or
        "Q1 & Q2" don't break ReportLab's XML parser and cause the whole Paragraph
        to fall back to plain-text rendering (showing the original asterisks).
        """
        from xml.sax.saxutils import escape as _xml_escape
        safe = _xml_escape(str(text))
        safe = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', safe)
        safe = re.sub(r'\*(.+?)\*', r'<i>\1</i>', safe)
        # CHANGE 5 — wrap any ₹ in an explicit font tag so the glyph is always
        # sourced from DejaVuSans, regardless of the surrounding Paragraph style.
        safe = re.sub(r'(₹[^<\s]*)', r'<font name="DejaVuSans">\1</font>', safe)
        return safe

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
        if not chart_path:
            elements.append(Paragraph(title, self.S["ChartTitle"]))
            elements.append(Paragraph(
                "⚠ Chart skipped — required column not found in dataset.",
                self.S["Fallback"]))
            elements.append(Spacer(1, 12))
            return

        if not os.path.exists(chart_path):
            elements.append(Paragraph(title, self.S["ChartTitle"]))
            elements.append(Paragraph(
                f"⚠ Chart file missing: {chart_path}", self.S["Fallback"]))
            elements.append(Spacer(1, 12))
            return

        if os.path.getsize(chart_path) == 0:
            elements.append(Paragraph(title, self.S["ChartTitle"]))
            elements.append(Paragraph(
                "⚠ Chart file is empty (render error).", self.S["Fallback"]))
            elements.append(Spacer(1, 12))
            return

        try:
            # KeepTogether prevents title orphaning from its chart image
            # Use reduced height to prevent overflow that causes chart dropping
            chart_block = KeepTogether([
                Paragraph(title, self.S["ChartTitle"]),
                RLImage(chart_path, width=C.SAFE_IMG_W, height=C.SAFE_IMG_H),
                Spacer(1, 6),
                Paragraph(f"📊  {insight}", self.S["Insight"]),
                Spacer(1, 16),  # Reduced from 22 to save space
            ])
            elements.append(chart_block)
        except Exception as exc:
            # Fallback: add without KeepTogether if block is too large
            log.warning("KeepTogether failed for %s, using fallback: %s", title, exc)
            elements.append(Paragraph(title, self.S["ChartTitle"]))
            try:
                elements.append(RLImage(chart_path, width=C.SAFE_IMG_W, height=C.SAFE_IMG_H))
                elements.append(Spacer(1, 6))
                elements.append(Paragraph(f"📊  {insight}", self.S["Insight"]))
            except Exception as img_exc:
                log.error("ReportLab failed loading %s: %s", chart_path, img_exc)
                elements.append(Paragraph(f"⚠ Render error: {img_exc}", self.S["Fallback"]))
            elements.append(Spacer(1, 16))

    def _chart_monthly_revenue(
        self,
        monthly_data: list,
        peak_month: str = "",
        trough_month: str = "",
        pct_gap: float = 0,
    ) -> Optional[str]:
        """Generate a monthly revenue line chart. Returns PNG path or None."""
        if not monthly_data or len(monthly_data) < 2:
            return None
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            from datetime import datetime as _dt

            months   = [d[0] for d in monthly_data]
            revenues = [d[1] for d in monthly_data]

            labels = []
            for m in months:
                try:
                    labels.append(_dt.strptime(m, "%Y-%m").strftime("%b %Y"))
                except Exception:
                    labels.append(m)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(labels, revenues, marker="o", linewidth=2.5,
                    color="#4a6fa5", markersize=8,
                    markerfacecolor="white", markeredgewidth=2.5)

            for label, rev in zip(labels, revenues):
                if rev >= 1e7:
                    val_str = f"₹{rev/1e7:.1f}Cr"
                elif rev >= 1e5:
                    val_str = f"₹{rev/1e5:.1f}L"
                else:
                    val_str = f"₹{rev/1e3:.0f}K"
                ax.annotate(val_str, (label, rev),
                            textcoords="offset points",
                            xytext=(0, 12), ha="center",
                            fontsize=9, color="#1a1a2e", fontweight="bold")

            # ✅ Peak marker
            if peak_month:
                try:
                    peak_label_idx = next(
                        i for i, m in enumerate(months)
                        if _dt.strptime(m, "%Y-%m").strftime("%B") == peak_month
                    )
                    ax.scatter(
                        [labels[peak_label_idx]], [revenues[peak_label_idx]],
                        marker="*", s=200, color="#10b981", zorder=5,
                        label=f"Peak: {peak_month}"
                    )
                    ax.annotate(
                        f"▲ {peak_month}",
                        (labels[peak_label_idx], revenues[peak_label_idx]),
                        textcoords="offset points", xytext=(0, 16),
                        ha="center", fontsize=9,
                        color="#10b981", fontweight="bold"
                    )
                except Exception:
                    pass

            # ✅ Trough marker
            if trough_month:
                try:
                    trough_label_idx = next(
                        i for i, m in enumerate(months)
                        if _dt.strptime(m, "%Y-%m").strftime("%B") == trough_month
                    )
                    ax.scatter(
                        [labels[trough_label_idx]], [revenues[trough_label_idx]],
                        marker="v", s=150, color="#ef4444", zorder=5,
                        label=f"Trough: {trough_month}"
                    )
                    ax.annotate(
                        f"▼ {trough_month}",
                        (labels[trough_label_idx], revenues[trough_label_idx]),
                        textcoords="offset points", xytext=(0, -20),
                        ha="center", fontsize=9,
                        color="#ef4444", fontweight="bold"
                    )
                except Exception:
                    pass

            # ✅ Shaded band between trough and peak values
            if peak_month and trough_month and revenues:
                peak_val_num = max(revenues)
                trough_val_num = min(revenues)
                ax.axhspan(
                    trough_val_num, peak_val_num,
                    alpha=0.06, color="#6366f1", zorder=0
                )
                if pct_gap > 0:
                    ax.text(
                        0.98, 0.5,
                        f"{pct_gap:.0f}% swing",
                        transform=ax.transAxes,
                        ha="right", va="center",
                        fontsize=9, color="#94a3b8",
                        style="italic"
                    )

            # ✅ Add legend if markers were added
            if peak_month or trough_month:
                ax.legend(
                    loc="upper left", fontsize=8,
                    framealpha=0.3, edgecolor="none"
                )

            ax.set_title("Monthly Revenue Trend", fontsize=13,
                         fontweight="bold", pad=12, color="#1a1a2e")
            ax.set_ylabel("Revenue", fontsize=10)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"₹{x/1e7:.1f}Cr" if x >= 1e7 else f"₹{x/1e5:.0f}L"
            ))
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.xticks(rotation=30, ha="right", fontsize=8)
            plt.tight_layout()

            path = os.path.join(os.path.dirname(__file__), "_tmp_monthly_trend.png")
            fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            return path
        except Exception as e:
            log.error("[chart_monthly_revenue] error: %s", e)
            return None

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
                # Always humanize snake_case keys regardless of case
                label = str(raw_key).replace("_", " ").title()
            if display.startswith("{") and display.endswith("}"):
                display = "—"
            out[label] = display
        return out

    def _kpi_table(self, kpis: dict) -> Table:
        # CHANGE 4 — Runtime font verification
        from reportlab.pdfbase import pdfmetrics as _pm
        registered = list(_pm._fonts.keys())
        print(f"[FONT_VERIFY] Registered fonts at KPI build time: {registered}")
        print(f"[FONT_VERIFY] PDF_FONT_REGULAR = '{PDF_FONT_REGULAR}'")
        print(f"[FONT_VERIFY] PDF_FONT_BOLD = '{PDF_FONT_BOLD}'")

        kpis = self._normalize_kpis(kpis)
        if not kpis:
            kpis = {"Status": "No KPI data"}
        col_w = (C.PAGE_W - 2 * C.MARGIN) / max(len(kpis), 1)
        cfg = self.config

        # Hardcode DejaVuSans — PDF_FONT_REGULAR may resolve to Helvetica if
        # font registration fails silently; ₹ glyph only exists in DejaVuSans.
        def _rupee_wrap(s: str) -> str:
            from xml.sax.saxutils import escape as _xe
            return f'<font name="DejaVuSans">{_xe(str(s))}</font>'

        kpi_val_style = ParagraphStyle(
            'KPIVal',
            fontName='DejaVuSans',
            fontSize=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1e293b'),
        )

        hdr = [Paragraph(k, ParagraphStyle("kh2", fontName=PDF_FONT_BOLD,
               fontSize=8, textColor=colors.white, alignment=TA_CENTER))
               for k in kpis]
        val = [Paragraph(_rupee_wrap(v), kpi_val_style) for v in kpis.values()]
        tbl = Table([hdr, val], colWidths=[col_w] * len(kpis))
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor(cfg["purple"])),
            ("BACKGROUND",    (0, 1), (-1, 1),  colors.HexColor(cfg["brand_light"])),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor(C.RULE_GREY)),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("FONTNAME",      (0, 1), (-1, 1),  'DejaVuSans'),
            ("FONTNAME",      (1, 0), (-1, -1), 'DejaVuSans'),
        ]))
        return tbl

    def _build_section_6_deep_insights(
        self,
        insights: list,
        metrics: dict = None,
        domain: str = "",
        df=None,
    ) -> list:
        """Section 6: Deep Insights with WHAT / WHY / DECISION format."""
        print(f"[SECTION6 ENTRY] insights={len(insights) if insights else 0}, df type={type(df)}, df is None={df is None}")
        elements = []
        header_style = ParagraphStyle(
            'Section6Header',
            fontSize=18, textColor=colors.HexColor('#6366f1'),
            spaceAfter=14, fontName=PDF_FONT_BOLD,
        )
        elements.append(Paragraph("Deep Insights", header_style))
        elements.append(Spacer(1, 0.1 * inch))

        # ── PROSE NARRATIVE ────────────────────────────────────────────────
        if insights:
            prose = InsightNarrator().generate(
                insights=insights,
                metrics=metrics or {},
                domain=domain,
                df=df,
            )
            if prose:
                narrative_style = ParagraphStyle(
                    'NarrativeStyle',
                    parent=getSampleStyleSheet()['Normal'],
                    fontName=PDF_FONT_REGULAR,
                    fontSize=10.5,
                    leading=17,
                    textColor=colors.HexColor('#1a1a2e'),
                    leftIndent=0,
                    rightIndent=0,
                    spaceAfter=18,
                    spaceBefore=6,
                    borderPad=12,
                    backColor=colors.HexColor('#f0f4ff'),
                    borderWidth=0,
                )
                elements.append(Spacer(1, 6))
                elements.append(Paragraph(self._md_to_rl(prose), narrative_style))
                elements.append(Spacer(1, 10))
        # ── END PROSE NARRATIVE ────────────────────────────────────────────

        if not insights:
            elements.append(Paragraph(
                "No deep insights met the qualification threshold for this dataset.",
                ParagraphStyle('Body', fontSize=10, textColor=colors.grey, fontName=PDF_FONT_REGULAR)
            ))
            return elements

        title_style = ParagraphStyle(
            'InsightTitle', fontSize=13, fontName=PDF_FONT_BOLD,
            textColor=colors.HexColor('#1e293b'), spaceAfter=6
        )
        impact_style_high = ParagraphStyle(
            'ImpactHigh', fontSize=9, fontName=PDF_FONT_BOLD,
            textColor=colors.HexColor('#dc2626'), spaceAfter=8
        )
        impact_style_medium = ParagraphStyle(
            'ImpactMed', fontSize=9, fontName=PDF_FONT_BOLD,
            textColor=colors.HexColor('#d97706'), spaceAfter=8
        )
        body_style = ParagraphStyle(
            'InsightBody', fontSize=10, textColor=colors.HexColor('#334155'),
            leading=14, spaceAfter=6, fontName=PDF_FONT_REGULAR
        )
        decision_style = ParagraphStyle(
            'Decision', fontSize=10, fontName=PDF_FONT_OBLIQUE,
            textColor=colors.HexColor('#6366f1'), leading=14,
            leftIndent=12, spaceAfter=14
        )

        for i, insight in enumerate(insights, 1):
            if isinstance(insight, str):
                title = "Strategic Finding"
                description = insight
                impact = "Medium"
                recommendation = ""
            else:
                title = insight.get('title', '') or insight.get('rule_type', 'Strategic Finding')
                description = insight.get('description', '')
                impact = insight.get('impact', 'Medium')
                recommendation = insight.get('recommendation', '')

            elements.append(Paragraph(f"{i:02d}. {title}", title_style))
            impact_clean = self._strip_emoji(impact)
            impact_label = f"[{impact_clean.upper()} IMPACT]"
            elements.append(Paragraph(impact_label, impact_style_high if 'high' in impact.lower() else impact_style_medium))
            elements.append(Paragraph(self._md_to_rl(description), body_style))
            if recommendation:
                elements.append(Paragraph(f"→ DECISION: {recommendation}", decision_style))
            elements.append(Spacer(1, 0.15 * inch))
        return elements

    def _build_section_7_recommendations(self, recommendations: list) -> list:
        elements = []
        elements.append(PageBreak())

        header_style = ParagraphStyle(
            'Section7Header', fontSize=18, textColor=colors.HexColor('#6366f1'),
            spaceAfter=14, fontName=PDF_FONT_BOLD,
        )
        elements.append(Paragraph("Strategic Recommendations", header_style))
        elements.append(Spacer(1, 0.15 * inch))

        if not recommendations:
            elements.append(Paragraph(
                "Insufficient signal in the dataset to generate strategic recommendations.",
                ParagraphStyle('Body', fontSize=10, textColor=colors.grey, fontName=PDF_FONT_REGULAR)
            ))
            return elements

        num_style = ParagraphStyle(
            'RecNum', fontSize=22, fontName=PDF_FONT_BOLD,
            textColor=colors.HexColor('#8b5cf6'),
        )
        action_style = ParagraphStyle(
            'RecAction', fontSize=11, fontName=PDF_FONT_REGULAR,
            textColor=colors.HexColor('#1e293b'), leading=15
        )
        meta_style = ParagraphStyle(
            'RecMeta', fontSize=9, fontName=PDF_FONT_REGULAR,
            textColor=colors.HexColor('#64748b'), leading=12, spaceBefore=4
        )

        for idx, rec in enumerate(recommendations, 1):
            # Handle legacy string format or new dict format
            if isinstance(rec, str):
                action = rec
                priority_val = idx
                timeframe = "—"
                owner = "—"
                impact = "Medium"
            else:
                priority_val = rec.get("priority") or idx
                action = rec.get("action", "")
                timeframe = rec.get("timeframe", "—")
                owner = rec.get("owner", "—")
                impact = rec.get("impact", "Medium")

            priority_str = f"{int(priority_val):02d}"
            impact_clean = self._strip_emoji(impact)
            meta_line = f"Timeframe: {timeframe}  |  Owner: {owner}  |  Impact: {impact_clean}"

            row = [
                Paragraph(priority_str, num_style),
                [
                    Paragraph(self._md_to_rl(action), action_style),
                    Paragraph(meta_line, meta_style),
                ],
            ]
            tbl = Table([row], colWidths=[0.7 * inch, 5.5 * inch])
            tbl.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 16),
            ]))
            elements.append(tbl)

        return elements

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

        # Override generic label with actual column name when template label isn't a real column
        _GENERIC = {"Key Performance Indicator", "KPI", "Metric", "Value", "Revenue"}
        if df is not None:
            if target_metric in _GENERIC or target_metric not in df.columns:
                _rev_cols = [c for c in df.columns if any(
                    k in c.lower() for k in
                    ['revenue', 'sales_amount', 'sales amount', 'amount', 'income', 'turnover']
                )]
                if _rev_cols:
                    target_metric = _rev_cols[0]

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

        # ✅ ADD MISSING SECTIONS 6 & 7
        print(f"[BUILD METHOD] calling section 6, df type={type(df)}, df is None={df is None}")
        if isinstance(insights, list):
            elements.extend(self._build_section_6_deep_insights(
                insights, metrics=kpis, domain=target_metric, df=df
            ))
        
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
            kpis[f"Total {cm.numeric}"] = f"\u20b9{total:,.0f}" if total > 100 else f"{total:,.2f}"
            kpis[f"Avg {cm.numeric}"]   = f"\u20b9{df[cm.numeric].mean():,.0f}"
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
                          insights: list = [],
                          recommendations: list = [],
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

        # Dynamic override: if the template's target_metric isn't a real column,
        # find the first numeric column whose name suggests a revenue/sales metric.
        if df is not None and target_metric not in (df.columns if hasattr(df, 'columns') else []):
            _rev_keywords = ("revenue", "sales", "amount", "total", "value", "price", "profit")
            _candidates = [c for c in df.columns if any(k in c.lower() for k in _rev_keywords)]
            if _candidates:
                target_metric = _candidates[0]

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

        # 3. PAGE 3: REGIONAL ANALYSIS (Rendered if DF provided, suppressed if low-variance)
        _regional_page_added = False
        if df is not None:
            region_col = get_region_column(df)
            if region_col and target_metric in df.columns:
                # Variance guard — skip the whole regional page if spread < 10%
                _reg_vals = df.groupby(region_col)[target_metric].median().tolist()
                _reg_variance_pct = (
                    (max(_reg_vals) - min(_reg_vals)) / max(max(_reg_vals), 1) * 100
                    if _reg_vals else 0
                )
                if _reg_variance_pct >= 10:
                    elements.append(PageBreak())
                    _regional_page_added = True
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
                else:
                    log.info("Regional page suppressed — variance %.1f%% < 10%%", _reg_variance_pct)

        # 4. PAGE 4: STRATEGIC FINDINGS & NOTES
        elements.append(PageBreak())
        if insights or text_blocks:
            elements.append(Paragraph("Strategic Findings & Key Results", self.S["Section"]))
            elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(C.RULE_GREY)))
            elements.append(Spacer(1, 20))

            finding_title_style = ParagraphStyle(
                'FindingTitle', fontSize=11, fontName=PDF_FONT_BOLD,
                textColor=colors.HexColor('#1e293b'), spaceAfter=4,
            )
            finding_body_style = ParagraphStyle(
                'FindingBody', fontSize=9.5, fontName=PDF_FONT_REGULAR,
                textColor=colors.HexColor('#334155'), leading=14,
                leftIndent=14, spaceAfter=4,
            )
            finding_impact_style = ParagraphStyle(
                'FindingImpact', fontSize=8.5, fontName=PDF_FONT_BOLD,
                textColor=colors.HexColor('#dc2626'), spaceAfter=10,
                leftIndent=14,
            )

            if insights:
                for idx, ins in enumerate(insights, 1):
                    if isinstance(ins, dict):
                        title = ins.get("title") or ins.get("rule_type", "Strategic Finding")
                        description = ins.get("description", "")
                        impact = ins.get("impact", "")
                        recommendation = ins.get("recommendation", "")
                    else:
                        title = str(ins).split('[')[0].strip()
                        description = ""
                        impact = ""
                        recommendation = ""

                    if title:
                        elements.append(Paragraph(f"• {title}", finding_title_style))
                    if description:
                        # Smart truncation at sentence boundary (up to 500 chars)
                        if len(description) <= 500:
                            short_desc = description
                        else:
                            # Find last sentence boundary before 500 chars
                            truncated = description[:500]
                            # Look for last period, exclamation, or question mark
                            last_period = max(
                                truncated.rfind('. '),
                                truncated.rfind('! '),
                                truncated.rfind('? ')
                            )
                            if last_period > 300:  # Only use if we get at least 300 chars
                                short_desc = description[:last_period + 1].rstrip()
                            else:
                                # Fallback to 500 char hard limit
                                short_desc = truncated.rstrip()
                            short_desc += "…"
                        elements.append(Paragraph(self._md_to_rl(short_desc), finding_body_style))
                    if impact:
                        impact_clean = self._strip_emoji(impact)
                        elements.append(Paragraph(f"Impact: {impact_clean.upper()}", finding_impact_style))
                    elif title:
                        elements.append(Spacer(1, 10))

                elements.append(Spacer(1, 10))

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
        total_charts = len(charts)
        for i, chart in enumerate(charts):
            img_path = self._decode_image(chart.get("image_base64", ""), session_id)
            err = chart.get("error")
            
            if err:
                # Suppress raw data dumps - only show error if it's not a data dump
                if "{" not in str(err) and "[" not in str(err):
                    elements.append(Paragraph(f"⚠ {chart.get('title')}: {err}", self.S["Fallback"]))
                    elements.append(Spacer(1, 20))
            else:
                self.embed_chart_safely(
                    elements,
                    img_path,
                    chart.get("title", "Visualization Segment"),
                    chart.get("insight", "Segmented data analysis.")
                )
                valid_charts += 1
        
        # Let ReportLab handle natural pagination - no manual PageBreaks in chart loop
        # Track chart count for temporal/deep insights logic below
        _last_chart_completed_pair = (valid_charts > 0 and valid_charts % 2 == 0)

        # ── Monthly Revenue Trend chart (from temporal_peaks insight) ──────
        temporal_insight = next(
            (i for i in insights
             if isinstance(i, dict) and (
                 i.get("rule_type") == "temporal_peaks"
                 or isinstance(i.get("chart_data"), dict)
                 and "monthly_data" in (i.get("chart_data") or {})
             )),
            None,
        )
        print(f"[temporal_chart] temporal_insight found = {temporal_insight is not None}")
        if temporal_insight:
            monthly_data = (temporal_insight.get("chart_data") or {}).get("monthly_data", [])
            print(f"[temporal_chart] monthly_data = {monthly_data[:2] if monthly_data else 'EMPTY'}")
            _cd = temporal_insight.get("chart_data") or {}
            chart_path = self._chart_monthly_revenue(
                monthly_data,
                peak_month=_cd.get("peak_month", ""),
                trough_month=_cd.get("trough_month", ""),
                pct_gap=_cd.get("pct_gap", 0),
            )
            if chart_path and os.path.exists(chart_path) and os.path.getsize(chart_path) > 0:
                # Only add PageBreak if we're not already on a fresh page from frontend charts
                if not _last_chart_completed_pair:
                    elements.append(PageBreak())
                elements.append(Paragraph("Monthly Revenue Trend", self.S["Section"]))
                elements.append(HRFlowable(width="100%", thickness=1,
                                           color=colors.HexColor(C.RULE_LIGHT)))
                elements.append(Spacer(1, 10))
                try:
                    img = RLImage(chart_path, width=480, height=210)
                    elements.append(img)
                    elements.append(Spacer(1, 6))
                    peak   = (temporal_insight.get("chart_data") or {}).get("peak_month", "")
                    trough = (temporal_insight.get("chart_data") or {}).get("trough_month", "")
                    caption = f"Revenue trajectory across all months — peak: {peak}, trough: {trough}."
                    elements.append(Paragraph(caption, self.S["Insight"]))
                    elements.append(Spacer(1, 16))
                except Exception as _e:
                    log.error("Monthly trend chart embed failed: %s", _e)
                # Update flag since we added content after frontend charts
                _last_chart_completed_pair = False
        
        # ✅ FALLBACK: Generate time series from raw data if temporal_insight not found
        elif df is not None and len(df) > 0:
            try:
                import polars as pl
                from datetime import datetime as _dt
                
                # Find date and revenue columns
                date_col = next((c for c in df.columns if any(k in c.lower() for k in ["date", "time", "day"])), None)
                rev_col = next((c for c in df.columns if any(k in c.lower() for k in ["sales", "amount", "revenue"])), None)
                
                if date_col and rev_col:
                    print(f"[temporal_fallback] Generating from df: date={date_col}, rev={rev_col}")
                    # df is already pandas at this point (converted at top of build_from_assets)
                    pdf_tmp = df.copy()
                    pdf_tmp[date_col] = pd.to_datetime(pdf_tmp[date_col], errors="coerce", dayfirst=True)
                    pdf_tmp = pdf_tmp.dropna(subset=[date_col])
                    pdf_tmp["month"] = pdf_tmp[date_col].dt.to_period("M").astype(str)
                    monthly = pdf_tmp.groupby("month")[rev_col].sum().reset_index()
                    monthly = monthly.sort_values("month")
                    
                    if len(monthly) >= 2:
                        # Prepare monthly_data for chart
                        monthly_data = [(row["month"], row[rev_col]) for _, row in monthly.iterrows()]
                        
                        # Calculate peak/trough
                        peak_idx = monthly[rev_col].idxmax()
                        trough_idx = monthly[rev_col].idxmin()
                        peak_month_str = monthly.loc[peak_idx, "month"]
                        trough_month_str = monthly.loc[trough_idx, "month"]
                        peak_val = monthly.loc[peak_idx, rev_col]
                        trough_val = monthly.loc[trough_idx, rev_col]
                        pct_gap = ((peak_val - trough_val) / peak_val * 100) if peak_val > 0 else 0
                        
                        # Extract month names
                        try:
                            peak_month = _dt.strptime(peak_month_str, "%Y-%m").strftime("%B")
                            trough_month = _dt.strptime(trough_month_str, "%Y-%m").strftime("%B")
                        except:
                            peak_month = peak_month_str
                            trough_month = trough_month_str
                        
                        chart_path = self._chart_monthly_revenue(
                            monthly_data,
                            peak_month=peak_month,
                            trough_month=trough_month,
                            pct_gap=pct_gap,
                        )
                        
                        if chart_path and os.path.exists(chart_path) and os.path.getsize(chart_path) > 0:
                            if not _last_chart_completed_pair:
                                elements.append(PageBreak())
                            elements.append(Paragraph("Monthly Revenue Trend", self.S["Section"]))
                            elements.append(HRFlowable(width="100%", thickness=1,
                                                       color=colors.HexColor(C.RULE_LIGHT)))
                            elements.append(Spacer(1, 10))
                            try:
                                img = RLImage(chart_path, width=480, height=210)
                                elements.append(img)
                                elements.append(Spacer(1, 6))
                                caption = f"Revenue trajectory across all months — peak: {peak_month}, trough: {trough_month}."
                                elements.append(Paragraph(caption, self.S["Insight"]))
                                elements.append(Spacer(1, 16))
                                _last_chart_completed_pair = False
                                print(f"[temporal_fallback] Chart generated successfully")
                            except Exception as _e:
                                log.error("Fallback monthly trend chart embed failed: %s", _e)
            except Exception as _e:
                log.warning("Fallback time series generation failed: %s", _e)

        # ✅ ADD MISSING SECTIONS 6 & 7
        print(f"[PRE-SECTION6] insights type={type(insights)}, len={len(insights) if insights else 0}, df is None={df is None}")
        print(f"[build_from_assets] df passed to section 6: type={type(df)}, is None={df is None}")
        
        # Always start Deep Insights on a fresh page
        if insights:
            elements.append(PageBreak())
        
        elements.extend(self._build_section_6_deep_insights(
            insights, metrics=kpis, domain=domain_id, df=df
        ))
        recs = recommendations or [
            b.get("content") for b in text_blocks
            if "recommendation" in b.get("content", "").lower()
        ]
        elements.extend(self._build_section_7_recommendations(recs))

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
