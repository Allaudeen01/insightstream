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
import seaborn as sns

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Image as RLImage, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
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

    def generate_all(self, df: pd.DataFrame) -> tuple[dict[str, str], ColumnMap]:
        """
        Run all generators. Returns (charts_dict, ColumnMap).
        Only verified paths enter the dict — failed charts are silently omitted.
        """
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
        self._build_styles()

    def _build_styles(self):
        base = getSampleStyleSheet()
        self.S = {
            "Title": ParagraphStyle("RTitle", parent=base["Title"],
                fontSize=20, textColor=colors.HexColor(C.PURPLE),
                spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold"),
            "Subtitle": ParagraphStyle("RSub", parent=base["Normal"],
                fontSize=10, textColor=colors.HexColor(C.TEXT_GREY),
                spaceAfter=18, alignment=TA_CENTER),
            "Section": ParagraphStyle("RSection", parent=base["Heading2"],
                fontSize=12, textColor=colors.HexColor(C.BRAND_DARK),
                fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6),
            "ChartTitle": ParagraphStyle("RChartTitle", parent=base["Normal"],
                fontSize=10, textColor=colors.HexColor(C.TEXT_DARK),
                fontName="Helvetica-Bold", spaceAfter=4),
            "Insight": ParagraphStyle("RInsight", parent=base["Normal"],
                fontSize=9, textColor=colors.HexColor("#2C3E50"),
                backColor=colors.HexColor(C.BRAND_ACCENT),
                borderPad=8, leftIndent=10, rightIndent=10,
                spaceBefore=4, spaceAfter=4),
            "Fallback": ParagraphStyle("RFallback", parent=base["Normal"],
                fontSize=9, textColor=colors.HexColor(C.TEXT_MUTED),
                fontName="Helvetica-Oblique"),
            "Normal": base["Normal"],
        }

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

    def _kpi_table(self, kpis: dict) -> Table:
        col_w = (C.PAGE_W - 2 * C.MARGIN) / max(len(kpis), 1)
        hdr = [Paragraph(k, ParagraphStyle("kh", fontName="Helvetica-Bold",
               fontSize=8, textColor=colors.white, alignment=TA_CENTER))
               for k in kpis]
        val = [Paragraph(str(v), ParagraphStyle("kv", fontName="Helvetica-Bold",
               fontSize=15, textColor=colors.HexColor(C.BRAND_DARK),
               alignment=TA_CENTER)) for v in kpis.values()]
        tbl = Table([hdr, val], colWidths=[col_w] * len(kpis))
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor(C.PURPLE)),
            ("BACKGROUND",    (0, 1), (-1, 1),  colors.HexColor(C.BRAND_LIGHT)),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor(C.RULE_GREY)),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return tbl

    def build(self, df: pd.DataFrame, charts: dict[str, str],
              insights: dict[str, str], output_path: str,
              cm: Optional[ColumnMap] = None,
              title: str = "InsightStream Analytics Report") -> str:

        doc = SimpleDocTemplate(output_path, pagesize=A4,
            leftMargin=C.MARGIN, rightMargin=C.MARGIN,
            topMargin=C.MARGIN,  bottomMargin=C.MARGIN)
        elements: list = []

        # Header
        elements.append(Paragraph(title, self.S["Title"]))
        elements.append(Paragraph(
            f"Authored by InsightStream AI  •  {date.today().strftime('%m/%d/%Y')}",
            self.S["Subtitle"]))
        elements.append(HRFlowable(width="100%", thickness=1,
                                   color=colors.HexColor(C.RULE_GREY)))
        elements.append(Spacer(1, 14))

        # KPIs
        kpis = self._derive_kpis(df, cm)
        if kpis:
            elements.append(Paragraph("Key Metrics", self.S["Section"]))
            elements.append(self._kpi_table(kpis))
            elements.append(Spacer(1, 20))

        # Charts
        num = cm.numeric  if cm and cm.numeric  else "Value"
        cat = cm.category if cm and cm.category else "Category"
        reg = cm.region   if cm and cm.region   else "Region"

        elements.append(Paragraph("Visual Analysis", self.S["Section"]))
        elements.append(HRFlowable(width="100%", thickness=0.5,
                                   color=colors.HexColor(C.RULE_LIGHT)))
        elements.append(Spacer(1, 10))

        chart_config = [
            ("category",     f"Total {num} by {cat}"),
            ("region",       f"{num} by {reg} & {cat}"),
            ("order_count",  f"Order Volume by {cat}"),
            ("distribution", f"Distribution of {num}"),
            ("correlation",  "Numeric Column Correlations"),
        ]
        for key, chart_title in chart_config:
            self.embed_chart_safely(
                elements,
                charts.get(key),
                chart_title,
                insights.get(key, "Automated insight pending for this chart."),
            )

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
