"""
engine/analyzer.py — LLM JSON Spec Engine (Phase 1, 10/10)
===========================================================
  Step 1  — detect_domain()       column-pattern domain detection
  Step 2  — SAFE_BUILTINS         kept for reference (no exec needed)
  Step 3  — GROQ_MODEL            correct model name
  Step 4  — _build_data_quality() pre-processing layer
  Step 5  — _generate_prompt()    JSON-spec prompt (no Python code)
  Step 6  — _render_from_spec()   chart renderer (no exec, no sandbox)
  Step 7  — caching               SHA256 fingerprint cache
  Step 8  — _validate_results()   output validation + safe fallback
  Step 9  — wired via routers/analyze.py
  Step 10 — analyze_dataset()     complete orchestration
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Correct Groq model names
# ─────────────────────────────────────────────────────────────────────────────
GROQ_MODELS = {
    "code_gen": "meta-llama/llama-4-scout-17b-16e-instruct",  # 30K TPM, 500K TPD
    "smart":    "llama-3.3-70b-versatile",                    # 1K RPD, smartest
    "fast":     "llama-3.1-8b-instant",                       # 14.4K RPD, lightweight
}
GROQ_MODEL = GROQ_MODELS["code_gen"]  # default for code generation

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Safe builtins whitelist
# ─────────────────────────────────────────────────────────────────────────────
SAFE_BUILTINS = {
    # Iteration
    "range":      range,
    "enumerate":  enumerate,
    "zip":        zip,
    "map":        map,
    "filter":     filter,
    "sorted":     sorted,
    "reversed":   reversed,
    # Types
    "str":        str,
    "int":        int,
    "float":      float,
    "bool":       bool,
    "list":       list,
    "dict":       dict,
    "tuple":      tuple,
    "set":        set,
    # Math
    "abs":        abs,
    "round":      round,
    "sum":        sum,
    "min":        min,
    "max":        max,
    "len":        len,
    "pow":        pow,
    "divmod":     divmod,
    # Inspection
    "print":      print,
    "repr":       repr,
    "type":       type,
    "isinstance": isinstance,
    "hasattr":    hasattr,
    "getattr":    getattr,
    # NO: open, __import__, exec, eval, compile, globals, locals
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SHA256 caching layer
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / ".analyzer_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of dataset schema and sample. Ignores row order."""
    meta = {
        "columns": sorted(df.columns.tolist()),
        "dtypes":  df.dtypes.astype(str).to_dict(),
        "shape":   list(df.shape),
        "head":    df.head(5).fillna("__NULL__").to_dict(orient="list"),
    }
    content = json.dumps(meta, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _cache_set(fp: str, results: dict) -> None:
    # Never cache a result with no charts — it's incomplete.
    chart_count = len(results.get("charts", []))
    if chart_count == 0:
        print(f"[analyzer] Cache write skipped — 0 charts in results (fp={fp})")
        return
    path = CACHE_DIR / f"{fp}.json"
    # Save everything except non-serializable Plotly figures.
    # Do NOT write a "charts" key — empty list causes false-positive cache hits.
    save = {k: v for k, v in results.items() if k != "charts"}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save, f, indent=2, default=str)
        print(f"[analyzer] Cache written: {fp} ({chart_count} charts, "
              f"{len(results.get('insights', []))} insights)")
    except Exception as e:
        print(f"[analyzer] Cache write failed: {e}")


def _cache_get(fp: str) -> Optional[dict]:
    path = CACHE_DIR / f"{fp}.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Reject stale/incomplete entries.
            # Valid entry: has insights with real text (>= 20 chars each).
            # The "charts" key is never stored (figures aren't serializable),
            # so we validate on insight quality instead.
            insights = data.get("insights", [])
            if not insights:
                print(f"[analyzer] Cache invalid (no insights): {fp} — deleting")
                path.unlink(missing_ok=True)
                return None
            has_text = any(
                len(ins.get("text", "")) >= 20
                for ins in insights if isinstance(ins, dict)
            )
            if not has_text:
                print(f"[analyzer] Cache invalid (empty insight text): {fp} — deleting")
                path.unlink(missing_ok=True)
                return None
            return data
        except Exception:
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Domain detection function
# ─────────────────────────────────────────────────────────────────────────────
VALIDATED_DOMAINS_COLS = {
    "sports": [
        ["winner", "toss_winner"],          # IPL pattern
        ["team_1", "team_2", "winner"],     # PSL pattern
        ["home_team", "away_team"],         # generic match pattern
        ["batting_team", "bowling_team"],   # cricket innings pattern
    ],
    "entertainment": [
        ["type", "listed_in", "date_added"],    # Netflix / Amazon
        ["type", "rating", "release_year"],     # Disney+
        ["genre", "release_year"],              # generic streaming
        ["show_id", "listed_in"],               # Netflix variant
    ],
}


def detect_domain(df: pd.DataFrame, filename: str = "") -> str:
    """
    Returns domain string: "sports", "entertainment", or "unknown".
    Checks column names first (pattern matching), then filename keywords.
    "unknown" routes to the LLM analyzer.
    """
    # Normalise column names: lowercase, strip underscores and spaces
    cols_lower = {
        c.lower().replace("_", "").replace(" ", "")
        for c in df.columns
    }

    for domain, patterns in VALIDATED_DOMAINS_COLS.items():
        for pattern in patterns:
            normalized = [p.lower().replace("_", "") for p in pattern]
            if all(p in cols_lower for p in normalized):
                return domain

    # Fallback: check filename keywords
    fname = filename.lower()
    if any(k in fname for k in ["ipl", "psl", "cricket", "match", "football", "nba", "nfl"]):
        return "sports"
    if any(k in fname for k in ["netflix", "disney", "prime", "movies", "shows", "streaming"]):
        return "entertainment"

    return "unknown"  # Route to LLM analyzer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Data quality pre-processing layer
# ─────────────────────────────────────────────────────────────────────────────
def _build_data_quality(df: pd.DataFrame) -> dict:
    """
    Pre-compute data quality metrics.
    Injected into prompt so LLM automatically generates correct caveats
    without being told explicitly.
    """
    quality: dict = {}

    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        q: dict = {"missing_pct": round(missing_pct, 1)}

        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            mean = df[col].mean()
            std  = df[col].std()
            if pd.notna(mean) and pd.notna(std) and std > 0:
                outliers = int(((df[col] - mean).abs() > 3 * std).sum())
            else:
                outliers = 0
            q["outliers"] = outliers
            q["mean"]     = round(float(mean), 2) if pd.notna(mean) else None
            q["std"]      = round(float(std),  2) if pd.notna(std)  else None
        else:
            q["unique_count"] = int(df[col].nunique())
            mode_vals = df[col].mode()
            q["top_value"] = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else "N/A"

        quality[col] = q

    # Flag columns with >50% missing
    high_missing = [c for c, v in quality.items() if v["missing_pct"] > 50]

    return {
        "per_column":           quality,
        "high_missing_columns": high_missing,
        "total_rows":           len(df),
        "complete_rows":        int(df.dropna().shape[0]),
        "complete_pct":         round(df.dropna().shape[0] / max(len(df), 1) * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Context builder (used by analyze_dataset)
# ─────────────────────────────────────────────────────────────────────────────
def _build_context(df: pd.DataFrame) -> dict:
    """Build a compact dataset context dict for the LLM prompt."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols     = df.select_dtypes(exclude="number").columns.tolist()

    sample_values: dict = {}
    for col in df.columns[:20]:  # limit to first 20 cols to keep prompt size sane
        vals = df[col].dropna().head(3).tolist()
        sample_values[col] = [str(v) for v in vals]

    stats: dict = {}
    for col in numeric_cols[:15]:
        try:
            stats[col] = {
                "min":    round(float(df[col].min()), 2),
                "max":    round(float(df[col].max()), 2),
                "mean":   round(float(df[col].mean()), 2),
                "median": round(float(df[col].median()), 2),
            }
        except Exception:
            pass

    return {
        "columns":      df.columns.tolist(),
        "shape":        list(df.shape),
        "numeric_cols": numeric_cols,
        "cat_cols":     cat_cols,
        "sample_values": sample_values,
        "numeric_stats": stats,
        "dtypes":       df.dtypes.astype(str).to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — JSON-spec prompt (replaces Python code generation)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_prompt(context: dict, quality: dict) -> str:
    """
    Ask the LLM for a JSON spec, not Python code.
    InsightStream renders the charts itself from the spec — zero exec() risk.
    """
    high_missing = quality.get("high_missing_columns", [])
    missing_note = ""
    if high_missing:
        per_col = quality.get("per_column", {})
        details = [
            f"{c} ({per_col.get(c, {}).get('missing_pct', '?')}% missing)"
            for c in high_missing
        ]
        missing_note = (
            f"For columns with >50% missing data: {details}\n"
            f"Add a caveat in the insight text: "
            f'"Note: X% of records are missing this field."'
        )

    # Truncate context to keep prompt under token limits
    context_str = json.dumps(context, indent=2, default=str)[:3000]

    return f"""You are a senior data analyst. Analyze this dataset and return ONLY a valid JSON object — no markdown, no code, no explanation.

The JSON must have these exact keys:
{{
  "domain": "one word domain type (HR/Sales/Sports/Health/Entertainment/Finance)",
  "title": "report title string",
  "insights": [
    {{
      "title": "short title",
      "text": "2-3 sentences with specific numbers from the data",
      "impact": "CRITICAL or IMPORTANT or MINOR"
    }}
  ],
  "charts": [
    {{
      "type": "bar or histogram or line or scatter or pie",
      "title": "chart title",
      "x_column": "exact column name from dataset",
      "y_column": "exact column name or null for histogram/pie",
      "agg": "count or sum or mean or null",
      "color_column": "column name or null"
    }}
  ],
  "recommendations": [
    {{
      "text": "specific actionable recommendation",
      "timeframe": "Next 14 days or Next 30 days or Next quarter",
      "owner": "team name",
      "impact": "Critical or Important or Minor"
    }}
  ]
}}

Rules:
- Return 5-7 insights with real statistics derived from the data summary below
- Return EXACTLY 3-4 charts — this field is REQUIRED and must not be empty
- You MUST include at least one histogram of a numeric column and one bar chart of a categorical column
- Return 3-5 recommendations tied to actual findings
- x_column and y_column must be exact column names from the list below
- If a column has >50% missing data, mention it in the insight text
- The "charts" array MUST contain 3-4 objects — an empty charts array is invalid
- Each insight MUST have a non-empty "text" field with 2-3 sentences including specific numbers
- Recommendations MUST be SPECIFIC to this dataset. Each recommendation must:
  * Reference at least one column name from: {context["columns"]}
  * Use actual numbers derived from the data summary
  * NEVER include phrases like "case fatality rate", "recovery rate", "public health",
    "disease outbreak", "pandemic", "COVID", "corona", "virus", "health ministry",
    "high-burden nations", "international aid", "treatment protocol", or any content
    not directly related to the columns provided
- If you cannot derive a recommendation directly from the data, omit it entirely.
  Fewer relevant recommendations are better than irrelevant ones.

Example of a correct insight:
{{"title": "Termination Rate", "text": "The termination rate is 35.5% (110 out of 311 employees). This exceeds the industry benchmark of 13-15% for technology firms.", "impact": "CRITICAL"}}

Example of a correct recommendation:
{{"text": "Review compensation for the Production department where average Salary is $58,000 — 12% below the company median.", "timeframe": "Next 30 days", "owner": "HR team", "impact": "Important"}}

Do NOT return insights with empty or missing text fields.
Do NOT return recommendations that mention topics unrelated to the dataset columns.
{missing_note}

Available columns: {context["columns"]}

Dataset summary:
{context_str}
"""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — JSON spec renderer (replaces exec-based sandbox entirely)
# ─────────────────────────────────────────────────────────────────────────────

# Numeric ID columns — skip when choosing a fallback histogram column
_SKIP_COLS = {"EmpID", "ManagerID", "PositionID", "Zip", "DeptID",
              "empid", "managerid", "positionid", "zip", "deptid"}


def _render_from_spec(spec: dict, df: pd.DataFrame) -> dict:
    """
    Render Plotly charts from a JSON spec produced by the LLM.
    No exec(), no code generation, no sandbox needed.
    InsightStream owns all chart-rendering logic — the LLM only supplies data.
    """
    import plotly.express as px

    valid_charts = []

    for chart_spec in spec.get("charts", []):
        try:
            chart_type = chart_spec.get("type", "bar")
            x          = chart_spec.get("x_column")
            y          = chart_spec.get("y_column")
            title      = chart_spec.get("title", "")
            agg        = chart_spec.get("agg", "count")
            color      = chart_spec.get("color_column")

            # ── Column validation ─────────────────────────────────────────
            if x and x not in df.columns:
                raise ValueError(f"Column {x!r} not in df")
            if y and y not in df.columns:
                y = None
            if color and color not in df.columns:
                color = None

            # ── Fix: categorical y can't be aggregated → count-based bar ──
            if y and df[y].dtype in ["object", "category", "string"]:
                y = None

            # ── Render ────────────────────────────────────────────────────
            if chart_type == "histogram" and x:
                fig = px.histogram(df, x=x, title=title, nbins=20, color=color)

            elif chart_type == "bar" and x and y:
                plot_df = df.groupby(x)[y].agg(agg).reset_index()
                fig = px.bar(plot_df, x=x, y=y, title=title)

            elif chart_type == "bar" and x:
                plot_df = df[x].value_counts().head(15).reset_index()
                plot_df.columns = [x, "count"]
                fig = px.bar(plot_df, x=x, y="count", title=title)

            elif chart_type == "scatter" and x and y:
                fig = px.scatter(df, x=x, y=y, title=title, color=color)

            elif chart_type == "pie" and x:
                counts = df[x].value_counts().head(10).reset_index()
                counts.columns = [x, "count"]
                fig = px.pie(counts, names=x, values="count", title=title)

            else:
                raise ValueError(f"Cannot render chart type={chart_type!r} with x={x!r} y={y!r}")

            # ── Validate chart has actual data ────────────────────────────
            if len(fig.data) == 0:
                raise ValueError("Empty chart — no traces")
            first = fig.data[0]
            if hasattr(first, "y") and first.y is not None:
                if all(v == 0 for v in first.y if v is not None):
                    raise ValueError("All y values are zero")

            valid_charts.append(fig)

        except Exception as e:
            print(f"[analyzer] Chart spec failed ({e}), trying fallback")
            # Fallback: histogram of first meaningful numeric column
            for col in df.select_dtypes(include="number").columns:
                if col not in _SKIP_COLS:
                    try:
                        fb = px.histogram(
                            df, x=col,
                            title=f"{col.replace('_', ' ').title()} Distribution",
                            nbins=20,
                        )
                        valid_charts.append(fb)
                        break
                    except Exception:
                        continue

    print(f"[analyzer] Rendered {len(valid_charts)} charts")

    # ── Guaranteed fallback: always produce at least 2 charts ────────────
    # If the LLM spec produced nothing (empty charts array, all specs failed,
    # or all-zero data), generate reliable charts directly from the df.
    if len(valid_charts) < 2:
        print(f"[analyzer] Only {len(valid_charts)} charts from spec — adding guaranteed fallbacks")
        numeric_cols = [c for c in df.select_dtypes(include="number").columns
                        if c not in _SKIP_COLS]
        cat_cols     = [c for c in df.select_dtypes(include="object").columns]

        # Histogram of first meaningful numeric column
        if numeric_cols and len(valid_charts) < 2:
            try:
                col = numeric_cols[0]
                fb  = px.histogram(df, x=col, nbins=20,
                                   title=f"{col.replace('_', ' ').title()} Distribution")
                valid_charts.append(fb)
                print(f"[analyzer] Fallback chart added: histogram of {col!r}")
            except Exception as _fe:
                print(f"[analyzer] Fallback histogram failed: {_fe}")

        # Bar chart of first categorical column
        if cat_cols and len(valid_charts) < 2:
            try:
                col     = cat_cols[0]
                plot_df = df[col].value_counts().head(15).reset_index()
                plot_df.columns = [col, "count"]
                fb = px.bar(plot_df, x=col, y="count",
                            title=f"{col.replace('_', ' ').title()} Breakdown")
                valid_charts.append(fb)
                print(f"[analyzer] Fallback chart added: bar of {col!r}")
            except Exception as _fe:
                print(f"[analyzer] Fallback bar chart failed: {_fe}")

        # Second numeric histogram if still short
        if len(numeric_cols) > 1 and len(valid_charts) < 2:
            try:
                col = numeric_cols[1]
                fb  = px.histogram(df, x=col, nbins=20,
                                   title=f"{col.replace('_', ' ').title()} Distribution")
                valid_charts.append(fb)
                print(f"[analyzer] Fallback chart added: histogram of {col!r}")
            except Exception:
                pass

    print(f"[analyzer] Final chart count: {len(valid_charts)}")
    return {
        "insights":        spec.get("insights", []),
        "charts":          valid_charts,
        "recommendations": spec.get("recommendations", []),
        "domain":          spec.get("domain", "General"),
        "title":           spec.get("title", "Data Analysis Report"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Output validation + safe fallback
# ─────────────────────────────────────────────────────────────────────────────
def _validate_results(results: dict, df: pd.DataFrame) -> dict:
    """
    Validate and sanitize LLM results before passing to report_writer.
    Fixes common LLM mistakes without crashing.
    """
    VALID_IMPACTS = {"CRITICAL", "IMPORTANT", "MINOR"}

    validated = {
        "insights":        results.get("insights", []),
        "charts":          results.get("charts", []),
        "recommendations": results.get("recommendations", []),
        "domain":          results.get("domain", "General"),
        "title":           results.get("title", "Data Analysis Report"),
    }

    # Validate insights structure
    clean_insights = []
    for ins in validated["insights"]:
        if isinstance(ins, dict) and "title" in ins and "text" in ins:
            ins["impact"] = ins.get("impact", "IMPORTANT")
            if ins["impact"] not in VALID_IMPACTS:
                ins["impact"] = "IMPORTANT"
            # Ensure text is non-empty and meaningful
            if not ins.get("text") or len(ins["text"]) < 20:
                old_text = ins.get("text", "")
                ins["text"] = f"Analysis of {ins['title']} across {len(df):,} records."
                print(f"[validate] Fixed empty text for {ins['title']!r}: "
                      f"{old_text!r} → {ins['text'][:60]!r}")
            clean_insights.append(ins)
    validated["insights"] = clean_insights

    # Validate recommendations structure
    clean_recs = []
    for rec in validated["recommendations"]:
        if isinstance(rec, dict) and "text" in rec:
            rec.setdefault("timeframe", "Next 30 days")
            rec.setdefault("owner", "Strategy team")
            rec.setdefault("impact", "Important")
            clean_recs.append(rec)
    validated["recommendations"] = clean_recs

    # Minimum content guard — ensure at least 3 insights
    if len(validated["insights"]) < 3:
        validated["insights"].append({
            "title":  "Dataset Overview",
            "text":   (
                f"Dataset contains {len(df):,} records across "
                f"{len(df.columns)} columns. "
                f"Automated analysis produced limited insights — "
                f"consider reviewing data completeness."
            ),
            "impact": "IMPORTANT",
        })

    # Validate charts are actually Plotly figures
    validated["charts"] = [
        c for c in validated["charts"]
        if hasattr(c, "to_json") or hasattr(c, "data")
    ]

    return validated


def _safe_fallback(df: pd.DataFrame) -> dict:
    """
    Minimal fallback if LLM completely fails.
    Returns basic stats-based report — never crashes.
    """
    import plotly.express as px

    row_count = len(df) if df is not None and not df.empty else 0
    col_count = len(df.columns) if df is not None and not df.empty else 0

    insights = [{
        "title":  f"Dataset: {row_count:,} Records, {col_count} Columns",
        "text":   (
            f"The dataset contains {row_count:,} rows and {col_count} columns. "
            f"Automated LLM analysis was not available for this upload. "
            f"Please retry or check your API key configuration."
        ),
        "impact": "IMPORTANT",
    }]

    charts = []
    if df is not None and not df.empty:
        for col in df.select_dtypes(include="number").columns[:1]:
            try:
                fig = px.histogram(df, x=col, title=f"{col} Distribution")
                charts.append(fig)
            except Exception:
                pass

    return {
        "insights": insights,
        "charts":   charts,
        "recommendations": [{
            "text":      "Review dataset structure and re-upload for full LLM analysis.",
            "timeframe": "Next 30 days",
            "owner":     "Data team",
            "impact":    "Important",
        }],
        "domain": "Unknown",
        "title":  "Data Analysis Report",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation filter — removes hallucinated / off-topic recommendations
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that indicate the LLM hallucinated content unrelated to the dataset
_FORBIDDEN_REC_KEYWORDS = [
    "case fatality", "recovery rate", "public health", "disease outbreak",
    "pandemic", "covid", "corona", "virus", "health ministry",
    "public health authority", "high-burden nations", "international aid",
    "treatment protocol", "surge planning", "fatality rate", "outbreak",
    "vaccination", "quarantine", "epidemi", "mortality rate",
]


def _filter_recommendations(recommendations: list, df_columns) -> list:
    """
    Remove recommendations that:
    1. Contain forbidden keywords (hallucinated health/pandemic content)
    2. Do not reference any column name from the actual dataset
    """
    col_names = [c.lower() for c in df_columns if len(c) > 2]

    # Build a set of column name variants to match against recommendation text.
    # Handles: exact match, camelCase split ("SibSp" → "sib sp"), underscore strip.
    import re as _re
    col_variants: set = set()
    for col in df_columns:
        if len(col) <= 2:
            continue
        col_lower = col.lower()
        col_variants.add(col_lower)
        # camelCase → space-separated: "SibSp" → "sib sp"
        spaced = _re.sub(r'([a-z])([A-Z])', r'\1 \2', col).lower()
        col_variants.add(spaced)
        # strip underscores/spaces: "sib_sp" → "sibsp"
        col_variants.add(col_lower.replace("_", "").replace(" ", ""))

    def is_valid(rec: dict) -> bool:
        text = rec.get("text", "").lower()
        if not text:
            return False
        # Reject forbidden keywords
        for kw in _FORBIDDEN_REC_KEYWORDS:
            if kw in text:
                print(f"[filter_recs] Removed — forbidden keyword {kw!r}: {text[:60]}")
                return False
        # Reject if no dataset column variant is mentioned
        if not any(v in text for v in col_variants if len(v) > 2):
            print(f"[filter_recs] Removed — no column reference: {text[:60]}")
            return False
        return True

    cleaned = [r for r in recommendations if is_valid(r)]
    removed = len(recommendations) - len(cleaned)
    if removed:
        print(f"[filter_recs] Filtered {removed} bad recommendation(s), kept {len(cleaned)}")
    return cleaned# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Complete analyze_dataset() function
# ─────────────────────────────────────────────────────────────────────────────
def analyze_dataset(df: pd.DataFrame, force_refresh: bool = False) -> dict:
    """
    Main entry point for LLM-based analysis of unknown-domain datasets.
    Returns a dict compatible with the InsightStream pipeline.

    Orchestration order:
      1.  Empty df guard → safe fallback
      2.  Cache check (SHA256 fingerprint)
      3.  Build context + data quality
      4.  Generate JSON-spec prompt
      5.  Call Groq API (max 2 attempts, temperature=0.1)
      6.  Strip markdown fences, parse JSON response
      7.  Render charts from spec via _render_from_spec() — no exec()
      8.  On JSON parse error: retry Groq asking for valid JSON
      9.  On second failure: safe fallback
      10. Validate results
      11. Cache successful result
      12. Return validated results
    """
    # ── 1. Empty guard ────────────────────────────────────────────────────
    if df is None or df.empty:
        print("[analyzer] Empty dataframe — returning safe fallback")
        return _safe_fallback(pd.DataFrame())

    # ── 2. Cache check ────────────────────────────────────────────────────
    fp = _fingerprint(df)
    if not force_refresh:
        cached = _cache_get(fp)
        if cached is not None:
            print(f"[analyzer] Cache hit: {fp}")
            return cached

    # ── 3. Pre-process ────────────────────────────────────────────────────
    context = _build_context(df)
    quality = _build_data_quality(df)

    # ── 4. Generate JSON prompt ───────────────────────────────────────────
    prompt = _generate_prompt(context, quality)

    # ── 5. Call Groq API ──────────────────────────────────────────────────
    try:
        from groq import Groq
    except ImportError:
        print("[analyzer] groq package not installed — returning safe fallback")
        return _safe_fallback(df)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[analyzer] GROQ_API_KEY not set — returning safe fallback")
        return _safe_fallback(df)

    client = Groq(api_key=api_key)

    def _call_groq(messages: list) -> str:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.0,   # deterministic — reduces chance of malformed JSON
            max_tokens=3000,
        )
        return resp.choices[0].message.content

    def _extract_json(raw: str) -> dict:
        """Strip markdown fences and parse JSON. Raises ValueError on failure."""
        # Remove ```json ... ``` or ``` ... ``` fences
        text = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        # Find the outermost { ... } block in case there's surrounding text
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")
        return json.loads(text[start:end])

    raw_response: Optional[str] = None
    last_error:   Optional[Exception] = None

    for attempt in range(2):
        try:
            raw_response = _call_groq([{"role": "user", "content": prompt}])
            break
        except Exception as e:
            last_error = e
            print(f"[analyzer] Groq attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(2)

    if not raw_response:
        print(f"[analyzer] Groq failed after 2 attempts ({last_error}) — safe fallback")
        return _safe_fallback(df)

    # ── 6. Parse JSON response ────────────────────────────────────────────
    try:
        spec = _extract_json(raw_response)
        print(f"[analyzer] JSON parsed OK — domain={spec.get('domain')!r}")

    except (json.JSONDecodeError, ValueError) as parse_err:
        print(f"[analyzer] JSON parse failed: {parse_err} — retrying Groq")

        # ── 8. Retry with explicit repair instruction ─────────────────────
        retry_prompt = (
            "Your previous response was not valid JSON. "
            "Return ONLY the raw JSON object — no markdown, no code fences, "
            "no explanation. Start your response with { and end with }.\n\n"
            f"Original request:\n{prompt}"
        )
        try:
            raw2 = _call_groq([{"role": "user", "content": retry_prompt}])
            spec = _extract_json(raw2)
            print(f"[analyzer] JSON retry parsed OK — domain={spec.get('domain')!r}")
        except Exception as retry_err:
            # ── 9. Both attempts failed → safe fallback ───────────────────
            print(f"[analyzer] JSON retry also failed: {retry_err} — safe fallback")
            return _safe_fallback(df)

    # ── 7. Render charts from spec (no exec) ─────────────────────────────
    results = _render_from_spec(spec, df)
    print(f"[analyzer] Rendered {len(results['charts'])} charts, "
          f"{len(results['insights'])} insights")

    # ── 7b. Post-render chart guarantee ──────────────────────────────────
    # _render_from_spec already adds fallbacks internally, but if somehow
    # charts is still empty (e.g. df has no numeric columns), add one more
    # safety net here so we never store 0 charts to the DB.
    if not results.get("charts"):
        import plotly.express as px
        print("[analyzer] Post-render: 0 charts — adding emergency fallback")
        for col in df.select_dtypes(include="number").columns:
            if col not in _SKIP_COLS:
                try:
                    results["charts"] = [
                        px.histogram(df, x=col, nbins=20,
                                     title=f"{col.replace('_', ' ').title()} Distribution")
                    ]
                    break
                except Exception:
                    pass

    # ── 10. Validate output ───────────────────────────────────────────────
    results = _validate_results(results, df)

    # ── 10b. Filter hallucinated recommendations ──────────────────────────
    results["recommendations"] = _filter_recommendations(
        results.get("recommendations", []), df.columns
    )

    # ── 11. Cache successful result ───────────────────────────────────────
    _cache_set(fp, results)

    # ── 12. Return ────────────────────────────────────────────────────────
    return results
