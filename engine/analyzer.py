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


# Known dataset fingerprints for refined domain labelling (logging only)
_DOMAIN_FINGERPRINTS = {
    "disaster_survival": [
        ["survived", "pclass", "sex", "age", "fare"],          # Titanic
        ["survived", "pclass", "embarked"],
    ],
    "hr_analytics": [
        ["attrition", "department", "salary"],
        ["empstatus", "deptid", "salary"],
    ],
    "real_estate": [
        ["saleprice", "grlivarea", "yearbuilt"],
        ["price", "bedrooms", "bathrooms", "sqft"],
    ],
}


def _refine_domain(domain: str, df: pd.DataFrame) -> str:
    """
    Post-process the detected domain for better logging and user-facing labels.
    Does NOT change routing — 'unknown' domains still go to the LLM analyzer.
    Returns a refined label string for display/logging purposes only.
    """
    cols_lower = {
        c.lower().replace("_", "").replace(" ", "")
        for c in df.columns
    }
    for refined, patterns in _DOMAIN_FINGERPRINTS.items():
        for pattern in patterns:
            normalized = [p.lower().replace("_", "") for p in pattern]
            if all(p in cols_lower for p in normalized):
                print(f"[DOMAIN DETECTOR] Refined domain: {refined} "
                      f"(routing domain stays: {domain!r})")
                return refined
    return domain


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
# INTELLIGENCE MODULES — Anomaly detection, statistical tests, feature importance
# Only activate when a binary target column is detected (fraud, attrition, etc.)
# ─────────────────────────────────────────────────────────────────────────────

_BINARY_TARGET_KEYWORDS = [
    "fraud", "attrition", "survived", "churn", "default",
    "flag", "dark_web", "is_fraud", "is_default", "target",
    "label", "class", "outcome", "event", "failure",
]

# Canonical mapping for string binary values → 0/1
_BINARY_STRING_MAP = {
    "yes": 1, "no": 0,
    "true": 1, "false": 0,
    "y": 1, "n": 0,
    "t": 1, "f": 0,
    "1": 1, "0": 0,
    "positive": 1, "negative": 0,
    "present": 1, "absent": 0,
}


def _to_binary_series(series: pd.Series) -> Optional[pd.Series]:
    """
    Convert a series to float 0/1.
    Handles numeric 0/1 and string variants (Yes/No, True/False, etc.).
    Returns None if conversion is not possible.
    """
    if series.dtype in ("int64", "float64", "int32", "float32"):
        vals = set(series.dropna().unique())
        if vals <= {0, 1}:
            return series.astype(float)
        return None
    if series.dtype in ("object", "string"):
        mapped = series.dropna().str.strip().str.lower().map(_BINARY_STRING_MAP)
        if mapped.isna().sum() == 0:  # all values mapped successfully
            return series.str.strip().str.lower().map(_BINARY_STRING_MAP).astype(float)
    return None


def _detect_binary_target(df: pd.DataFrame) -> Optional[str]:
    """
    Return the first column that is binary (0/1, Yes/No, True/False, etc.)
    and whose name hints at being a classification target.
    Supports both numeric and string binary columns.
    """
    for col in df.columns:
        col_lower = col.lower().replace("_", "").replace(" ", "")
        if not any(kw.replace("_", "") in col_lower
                   for kw in _BINARY_TARGET_KEYWORDS):
            continue
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) != 2:
            continue
        # Numeric 0/1
        if df[col].dtype in ("int64", "float64", "int32", "float32"):
            if set(unique_vals) <= {0, 1}:
                return col
        # String binary (Yes/No, True/False, etc.)
        elif df[col].dtype in ("object", "string"):
            vals_lower = {str(v).strip().lower() for v in unique_vals}
            if vals_lower <= set(_BINARY_STRING_MAP.keys()):
                return col
    return None


def _detect_anomalies(df: pd.DataFrame, numeric_cols: list,
                      contamination: float = 0.05):
    """
    Run Isolation Forest on numeric columns.
    Returns (scores_array, anomaly_index_list) or (None, None) on failure.
    """
    if len(numeric_cols) < 2:
        return None, None
    try:
        from sklearn.ensemble import IsolationForest
        X = df[numeric_cols].fillna(df[numeric_cols].mean()).values
        iso = IsolationForest(contamination=contamination, random_state=42,
                              n_estimators=100)
        preds  = iso.fit_predict(X)
        scores = iso.score_samples(X)
        anomaly_indices = df.index[preds == -1].tolist()
        return scores, anomaly_indices
    except Exception as e:
        print(f"[intelligence] Anomaly detection failed: {e}")
        return None, None


def _statistical_tests(df: pd.DataFrame, target_col: str) -> list:
    """
    Run t-tests (numeric) and chi-square tests (categorical) comparing
    target=0 vs target=1 groups. Returns list of significant results (p<0.05).
    Supports both numeric and string binary target columns.
    """
    results = []
    try:
        from scipy.stats import ttest_ind, chi2_contingency

        # Convert target to numeric 0/1 for grouping
        target_series = _to_binary_series(df[target_col])
        if target_series is None:
            print(f"[intelligence] Cannot convert {target_col!r} to binary — skipping tests")
            return []

        for col in df.columns:
            if col == target_col:
                continue
            try:
                if df[col].dtype in ("int64", "float64", "int32", "float32"):
                    g0 = df[target_series == 0][col].dropna()
                    g1 = df[target_series == 1][col].dropna()
                    if len(g0) > 1 and len(g1) > 1:
                        _, p = ttest_ind(g0, g1, equal_var=False)
                        if p < 0.05:
                            results.append({
                                "column":         col,
                                "test":           "t-test",
                                "p_value":        round(float(p), 4),
                                "mean_0":         round(float(g0.mean()), 2),
                                "mean_1":         round(float(g1.mean()), 2),
                                "interpretation": (
                                    f"Significant difference in {col} between "
                                    f"target=0 (mean={g0.mean():.2f}) and "
                                    f"target=1 (mean={g1.mean():.2f}) — p={p:.4f}"
                                ),
                            })
                elif df[col].dtype in ("object", "string"):
                    ct = pd.crosstab(df[col], target_series)
                    if ct.shape[0] > 1 and ct.shape[1] > 1:
                        chi2, p, _, _ = chi2_contingency(ct)
                        if p < 0.05:
                            results.append({
                                "column":         col,
                                "test":           "chi-square",
                                "p_value":        round(float(p), 4),
                                "interpretation": (
                                    f"Significant association between {col} "
                                    f"and {target_col} — p={p:.4f}"
                                ),
                            })
            except Exception:
                continue
        results.sort(key=lambda x: x["p_value"])
    except Exception as e:
        print(f"[intelligence] Statistical tests failed: {e}")
    return results


def _feature_importance(df: pd.DataFrame, target_col: str,
                        numeric_cols: list) -> list:
    """
    Train a shallow Decision Tree to get feature importances for target_col.
    Returns top-5 (feature_name, importance) pairs sorted descending.
    Supports both numeric and string binary target columns.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import LabelEncoder

        # Convert target to numeric 0/1
        y_series = _to_binary_series(df[target_col])
        if y_series is None:
            print(f"[intelligence] Cannot convert {target_col!r} to binary for feature importance")
            return []
        y = y_series.fillna(0)

        X = df.copy()
        # Encode categorical columns
        for col in X.select_dtypes(include=["object", "string"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        # Use all numeric columns except the target
        feature_cols = [c for c in X.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns if c != target_col]
        if not feature_cols:
            return []
        X_feat = X[feature_cols].fillna(0)
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_feat, y)
        pairs = sorted(
            zip(feature_cols, clf.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        return [(str(f), round(float(i), 4)) for f, i in pairs[:5] if i > 0]
    except Exception as e:
        print(f"[intelligence] Feature importance failed: {e}")
        return []


def _categorical_group_analysis(df: pd.DataFrame,
                                numeric_target_col: str = None) -> str:
    """
    For every categorical column (≤20 unique values), compute group-by stats
    against the best numeric target column. Returns a formatted text block.
    """
    # Pick numeric target: prefer named columns, else highest-variance numeric
    if numeric_target_col and numeric_target_col in df.columns:
        target = numeric_target_col
    else:
        preferred = ["popularity", "score", "sales", "price", "value",
                     "rating", "revenue", "salary", "weekly_sales"]
        target = next(
            (c for c in preferred if c.lower() in
             [x.lower() for x in df.columns]),
            None
        )
        if not target:
            num_cols = [c for c in df.select_dtypes(include="number").columns
                        if not _is_id_column(df, c)]
            if num_cols:
                target = max(num_cols, key=lambda c: df[c].std() / max(df[c].mean(), 1e-9)
                             if df[c].mean() != 0 else 0)
    if not target:
        return ""

    cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns
                if df[c].nunique() <= 20 and not _is_id_column(df, c)]
    if not cat_cols:
        return ""

    lines = [f"\n=== Group-by Analysis (target: {target}) ==="]
    for cat in cat_cols[:3]:  # limit to 3 categorical columns
        try:
            grp = df.groupby(cat)[target].agg(
                count="count", mean="mean", median="median"
            ).reset_index().sort_values("count", ascending=False)

            total = grp["count"].sum()
            lines.append(f"\n{cat} breakdown (n={total:,}):")

            # Top 5 categories
            for _, row in grp.head(5).iterrows():
                pct  = row["count"] / total * 100
                warn = " ⚠ small sample" if row["count"] < 30 else ""
                lines.append(
                    f"  {row[cat]}: n={int(row['count'])} ({pct:.1f}%), "
                    f"mean={row['mean']:.2f}, median={row['median']:.2f}{warn}"
                )

            # Narrative highlights
            top_count = grp.iloc[0]
            top_mean  = grp.loc[grp["mean"].idxmax()]
            low_mean  = grp.loc[grp["mean"].idxmin()]

            lines.append(
                f"  → Dominant: {top_count[cat]} ({top_count['count']/total*100:.0f}% of records)"
            )
            if top_mean["count"] < 30:
                lines.append(
                    f"  → Highest mean: {top_mean[cat]} ({top_mean['mean']:.2f}) "
                    f"⚠ only {int(top_mean['count'])} records — may be driven by outliers"
                )
            else:
                lines.append(
                    f"  → Highest mean: {top_mean[cat]} ({top_mean['mean']:.2f})"
                )
            lines.append(
                f"  → Lowest mean: {low_mean[cat]} ({low_mean['mean']:.2f})"
            )
        except Exception as e:
            print(f"[group_analysis] Failed for {cat}: {e}")
            continue

    lines.append("=== End Group-by Analysis ===\n")
    result = "\n".join(lines)
    print(f"[analyzer] Group-by analysis: {len(cat_cols)} categorical cols vs {target!r}")
    return result


def _explain_outlier_impact(df: pd.DataFrame, numeric_col: str,
                            categorical_cols: list) -> str:
    """
    For each categorical column, compare mean of numeric_col with and without
    outliers (IQR method). Flags categories where mean changes >50%.
    """
    if numeric_col not in df.columns:
        return ""
    try:
        q1, q3 = df[numeric_col].quantile(0.25), df[numeric_col].quantile(0.75)
        iqr     = q3 - q1
        lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask_clean = (df[numeric_col] >= lo) & (df[numeric_col] <= hi)
        n_outliers = (~mask_clean).sum()
        if n_outliers == 0:
            return ""

        lines = [
            f"\n=== Outlier Impact Analysis ({numeric_col}: {n_outliers} outliers) ==="
        ]
        found_any = False
        for cat in categorical_cols[:3]:
            if cat not in df.columns:
                continue
            try:
                for val in df[cat].dropna().unique():
                    mask_cat   = df[cat] == val
                    mean_all   = df.loc[mask_cat, numeric_col].mean()
                    mean_clean = df.loc[mask_cat & mask_clean, numeric_col].mean()
                    n_cat      = mask_cat.sum()
                    if pd.isna(mean_all) or pd.isna(mean_clean) or mean_all == 0:
                        continue
                    change_pct = abs(mean_all - mean_clean) / abs(mean_all) * 100
                    abs_diff   = abs(mean_all - mean_clean)
                    if change_pct > 30 or (abs_diff > 0.5 and n_cat < 50):
                        found_any = True
                        lines.append(
                            f"  {cat}={val!r}: mean={mean_all:.2f} with outliers, "
                            f"{mean_clean:.2f} without — {change_pct:.0f}% change. "
                            f"n={n_cat}" +
                            (" ⚠ small sample" if n_cat < 30 else "")
                        )
            except Exception:
                continue

        if not found_any:
            return ""
        lines.append("=== End Outlier Impact ===\n")
        print(f"[analyzer] Outlier impact: {n_outliers} outliers in {numeric_col!r}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[analyzer] Outlier impact failed: {e}")
        return ""


def _duplicate_insight(df: pd.DataFrame) -> str:
    """
    Detect duplicate rows and return an alert string if >1% duplicates found.
    """
    try:
        dup_count = int(df.duplicated().sum())
        dup_pct   = dup_count / max(len(df), 1) * 100
        if dup_pct <= 1.0:
            return ""

        # Find columns most responsible for duplicates
        top_dup_cols = []
        for col in df.columns[:10]:
            col_dup = int(df.duplicated(subset=[col]).sum())
            if col_dup > 0:
                top_dup_cols.append((col, col_dup))
        top_dup_cols.sort(key=lambda x: x[1], reverse=True)
        top_names = ", ".join(c for c, _ in top_dup_cols[:3])

        block = (
            f"\n=== Data Quality Alert: Duplicates ===\n"
            f"{dup_count:,} duplicate rows ({dup_pct:.1f}% of data) detected. "
            f"Consider deduplicating before analysis to avoid over-representing "
            f"the same entities. Most duplicated columns: {top_names}.\n"
            f"=== End Duplicate Alert ===\n"
        )
        print(f"[analyzer] Duplicates: {dup_count:,} ({dup_pct:.1f}%)")
        return block
    except Exception as e:
        print(f"[analyzer] Duplicate check failed: {e}")
        return ""


def _build_intel_block(df: pd.DataFrame) -> str:
    """
    Run all intelligence modules and return a formatted string to inject
    into the LLM prompt. Returns empty string if no binary target found.
    """
    target_col = _detect_binary_target(df)
    if not target_col:
        return ""

    print(f"[INTELLIGENCE] Binary target detected: {target_col!r}")
    lines = [f"\n=== AI Intelligence Analysis (target: {target_col}) ==="]

    # Anomaly detection
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != target_col
        # Only skip integer ID columns (not continuous floats like credit_limit)
        and not ("id" in c.lower() and df[c].dtype in ("int64", "int32")
                 and len(df) > 0 and df[c].nunique() / len(df) > 0.95)
    ]
    scores, anomalies = _detect_anomalies(df, numeric_cols)
    if anomalies is not None:
        pct = round(len(anomalies) / max(len(df), 1) * 100, 1)
        lines.append(
            f"Anomaly detection (Isolation Forest): {len(anomalies)} anomalous "
            f"rows ({pct}% of data). These rows have unusual feature combinations "
            f"and warrant investigation."
        )
        print(f"[INTELLIGENCE] Anomalies: {len(anomalies)} ({pct}%)")

    # Statistical tests
    stat_tests = _statistical_tests(df, target_col)
    if stat_tests:
        lines.append(f"\nStatistically significant features (p<0.05) for {target_col}:")
        for t in stat_tests[:5]:
            lines.append(f"  - {t['interpretation']}")
        print(f"[INTELLIGENCE] Significant features: {len(stat_tests)}")

    # Feature importance
    importance = _feature_importance(df, target_col, numeric_cols)
    if importance:
        lines.append(f"\nTop predictors for {target_col} (Decision Tree importance):")
        for feat, imp in importance:
            lines.append(f"  - {feat}: {imp:.3f}")
        lines.append(
            "Use these features to build predictive models or focus risk mitigation."
        )
        print(f"[INTELLIGENCE] Top feature: {importance[0][0]} ({importance[0][1]:.3f})")

    lines.append("=== End Intelligence Analysis ===\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation pre-computation (injected into prompt so LLM uses real r values)
# ─────────────────────────────────────────────────────────────────────────────
def _build_correlations(df: pd.DataFrame) -> list:
    """
    Compute Pearson correlation coefficients for all pairs of numeric columns.
    Returns a list of dicts sorted by |r| descending, capped at 10 pairs.
    Only includes pairs where both columns have at least 30 non-null values.
    Excludes ID columns (name contains 'id' OR integer column with 100% unique values).
    """
    def _is_corr_skip(col: str) -> bool:
        """Skip column for correlation if it's an ID (not just high-cardinality continuous)."""
        if "id" in col.lower():
            return True
        # Only skip integer columns with near-100% uniqueness (true IDs)
        # Continuous floats (Temperature, Sales) are high-cardinality but meaningful
        if df[col].dtype in ("int64", "int32") and len(df) > 0:
            if df[col].nunique() / len(df) > 0.95:
                return True
        return False

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if not _is_corr_skip(c) and df[c].dropna().shape[0] >= 30
    ]
    pairs = []
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1:]:
            try:
                r = df[[col_a, col_b]].dropna()
                if len(r) < 30:
                    continue
                r_val = round(float(r[col_a].corr(r[col_b])), 2)
                if pd.isna(r_val):
                    continue
                abs_r = abs(r_val)
                strength = (
                    "strong"   if abs_r > 0.7 else
                    "moderate" if abs_r > 0.3 else
                    "weak"
                )
                pairs.append({
                    "col_a":    col_a,
                    "col_b":    col_b,
                    "r":        r_val,
                    "strength": strength,
                })
            except Exception:
                continue
    # Sort by |r| descending, return top 10
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return pairs[:10]


# ─────────────────────────────────────────────────────────────────────────────
# Chart summarizer — generates natural-language captions for each chart type
# ─────────────────────────────────────────────────────────────────────────────
def _generate_chart_summary(chart_type: str, x_col: Optional[str],
                             y_col: Optional[str], df: pd.DataFrame) -> str:
    """
    Generate a natural-language summary sentence for a rendered chart.
    Used as the caption/insight text below each chart in the PDF report.

    Supports: histogram, bar, scatter, line, pie.
    Falls back to a generic description for unknown types.
    """
    import numpy as np
    try:
        from scipy.stats import skew as _skew
    except ImportError:
        _skew = None

    try:
        # ── Histogram (distribution of a numeric column) ──────────────────
        if chart_type == "histogram" and x_col and x_col in df.columns:
            data = df[x_col].dropna()
            if len(data) == 0:
                return f"No data available for {x_col}."
            mean_val   = float(data.mean())
            median_val = float(data.median())
            min_val    = float(data.min())
            max_val    = float(data.max())

            # Skewness
            if _skew is not None and len(data) >= 3:
                skewness = float(_skew(data))
                if skewness > 1:
                    shape = "right-skewed"
                elif skewness < -1:
                    shape = "left-skewed"
                else:
                    shape = "approximately symmetric"
            else:
                diff = mean_val - median_val
                if abs(diff) < 0.05 * max(abs(mean_val), 1):
                    shape = "approximately symmetric"
                elif diff > 0:
                    shape = "right-skewed"
                else:
                    shape = "left-skewed"

            # Outliers via IQR
            q1  = float(data.quantile(0.25))
            q3  = float(data.quantile(0.75))
            iqr = q3 - q1
            outlier_count = int(((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum())
            outlier_note  = f" with {outlier_count} outliers beyond the whiskers" if outlier_count > 0 else ""

            label = x_col.replace("_", " ").title()
            return (
                f"The distribution of {label} is {shape} "
                f"(mean = {mean_val:.2f}, median = {median_val:.2f}), "
                f"ranging from {min_val:.2f} to {max_val:.2f}{outlier_note}."
            )

        # ── Bar chart (categorical counts or aggregated values) ───────────
        elif chart_type == "bar" and x_col and x_col in df.columns:
            if y_col and y_col in df.columns:
                # Aggregated bar: describe top category by y value
                grp = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
                if len(grp) == 0:
                    return f"Bar chart of {y_col} by {x_col}."
                top_cat = str(grp.index[0])
                top_val = float(grp.iloc[0])
                total   = float(grp.sum())
                pct     = top_val / total * 100 if total > 0 else 0
                x_label = x_col.replace("_", " ").title()
                y_label = y_col.replace("_", " ").title()
                if pct > 50:
                    return (
                        f"The {x_label} category is dominated by '{top_cat}', "
                        f"accounting for {pct:.1f}% of total {y_label}."
                    )
                else:
                    return (
                        f"The most common {x_label} by {y_label} is '{top_cat}' "
                        f"({pct:.1f}%), with a diverse distribution across categories."
                    )
            else:
                # Count-based bar
                counts  = df[x_col].value_counts()
                if len(counts) == 0:
                    return f"No data for {x_col}."
                top_cat = str(counts.index[0])
                top_pct = float(counts.iloc[0]) / len(df) * 100
                x_label = x_col.replace("_", " ").title()
                if top_pct > 50:
                    return (
                        f"The {x_label} category is dominated by '{top_cat}', "
                        f"accounting for {top_pct:.1f}% of records."
                    )
                else:
                    return (
                        f"The most common {x_label} is '{top_cat}' "
                        f"({top_pct:.1f}% of records), with a diverse distribution."
                    )

        # ── Scatter plot (correlation between two numeric columns) ─────────
        elif chart_type == "scatter" and x_col and y_col \
                and x_col in df.columns and y_col in df.columns:
            pair = df[[x_col, y_col]].dropna()
            if len(pair) < 5:
                return f"Scatter plot of {y_col} vs {x_col}."
            r_val   = float(pair[x_col].corr(pair[y_col]))
            abs_r   = abs(r_val)
            strength  = "strong" if abs_r > 0.7 else "moderate" if abs_r > 0.3 else "weak"
            direction = "positive" if r_val > 0 else "negative"
            x_label   = x_col.replace("_", " ").title()
            y_label   = y_col.replace("_", " ").title()
            return (
                f"The scatter plot reveals a {strength} {direction} correlation "
                f"(r = {r_val:.2f}) between {x_label} and {y_label}."
            )

        # ── Line chart (time series trend) ────────────────────────────────
        elif chart_type == "line" and x_col and y_col \
                and x_col in df.columns and y_col in df.columns:
            df_sorted = df.sort_values(x_col)
            y_vals    = df_sorted[y_col].dropna()
            if len(y_vals) < 2:
                return f"Line chart of {y_col} over {x_col}."
            peak_val   = float(y_vals.max())
            trough_val = float(y_vals.min())
            first_val  = float(y_vals.iloc[0])
            last_val   = float(y_vals.iloc[-1])
            trend = (
                "increasing" if last_val > first_val * 1.02 else
                "decreasing" if last_val < first_val * 0.98 else
                "stable"
            )
            y_label = y_col.replace("_", " ").title()
            x_label = x_col.replace("_", " ").title()
            return (
                f"Over time, {y_label} shows a {trend} trend, "
                f"peaking at {peak_val:.2f} and troughing at {trough_val:.2f} "
                f"(range: {peak_val - trough_val:.2f})."
            )

        # ── Pie chart (categorical proportions) ───────────────────────────
        elif chart_type == "pie" and x_col and x_col in df.columns:
            counts  = df[x_col].value_counts()
            if len(counts) == 0:
                return f"Pie chart of {x_col}."
            top_cat = str(counts.index[0])
            top_pct = float(counts.iloc[0]) / len(df) * 100
            x_label = x_col.replace("_", " ").title()
            return (
                f"The pie chart shows {x_label} distribution; "
                f"'{top_cat}' is the largest segment at {top_pct:.1f}%."
            )

    except Exception as _e:
        print(f"[chart_summary] Failed for type={chart_type!r} x={x_col!r} y={y_col!r}: {_e}")

    # ── Fallback ──────────────────────────────────────────────────────────
    if x_col and y_col:
        return (
            f"Chart displays the relationship between "
            f"{x_col.replace('_', ' ').title()} and {y_col.replace('_', ' ').title()}."
        )
    if x_col:
        return f"Chart shows the distribution of {x_col.replace('_', ' ').title()}."
    return f"Chart shows {chart_type} of the data."


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
        "correlations": _build_correlations(df),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — JSON-spec prompt (replaces Python code generation)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_prompt(context: dict, quality: dict, domain_risk: str = None,
                     intel_block: str = "",
                     group_block: str = "",
                     outlier_block: str = "",
                     duplicate_block: str = "",
                     domain: str = "GENERIC_TABULAR") -> str:
    """
    Ask the LLM for a JSON spec, not Python code.
    InsightStream renders the charts itself from the spec — zero exec() risk.
    domain_risk: "sports" | "entertainment" | None — adds extra rule 8 if set.
    intel_block: pre-computed intelligence results to inject into the prompt.
    group_block: categorical group-by analysis results.
    outlier_block: outlier impact explanation per category.
    duplicate_block: duplicate row detection alert.
    domain: classified domain (PEOPLE_CATALOG, MEDIA_CATALOG, etc.)
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

    # Rule 8: domain-specific financial language ban
    financial_ban = ""
    if domain_risk in ("sports", "entertainment"):
        financial_ban = f"""
- Rule 8 — This is a {domain_risk.upper()} dataset. Do NOT use financial language.
  NEVER include words like "revenue", "profit", "sales", "customer acquisition",
  "product pricing", "cost reduction", "profit margin", "ROI", "gross margin",
  "monetize", or any business/financial term not present in the column list.
  Use domain-appropriate language: match outcomes, player performance, venue stats,
  content ratings, viewership, etc.
"""

    # Domain-specific forbidden concepts block
    domain_forbidden_block = ""
    try:
        from classifiers.domain import DOMAIN_FORBIDDEN_CONCEPTS
        _forbidden = DOMAIN_FORBIDDEN_CONCEPTS.get(domain.upper(), [])
        if _forbidden:
            domain_forbidden_block = (
                f"\nThis is a {domain.upper()} dataset. "
                f"The following concepts are FORBIDDEN in recommendations "
                f"(they do not exist in this dataset):\n"
                + "\n".join(f"  - {c}" for c in _forbidden)
                + "\n"
            )
    except Exception:
        pass

    # Rules 9 & 10: correlation reporting + actionable recommendations
    correlations = context.get("correlations", [])
    corr_block = ""
    if correlations:
        corr_lines = []
        for p in correlations[:5]:  # top 5 pairs in prompt
            corr_lines.append(
                f"  {p['col_a']} vs {p['col_b']}: r={p['r']} ({p['strength']})"
            )
        corr_block = "Pre-computed Pearson correlations (use these exact values):\n" + "\n".join(corr_lines)
        print(f"[analyzer] Injected top {len(correlations[:5])} correlation pairs into prompt")

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
- Rule 9 — Correlations (MANDATORY): You are provided with pre-computed Pearson
  correlation pairs below. You MUST include at least 3 insights that reference the
  top 3 absolute correlation pairs. For each, report the exact r value (rounded to
  2 decimals) and the strength label. Example: "Weekly_Sales shows a strong positive
  correlation with Temperature (r=0.97)." Do NOT invent correlations — use only the
  exact values from the injected list. If a top pair involves a store/ID column
  (e.g., Store vs Weekly_Sales), still report it.
- Rule 10 — Recommendations (MANDATORY column references): Generate 3-5 actionable
  recommendations. Each recommendation MUST contain at least one actual column name
  from the dataset (e.g., Weekly_Sales, Store, Temperature, Holiday_Flag, Date,
  Fuel_Price, CPI, Unemployment). Recommendations without any column name will be
  rejected. Do NOT use generic phrases like "investigate underlying drivers" or
  "optimise operations". Be specific with concrete numbers from the data.
  Example: "Increase inventory in May to prepare for the June peak in Weekly_Sales
  (historical lift 72%)."
- Rule 11 — Domain-appropriate recommendations only. Each recommendation must be
  directly derived from the columns and statistics in THIS dataset. Do NOT include
  any recommendation that mentions:
  * TV-MA, TV-PG, G, PG, or any rating system for movies/TV shows
  * Content origin (India, UK, Canada, etc.) unless a country/origin column exists
  * Catalogue refresh, licensing, release year (unless 'release_year' is a column)
  * Parental controls, kids profiles, streaming tiers, subscription models
  * Any term or concept that does NOT appear in the column names or data summary
  For celebrity/people datasets: focus on department analysis, outliers, duplicates,
  gender differences, or popularity trends. Never invent entertainment-platform advice.

Few-shot examples of correct recommendations (adapt to this dataset's columns):
{{"text": "Analyse Weekly_Sales performance during weeks where Holiday_Flag = 1 to quantify the sales lift (currently 7% of weeks are holidays).", "timeframe": "Next 14 days", "owner": "Analytics team", "impact": "Important"}}
{{"text": "Review stores with the lowest mean Weekly_Sales (e.g., Store 45: $381,869 vs Store 1: $1,641,690) to identify operational or demographic factors.", "timeframe": "Next 30 days", "owner": "Regional manager", "impact": "Critical"}}
{{"text": "Use the strong correlation between Weekly_Sales and Temperature (r=0.97) to align staffing and inventory with seasonal temperature changes.", "timeframe": "Next quarter", "owner": "Supply chain team", "impact": "Important"}}

Do NOT return insights with empty or missing text fields.
Do NOT return recommendations that mention topics unrelated to the dataset columns.
{domain_forbidden_block}
- Key Takeaway (MANDATORY): The FIRST insight in the list must be titled "Key Takeaway"
  and answer: "If the user remembers only one thing from this report, what should it be?"
  It must be a single, striking sentence with the most important finding and a specific number.
  Example: "93% of records are actors — all popularity trends are actor-driven; small
  departments like Visual Effects appear popular only due to single outliers."
  Also ensure that all 3-5 recommendations are strictly about the dataset at hand.
  Do not import recommendations from other contexts (e.g., do not suggest streaming
  platform features for a celebrity dataset, or health protocols for a sales dataset).
{missing_note}
{financial_ban}
Available columns: {context["columns"]}

{corr_block}
{group_block}
{outlier_block}
{duplicate_block}
{intel_block}
Dataset summary:
{context_str}
"""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — JSON spec renderer (replaces exec-based sandbox entirely)
# ─────────────────────────────────────────────────────────────────────────────

# Numeric ID columns — skip when choosing a fallback histogram column
_SKIP_COLS = {"EmpID", "ManagerID", "PositionID", "Zip", "DeptID",
              "empid", "managerid", "positionid", "zip", "deptid"}


def _is_id_column(df: pd.DataFrame, col: str) -> bool:
    """
    Return True if a column is likely an ID column and should be skipped for charting.
    Heuristics:
      1. Column name contains 'id' (case-insensitive)
      2. More than 90% of values are unique (high-cardinality identifier)
    """
    if "id" in col.lower():
        return True
    if len(df) > 0 and df[col].nunique() / len(df) > 0.9:
        return True
    return False


def _render_from_spec(spec: dict, df: pd.DataFrame) -> dict:
    """
    Render Plotly charts from a JSON spec produced by the LLM.
    No exec(), no code generation, no sandbox needed.
    InsightStream owns all chart-rendering logic — the LLM only supplies data.

    Returns charts as a list of dicts:
        {"fig": <Plotly Figure>, "summary": <str>, "x_col": <str>, "y_col": <str>,
         "chart_type": <str>, "title": <str>}
    """
    import plotly.express as px

    valid_charts = []

    def _make_chart_dict(fig, chart_type, x, y, title):
        """Wrap a Plotly figure with its natural-language summary."""
        summary = _generate_chart_summary(chart_type, x, y, df)
        return {
            "fig":        fig,
            "summary":    summary,
            "x_col":      x or "",
            "y_col":      y or "",
            "chart_type": chart_type,
            "title":      title,
        }

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

            # ── Skip ID columns — they produce meaningless charts ─────────
            if x and _is_id_column(df, x):
                print(f"[analyzer] Skipping ID column: {x}")
                raise ValueError(f"Column {x!r} is an ID column — skipping")

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

            valid_charts.append(_make_chart_dict(fig, chart_type, x, y, title))

        except Exception as e:
            print(f"[analyzer] Chart spec failed ({e}), trying fallback")
            # Fallback: histogram of first meaningful numeric column (skip IDs)
            for col in df.select_dtypes(include="number").columns:
                if not _is_id_column(df, col):
                    try:
                        fb_title = f"{col.replace('_', ' ').title()} Distribution"
                        fb = px.histogram(df, x=col, title=fb_title, nbins=20)
                        valid_charts.append(_make_chart_dict(fb, "histogram", col, None, fb_title))
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
                        if not _is_id_column(df, c)]
        cat_cols     = [c for c in df.select_dtypes(include="object").columns
                        if not _is_id_column(df, c)]

        # Histogram of first meaningful numeric column
        if numeric_cols and len(valid_charts) < 2:
            try:
                col      = numeric_cols[0]
                fb_title = f"{col.replace('_', ' ').title()} Distribution"
                fb       = px.histogram(df, x=col, nbins=20, title=fb_title)
                valid_charts.append(_make_chart_dict(fb, "histogram", col, None, fb_title))
                print(f"[analyzer] Fallback chart added: histogram of {col!r}")
            except Exception as _fe:
                print(f"[analyzer] Fallback histogram failed: {_fe}")

        # Bar chart of first categorical column
        if cat_cols and len(valid_charts) < 2:
            try:
                col     = cat_cols[0]
                plot_df = df[col].value_counts().head(15).reset_index()
                plot_df.columns = [col, "count"]
                fb_title = f"{col.replace('_', ' ').title()} Breakdown"
                fb = px.bar(plot_df, x=col, y="count", title=fb_title)
                valid_charts.append(_make_chart_dict(fb, "bar", col, None, fb_title))
                print(f"[analyzer] Fallback chart added: bar of {col!r}")
            except Exception as _fe:
                print(f"[analyzer] Fallback bar chart failed: {_fe}")

        # Second numeric histogram if still short
        if len(numeric_cols) > 1 and len(valid_charts) < 2:
            try:
                col      = numeric_cols[1]
                fb_title = f"{col.replace('_', ' ').title()} Distribution"
                fb       = px.histogram(df, x=col, nbins=20, title=fb_title)
                valid_charts.append(_make_chart_dict(fb, "histogram", col, None, fb_title))
                print(f"[analyzer] Fallback chart added: histogram of {col!r}")
            except Exception:
                pass

    print(f"[analyzer] Final chart count: {len(valid_charts)}")

    # Extract raw Plotly figures for downstream compatibility
    # (validate_results and the post-render guard expect fig objects)
    raw_figs = [c["fig"] for c in valid_charts]

    return {
        "insights":        spec.get("insights", []),
        "charts":          raw_figs,
        "chart_metas":     valid_charts,   # full dicts with summary, x_col, y_col, etc.
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
        "chart_metas":     results.get("chart_metas", []),
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
    # Health / pandemic hallucinations — always forbidden
    "case fatality", "recovery rate", "public health", "disease outbreak",
    "pandemic", "covid", "corona", "virus", "health ministry",
    "public health authority", "high-burden nations", "international aid",
    "treatment protocol", "surge planning", "fatality rate", "outbreak",
    "vaccination", "quarantine", "epidemi", "mortality rate",
    # Entertainment-platform hallucinations — forbidden for non-streaming datasets
    "tv-ma", "tv-14", "tv-pg", "tv-g", "parental controls", "kids profiles",
    "content origin", "catalogue refresh", "licensing", "release window",
    "content acquisition", "original tv show", "family-friendly titles",
    "streaming platform", "content library", "subscription tier",
]

# Financial terms — only forbidden for non-financial datasets (sports, entertainment, HR, etc.)
_FINANCIAL_KEYWORDS = [
    "revenue", "profit", "sales volume", "customer acquisition",
    "product pricing", "cost reduction", "profit margin",
    "return on investment", " roi ", "gross margin", "net revenue",
    "monetize", "monetisation", "monetization",
]

# Column patterns that indicate a financial dataset (safe to use financial terms)
_FINANCIAL_COL_SIGNALS = {
    "revenue", "profit", "sales", "price", "cost", "margin",
    "revenue_total", "gross_profit", "net_income", "cogs",
}


def _detect_financial_language_risk(df: pd.DataFrame):
    """
    Detect if the dataset is NON-financial but the LLM might hallucinate
    financial language (e.g., cricket/sports data → LLM writes 'revenue').

    Returns:
      "sports"          — dataset looks like sports/cricket data
      "entertainment"   — dataset looks like streaming/entertainment data
      None              — dataset is financial or unknown (no extra rule needed)
    """
    cols_lower = {c.lower().replace("_", "").replace(" ", "") for c in df.columns}

    # If the dataset already has financial columns, financial language is fine
    if any(sig in cols_lower for sig in _FINANCIAL_COL_SIGNALS):
        return None

    # Sports/cricket signals
    sports_signals = {"winner", "tosswinner", "team1", "team2", "venue",
                      "battingteam", "bowlingteam", "runs", "wickets",
                      "homerun", "awayrun", "innings", "over"}
    if len(cols_lower & sports_signals) >= 2:
        return "sports"

    # Entertainment signals
    entertainment_signals = {"listedin", "dateadded", "releaseyear",
                             "showid", "genre", "rating"}
    if len(cols_lower & entertainment_signals) >= 2:
        return "entertainment"

    return None


def validate_recommendation(rec_text: str, df_columns: list,
                            computed_stats: dict = None) -> bool:
    """
    Reject recommendations that cite statistics not computable from the
    actual dataframe, or that reference concepts from the wrong domain.

    Checks:
    1. Any percentage cited in the text must appear in computed_stats
       (within ±1 percentage point tolerance).
    2. Column reference check is handled by _filter_recommendations.
    """
    import re as _re
    if not rec_text:
        return False

    if computed_stats:
        # Extract all percentages mentioned in the recommendation
        cited_pcts = [float(p.rstrip("%")) for p in _re.findall(r'\d+\.?\d*%', rec_text)]
        if cited_pcts:
            # Build the set of valid percentages from computed stats
            valid_pcts: set = set()
            for v in computed_stats.values():
                if isinstance(v, (int, float)):
                    # Raw fraction (0-1) → percentage
                    if 0 < v <= 1:
                        valid_pcts.add(round(v * 100, 1))
                    # Already a percentage (1-100)
                    elif 1 < v <= 100:
                        valid_pcts.add(round(float(v), 1))

            for pct in cited_pcts:
                # Allow ±1 pp tolerance
                if not any(abs(pct - vp) <= 1.0 for vp in valid_pcts):
                    print(f"[validate_rec] Rejected — hallucinated stat {pct}%: "
                          f"{rec_text[:60]}")
                    return False

    return True


def _filter_recommendations(recommendations: list, df_columns,
                            domain: str = "GENERIC_TABULAR",
                            computed_stats: dict = None) -> list:
    """
    Remove recommendations that:
    1. Contain forbidden keywords (hallucinated health/pandemic/entertainment content)
    2. Contain financial keywords when the dataset is non-financial
    3. Contain domain-specific forbidden concepts (e.g., TV ratings for people datasets)
    4. Do not reference any column name from the actual dataset
    5. Cite statistics not computable from the actual data (via validate_recommendation)
    """
    import re as _re

    # Load domain-specific forbidden concepts
    try:
        from classifiers.domain import DOMAIN_FORBIDDEN_CONCEPTS
        domain_forbidden = [
            kw.lower() for kw in
            DOMAIN_FORBIDDEN_CONCEPTS.get(domain.upper(), [])
        ]
    except Exception:
        domain_forbidden = []

    # Build column variant set
    col_variants: set = set()
    for col in df_columns:
        if len(col) <= 2:
            continue
        col_lower = col.lower()
        col_variants.add(col_lower)
        col_variants.add(_re.sub(r'([a-z])([A-Z])', r'\1 \2', col).lower())
        col_variants.add(col_lower.replace("_", "").replace(" ", ""))

    # Determine if financial language is appropriate for this dataset
    cols_lower_set = {c.lower().replace("_","").replace(" ","") for c in df_columns}
    dataset_is_financial = any(sig in cols_lower_set for sig in _FINANCIAL_COL_SIGNALS)

    # Build the active forbidden list
    active_forbidden = list(_FORBIDDEN_REC_KEYWORDS)
    if not dataset_is_financial:
        # Add financial terms only for non-financial datasets
        for kw in _FINANCIAL_KEYWORDS:
            if kw not in active_forbidden:
                active_forbidden.append(kw)

    def is_valid(rec: dict) -> bool:
        # Support both 'text' (LLM format) and 'action' (engine format)
        text = rec.get("text", "") or rec.get("action", "")
        text_lower = text.lower()
        if not text_lower:
            return False
        # Reject global forbidden keywords (health/pandemic/entertainment-platform)
        for kw in active_forbidden:
            if kw in text_lower:
                kw_stripped = kw.strip().replace(" ", "")
                if kw_stripped in cols_lower_set:
                    continue
                print(f"[filter_recs] Removed — forbidden keyword {kw!r}: {text_lower[:60]}")
                return False
        # Reject domain-specific forbidden concepts
        for kw in domain_forbidden:
            if kw in text_lower:
                print(f"[filter_recs] Removed — domain concept {kw!r}: {text_lower[:60]}")
                return False
        # Reject if no dataset column variant is mentioned
        if not any(v in text_lower for v in col_variants if len(v) > 2):
            print(f"[filter_recs] Removed — no column reference: {text_lower[:60]}")
            return False
        # Reject if cited statistics are not computable from the data
        if computed_stats and not validate_recommendation(text, list(df_columns), computed_stats):
            return False
        print(f"[filter_recs] KEPT (col ref): {text_lower[:80]}")
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
    n_corr  = len(context.get("correlations", []))
    print(f"[analyzer] Building context with {n_corr} correlation pairs")

    # ── 3b. Intelligence modules (anomaly, stats, feature importance) ─────
    intel_block = _build_intel_block(df)

    # ── 3c. Group-by, outlier impact, duplicate detection ─────────────────
    group_block    = _categorical_group_analysis(df)
    # Pick best numeric col for outlier impact (same logic as group analysis)
    _num_cols = [c for c in df.select_dtypes(include="number").columns
                 if not _is_id_column(df, c)]
    _preferred = ["popularity", "score", "sales", "price", "value",
                  "rating", "revenue", "salary", "weekly_sales"]
    _oc = next((c for c in _preferred
                if c.lower() in [x.lower() for x in df.columns]), None)
    if not _oc and _num_cols:
        _oc = max(_num_cols, key=lambda c: df[c].std() / max(abs(df[c].mean()), 1e-9))
    _cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns
                 if df[c].nunique() <= 20 and not _is_id_column(df, c)]
    outlier_block  = _explain_outlier_impact(df, _oc, _cat_cols) if _oc else ""
    duplicate_block = _duplicate_insight(df)

    # ── 3d. Domain classification (rule-based fast path, no LLM yet) ──────
    # Initialize with safe default — LLM-based classification happens after
    # the Groq client is created (step 5), but we need domain for the prompt.
    domain_info     = {"category": "GENERIC_TABULAR", "confidence": 0.5, "reason": ""}
    domain_template = {}
    try:
        from classifiers.domain import _rule_based_classify, get_domain_template
        _rb = _rule_based_classify(df)
        if _rb:
            domain_info     = {"category": _rb, "confidence": 0.95,
                               "reason": "Rule-based classification"}
            domain_template = get_domain_template(_rb)
            context["domain_info"]     = domain_info
            context["domain_template"] = domain_template
            print(f"[analyzer] Domain (rule-based): {_rb}")
    except Exception as _rbe:
        print(f"[analyzer] Rule-based domain classification failed: {_rbe}")

    # ── 4. Generate JSON prompt ───────────────────────────────────────────
    domain_risk = _detect_financial_language_risk(df)
    if domain_risk:
        print(f"[analyzer] Financial language risk detected: {domain_risk!r} — adding rule 8")
    prompt = _generate_prompt(context, quality, domain_risk=domain_risk,
                              intel_block=intel_block,
                              group_block=group_block,
                              outlier_block=outlier_block,
                              duplicate_block=duplicate_block,
                              domain=domain_info.get("category", "GENERIC_TABULAR"))

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

    # ── 5b. Refine domain with LLM if rule-based returned GENERIC_TABULAR ─
    if domain_info["category"] == "GENERIC_TABULAR":
        try:
            from classifiers.domain import classify_domain, get_domain_template
            domain_info     = classify_domain(df, client)
            domain_template = get_domain_template(domain_info["category"])
            context["domain_info"]     = domain_info
            context["domain_template"] = domain_template
            print(f"[analyzer] Domain (LLM): {domain_info['category']} "
                  f"(confidence={domain_info.get('confidence', '?'):.2f})")
        except Exception as _de:
            print(f"[analyzer] LLM domain classification failed: {_de}")

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
        results.get("recommendations", []), df.columns,
        domain=domain_info.get("category", "GENERIC_TABULAR"),
    )

    # ── 11. Cache successful result ───────────────────────────────────────
    _cache_set(fp, results)

    # ── 12. Return ────────────────────────────────────────────────────────
    # Attach domain classification info for downstream use
    results["domain_info"]     = context.get("domain_info", {})
    results["domain_template"] = context.get("domain_template", {})
    return results
