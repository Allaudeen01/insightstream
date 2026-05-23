"""
engine/classifiers/domain.py
============================
LLM-based domain classifier for dataset type detection.
Uses column schema + sample rows to classify into one of six categories.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Classifier prompt
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_CLASSIFIER_PROMPT = """You are a dataset domain classifier. Given the column schema and 3 sample rows, classify the dataset into EXACTLY ONE category.

CATEGORIES (with strong signals):

1. PEOPLE_CATALOG
   Signals: columns like `name`, `first_name`, `last_name`, `original_name`,
   `gender`, `birth_date`, `nationality`, `occupation`, `profession`,
   `known_for_*`, `dob`, `age`, `popularity`.
   Examples: celebrities, employees, customers, authors, athletes.

2. MEDIA_CATALOG
   Signals: columns like `title`, `release_date`, `runtime`, `genre`,
   `episode_count`, `season`, `director`, `cast`, `box_office`, `imdb_id`,
   `media_type`, `streaming_service`, `listed_in`, `date_added`.
   Examples: movies, TV shows, books, songs, podcasts.

3. ECOMMERCE_TRANSACTIONS
   Signals: `order_id`, `customer_id`, `product`, `sku`, `quantity`, `price`,
   `revenue`, `discount`, `transaction_date`, `payment_method`, `weekly_sales`.

4. FINANCIAL_TIMESERIES
   Signals: `date`, `open`, `high`, `low`, `close`, `volume`, `ticker`, `symbol`.

5. OPERATIONAL_METRICS
   Signals: `timestamp`, `metric_name`, `value`, `host`, `service`, `latency`,
   `error_rate`.

6. GENERIC_TABULAR (fallback when no category dominates)

CLASSIFICATION RULES:
- If `name` AND (`gender` OR `original_name` OR `birth_*` OR `known_for_*` OR `popularity`) are present → PEOPLE_CATALOG.
  This rule beats all other heuristics. People are not media.
- `title` without person-name columns → MEDIA_CATALOG.
- Never infer revenue/monetary semantics from a column unless its name contains
  `price`, `revenue`, `cost`, `amount`, `value`, `sales`, `gmv`, or currency symbols.
- Output ONLY the category name + confidence (0-1) + 1-sentence reason. No prose.

SCHEMA:
{schema}

SAMPLE_ROWS:
{sample_rows}

OUTPUT (JSON only, no markdown):
{{"category": "...", "confidence": 0.0, "reason": "..."}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Domain template map
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_TEMPLATE_MAP: Dict[str, Dict[str, Any]] = {
    "PEOPLE_CATALOG": {
        "report_title":          "People Dataset Analysis Report",
        "primary_entity_label":  "Total People",
        "forbidden_chart_types": [
            "pareto_revenue", "sales_funnel", "gmv_trend", "revenue_treemap",
            "revenue_by_category", "revenue_by_segment",
        ],
        "required_sections": [
            "top_n_table", "categorical_breakdown", "distribution_analysis",
        ],
    },
    "MEDIA_CATALOG": {
        "report_title":          "Content Library Analysis Report",
        "primary_entity_label":  "Total Titles",
        "forbidden_chart_types": ["demographic_pyramid"],
        "required_sections":     ["genre_breakdown", "release_timeline"],
    },
    "ECOMMERCE_TRANSACTIONS": {
        "report_title":          "Transaction Analytics Report",
        "primary_entity_label":  "Total Orders",
        "forbidden_chart_types": [],
        "required_sections":     ["pareto_revenue", "top_products", "cohort_analysis"],
    },
    "FINANCIAL_TIMESERIES": {
        "report_title":          "Financial Time Series Report",
        "primary_entity_label":  "Total Periods",
        "forbidden_chart_types": ["demographic_pyramid"],
        "required_sections":     ["trend_analysis", "volatility"],
    },
    "OPERATIONAL_METRICS": {
        "report_title":          "Operational Metrics Report",
        "primary_entity_label":  "Total Records",
        "forbidden_chart_types": [],
        "required_sections":     ["latency_distribution", "error_rates"],
    },
    "GENERIC_TABULAR": {
        "report_title":          "Data Analysis Report",
        "primary_entity_label":  "Total Records",
        "forbidden_chart_types": [],
        "required_sections":     [],
    },
}

VALID_CATEGORIES = set(DOMAIN_TEMPLATE_MAP.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific forbidden concepts for recommendation validation
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_FORBIDDEN_CONCEPTS: dict = {
    "PEOPLE_CATALOG": [
        "content ratings (TV-MA, PG, etc.)", "TV-MA", "TV-PG", "TV-14", "R-rated",
        "mature content", "release date", "catalogue freshness", "catalogue refresh",
        "regional origin", "local originals", "international growth", "regional content",
        "subscriber", "churn", "household account", "kids profile", "kids profiles",
        "revenue", "licensing", "box office", "episode count", "seasons",
        "post-2020", "post-2019", "newer releases", "streaming accounts",
    ],
    "MEDIA_CATALOG": [
        "demographic pyramid", "attrition", "employee turnover",
        "salary", "payroll", "headcount",
    ],
    "ECOMMERCE_TRANSACTIONS": [
        "case fatality", "recovery rate", "pandemic", "demographic pyramid",
    ],
    "FINANCIAL_TIMESERIES": [
        "demographic pyramid", "attrition", "content ratings",
    ],
    "OPERATIONAL_METRICS": [
        "content ratings", "box office", "demographic pyramid",
    ],
    "GENERIC_TABULAR": [],
}


def get_domain_template(domain: str) -> Dict[str, Any]:
    """Return the template for a given domain (defaults to GENERIC_TABULAR)."""
    return DOMAIN_TEMPLATE_MAP.get(domain, DOMAIN_TEMPLATE_MAP["GENERIC_TABULAR"])


def _rule_based_classify(df: pd.DataFrame) -> Optional[str]:
    """
    Fast rule-based pre-classification before calling the LLM.
    Returns a category string or None (meaning: use LLM).
    """
    cols_lower = {c.lower().replace("_", "").replace(" ", "") for c in df.columns}

    # PEOPLE_CATALOG: name + (gender | known_for_* | popularity | birth_*)
    has_name = any(c in cols_lower for c in ("name", "firstname", "lastname", "originalname"))
    has_person_signal = any(
        c in cols_lower for c in
        ("gender", "popularity", "dob", "birthdate", "age", "nationality",
         "occupation", "profession")
    ) or any("knownfor" in c for c in cols_lower)
    if has_name and has_person_signal:
        return "PEOPLE_CATALOG"

    # FINANCIAL_TIMESERIES: open/high/low/close/volume
    fin_ts = {"open", "high", "low", "close", "volume"}
    if len(cols_lower & fin_ts) >= 3:
        return "FINANCIAL_TIMESERIES"

    return None  # fall through to LLM


def classify_domain(df: pd.DataFrame, llm_client) -> Dict[str, Any]:
    """
    Classify the dataset domain using rule-based heuristics first,
    then the LLM as a fallback.

    Returns: {"category": str, "confidence": float, "reason": str}
    """
    # Fast rule-based check first (no LLM call needed)
    rule_result = _rule_based_classify(df)
    if rule_result:
        print(f"[domain_classifier] Rule-based: {rule_result}")
        return {
            "category":   rule_result,
            "confidence": 0.95,
            "reason":     f"Rule-based classification: column pattern matches {rule_result}",
        }

    # LLM classification
    try:
        schema      = {col: str(dtype) for col, dtype in df.dtypes.items()}
        sample_rows = df.head(3).fillna("").to_dict(orient="records")
        prompt      = DOMAIN_CLASSIFIER_PROMPT.format(
            schema=json.dumps(schema, indent=2),
            sample_rows=json.dumps(sample_rows, indent=2, default=str),
        )
        response = llm_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        result = json.loads(raw.strip())

        # Validate category
        if result.get("category") not in VALID_CATEGORIES:
            result["category"] = "GENERIC_TABULAR"

        print(f"[domain_classifier] LLM: {result['category']} "
              f"(confidence={result.get('confidence', '?')})")
        return result

    except Exception as e:
        print(f"[domain_classifier] Failed: {e} — falling back to GENERIC_TABULAR")
        return {
            "category":   "GENERIC_TABULAR",
            "confidence": 0.5,
            "reason":     f"Classification error: {e}",
        }
