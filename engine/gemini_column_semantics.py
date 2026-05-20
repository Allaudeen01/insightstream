# Groq Column Semantics
# Save as: engine/gemini_column_semantics.py

"""
Column semantic analysis using Groq (free tier).
Understands ANY column name regardless of naming convention.
Replaces brittle rule-based detection for HR, Sales, Health, etc.
"""

import os
import json
import re
import time
import logging
from typing import Optional
from groq import Groq

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME   = "llama-3.3-70b-versatile"   # best quality, 1000 RPD free
FAST_MODEL   = "llama-3.1-8b-instant"      # 14,400 RPD, fast
MAX_RETRIES  = 3
RETRY_DELAY  = 2  # seconds between retries


# ── Semantic type definitions ─────────────────────────────────────────────────
SEMANTIC_TYPES = {
    # HR
    "attrition":       "Employee left/terminated status (Yes/No, 0/1, Termd)",
    "satisfaction":    "Employee satisfaction/engagement score",
    "salary":          "Employee salary, income, compensation",
    "department":      "Business unit, team, division",
    "job_role":        "Job title, position, role",
    "hire_date":       "Date employee joined",
    "tenure":          "Years/months at company",
    "performance":     "Performance rating or score",

    # Sales/Revenue
    "revenue":         "Sales amount, revenue, income, weekly_sales",
    "quantity":        "Units sold, quantity, count",
    "price":           "Unit price, cost, rate",
    "discount":        "Discount amount or percentage",
    "category":        "Product category, type, segment",
    "customer_id":     "Customer identifier",
    "order_id":        "Order or transaction identifier",
    "region":          "Geographic region, territory, state",

    # Health
    "cases":           "Confirmed cases, infected count",
    "deaths":          "Death count, fatalities",
    "recovered":       "Recovered patients count",
    "country":         "Country or geographic entity",

    # Sports
    "team":            "Sports team name",
    "winner":          "Match winner",
    "score":           "Match score or runs",
    "venue":           "Match venue or stadium",

    # Entertainment
    "content_type":    "Movie/TV Show/Series type",
    "rating":          "Content rating (PG, R, etc)",
    "genre":           "Content genre",

    # General
    "date":            "Date or timestamp column",
    "id":              "Identifier column (skip for analysis)",
    "text":            "Free text, name, description",
    "boolean":         "True/False, Yes/No flag",
    "numeric_other":   "Numeric column not matching above",
    "other":           "Cannot classify"
}


# ── Groq Client ───────────────────────────────────────────────────────────────
def _get_client():
    if not GROQ_API_KEY:
        log.warning("[Groq] GROQ_API_KEY not set — skipping semantic analysis")
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        log.warning(f"[Groq] Client error: {e}")
        return None


# ── Main semantic analysis function ──────────────────────────────────────────
def analyze_column_semantics(df, max_sample_rows=5) -> dict:
    """
    Analyze columns using Groq.
    Returns dict: {column_name: semantic_type}

    Falls back to rule-based detection if Groq unavailable.
    """
    client = _get_client()
    if not client:
        return _rule_based_fallback(df)

    columns = list(df.columns)
    try:
        sample = df.head(max_sample_rows).to_dict(orient="records")
    except Exception:
        sample = []

    sample_str = json.dumps(sample, default=str)[:2000]

    prompt = f"""Analyze these dataset columns and classify each one.

Columns: {json.dumps(columns)}
Sample data: {sample_str}

Classify each column into exactly one type:
attrition, satisfaction, salary, performance, department, job_role,
revenue, quantity, price, category, customer_id, region,
cases, deaths, recovered, country, team, date, id, text,
boolean, numeric_other, other

Rules:
- Termd, terminated, left, is_active → attrition
- EmpSatisfaction, JobSatisfaction, EngagementSurvey → satisfaction
- Weekly_Sales, Revenue, Amount, Sales → revenue
- CustomerID, EmpID, InvoiceNo → id
- Date columns → date
- Salary, MonthlySalary, AnnualSalary, Wage, Pay, Compensation → salary
- PerformanceScore, PerfScore, Rating, PerformanceRating → performance

Return ONLY valid JSON:
{{"column_name": "semantic_type"}}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            result_text = response.choices[0].message.content
            result = _parse_json_response(result_text)
            if result:
                log.info(f"[Groq] Classified {len(result)} columns")
                return result
        except Exception as e:
            if "429" in str(e):
                log.warning(f"[Groq] Rate limit — retrying...")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                log.warning(f"[Groq] Error: {e}")
                break

    log.warning("[Groq] All attempts failed — using rule-based fallback")
    return _rule_based_fallback(df)


def _parse_json_response(text: str) -> Optional[dict]:
    """Parse JSON from response, handling markdown fences."""
    try:
        clean = re.sub(r"```(?:json)?", "", text).strip()
        clean = clean.strip("`").strip()
        result = json.loads(clean)
        if isinstance(result, dict):
            return result
    except Exception:
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


# ── Rule-based fallback ───────────────────────────────────────────────────────
def _rule_based_fallback(df) -> dict[str, str]:
    """Fast rule-based classification when Groq unavailable."""
    result = {}
    columns = list(df.columns)

    _RULES = {
        "attrition":    ["attrition", "termd", "terminated", "turnover",
                         "resigned", "left", "empstatus", "employment_status",
                         "separation", "departure"],
        "satisfaction": ["satisfaction", "engage", "morale", "happiness",
                         "empsatisfaction", "jobsatisfaction"],
        "salary":       ["salary", "income", "wage", "compensation",
                         "monthlysalary", "annualsalary", "pay"],
        "department":   ["department", "dept", "division", "team", "unit"],
        "job_role":     ["position", "jobtitle", "role", "title", "job"],
        "revenue":      ["revenue", "sales", "amount", "weekly_sales",
                         "daily_sales", "monthly_sales", "gmv", "turnover"],
        "quantity":     ["quantity", "qty", "units", "volume", "count"],
        "price":        ["price", "cost", "rate", "unitprice", "fee"],
        "category":     ["category", "type", "segment", "class", "group"],
        "region":       ["region", "state", "country", "city", "territory",
                         "province", "location", "area"],
        "date":         ["date", "time", "year", "month", "week"],
        "customer_id":  ["customerid", "customer_id", "custid", "client_id"],
        "order_id":     ["orderid", "order_id", "invoiceno", "transactionid"],
        "cases":        ["cases", "confirmed", "infected", "positive"],
        "deaths":       ["deaths", "fatalities", "deceased", "mortality"],
        "recovered":    ["recovered", "recovery"],
        "team":         ["team", "club", "franchise", "squad"],
    }

    for col in columns:
        col_lower = col.lower().replace(" ", "_").replace("\n", "")
        classified = "other"

        for sem_type, keywords in _RULES.items():
            if any(kw in col_lower for kw in keywords):
                classified = sem_type
                break

        # Check if ID column (high cardinality unique values)
        if classified == "other":
            try:
                if df[col].nunique() / len(df) > 0.9:
                    classified = "id"
            except Exception:
                pass

        result[col] = classified

    return result


# ── Apply semantics to InsightStream ─────────────────────────────────────────
def apply_semantics_to_engine(df, semantics: dict) -> dict:
    """
    Convert semantic analysis to InsightStream ColumnMap format.
    Returns override dict that patches the existing ColumnMap.
    """
    overrides = {}

    def find_col(sem_type: str) -> Optional[str]:
        return next(
            (col for col, sem in semantics.items() if sem == sem_type),
            None
        )

    overrides["revenue_col"]      = find_col("revenue")
    overrides["price_col"]        = find_col("price")
    overrides["qty_col"]          = find_col("quantity")
    overrides["category_col"]     = find_col("category")
    overrides["region_col"]       = find_col("region") or find_col("country")
    overrides["date_col"]         = find_col("date") or find_col("hire_date")
    overrides["customer_col"]     = find_col("customer_id")
    overrides["attrition_col"]    = find_col("attrition")
    overrides["satisfaction_col"] = find_col("satisfaction")
    overrides["salary_col"]       = find_col("salary")
    overrides["dept_col"]         = find_col("department")

    overrides = {k: v for k, v in overrides.items() if v is not None}

    log.info(f"[Groq Semantics] ColumnMap overrides: {overrides}")
    return overrides


# ── Currency detection using Groq ─────────────────────────────────────────────
def detect_currency_gemini(df, columns: list) -> str:
    """
    Detect currency using Groq instead of rule-based detection.
    Returns currency symbol: ₹, $, £, €, AED, S$, ¥
    """
    client = _get_client()
    if not client:
        return "₹"
    try:
        numeric_sample = {}
        for col in df.select_dtypes(include=["number"]).columns[:5]:
            numeric_sample[col] = df[col].dropna().head(3).tolist()

        response = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": f"""
What currency are values in this dataset?
Columns: {columns}
Numeric samples: {json.dumps(numeric_sample, default=str)}
Reply with ONLY one symbol: ₹ $ £ € AED S$ ¥"""}],
            temperature=0,
            max_tokens=5,
        )
        sym = response.choices[0].message.content.strip()
        if sym in {"₹", "$", "£", "€", "AED", "S$", "¥"}:
            log.info(f"[Groq Currency] Detected: {sym}")
            return sym
    except Exception as e:
        log.warning(f"[Groq Currency] Error: {e}")
    return "₹"


# ── Integration helper ────────────────────────────────────────────────────────
def get_column_semantic_type(col_name: str, df) -> str:
    """Quick single-column semantic lookup."""
    semantics = analyze_column_semantics(df)
    return semantics.get(col_name, "other")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    test_df = pd.DataFrame({
        "EmpID":             [1, 2, 3],
        "Employee_Name":     ["John", "Jane", "Bob"],
        "Termd":             [0, 1, 0],
        "EmpSatisfaction":   [4, 2, 3],
        "Salary":            [75000, 82000, 65000],
        "Department":        ["Sales", "IT", "HR"],
        "DateofHire":        ["2020-01-15", "2019-03-22", "2021-06-01"],
        "PerformanceScore":  ["Exceeds", "Fully Meets", "Needs Improvement"],
    })

    print("Testing Groq Column Semantics...")
    print(f"GROQ_API_KEY set: {bool(GROQ_API_KEY)}")
    print()

    semantics = analyze_column_semantics(test_df)
    print("Results:")
    for col, sem in semantics.items():
        print(f"  {col:25} → {sem}")

    print()
    overrides = apply_semantics_to_engine(test_df, semantics)
    print("ColumnMap overrides:")
    for k, v in overrides.items():
        print(f"  {k:20} → {v}")
