# Gemini Flash-Lite Column Semantics Setup
# Save as: engine/gemini_column_semantics.py

"""
Column semantic analysis using Gemini Flash-Lite (free tier).
Understands ANY column name regardless of naming convention.
Replaces brittle rule-based detection for HR, Sales, Health, etc.
"""

import os
import json
import re
import time
import logging
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME     = "gemini-2.5-flash-lite-preview-06-17"  # free: 1000 RPD, 15 RPM
FALLBACK_MODEL = "gemini-2.5-flash"                     # free: 250 RPD, 10 RPM
MAX_RETRIES    = 3
RETRY_DELAY    = 2  # seconds between retries


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


# ── Gemini Client ─────────────────────────────────────────────────────────────
def _get_gemini_client():
    """Initialize Gemini client. Returns None if API key not set."""
    if not GEMINI_API_KEY:
        log.warning("[Gemini] GEMINI_API_KEY not set — skipping semantic analysis")
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        return genai
    except ImportError:
        log.warning("[Gemini] google-generativeai not installed")
        return None


# ── Main semantic analysis function ──────────────────────────────────────────
def analyze_column_semantics(
    df,
    max_sample_rows: int = 5
) -> dict[str, str]:
    """
    Analyze columns using Gemini Flash-Lite.
    Returns dict: {column_name: semantic_type}
    
    Falls back to rule-based detection if Gemini unavailable.
    """
    genai = _get_gemini_client()
    if not genai:
        return _rule_based_fallback(df)

    # Build column + sample data for prompt
    columns = list(df.columns)
    try:
        # Get 5 sample rows (non-null where possible)
        sample = df.head(max_sample_rows).to_dict(orient="records")
    except Exception:
        sample = []

    # Truncate to avoid token limits
    sample_str = json.dumps(sample, default=str)[:2000]
    columns_str = json.dumps(columns)

    prompt = f"""Analyze these dataset columns and classify each one.

Columns: {columns_str}

Sample data (first {max_sample_rows} rows):
{sample_str}

Classify each column into exactly one semantic type:
attrition, satisfaction, salary, department, job_role, hire_date,
tenure, performance, revenue, quantity, price, discount, category,
customer_id, order_id, region, cases, deaths, recovered, country,
team, winner, score, venue, content_type, rating, genre,
date, id, text, boolean, numeric_other, other

Rules:
- "Termd", "terminated", "left", "is_active" → attrition
- "EmpSatisfaction", "JobSatisfaction", "EngagementSurvey" → satisfaction
- "Weekly_Sales", "Revenue", "Amount", "Sales" → revenue
- "CustomerID", "EmpID", "InvoiceNo" → id
- Columns with dates → date
- Name/description columns → text

Return ONLY valid JSON, no explanation, no markdown:
{{"column_name": "semantic_type", "column_name2": "semantic_type2"}}"""

    # Try with retries
    for attempt in range(MAX_RETRIES):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,        # deterministic
                    "max_output_tokens": 500,
                    "response_mime_type": "application/json",
                }
            )
            result_text = response.text.strip()

            # Parse JSON response
            result = _parse_json_response(result_text)
            if result:
                log.info(
                    f"[Gemini Semantics] Classified {len(result)} columns "
                    f"using {MODEL_NAME}"
                )
                # Log key findings
                for col, sem in result.items():
                    if sem not in ("other", "text", "id", "numeric_other"):
                        log.info(f"  → {col}: {sem}")
                return result

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                log.warning(
                    f"[Gemini] Rate limit hit (attempt {attempt+1}/{MAX_RETRIES})"
                    f" — retrying in {RETRY_DELAY}s"
                )
                time.sleep(RETRY_DELAY * (attempt + 1))
                # Try fallback model on last attempt
                if attempt == MAX_RETRIES - 1:
                    return _try_fallback_model(genai, prompt) or \
                           _rule_based_fallback(df)
            else:
                log.warning(f"[Gemini] Error: {e}")
                break

    log.warning("[Gemini] All attempts failed — using rule-based fallback")
    return _rule_based_fallback(df)


def _try_fallback_model(genai, prompt: str) -> Optional[dict]:
    """Try with fallback model if primary hits rate limits."""
    try:
        log.info(f"[Gemini] Trying fallback model: {FALLBACK_MODEL}")
        model = genai.GenerativeModel(FALLBACK_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 500}
        )
        return _parse_json_response(response.text)
    except Exception as e:
        log.warning(f"[Gemini] Fallback model failed: {e}")
        return None


def _parse_json_response(text: str) -> Optional[dict]:
    """Parse JSON from Gemini response, handling markdown fences."""
    try:
        # Remove markdown fences if present
        clean = re.sub(r"```(?:json)?", "", text).strip()
        clean = clean.strip("`").strip()
        result = json.loads(clean)
        if isinstance(result, dict):
            return result
    except Exception:
        # Try extracting JSON object with regex
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


# ── Rule-based fallback ───────────────────────────────────────────────────────
def _rule_based_fallback(df) -> dict[str, str]:
    """
    Fast rule-based classification when Gemini unavailable.
    Same as current InsightStream logic.
    """
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
    Convert Gemini semantic analysis to InsightStream ColumnMap format.
    Returns override dict that patches the existing ColumnMap.
    """
    overrides = {}

    # Find columns by semantic type
    def find_col(sem_type: str) -> Optional[str]:
        return next(
            (col for col, sem in semantics.items() if sem == sem_type),
            None
        )

    # Map semantic types to InsightStream fields
    overrides["revenue_col"]   = find_col("revenue")
    overrides["price_col"]     = find_col("price")
    overrides["qty_col"]       = find_col("quantity")
    overrides["category_col"]  = find_col("category")
    overrides["region_col"]    = find_col("region") or find_col("country")
    overrides["date_col"]      = find_col("date") or find_col("hire_date")
    overrides["customer_col"]  = find_col("customer_id")
    overrides["attrition_col"] = find_col("attrition")
    overrides["satisfaction_col"] = find_col("satisfaction")
    overrides["salary_col"]    = find_col("salary")
    overrides["dept_col"]      = find_col("department")

    # Remove None values
    overrides = {k: v for k, v in overrides.items() if v is not None}

    log.info(f"[Gemini Semantics] ColumnMap overrides: {overrides}")
    return overrides


# ── Currency detection using Gemini ──────────────────────────────────────────
def detect_currency_gemini(df, columns: list) -> str:
    """
    Detect currency using Gemini instead of rule-based detection.
    Returns currency symbol: ₹, $, £, €, AED, S$, ¥
    """
    genai = _get_gemini_client()
    if not genai:
        return "₹"  # fallback to default

    # Sample numeric values
    try:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        sample_vals  = {}
        for col in numeric_cols[:5]:
            vals = df[col].dropna().head(3).tolist()
            sample_vals[col] = vals
    except Exception:
        sample_vals = {}

    prompt = f"""What currency are the numeric values in this dataset?

Column names: {columns}
Sample numeric values: {json.dumps(sample_vals, default=str)}

Consider:
- Country column values (if any)
- Dataset context (UK retail, US sales, Indian company)
- Value ranges ($229 could be USD; £229 could be GBP)
- Column names (UnitPrice, Weekly_Sales, Salary)

Reply with ONLY the currency symbol, nothing else.
Valid responses: ₹  $  £  €  AED  S$  ¥"""

    try:
        model   = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 10}
        )
        symbol = response.text.strip()
        # Validate it's a known symbol
        valid  = {"₹", "$", "£", "€", "AED", "S$", "¥"}
        if symbol in valid:
            log.info(f"[Gemini Currency] Detected: {symbol}")
            return symbol
    except Exception as e:
        log.warning(f"[Gemini Currency] Error: {e}")

    return "₹"  # safe default


# ── Integration helper ────────────────────────────────────────────────────────
def get_column_semantic_type(col_name: str, df) -> str:
    """
    Quick single-column semantic lookup.
    Used when you just need one column classified.
    """
    semantics = analyze_column_semantics(df)
    return semantics.get(col_name, "other")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    # Test with HRDataset columns
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

    print("Testing Gemini Column Semantics...")
    print(f"GEMINI_API_KEY set: {bool(GEMINI_API_KEY)}")
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
