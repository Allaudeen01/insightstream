from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import polars as pl
import io
import uuid
from typing import Optional

app = FastAPI(title="Virtual Data Scientist Engine")

# Allow CORS for Next.js frontend (localhost + production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Vercel, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== FILE-BASED SESSION STORAGE ==============
# Sessions are stored in /tmp/sessions/ as JSON (metadata) + Parquet (dataframe)
# This survives within a container's lifetime, unlike in-memory dicts

import os
import json
from pathlib import Path

SESSION_DIR = Path("/tmp/sessions")
SESSION_DIR.mkdir(parents=True, exist_ok=True)

def save_session(session_id: str, filename: str, df: pl.DataFrame) -> None:
    """Save session to disk."""
    session_path = SESSION_DIR / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {"filename": filename}
    with open(session_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Save DataFrame as Parquet (fast and efficient)
    df.write_parquet(session_path / "data.parquet")

def load_session(session_id: str) -> tuple[str, pl.DataFrame]:
    """Load session from disk. Raises FileNotFoundError if not found."""
    session_path = SESSION_DIR / session_id
    
    if not session_path.exists():
        raise FileNotFoundError(f"Session {session_id} not found")
    
    # Load metadata
    with open(session_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load DataFrame
    df = pl.read_parquet(session_path / "data.parquet")
    
    return metadata["filename"], df

def session_exists(session_id: str) -> bool:
    """Check if a session exists."""
    return (SESSION_DIR / session_id / "data.parquet").exists()

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int

class DatasetSchema(BaseModel):
    session_id: str
    filename: str
    row_count: int
    column_count: int
    columns: list[ColumnInfo]
    preview: list[dict]  # First 10 rows

@app.get("/")
def read_root():
    return {"status": "online", "message": "Virtual Data Scientist Engine is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload", response_model=DatasetSchema)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset (CSV or Excel) and get back the schema.
    This is the core "Data Collection" step.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    extension = file.filename.split(".")[-1].lower()
    if extension not in ["csv", "xlsx", "xls"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Use CSV or Excel."
        )
    
    try:
        contents = await file.read()
        
        # Parse with Polars (faster than Pandas for large files)
        if extension == "csv":
            df = pl.read_csv(io.BytesIO(contents))
        else:
            # For Excel, use Pandas as intermediary (Polars has limited Excel support)
            import pandas as pd
            pandas_df = pd.read_excel(io.BytesIO(contents))
            df = pl.from_pandas(pandas_df)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Store dataframe in session (file-based for production)
        save_session(session_id, file.filename, df)
        
        # Build schema response
        columns_info = []
        for col in df.columns:
            col_data = df[col]
            null_count = col_data.null_count()
            columns_info.append(ColumnInfo(
                name=col,
                dtype=str(col_data.dtype),
                missing_count=null_count,
                missing_pct=round(null_count / len(df) * 100, 2) if len(df) > 0 else 0,
                unique_count=col_data.n_unique()
            ))
        
        # Generate preview (first 10 rows)
        preview = df.head(10).to_dicts()
        
        return DatasetSchema(
            session_id=session_id,
            filename=file.filename,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns_info,
            preview=preview
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Get info about an existing session."""
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "filename": filename,
        "row_count": len(df),
        "column_count": len(df.columns)
    }

# ============== PHASE 3: HEALTH CHECK & EDA ==============

class IssueItem(BaseModel):
    column: str
    issue_type: str  # "missing" | "duplicate" | "outlier"
    severity: str    # "low" | "medium" | "high"
    description: str
    count: int
    percentage: float

class HealthCheckResponse(BaseModel):
    session_id: str
    quality_score: str  # "A" | "B" | "C" | "D"
    row_count: int
    column_count: int
    duplicate_rows: int
    issues: list[IssueItem]

@app.get("/health-check/{session_id}", response_model=HealthCheckResponse)
def get_health_check(session_id: str):
    """
    Analyze dataset for data quality issues.
    This is the "Data Health Check" step - what a real Data Scientist does first.
    """
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    issues = []
    
    # 1. Check for duplicate rows
    duplicate_count = len(df) - df.unique().height
    if duplicate_count > 0:
        dup_pct = round(duplicate_count / len(df) * 100, 2)
        severity = "high" if dup_pct > 10 else ("medium" if dup_pct > 5 else "low")
        issues.append(IssueItem(
            column="_all_",
            issue_type="duplicate",
            severity=severity,
            description=f"{duplicate_count} duplicate rows detected ({dup_pct}%)",
            count=duplicate_count,
            percentage=dup_pct
        ))
    
    # 2. Check for missing values per column
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            null_pct = round(null_count / len(df) * 100, 2)
            severity = "high" if null_pct > 30 else ("medium" if null_pct > 10 else "low")
            issues.append(IssueItem(
                column=col,
                issue_type="missing",
                severity=severity,
                description=f"{null_count} missing values ({null_pct}%)",
                count=null_count,
                percentage=null_pct
            ))
    
    # 3. Check for outliers in numeric columns (IQR method)
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
    for col in numeric_cols:
        col_data = df[col].drop_nulls()
        if len(col_data) < 4:
            continue
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = len(col_data.filter((col_data < lower_bound) | (col_data > upper_bound)))
        if outlier_count > 0:
            outlier_pct = round(outlier_count / len(col_data) * 100, 2)
            severity = "high" if outlier_pct > 10 else ("medium" if outlier_pct > 5 else "low")
            issues.append(IssueItem(
                column=col,
                issue_type="outlier",
                severity=severity,
                description=f"{outlier_count} outliers detected ({outlier_pct}%)",
                count=outlier_count,
                percentage=outlier_pct
            ))
    
    # Calculate quality score
    high_issues = sum(1 for i in issues if i.severity == "high")
    medium_issues = sum(1 for i in issues if i.severity == "medium")
    
    if high_issues >= 3:
        quality_score = "D"
    elif high_issues >= 1:
        quality_score = "C"
    elif medium_issues >= 3:
        quality_score = "B"
    else:
        quality_score = "A"
    
    return HealthCheckResponse(
        session_id=session_id,
        quality_score=quality_score,
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_rows=duplicate_count,
        issues=issues
    )

class CleanAction(BaseModel):
    action: str  # "drop_duplicates" | "drop_column" | "impute_mean" | "impute_median" | "impute_mode" | "drop_nulls"
    column: Optional[str] = None

@app.post("/clean/{session_id}")
def apply_cleaning(session_id: str, actions: list[CleanAction]):
    """
    Apply cleaning actions to the dataset.
    """
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    for action in actions:
        if action.action == "drop_duplicates":
            df = df.unique()
        elif action.action == "drop_column" and action.column:
            df = df.drop(action.column)
        elif action.action == "drop_nulls" and action.column:
            df = df.filter(pl.col(action.column).is_not_null())
        elif action.action == "impute_mean" and action.column:
            mean_val = df[action.column].mean()
            df = df.with_columns(pl.col(action.column).fill_null(mean_val))
        elif action.action == "impute_median" and action.column:
            median_val = df[action.column].median()
            df = df.with_columns(pl.col(action.column).fill_null(median_val))
        elif action.action == "impute_mode" and action.column:
            mode_val = df[action.column].mode().to_list()[0] if len(df[action.column].mode()) > 0 else None
            if mode_val is not None:
                df = df.with_columns(pl.col(action.column).fill_null(mode_val))
    
    # Save the cleaned DataFrame back to disk
    save_session(session_id, filename, df)
    
    return {"status": "ok", "row_count": len(df), "column_count": len(df.columns)}

class EDAColumnStats(BaseModel):
    column: str
    dtype: str
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    unique_count: int
    top_values: list[dict]  # For categorical columns

class EDAResponse(BaseModel):
    session_id: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    date_columns: list[str]
    column_stats: list[EDAColumnStats]
    correlation_matrix: dict  # column -> {column: correlation}
    insights: list[str]

@app.get("/eda/{session_id}", response_model=EDAResponse)
def get_eda(session_id: str):
    """
    Perform Exploratory Data Analysis.
    This is the "Auto EDA" step - auto-generated statistics and insights.
    """
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Categorize columns
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    
    for col in df.columns:
        dtype = df[col].dtype
        if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            numeric_cols.append(col)
        elif dtype in [pl.Date, pl.Datetime]:
            date_cols.append(col)
        else:
            categorical_cols.append(col)
    
    # Compute column statistics
    column_stats = []
    for col in df.columns:
        dtype = df[col].dtype
        stats = EDAColumnStats(
            column=col,
            dtype=str(dtype),
            unique_count=df[col].n_unique(),
            top_values=[]
        )
        
        if col in numeric_cols:
            col_data = df[col].drop_nulls()
            if len(col_data) > 0:
                stats.mean = round(float(col_data.mean()), 2)
                stats.median = round(float(col_data.median()), 2)
                stats.std = round(float(col_data.std()), 2)
                stats.min_val = float(col_data.min())
                stats.max_val = float(col_data.max())
        
        if col in categorical_cols:
            value_counts = df[col].value_counts().head(5).to_dicts()
            stats.top_values = value_counts
        
        column_stats.append(stats)
    
    # Compute correlation matrix for numeric columns
    correlation_matrix = {}
    if len(numeric_cols) > 1:
        for col1 in numeric_cols:
            correlation_matrix[col1] = {}
            for col2 in numeric_cols:
                try:
                    corr = df.select(pl.corr(col1, col2)).item()
                    correlation_matrix[col1][col2] = round(float(corr), 3) if corr is not None else 0
                except:
                    correlation_matrix[col1][col2] = 0
    
    # Generate insights
    insights = []
    
    # Insight 1: High correlation pairs
    for col1 in correlation_matrix:
        for col2, corr in correlation_matrix[col1].items():
            if col1 != col2 and abs(corr) > 0.7:
                insights.append(f"Strong correlation ({corr}) between '{col1}' and '{col2}'")
    
    # Insight 2: Skewed distributions
    for stats in column_stats:
        if stats.mean and stats.median:
            skew_ratio = abs(stats.mean - stats.median) / max(stats.std, 0.01) if stats.std else 0
            if skew_ratio > 0.5:
                direction = "right-skewed" if stats.mean > stats.median else "left-skewed"
                insights.append(f"'{stats.column}' is {direction} (mean: {stats.mean}, median: {stats.median})")
    
    # Insight 3: Dominant categories
    for stats in column_stats:
        if stats.top_values and len(stats.top_values) > 0:
            top = stats.top_values[0]
            if "count" in top:
                pct = round(top["count"] / len(df) * 100, 1)
                if pct > 50:
                    insights.append(f"'{stats.column}': '{top.get(stats.column, 'N/A')}' dominates with {pct}% of values")
    
    return EDAResponse(
        session_id=session_id,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        date_columns=date_cols,
        column_stats=column_stats,
        correlation_matrix=correlation_matrix,
        insights=insights[:10]  # Limit to top 10 insights
    )

# ============== PHASE 4: INTELLIGENCE LAYER ==============

class InsightCard(BaseModel):
    title: str
    description: str
    chart_type: str  # "bar" | "pie" | "line" | "none"
    chart_data: Optional[dict] = None
    importance: str  # "high" | "medium" | "low"

class InsightsResponse(BaseModel):
    session_id: str
    executive_summary: str
    insights: list[InsightCard]
    recommendations: list[str]

@app.get("/insights/{session_id}", response_model=InsightsResponse)
def get_insights(session_id: str):
    """
    Generate business insights in plain English.
    This is the 'What is happening?' answer - real Data Scientist work.
    """
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    insights = []
    recommendations = []
    
    # Categorize columns
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
    categorical_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
    
    # INSIGHT 1: Top contributors (for numeric columns)
    for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
        for cat_col in categorical_cols[:1]:  # Pair with first categorical
            try:
                top_groups = df.group_by(cat_col).agg(pl.sum(num_col).alias("total")).sort("total", descending=True).head(3)
                if len(top_groups) >= 3:
                    total_sum = df[num_col].sum()
                    top_3_sum = top_groups["total"].sum()
                    pct = round(top_3_sum / total_sum * 100, 1) if total_sum > 0 else 0
                    
                    top_names = top_groups[cat_col].to_list()
                    insights.append(InsightCard(
                        title=f"Top 3 {cat_col} by {num_col}",
                        description=f"The top 3 {cat_col} values ({', '.join(str(n) for n in top_names[:3])}) account for {pct}% of total {num_col}.",
                        chart_type="bar",
                        chart_data={
                            "labels": [str(x) for x in top_groups[cat_col].to_list()],
                            "values": [float(x) for x in top_groups["total"].to_list()]
                        },
                        importance="high"
                    ))
                    recommendations.append(f"Focus resources on top-performing {cat_col} categories to maximize {num_col}.")
            except:
                pass
    
    # INSIGHT 2: Distribution analysis
    for col in numeric_cols[:3]:
        try:
            col_data = df[col].drop_nulls()
            if len(col_data) > 10:
                mean = col_data.mean()
                median = col_data.median()
                std = col_data.std()
                
                if abs(mean - median) / max(std, 0.01) > 0.3:
                    skew = "right-skewed (long tail of high values)" if mean > median else "left-skewed (long tail of low values)"
                    insights.append(InsightCard(
                        title=f"{col} Distribution Pattern",
                        description=f"The {col} data is {skew}. Mean: {round(mean, 2)}, Median: {round(median, 2)}. Most values cluster below the average.",
                        chart_type="none",
                        importance="medium"
                    ))
        except:
            pass
    
    # INSIGHT 3: Category dominance
    for col in categorical_cols[:2]:
        try:
            value_counts = df[col].value_counts()
            if len(value_counts) > 1:
                top_val = value_counts.head(1)
                top_name = top_val[col].item()
                top_count = top_val["count"].item()
                pct = round(top_count / len(df) * 100, 1)
                
                if pct > 40:
                    insights.append(InsightCard(
                        title=f"{col} Concentration",
                        description=f"'{top_name}' dominates the {col} category with {pct}% of all records. Consider if this concentration is desired.",
                        chart_type="pie",
                        chart_data={
                            "labels": [str(x) for x in value_counts[col].head(5).to_list()],
                            "values": [int(x) for x in value_counts["count"].head(5).to_list()]
                        },
                        importance="medium"
                    ))
        except:
            pass
    
    # INSIGHT 4: Correlation insights
    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols[:3]):
            for col2 in numeric_cols[i+1:4]:
                try:
                    corr = df.select(pl.corr(col1, col2)).item()
                    if corr and abs(corr) > 0.6:
                        direction = "increases" if corr > 0 else "decreases"
                        insights.append(InsightCard(
                            title=f"{col1} & {col2} Relationship",
                            description=f"When {col1} goes up, {col2} tends to {direction} (correlation: {round(corr, 2)}). This could indicate a causal relationship worth investigating.",
                            chart_type="none",
                            importance="high" if abs(corr) > 0.8 else "medium"
                        ))
                        recommendations.append(f"Investigate why {col1} and {col2} are {'positively' if corr > 0 else 'negatively'} correlated.")
                except:
                    pass
    
    # Generate executive summary
    high_insights = [i for i in insights if i.importance == "high"]
    if high_insights:
        exec_summary = f"Analysis of {len(df)} records across {len(df.columns)} columns reveals {len(high_insights)} critical findings. " + high_insights[0].description
    else:
        exec_summary = f"Dataset contains {len(df)} records with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns. No critical anomalies detected."
    
    return InsightsResponse(
        session_id=session_id,
        executive_summary=exec_summary,
        insights=insights[:8],  # Limit to 8 insights
        recommendations=recommendations[:5]  # Limit to 5 recommendations
    )


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    chart_data: Optional[dict] = None

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    chart_type: Optional[str] = None
    chart_data: Optional[dict] = None
    sql_equivalent: Optional[str] = None

@app.post("/chat/{session_id}", response_model=ChatResponse)
def chat_with_data(session_id: str, request: ChatRequest):
    """
    Natural language interface to query data.
    This is the 'Talk to Your Data' feature - converts questions to analysis.
    """
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    question = request.question.lower()
    
    # Simple NL parsing (in production, use LLM)
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
    categorical_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
    
    answer = ""
    chart_type = None
    chart_data = None
    sql_equivalent = None
    
    # Pattern: "total/sum of X"
    if any(word in question for word in ["total", "sum", "overall"]):
        for col in numeric_cols:
            if col.lower() in question:
                total = df[col].sum()
                answer = f"The total {col} is {total:,.2f}."
                sql_equivalent = f"SELECT SUM({col}) FROM dataset"
                break
        if not answer:
            answer = f"I found these numeric columns you can sum: {', '.join(numeric_cols)}. Try asking about a specific one."
    
    # Pattern: "average/mean of X"
    elif any(word in question for word in ["average", "mean", "avg"]):
        for col in numeric_cols:
            if col.lower() in question:
                avg = df[col].mean()
                answer = f"The average {col} is {avg:,.2f}."
                sql_equivalent = f"SELECT AVG({col}) FROM dataset"
                break
        if not answer:
            answer = f"I can calculate averages for: {', '.join(numeric_cols)}."
    
    # Pattern: "top/highest/best X"
    elif any(word in question for word in ["top", "highest", "best", "most"]):
        for cat_col in categorical_cols:
            if cat_col.lower() in question:
                for num_col in numeric_cols:
                    top = df.group_by(cat_col).agg(pl.sum(num_col).alias("total")).sort("total", descending=True).head(5)
                    labels = [str(x) for x in top[cat_col].to_list()]
                    values = [float(x) for x in top["total"].to_list()]
                    answer = f"Top 5 {cat_col} by {num_col}:\n" + "\n".join([f"  {i+1}. {labels[i]}: {values[i]:,.2f}" for i in range(len(labels))])
                    chart_type = "bar"
                    chart_data = {"labels": labels, "values": values}
                    sql_equivalent = f"SELECT {cat_col}, SUM({num_col}) FROM dataset GROUP BY {cat_col} ORDER BY SUM({num_col}) DESC LIMIT 5"
                    break
            if answer:
                break
        if not answer:
            answer = f"I can find top values by grouping these categories: {', '.join(categorical_cols)}."
    
    # Pattern: "how many/count"
    elif any(word in question for word in ["how many", "count", "number of"]):
        for col in categorical_cols:
            if col.lower() in question:
                counts = df[col].value_counts().head(10)
                labels = [str(x) for x in counts[col].to_list()]
                values = [int(x) for x in counts["count"].to_list()]
                answer = f"Counts by {col}:\n" + "\n".join([f"  {labels[i]}: {values[i]}" for i in range(len(labels))])
                chart_type = "pie"
                chart_data = {"labels": labels, "values": values}
                sql_equivalent = f"SELECT {col}, COUNT(*) FROM dataset GROUP BY {col}"
                break
        if not answer:
            answer = f"Total number of records: {len(df):,}. Ask about specific categories: {', '.join(categorical_cols)}."
    
    # Pattern: "why" questions (correlation analysis)
    elif "why" in question:
        if len(numeric_cols) >= 2:
            # Find strongest correlation
            best_corr = 0
            best_pair = None
            for i, col1 in enumerate(numeric_cols[:5]):
                for col2 in numeric_cols[i+1:6]:
                    try:
                        corr = df.select(pl.corr(col1, col2)).item()
                        if corr and abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_pair = (col1, col2)
                    except:
                        pass
            if best_pair:
                direction = "positively" if best_corr > 0 else "negatively"
                answer = f"I found that {best_pair[0]} and {best_pair[1]} are {direction} correlated ({best_corr:.2f}). When {best_pair[0]} increases, {best_pair[1]} tends to {'increase' if best_corr > 0 else 'decrease'}. This might help explain patterns in your data."
            else:
                answer = "I couldn't find strong correlations to explain patterns. Try asking about specific metrics."
        else:
            answer = "I need at least 2 numeric columns to find correlations and explain 'why' patterns occur."
    
    # Pattern: "predict/forecast"
    elif any(word in question for word in ["predict", "forecast", "estimate", "next"]):
        answer = "For predictions, go to the Smart Modeling screen (Screen 7) where you can build and train ML models. I can help you understand historical patterns here."
    
    # Default: describe the dataset
    else:
        answer = f"Your dataset has {len(df):,} rows and {len(df.columns)} columns.\n\nNumeric columns: {', '.join(numeric_cols[:5])}\nCategorical columns: {', '.join(categorical_cols[:5])}\n\nTry asking:\n• 'What is the total [column]?'\n• 'Show me top 5 [category]'\n• 'Why did [metric] change?'"
    
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        chart_type=chart_type,
        chart_data=chart_data,
        sql_equivalent=sql_equivalent
    )

# ============== PHASE 5: MODELING & REPORTING ==============

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
import numpy as np

class ModelConfig(BaseModel):
    target_column: str
    goal: str  # "predict" | "classify" | "segment"
    features: Optional[list[str]] = None  # If None, use all numeric

class ModelResult(BaseModel):
    model_name: str
    model_type: str  # "regression" | "classification"
    accuracy_or_r2: float
    secondary_metric: float  # MSE for regression, F1 for classification
    feature_importance: dict[str, float]

class ModelingResponse(BaseModel):
    session_id: str
    target_column: str
    goal: str
    best_model: str
    models: list[ModelResult]
    prediction_sample: list[dict]

@app.post("/model/{session_id}", response_model=ModelingResponse)
def train_models(session_id: str, config: ModelConfig):
    """
    AutoML Pipeline: Train and compare multiple models.
    This is the 'Smart Modeling' feature - what a Data Scientist does for predictions.
    """
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if config.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{config.target_column}' not found")
    
    # Prepare features
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
    
    if config.features:
        feature_cols = [f for f in config.features if f in numeric_cols and f != config.target_column]
    else:
        feature_cols = [c for c in numeric_cols if c != config.target_column]
    
    if len(feature_cols) < 1:
        raise HTTPException(status_code=400, detail="Need at least 1 numeric feature column")
    
    # Prepare data
    df_clean = df.select(feature_cols + [config.target_column]).drop_nulls()
    if len(df_clean) < 20:
        raise HTTPException(status_code=400, detail="Need at least 20 rows after removing nulls")
    
    X = df_clean.select(feature_cols).to_numpy()
    y = df_clean[config.target_column].to_numpy()
    
    # Determine if classification or regression
    target_dtype = df[config.target_column].dtype
    unique_values = df[config.target_column].n_unique()
    
    is_classification = (
        config.goal == "classify" or 
        target_dtype == pl.Utf8 or 
        (unique_values <= 10 and target_dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8])
    )
    
    # Encode target if classification with strings
    le = None
    if target_dtype == pl.Utf8:
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    results = []
    
    if is_classification:
        # Classification models
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
            ("Random Forest", RandomForestClassifier(n_estimators=50, max_depth=5))
        ]
        
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(feature_cols, [round(float(x), 4) for x in model.feature_importances_]))
                elif hasattr(model, 'coef_'):
                    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    importance = dict(zip(feature_cols, [round(abs(float(x)), 4) for x in coef]))
                else:
                    importance = {}
                
                results.append(ModelResult(
                    model_name=name,
                    model_type="classification",
                    accuracy_or_r2=round(acc, 4),
                    secondary_metric=round(f1, 4),
                    feature_importance=importance
                ))
            except Exception as e:
                pass
    else:
        # Regression models
        models = [
            ("Linear Regression", LinearRegression()),
            ("Decision Tree", DecisionTreeRegressor(max_depth=5)),
            ("Random Forest", RandomForestRegressor(n_estimators=50, max_depth=5))
        ]
        
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(feature_cols, [round(float(x), 4) for x in model.feature_importances_]))
                elif hasattr(model, 'coef_'):
                    importance = dict(zip(feature_cols, [round(abs(float(x)), 4) for x in model.coef_]))
                else:
                    importance = {}
                
                results.append(ModelResult(
                    model_name=name,
                    model_type="regression",
                    accuracy_or_r2=round(r2, 4),
                    secondary_metric=round(mse, 4),
                    feature_importance=importance
                ))
            except Exception as e:
                pass
    
    if not results:
        raise HTTPException(status_code=500, detail="Failed to train any models")
    
    # Find best model
    best = max(results, key=lambda x: x.accuracy_or_r2)
    
    # Generate sample predictions
    sample_indices = list(range(min(5, len(X_test))))
    predictions = []
    for i in sample_indices:
        pred_row = {"actual": float(y_test[i]) if not le else le.inverse_transform([int(y_test[i])])[0]}
        for name, model in models:
            try:
                pred = model.predict(X_test[i:i+1])[0]
                pred_row[name] = float(pred) if not le else le.inverse_transform([int(pred)])[0]
            except:
                pass
        predictions.append(pred_row)
    
    return ModelingResponse(
        session_id=session_id,
        target_column=config.target_column,
        goal=config.goal,
        best_model=best.model_name,
        models=results,
        prediction_sample=predictions
    )


class ReportSection(BaseModel):
    title: str
    content: str
    chart_type: Optional[str] = None
    chart_data: Optional[dict] = None

class ReportResponse(BaseModel):
    session_id: str
    title: str
    generated_at: str
    sections: list[ReportSection]

@app.get("/report/{session_id}", response_model=ReportResponse)
def generate_report(session_id: str):
    """
    Generate a comprehensive analysis report.
    This is the 'Communication' step - presenting findings to stakeholders.
    """
    from datetime import datetime
    
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sections = []
    
    # Section 1: Dataset Overview
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
    categorical_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
    
    sections.append(ReportSection(
        title="Dataset Overview",
        content=f"This report analyzes '{filename}' containing {len(df):,} records across {len(df.columns)} columns. The dataset includes {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns.",
    ))
    
    # Section 2: Data Quality
    missing_total = sum(df[col].null_count() for col in df.columns)
    duplicate_count = len(df) - df.unique().height
    quality_score = "Good" if missing_total < len(df) * 0.1 else ("Fair" if missing_total < len(df) * 0.3 else "Needs Attention")
    
    sections.append(ReportSection(
        title="Data Quality Assessment",
        content=f"Data Quality: {quality_score}. Missing values: {missing_total:,} ({round(missing_total/len(df)/len(df.columns)*100, 1)}% of cells). Duplicate rows: {duplicate_count}.",
    ))
    
    # Section 3: Key Statistics
    if numeric_cols:
        stats_text = "Key Metrics:\n"
        for col in numeric_cols[:3]:
            col_data = df[col].drop_nulls()
            if len(col_data) > 0:
                stats_text += f"• {col}: Mean = {col_data.mean():.2f}, Median = {col_data.median():.2f}, Range = [{col_data.min():.2f}, {col_data.max():.2f}]\n"
        sections.append(ReportSection(
            title="Statistical Summary",
            content=stats_text.strip(),
        ))
    
    # Section 4: Top Insights
    insights_text = "Key Findings:\n"
    insight_count = 0
    
    # Find top category
    for cat_col in categorical_cols[:1]:
        try:
            top = df[cat_col].value_counts().head(1)
            top_name = top[cat_col].item()
            top_pct = round(top["count"].item() / len(df) * 100, 1)
            insights_text += f"• '{top_name}' is the dominant {cat_col} ({top_pct}% of records)\n"
            insight_count += 1
        except:
            pass
    
    # Find correlations
    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols[:3]):
            for col2 in numeric_cols[i+1:4]:
                try:
                    corr = df.select(pl.corr(col1, col2)).item()
                    if corr and abs(corr) > 0.5:
                        direction = "positively" if corr > 0 else "negatively"
                        insights_text += f"• {col1} and {col2} are {direction} correlated ({corr:.2f})\n"
                        insight_count += 1
                        break
                except:
                    pass
            if insight_count >= 3:
                break
    
    if insight_count > 0:
        sections.append(ReportSection(
            title="Key Insights",
            content=insights_text.strip(),
        ))
    
    # Section 5: Recommendations
    recommendations = "Recommended Actions:\n"
    recommendations += "1. Address any missing values before modeling\n"
    recommendations += "2. Investigate correlations for potential causal relationships\n"
    recommendations += "3. Consider segmenting analysis by dominant categories\n"
    
    sections.append(ReportSection(
        title="Recommendations",
        content=recommendations.strip(),
    ))
    
    return ReportResponse(
        session_id=session_id,
        title=f"Analysis Report: {filename}",
        generated_at=datetime.now().isoformat(),
        sections=sections
    )

if __name__ == '__main__':
    import uvicorn
    import os
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host='0.0.0.0', port=port, proxy_headers=True, forwarded_allow_ips="*")
