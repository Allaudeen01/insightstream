from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import polars as pl
import io
import uuid
from typing import Optional, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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

@app.on_event("startup")
async def startup_event():
    """Create session directory on startup."""
    try:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Session directory created: {SESSION_DIR}")
    except Exception as e:
        print(f"Warning: Could not create session directory: {e}")

def save_session(session_id: str, filename: str, df: pl.DataFrame) -> None:
    """Save session to disk."""
    # Ensure base directory exists (fallback in case startup event didn't run)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    
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

@app.get("/data/{session_id}")
def get_raw_data(session_id: str, page: int = 1, page_size: int = 100):
    """Get paginated raw data for a session."""
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Limit page_size for performance
    page_size = min(page_size, 1000)
    page = max(1, page)
    
    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size
    
    # Calculate slice indices
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    # Get the slice of data
    data_slice = df.slice(start_idx, page_size).to_dicts()
    
    return {
        "session_id": session_id,
        "columns": df.columns,
        "data": data_slice,
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages
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
    enabled: bool = True  # Allow toggling actions

class CleaningPreview(BaseModel):
    before_rows: int
    before_columns: int
    before_score: str
    after_rows: int
    after_columns: int
    after_score: str
    changes: list[str]
    sample_affected: list[dict]

def calculate_quality_score(df: pl.DataFrame) -> str:
    """Calculate quality score for a DataFrame."""
    total_cells = len(df) * len(df.columns)
    if total_cells == 0:
        return "A"
    
    null_count = sum(df[col].null_count() for col in df.columns)
    null_pct = (null_count / total_cells) * 100
    
    duplicate_count = len(df) - len(df.unique())
    dup_pct = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
    
    if null_pct > 20 or dup_pct > 10:
        return "D"
    elif null_pct > 10 or dup_pct > 5:
        return "C"
    elif null_pct > 5 or dup_pct > 2:
        return "B"
    return "A"

def apply_actions_to_df(df: pl.DataFrame, actions: list[CleanAction]) -> tuple[pl.DataFrame, list[str]]:
    """Apply cleaning actions and return cleaned df with changelog."""
    changelog = []
    
    for action in actions:
        if not action.enabled:
            continue
            
        if action.action == "drop_duplicates":
            before = len(df)
            df = df.unique()
            removed = before - len(df)
            if removed > 0:
                changelog.append(f"Removed {removed} duplicate rows")
        elif action.action == "drop_column" and action.column:
            if action.column in df.columns:
                df = df.drop(action.column)
                changelog.append(f"Dropped column '{action.column}'")
        elif action.action == "drop_nulls" and action.column:
            before = len(df)
            df = df.filter(pl.col(action.column).is_not_null())
            removed = before - len(df)
            if removed > 0:
                changelog.append(f"Dropped {removed} rows with null '{action.column}'")
        elif action.action == "impute_mean" and action.column:
            null_count = df[action.column].null_count()
            if null_count > 0:
                mean_val = df[action.column].mean()
                df = df.with_columns(pl.col(action.column).fill_null(mean_val))
                changelog.append(f"Imputed {null_count} missing '{action.column}' with mean ({mean_val:.2f})")
        elif action.action == "impute_median" and action.column:
            null_count = df[action.column].null_count()
            if null_count > 0:
                median_val = df[action.column].median()
                df = df.with_columns(pl.col(action.column).fill_null(median_val))
                changelog.append(f"Imputed {null_count} missing '{action.column}' with median ({median_val:.2f})")
        elif action.action == "impute_mode" and action.column:
            null_count = df[action.column].null_count()
            if null_count > 0:
                mode_list = df[action.column].mode().to_list()
                mode_val = mode_list[0] if len(mode_list) > 0 else None
                if mode_val is not None:
                    df = df.with_columns(pl.col(action.column).fill_null(mode_val))
                    changelog.append(f"Imputed {null_count} missing '{action.column}' with mode ('{mode_val}')")
    
    return df, changelog

@app.post("/preview-clean/{session_id}")
def preview_cleaning(session_id: str, actions: list[CleanAction]):
    """
    Preview cleaning actions without applying them.
    Returns before/after stats and sample of affected rows.
    """
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Before stats
    before_rows = len(df)
    before_columns = len(df.columns)
    before_score = calculate_quality_score(df)
    
    # Apply to a copy
    df_copy = df.clone()
    cleaned_df, changelog = apply_actions_to_df(df_copy, actions)
    
    # After stats
    after_rows = len(cleaned_df)
    after_columns = len(cleaned_df.columns)
    after_score = calculate_quality_score(cleaned_df)
    
    # Get sample of affected rows (first 5)
    sample_affected = []
    try:
        sample_affected = cleaned_df.head(5).to_dicts()
    except Exception:
        pass
    
    return {
        "before_rows": before_rows,
        "before_columns": before_columns,
        "before_score": before_score,
        "after_rows": after_rows,
        "after_columns": after_columns, 
        "after_score": after_score,
        "row_delta": after_rows - before_rows,
        "column_delta": after_columns - before_columns,
        "changes": changelog,
        "sample_affected": sample_affected
    }

@app.post("/clean/{session_id}")
def apply_cleaning(session_id: str, actions: list[CleanAction]):
    """
    Apply cleaning actions to the dataset.
    Saves backup of original data for undo capability.
    """
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save backup before cleaning
    backup_path = SESSION_DIR / session_id / "backup.parquet"
    try:
        df.write_parquet(backup_path)
    except Exception as e:
        print(f"Warning: Could not save backup: {e}")
    
    # Apply cleaning actions
    cleaned_df, changelog = apply_actions_to_df(df, actions)
    
    # Save the cleaned DataFrame back to disk
    save_session(session_id, filename, cleaned_df)
    
    return {
        "status": "ok", 
        "row_count": len(cleaned_df), 
        "column_count": len(cleaned_df.columns),
        "changes": changelog,
        "can_undo": backup_path.exists()
    }

@app.post("/undo-clean/{session_id}")
def undo_cleaning(session_id: str):
    """
    Restore the original data from backup.
    """
    try:
        filename, _ = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    backup_path = SESSION_DIR / session_id / "backup.parquet"
    
    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="No backup available to restore")
    
    try:
        # Restore from backup
        df = pl.read_parquet(backup_path)
        save_session(session_id, filename, df)
        
        # Remove backup after restore
        backup_path.unlink()
        
        return {
            "status": "ok",
            "message": "Data restored to original state",
            "row_count": len(df),
            "column_count": len(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore backup: {str(e)}")



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


# ============== PHASE 5: ADVANCED VISUALIZATIONS ==============

class ChartData(BaseModel):
    chart_id: str
    chart_type: str
    title: str
    description: str
    plotly_json: dict
    columns_used: List[str]
    priority_score: float = 50.0  # Higher = more interesting (0-100)
    insight_reason: str = ""  # Why this chart was prioritized (for tooltip)
    interest_level: str = "standard"  # "high", "recommended", or "standard"

class VizResponse(BaseModel):
    session_id: str
    charts: List[ChartData]
    total_generated: int

@app.get("/generate-viz/{session_id}", response_model=VizResponse)
def generate_visualizations(session_id: str, max_charts: int = 10):
    """
    Auto-generate advanced interactive charts based on dataset characteristics.
    Returns Plotly JSON for frontend rendering with react-plotly.js.
    """
    try:
        filename, df = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Convert to pandas for Plotly
    pdf = df.to_pandas()
    charts = []
    
    # Categorize columns
    numeric_cols = [c for c in pdf.columns if pdf[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    categorical_cols = [c for c in pdf.columns if pdf[c].dtype == 'object' or pdf[c].nunique() < 10]
    
    # ============== SMART COLUMN SCORING ==============
    # Analyze data patterns to prioritize most interesting visualizations
    
    column_scores = {}
    column_insights = {}  # Track why each column is interesting
    
    def get_interest_level(score: float) -> str:
        """Convert priority score to interest level label."""
        if score >= 70:
            return "high"
        elif score >= 55:
            return "recommended"
        return "standard"
    
    # Score numeric columns by: variance, missing %, unique ratio
    for col in numeric_cols:
        score = 50
        reasons = []
        missing_pct = pdf[col].isna().sum() / len(pdf) * 100
        if missing_pct > 10:
            score += 15  # High missing = interesting distribution question
            reasons.append(f"{missing_pct:.0f}% missing values")
        try:
            cv = pdf[col].std() / (abs(pdf[col].mean()) + 0.001)
            if cv > 0.5:
                score += 10  # High coefficient of variation = spread worth showing
                reasons.append("High variance")
        except:
            pass
        column_scores[col] = min(score, 100)
        column_insights[col] = reasons if reasons else ["Standard distribution"]
    
    # Score categorical columns by: cardinality (3-8 ideal), balance
    for col in categorical_cols:
        score = 50
        reasons = []
        n_unique = pdf[col].nunique()
        if 3 <= n_unique <= 8:
            score += 20  # Ideal for pie/bar
            reasons.append(f"{n_unique} categories (ideal)")
        elif n_unique <= 2:
            score -= 10  # Binary less interesting
        else:
            reasons.append(f"{n_unique} categories")
        # Check balance (entropy-like)
        value_counts = pdf[col].value_counts(normalize=True)
        if value_counts.max() < 0.7:  # Not dominated by one value
            score += 10
            reasons.append("Balanced distribution")
        column_scores[col] = min(score, 100)
        column_insights[col] = reasons if reasons else ["Categorical variable"]
    
    # Find strong correlations for scatter prioritization
    strong_correlations = []
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = pdf[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr = abs(corr_matrix.loc[col1, col2])
                    if corr > 0.5:
                        strong_correlations.append((col1, col2, corr))
            strong_correlations.sort(key=lambda x: x[2], reverse=True)
        except:
            pass
    
    # Sort columns by interest score for chart generation
    numeric_cols = sorted(numeric_cols, key=lambda c: column_scores.get(c, 50), reverse=True)
    categorical_cols = sorted(categorical_cols, key=lambda c: column_scores.get(c, 50), reverse=True)
    
    # ============== PHASE 0: CORE BASICS ==============
    
    # 0.1 HISTOGRAM: Distribution of numeric columns (with optional color grouping)
    for col in numeric_cols[:2]:  # Limit to first 2 numeric columns
        if pdf[col].nunique() > 5 and len(charts) < max_charts:
            try:
                # Add color grouping if good categorical exists
                color_col = None
                if categorical_cols and pdf[categorical_cols[0]].nunique() <= 4:
                    color_col = categorical_cols[0]
                
                fig = px.histogram(
                    pdf, x=col,
                    color=color_col,
                    title=f"Distribution of {col}" + (f" by {color_col}" if color_col else ""),
                    marginal="rug",
                    barmode="overlay" if color_col else "relative",
                    opacity=0.7 if color_col else 1.0
                )
                fig.update_layout(template="plotly_dark", showlegend=bool(color_col))
                if not color_col:
                    fig.update_traces(marker_color='#6366f1')
                score = column_scores.get(col, 50) + (10 if color_col else 0)
                reasons = column_insights.get(col, [])
                charts.append(ChartData(
                    chart_id=f"histogram_{col}",
                    chart_type="histogram",
                    title=f"{col} Distribution",
                    description=f"Frequency distribution" + (f" colored by {color_col}" if color_col else ""),
                    plotly_json=fig.to_dict(),
                    columns_used=[col] + ([color_col] if color_col else []),
                    priority_score=score,
                    insight_reason=" • ".join(reasons) if reasons else "Distribution analysis",
                    interest_level=get_interest_level(score)
                ))
            except:
                pass
    
    # 0.2 BAR CHART: Categorical vs Numeric (median for robustness, horizontal for long labels)
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1 and len(charts) < max_charts:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        n_categories = pdf[cat_col].nunique()
        if n_categories <= 12 and cat_col != num_col:
            try:
                # Use median for robustness against outliers
                agg_df = pdf.groupby(cat_col)[num_col].median().reset_index()
                agg_df.columns = [cat_col, f'Median {num_col}']
                agg_df = agg_df.sort_values(f'Median {num_col}', ascending=True)
                
                # Use horizontal for many categories or long labels
                use_horizontal = n_categories > 6 or pdf[cat_col].astype(str).str.len().max() > 10
                
                fig = px.bar(
                    agg_df, 
                    x=f'Median {num_col}' if use_horizontal else cat_col,
                    y=cat_col if use_horizontal else f'Median {num_col}',
                    orientation='h' if use_horizontal else 'v',
                    title=f"Median {num_col} by {cat_col}",
                    text_auto='.1f',
                    color=f'Median {num_col}',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(template="plotly_dark", showlegend=False, coloraxis_showscale=False)
                score = (column_scores.get(cat_col, 50) + column_scores.get(num_col, 50)) / 2
                reasons = column_insights.get(cat_col, []) + column_insights.get(num_col, [])
                charts.append(ChartData(
                    chart_id=f"bar_{num_col}_{cat_col}",
                    chart_type="bar",
                    title=f"{num_col} by {cat_col}",
                    description=f"Median {num_col} comparison (robust to outliers)",
                    plotly_json=fig.to_dict(),
                    columns_used=[cat_col, num_col],
                    priority_score=score,
                    insight_reason=" • ".join(reasons[:2]) if reasons else "Category comparison",
                    interest_level=get_interest_level(score)
                ))
            except:
                pass
    
    # 0.3 PIE CHART: With auto-collapse small slices into "Other"
    if len(categorical_cols) >= 1 and len(charts) < max_charts:
        cat_col = categorical_cols[0]
        n_unique = pdf[cat_col].nunique()
        if n_unique <= 15:  # Allow more but collapse
            try:
                counts = pdf[cat_col].value_counts().reset_index()
                counts.columns = [cat_col, 'count']
                
                # Auto-collapse slices < 4% into "Other"
                total = counts['count'].sum()
                threshold = 0.04 * total
                
                # Mark small categories as "Other"
                counts['category'] = counts.apply(
                    lambda row: row[cat_col] if row['count'] >= threshold else 'Other',
                    axis=1
                )
                counts = counts.groupby('category')['count'].sum().reset_index()
                counts = counts.sort_values('count', ascending=False)
                
                # Limit to 8 slices max
                if len(counts) > 8:
                    top_7 = counts.head(7)
                    other_sum = counts.iloc[7:]['count'].sum()
                    other_row = pd.DataFrame({'category': ['Other'], 'count': [other_sum]})
                    counts = pd.concat([top_7, other_row], ignore_index=True)
                
                fig = px.pie(
                    counts, names='category', values='count',
                    title=f"Proportion of {cat_col}",
                    hole=0.4,  # Slightly larger hole for modern look
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(template="plotly_dark")
                fig.update_traces(textposition='inside', textinfo='percent+label', 
                                  pull=[0.02] * len(counts))  # Slight pull for emphasis
                charts.append(ChartData(
                    chart_id=f"pie_{cat_col}",
                    chart_type="pie",
                    title=f"{cat_col} Breakdown",
                    description=f"Percentage distribution (small slices grouped as 'Other')",
                    plotly_json=fig.to_dict(),
                    columns_used=[cat_col],
                    priority_score=column_scores.get(cat_col, 50) + 5  # Pie charts popular
                ))
            except:
                pass
    
    # 0.4 COUNT BAR: Stacked categorical frequency
    if len(categorical_cols) >= 2 and len(charts) < max_charts:
        cat1, cat2 = categorical_cols[0], categorical_cols[1]
        if cat1 != cat2 and pdf[cat1].nunique() <= 8 and pdf[cat2].nunique() <= 6:
            try:
                fig = px.histogram(
                    pdf, x=cat1, color=cat2,
                    title=f"{cat1} Counts by {cat2}",
                    barmode="stack"  # Stacked shows proportions better
                )
                fig.update_layout(template="plotly_dark")
                charts.append(ChartData(
                    chart_id=f"countbar_{cat1}_{cat2}",
                    chart_type="count_bar",
                    title=f"{cat1} by {cat2}",
                    description=f"Frequency counts of {cat1} grouped by {cat2}",
                    plotly_json=fig.to_dict(),
                    columns_used=[cat1, cat2]
                ))
            except:
                pass
    
    # ============== PHASE 1: STATISTICAL & DISTRIBUTION ==============
    
    # Find best numeric column (highest variance or missing %)
    best_num = None
    best_num_score = 0
    for col in numeric_cols:
        try:
            variance = pdf[col].var() or 0
            missing = pdf[col].isna().sum() / len(pdf)
            score = variance / (pdf[col].mean()**2 + 0.001) + missing * 50  # CV + missing bonus
            if score > best_num_score:
                best_num_score = score
                best_num = col
        except:
            pass
    if not best_num and numeric_cols:
        best_num = numeric_cols[0]
    
    # Find best categorical (3-8 unique values, balanced)
    best_cat = None
    best_cat_score = 0
    for col in categorical_cols:
        n_unique = pdf[col].nunique()
        if 3 <= n_unique <= 8:
            balance = 1 - pdf[col].value_counts(normalize=True).max()
            score = balance * 100 + (8 - abs(n_unique - 5)) * 5
            if score > best_cat_score:
                best_cat_score = score
                best_cat = col
    if not best_cat and categorical_cols:
        best_cat = categorical_cols[0]
    
    # 1.1 VIOLIN PLOT: Best numeric by best categorical
    if best_num and best_cat and len(charts) < max_charts:
        try:
            fig = px.violin(
                pdf, x=best_cat, y=best_num,
                box=True, points="all",
                color=best_cat,
                title=f"Distribution of {best_num} by {best_cat}",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(template="plotly_dark", showlegend=False)
            score = column_scores.get(best_num, 50) + 25  # Violin boost
            reasons = column_insights.get(best_num, []) + ["Shows distribution shape + outliers"]
            charts.append(ChartData(
                chart_id=f"violin_{best_num}_{best_cat}",
                chart_type="violin",
                title=f"{best_num} by {best_cat}",
                description=f"Full distribution with box stats and individual points",
                plotly_json=fig.to_dict(),
                columns_used=[best_num, best_cat],
                priority_score=score,
                insight_reason=" • ".join(reasons[:3]),
                interest_level=get_interest_level(score)
            ))
        except:
            pass
    
    # 1.2 SCATTER + MARGINALS: Strongest correlation pair
    if len(strong_correlations) > 0 and len(charts) < max_charts:
        # Use the strongest correlation pair
        col1, col2, corr_val = strong_correlations[0]
        color_col = best_cat if best_cat and pdf[best_cat].nunique() <= 4 else None
        try:
            fig = px.scatter(
                pdf, x=col1, y=col2,
                color=color_col,
                marginal_x="histogram", marginal_y="violin",
                title=f"{col1} vs {col2} (r = {corr_val:.2f})",
                opacity=0.7,
                trendline="ols" if not color_col else None  # Add OLS trendline when no color grouping
            )
            fig.update_layout(template="plotly_dark")
            fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
            # Style the trendline (if present)
            fig.update_traces(line=dict(color="#ef4444", dash="dash", width=2), selector=dict(mode="lines"))
            score = 60 + corr_val * 40  # Strong boost for high correlation
            charts.append(ChartData(
                chart_id=f"scatter_corr_{col1}_{col2}",
                chart_type="scatter_marginals",
                title=f"{col1} vs {col2}",
                description=f"Bivariate relationship with marginal distributions",
                plotly_json=fig.to_dict(),
                columns_used=[col1, col2] + ([color_col] if color_col else []),
                priority_score=score,
                insight_reason=f"Strongest correlation (r = {corr_val:.2f})",
                interest_level=get_interest_level(score)
            ))
        except:
            pass
    elif len(numeric_cols) >= 2 and len(charts) < max_charts:
        # Fallback: first two numerics
        col1, col2 = numeric_cols[0], numeric_cols[1]
        try:
            fig = px.scatter(
                pdf, x=col1, y=col2,
                marginal_x="histogram", marginal_y="violin",
                title=f"{col1} vs {col2} with Distributions"
            )
            fig.update_layout(template="plotly_dark")
            charts.append(ChartData(
                chart_id=f"scatter_marginal_{col1}_{col2}",
                chart_type="scatter_marginals",
                title=f"{col1} vs {col2}",
                description=f"Bivariate analysis with marginal distributions",
                plotly_json=fig.to_dict(),
                columns_used=[col1, col2],
                priority_score=55,
                insight_reason="Numeric relationship exploration",
                interest_level="recommended"
            ))
        except:
            pass
    
    # 1.3 BOX PLOT: With notches and points
    if best_num and best_cat and len(charts) < max_charts:
        try:
            fig = px.box(
                pdf, x=best_cat, y=best_num,
                notched=True, points="all",
                color=best_cat,
                title=f"Box Summary of {best_num} by {best_cat}",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(template="plotly_dark", showlegend=False)
            score = column_scores.get(best_num, 50) + 15  # Box plot boost
            charts.append(ChartData(
                chart_id=f"box_{best_num}_{best_cat}",
                chart_type="box",
                title=f"{best_num} Summary",
                description=f"Quartiles, median, and confidence intervals with data points",
                plotly_json=fig.to_dict(),
                columns_used=[best_num, best_cat],
                priority_score=score,
                insight_reason="Statistical summary with confidence notches",
                interest_level=get_interest_level(score)
            ))
        except:
            pass
    
    # 1.4 DENSITY HEATMAP: Joint distribution of top correlation pair
    if len(numeric_cols) >= 2 and len(charts) < max_charts:
        # Use correlation pair if available, else first two
        if strong_correlations:
            col1, col2, _ = strong_correlations[0]
        else:
            col1, col2 = numeric_cols[0], numeric_cols[1]
        try:
            fig = px.density_heatmap(
                pdf, x=col1, y=col2,
                title=f"Density: {col1} vs {col2}",
                color_continuous_scale="Viridis",
                marginal_x="histogram", marginal_y="histogram"
            )
            fig.update_layout(template="plotly_dark")
            charts.append(ChartData(
                chart_id=f"density_{col1}_{col2}",
                chart_type="density_heatmap",
                title=f"Joint Distribution",
                description=f"Density concentration showing where data clusters",
                plotly_json=fig.to_dict(),
                columns_used=[col1, col2],
                priority_score=58,
                insight_reason="Shows concentration patterns in joint distribution",
                interest_level="recommended"
            ))
        except:
            pass
    
    # ============== PHASE 2: HIERARCHICAL & SPECIALIZED ==============
    
    # Smart path detection for hierarchical charts
    path_candidates = []
    for col in categorical_cols:
        n_unique = pdf[col].nunique()
        if 3 <= n_unique <= 12:
            balance = 1 - pdf[col].value_counts(normalize=True).max()
            avg_label_len = pdf[col].astype(str).str.len().mean()
            score = balance * 50 + (12 - abs(n_unique - 6)) * 5
            path_candidates.append((col, n_unique, balance, avg_label_len, score))
    
    # Sort by score (best first)
    path_candidates.sort(key=lambda x: x[4], reverse=True)
    
    # 2.1 TREEMAP: Hierarchical categories (prefer when labels are long)
    if len(path_candidates) >= 2 and len(charts) < max_charts:
        path_cols = [p[0] for p in path_candidates[:3]]  # Top 2-3 columns
        val_col = best_num if best_num else None
        avg_label_len = sum(p[3] for p in path_candidates[:2]) / 2
        
        # Use treemap when labels are longer
        if avg_label_len > 6:
            try:
                if val_col:
                    treemap_df = pdf.dropna(subset=path_cols + [val_col])
                    fig = px.treemap(
                        treemap_df,
                        path=path_cols[:2],
                        values=val_col,
                        title=f"Breakdown of {val_col} by {' → '.join(path_cols[:2])}",
                        color_continuous_scale="Viridis"
                    )
                else:
                    treemap_df = pdf.dropna(subset=path_cols)
                    treemap_df['_count'] = 1
                    fig = px.treemap(
                        treemap_df,
                        path=path_cols[:2],
                        values='_count',
                        title=f"Hierarchy: {' → '.join(path_cols[:2])}"
                    )
                fig.update_layout(template="plotly_dark")
                depth = len(path_cols[:2])
                score = 60 + (depth * 15)  # Boost for depth
                charts.append(ChartData(
                    chart_id=f"treemap_{'_'.join(path_cols[:2])}",
                    chart_type="treemap",
                    title=f"Hierarchical Breakdown",
                    description=f"Part-to-whole analysis across {depth} levels",
                    plotly_json=fig.to_dict(),
                    columns_used=path_cols[:2] + ([val_col] if val_col else []),
                    priority_score=score,
                    insight_reason=f"{depth}-level hierarchy detected • Good cardinality",
                    interest_level=get_interest_level(score)
                ))
            except:
                pass
    
    # 2.2 SUNBURST: Radial hierarchy (prefer when labels are short)
    if len(path_candidates) >= 2 and len(charts) < max_charts:
        path_cols = [p[0] for p in path_candidates[:3]]
        val_col = best_num if best_num else None
        avg_label_len = sum(p[3] for p in path_candidates[:2]) / 2
        
        # Use sunburst when labels are shorter (fits better radially)
        if avg_label_len <= 10:
            try:
                if val_col:
                    sunburst_df = pdf.dropna(subset=path_cols[:2] + [val_col])
                    fig = px.sunburst(
                        sunburst_df,
                        path=path_cols[:2],
                        values=val_col,
                        title=f"Sunburst: {' → '.join(path_cols[:2])}",
                        color_continuous_scale="Viridis"
                    )
                else:
                    sunburst_df = pdf.dropna(subset=path_cols[:2])
                    sunburst_df['_count'] = 1
                    fig = px.sunburst(
                        sunburst_df,
                        path=path_cols[:2],
                        values='_count',
                        title=f"Sunburst: {' → '.join(path_cols[:2])}"
                    )
                fig.update_layout(template="plotly_dark")
                score = 58 + (len(path_cols[:2]) * 12)
                charts.append(ChartData(
                    chart_id=f"sunburst_{'_'.join(path_cols[:2])}",
                    chart_type="sunburst",
                    title=f"Radial Hierarchy",
                    description=f"Nested breakdown in radial format",
                    plotly_json=fig.to_dict(),
                    columns_used=path_cols[:2] + ([val_col] if val_col else []),
                    priority_score=score,
                    insight_reason="Compact labels suit radial display",
                    interest_level=get_interest_level(score)
                ))
            except:
                pass
    
    # 2.3 FUNNEL CHART: Detect stage-like columns
    stage_keywords = ['stage', 'step', 'phase', 'funnel', 'conversion', 'status', 'level', 'tier']
    ordinal_patterns = ['low', 'medium', 'high', 'small', 'large', 'start', 'end', 'begin', 'complete']
    
    funnel_col = None
    funnel_score = 0
    for col in categorical_cols:
        col_lower = col.lower()
        # Check if column name matches stage keywords
        if any(kw in col_lower for kw in stage_keywords):
            funnel_col = col
            funnel_score = 85  # High priority for explicit stage columns
            break
        # Check if values look ordinal
        values = pdf[col].dropna().astype(str).str.lower().unique()
        if len(values) >= 2 and len(values) <= 8:
            if any(p in ' '.join(values) for p in ordinal_patterns):
                funnel_col = col
                funnel_score = 70
    
    if funnel_col and len(charts) < max_charts:
        try:
            funnel_data = pdf[funnel_col].value_counts().reset_index()
            funnel_data.columns = [funnel_col, 'count']
            # Sort by count descending for proper funnel shape
            funnel_data = funnel_data.sort_values('count', ascending=False)
            
            fig = px.funnel(
                funnel_data,
                x='count',
                y=funnel_col,
                title=f"Funnel: {funnel_col}",
                color_discrete_sequence=['#6366f1']
            )
            fig.update_layout(template="plotly_dark")
            charts.append(ChartData(
                chart_id=f"funnel_{funnel_col}",
                chart_type="funnel",
                title=f"{funnel_col} Funnel",
                description="Stage-by-stage breakdown showing progression",
                plotly_json=fig.to_dict(),
                columns_used=[funnel_col],
                priority_score=funnel_score,
                insight_reason="Stage-like column detected" if funnel_score > 75 else "Ordinal values suggest progression",
                interest_level=get_interest_level(funnel_score)
            ))
        except:
            pass
    
    # 2.4 ICICLE: For deep hierarchies (3+ levels)
    if len(path_candidates) >= 3 and len(charts) < max_charts:
        path_cols = [p[0] for p in path_candidates[:3]]
        val_col = best_num if best_num else None
        
        try:
            if val_col:
                icicle_df = pdf.dropna(subset=path_cols + [val_col])
                fig = px.icicle(
                    icicle_df,
                    path=path_cols,
                    values=val_col,
                    title=f"Deep Hierarchy: {' → '.join(path_cols)}",
                    color_continuous_scale="Blues"
                )
            else:
                icicle_df = pdf.dropna(subset=path_cols)
                icicle_df['_count'] = 1
                fig = px.icicle(
                    icicle_df,
                    path=path_cols,
                    values='_count',
                    title=f"Deep Hierarchy: {' → '.join(path_cols)}"
                )
            fig.update_layout(template="plotly_dark")
            score = 55 + (len(path_cols) * 10)
            charts.append(ChartData(
                chart_id=f"icicle_{'_'.join(path_cols)}",
                chart_type="icicle",
                title=f"Icicle Chart",
                description=f"Deep {len(path_cols)}-level hierarchical breakdown",
                plotly_json=fig.to_dict(),
                columns_used=path_cols + ([val_col] if val_col else []),
                priority_score=score,
                insight_reason=f"{len(path_cols)}-level deep hierarchy",
                interest_level=get_interest_level(score)
            ))
        except:
            pass
    
    # ============== PHASE 3: ADVANCED CHARTS ==============
    
    # 3.1 PARALLEL COORDINATES: Multi-numeric exploration (enhanced)
    if len(numeric_cols) >= 4 and len(charts) < max_charts:
        dims = numeric_cols[:6]  # Limit to 6 dimensions
        color_col = best_cat if best_cat else (categorical_cols[0] if categorical_cols else None)
        try:
            if color_col and pdf[color_col].nunique() <= 8:
                pdf_temp = pdf.copy()
                pdf_temp[f"{color_col}_encoded"] = pdf_temp[color_col].astype('category').cat.codes
                fig = px.parallel_coordinates(
                    pdf_temp, dimensions=dims,
                    color=f"{color_col}_encoded",
                    title=f"Multi-Dimensional Comparison (colored by {color_col})",
                    color_continuous_scale="Viridis"
                )
            else:
                fig = px.parallel_coordinates(
                    pdf, dimensions=dims,
                    title="Multi-Dimensional Analysis"
                )
            fig.update_layout(template="plotly_dark")
            score = 50 + len(dims) * 8  # More dimensions = more interesting
            charts.append(ChartData(
                chart_id="parallel_coords",
                chart_type="parallel_coordinates",
                title="Parallel Coordinates",
                description=f"Compare {len(dims)} numeric dimensions simultaneously",
                plotly_json=fig.to_dict(),
                columns_used=dims,
                priority_score=score,
                insight_reason=f"{len(dims)} numeric columns suitable for parallel comparison",
                interest_level=get_interest_level(score)
            ))
        except:
            pass
    
    # 3.2 SANKEY DIAGRAM: Flow between categorical columns
    if len(categorical_cols) >= 2 and len(charts) < max_charts:
        # Find best source-target pair with good cardinality
        source_col = None
        target_col = None
        best_flow_score = 0
        
        for i, col1 in enumerate(categorical_cols[:4]):
            for col2 in categorical_cols[i+1:4]:
                n1, n2 = pdf[col1].nunique(), pdf[col2].nunique()
                if 2 <= n1 <= 10 and 2 <= n2 <= 10:
                    # Score based on cardinality balance and distinct values
                    flow_score = min(n1, n2) * max(n1, n2) / (abs(n1 - n2) + 1)
                    if flow_score > best_flow_score:
                        best_flow_score = flow_score
                        source_col, target_col = col1, col2
        
        if source_col and target_col:
            try:
                # Create source-target-value aggregation
                flow_df = pdf.groupby([source_col, target_col]).size().reset_index(name='count')
                
                # Create node labels (truncate long names with ellipsis)
                def truncate_label(name, max_len=12):
                    s = str(name).replace('_target', '')
                    return s[:max_len-1] + '…' if len(s) > max_len else s
                
                sources = flow_df[source_col].unique().tolist()
                targets = flow_df[target_col].unique().tolist()
                all_nodes = sources + [f"{t}_target" for t in targets]
                
                # Generate gradient colors for nodes
                n_nodes = len(all_nodes)
                node_colors = [f'hsl({i * 360 // n_nodes}, 70%, 50%)' for i in range(n_nodes)]
                
                # Create link indices
                source_idx = [sources.index(s) for s in flow_df[source_col]]
                target_idx = [len(sources) + targets.index(t) for t in flow_df[target_col]]
                
                fig = go.Figure(go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="rgba(255,255,255,0.3)", width=0.5),
                        label=[truncate_label(n) for n in all_nodes],
                        color=node_colors,
                        hovertemplate='%{label}<br>Total: %{value}<extra></extra>'
                    ),
                    link=dict(
                        source=source_idx,
                        target=target_idx,
                        value=flow_df['count'].tolist(),
                        color='rgba(99, 102, 241, 0.6)',
                        hovertemplate='%{source.label} → %{target.label}<br>Count: %{value}<extra></extra>'
                    )
                ))
                fig.update_layout(
                    title=f"Flow: {source_col} → {target_col}",
                    template="plotly_dark",
                    hovermode="closest"
                )
                score = 70 + min(best_flow_score, 20)  # High boost for Sankey
                charts.append(ChartData(
                    chart_id=f"sankey_{source_col}_{target_col}",
                    chart_type="sankey",
                    title=f"Flow Diagram",
                    description=f"Flow relationships between {source_col} and {target_col}",
                    plotly_json=fig.to_dict(),
                    columns_used=[source_col, target_col],
                    priority_score=score,
                    insight_reason=f"Flow structure detected between {source_col} and {target_col}",
                    interest_level=get_interest_level(score)
                ))
            except:
                pass
    
    # 3.3 RADAR / POLAR CHART: Multi-attribute comparison
    if len(numeric_cols) >= 4 and len(numeric_cols) <= 8 and len(charts) < max_charts:
        radar_cols = numeric_cols[:8]
        try:
            # Normalize values for radar (0-1 scale)
            radar_df = pdf[radar_cols].copy()
            for col in radar_cols:
                min_val, max_val = radar_df[col].min(), radar_df[col].max()
                if max_val > min_val:
                    radar_df[col] = (radar_df[col] - min_val) / (max_val - min_val)
            
            # Use median values for the radar
            median_values = radar_df.median().tolist()
            median_values.append(median_values[0])  # Close the polygon
            radar_cols_closed = radar_cols + [radar_cols[0]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=median_values,
                theta=radar_cols_closed,
                fill='toself',
                fillcolor='rgba(99, 102, 241, 0.3)',
                line=dict(color='#6366f1', width=2),
                name='Median'
            ))
            fig.update_layout(
                title=f"Attribute Comparison Radar ({len(radar_cols)} dimensions)",
                template="plotly_dark",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    bgcolor='rgba(0,0,0,0)'
                )
            )
            score = 55 + len(radar_cols) * 5
            charts.append(ChartData(
                chart_id="radar_attributes",
                chart_type="radar",
                title=f"Attribute Radar",
                description=f"Multi-attribute comparison across {len(radar_cols)} dimensions",
                plotly_json=fig.to_dict(),
                columns_used=radar_cols,
                priority_score=score,
                insight_reason=f"{len(radar_cols)} numeric attributes suitable for radar comparison",
                interest_level=get_interest_level(score)
            ))
        except:
            pass
    
    # 3.4 WATERFALL CHART: For numeric with positive/negative values
    waterfall_col = None
    waterfall_score = 0
    for col in numeric_cols:
        try:
            values = pdf[col].dropna()
            has_positive = (values > 0).any()
            has_negative = (values < 0).any()
            if has_positive and has_negative:
                # Good candidate - has both positive and negative
                ratio = min(abs(values[values > 0].sum()), abs(values[values < 0].sum())) / max(abs(values.sum()), 1)
                if ratio > 0.1:  # At least 10% contribution from both sides
                    col_score = 80 + ratio * 20
                    if col_score > waterfall_score:
                        waterfall_score = col_score
                        waterfall_col = col
        except:
            pass
    
    if waterfall_col and len(charts) < max_charts:
        try:
            # Create waterfall from top values
            cat_col = best_cat if best_cat else (categorical_cols[0] if categorical_cols else None)
            if cat_col and pdf[cat_col].nunique() <= 10:
                waterfall_data = pdf.groupby(cat_col)[waterfall_col].sum().sort_values(ascending=False).head(10)
                
                fig = go.Figure(go.Waterfall(
                    x=waterfall_data.index.tolist(),
                    y=waterfall_data.values.tolist(),
                    connector=dict(line=dict(color='rgba(99, 102, 241, 0.5)', width=1.5, dash='dash')),
                    increasing=dict(marker=dict(color='#10b981', line=dict(color='#059669', width=1))),
                    decreasing=dict(marker=dict(color='#ef4444', line=dict(color='#dc2626', width=1))),
                    totals=dict(marker=dict(color='#6366f1', line=dict(color='#4f46e5', width=1))),
                    textposition="outside",
                    hovertemplate='%{x}<br>%{y:+,.0f}<extra></extra>'
                ))
                net_value = waterfall_data.sum()
                fig.update_layout(
                    title=f"Waterfall: {waterfall_col} by {cat_col}",
                    template="plotly_dark",
                    annotations=[dict(
                        x=1, y=1.1, xref="paper", yref="paper",
                        text=f"Net: {net_value:+,.0f}",
                        showarrow=False,
                        font=dict(size=14, color='#10b981' if net_value >= 0 else '#ef4444')
                    )]
                )
                charts.append(ChartData(
                    chart_id=f"waterfall_{waterfall_col}",
                    chart_type="waterfall",
                    title=f"Waterfall Chart",
                    description=f"Cumulative {waterfall_col} changes by {cat_col}",
                    plotly_json=fig.to_dict(),
                    columns_used=[waterfall_col, cat_col],
                    priority_score=waterfall_score,
                    insight_reason="Clear positive/negative value pattern detected",
                    interest_level=get_interest_level(waterfall_score)
                ))
        except:
            pass
    
    # 3.5 CORRELATION HEATMAP (enhanced with priority)
    if len(numeric_cols) >= 3 and len(charts) < max_charts:
        try:
            corr_matrix = pdf[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.columns.tolist(),
                colorscale='RdBu_r',
                zmin=-1, zmax=1,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            fig.update_layout(
                title="Correlation Matrix",
                template="plotly_dark"
            )
            # Score based on how many strong correlations exist
            strong_corr_count = ((corr_matrix.abs() > 0.5) & (corr_matrix.abs() < 1)).sum().sum() / 2
            score = 50 + min(strong_corr_count * 5, 30)
            charts.append(ChartData(
                chart_id="correlation_heatmap",
                chart_type="heatmap",
                title="Correlation Matrix",
                description="Pairwise correlations between numeric columns",
                plotly_json=fig.to_dict(),
                columns_used=numeric_cols,
                priority_score=score,
                insight_reason=f"{int(strong_corr_count)} strong correlations found" if strong_corr_count > 0 else "Overview of variable relationships",
                interest_level=get_interest_level(score)
            ))
        except:
            pass
    
    # Sort charts by priority score (most interesting first)
    charts.sort(key=lambda c: c.priority_score, reverse=True)
    
    return VizResponse(
        session_id=session_id,
        charts=charts[:max_charts],
        total_generated=len(charts)
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
# NOTE: sklearn imports are deferred to function level to avoid slow startup

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
    # Deferred imports to avoid slow startup on Railway
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
    import numpy as np
    
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
