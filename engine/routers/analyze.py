from fastapi import APIRouter, UploadFile, Depends, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
import io
import os
import pandas as pd
from db_async import get_db
from auth import get_current_user
from models import User
from services.session_service import create_session, save_results, mark_failed
from insight_engine import run_insight_engine

router = APIRouter(tags=["analyze"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")

@router.post("/analyze")
async def analyze(
    file: UploadFile,
    currency: str = Form("auto"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. Create session record immediately
    session_record = await create_session(
        db=db,
        user_id=current_user.id,
        filename=file.filename,
        original_filename=file.filename,
        row_count=0,
        column_count=0,
        file_size_bytes=0,
    )

    try:
        contents = await file.read()
        filename = (file.filename or "").lower()

        # Persist raw upload so the old /insights and /generate-viz endpoints can load it
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        safe_name = (file.filename or "upload").replace("/", "_").replace("\\", "_")
        upload_path = os.path.join(UPLOAD_DIR, f"session_{session_record.id}_{safe_name}")
        with open(upload_path, "wb") as fh:
            fh.write(contents)

        if filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        elif filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")
        else:
            raise HTTPException(400, "Unsupported file type. Please upload CSV or Excel.")

        # Reset explicit flag so auto-detect works when currency=="auto"
        from insight_engine import _set_currency_symbol
        import insight_engine as _ie
        _ie._CURRENCY_EXPLICIT = False

        # Override currency if user specified
        if currency != "auto":
            _CURRENCY_MAP = {
                "INR": "₹", "USD": "$", "GBP": "£",
                "EUR": "€", "AED": "AED", "SGD": "S$",
                "JPY": "¥",
            }
            sym = _CURRENCY_MAP.get(currency, "₹")
            _set_currency_symbol(sym, explicit=True)
            print(f"[CURRENCY] User override: {currency} → {sym}")
        else:
            try:
                from gemini_column_semantics import detect_currency_gemini
                detected_sym = detect_currency_gemini(df, list(df.columns))
                _set_currency_symbol(detected_sym)
                print(f"[Gemini Currency] Auto-detected: {detected_sym}")
            except Exception as _ce:
                print(f"[Gemini Currency] Failed: {_ce} — using rule-based")

        session_record.row_count = len(df)
        session_record.column_count = len(df.columns)
        session_record.file_size_bytes = len(contents)
        session_record.status = "processing"
        
        # Store currency selection in session
        if currency and currency != "auto":
            session_record.currency = currency
            print(f"[SESSION] Stored currency: {currency}")
        
        await db.commit()

        results = run_insight_engine(df)

        # Mock PDF generation for now, wire it properly later
        report_path = f"/tmp/Report_{session_record.id}.pdf"

        domain_raw = results.get("domain", {})
        domain_name = domain_raw.get("name", "general") if isinstance(domain_raw, dict) else str(domain_raw)

        await save_results(
            db=db,
            session_id=session_record.id,
            kpis=results.get("computed_metrics", {}),
            insights=results.get("strategic_brief", []),
            recommendations=results.get("recommendations", []),
            detected_domain=domain_name,
            report_path=report_path,
        )

        return {
            "session_id": session_record.id,
            **results
        }

    except Exception as e:
        await db.rollback()
        await mark_failed(db, session_record.id, str(e))
        raise HTTPException(500, f"Analysis failed: {str(e)}")
