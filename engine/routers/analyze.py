from fastapi import APIRouter, UploadFile, Depends, HTTPException
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

        session_record.row_count = len(df)
        session_record.column_count = len(df.columns)
        session_record.file_size_bytes = len(contents)
        session_record.status = "processing"
        await db.commit()

        results = run_insight_engine(df)
        print("=== RESULTS KEYS ===", list(results.keys()))

        # Mock PDF generation for now, wire it properly later
        report_path = f"/tmp/Report_{session_record.id}.pdf"

        # DEBUG — remove after inspection
        if results.get("insights"):
            first = results["insights"][0]
            print("=== INSIGHT DEBUG ===")
            print("TYPE:", type(first))
            print("KEYS:", first.keys() if hasattr(first, 'keys') else dir(first))
            print("FULL FIRST INSIGHT:", first)
            print("=== END DEBUG ===")

        if results.get("recommendations"):
            first_rec = results["recommendations"][0]
            print("=== REC DEBUG ===")
            print("TYPE:", type(first_rec))
            print("KEYS:", first_rec.keys() if hasattr(first_rec, 'keys') else dir(first_rec))
            print("FULL FIRST REC:", first_rec)
            print("=== END REC DEBUG ===")

        await save_results(
            db=db,
            session_id=session_record.id,
            kpis=results.get("kpis", {}),
            insights=results.get("insights", []),
            recommendations=results.get("recommendations", []),
            detected_domain=results.get("domain", "general"),
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
