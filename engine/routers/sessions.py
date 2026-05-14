from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from db_async import get_db
from auth import get_current_user
from models import User
from services.session_service import (
    get_user_sessions, get_session_detail, delete_session
)
from schemas import SessionListResponse, SessionDetailResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await get_user_sessions(db, current_user.id, page, page_size)


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = await get_session_detail(db, session_id, current_user.id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@router.delete("/{session_id}", status_code=204)
async def delete_session_endpoint(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    deleted = await delete_session(db, session_id, current_user.id)
    if not deleted:
        raise HTTPException(404, "Session not found")


@router.get("/{session_id}/report")
async def download_report(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = await get_session_detail(db, session_id, current_user.id)
    if not session:
        raise HTTPException(404, "Session not found")
    if not session.report_path:
        raise HTTPException(404, "Report not yet generated")

    return FileResponse(
        path=session.report_path,
        filename=f"InsightStream_{session.original_filename}_Report.pdf",
        media_type="application/pdf",
    )
