import json
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from sqlalchemy.orm import selectinload

from models import AnalysisSession, SessionInsight, SessionRecommendation


def _normalize_impact(raw: str) -> str:
    s = raw.lower()
    if "critical" in s: return "critical"
    if "important" in s or "high" in s: return "important"
    return "minor"


async def create_session(
    db: AsyncSession,
    user_id: int,
    filename: str,
    original_filename: str,
    row_count: int,
    column_count: int,
    file_size_bytes: int,
) -> AnalysisSession:
    session = AnalysisSession(
        user_id=user_id,
        filename=filename,
        original_filename=original_filename,
        row_count=row_count,
        column_count=column_count,
        file_size_bytes=file_size_bytes,
        status="pending",
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def save_results(
    db: AsyncSession,
    session_id: int,
    kpis: dict,
    insights: list[dict],
    recommendations: list[dict],
    detected_domain: str,
    report_path: Optional[str] = None,
) -> AnalysisSession:
    session = await db.get(AnalysisSession, session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    # Update session metadata
    session.kpis = kpis
    if isinstance(detected_domain, dict):
        session.detected_domain = detected_domain.get("name", "general")
    else:
        session.detected_domain = str(detected_domain) if detected_domain else "general"
    session.report_path = report_path
    session.status = "complete"
    session.completed_at = datetime.now(timezone.utc)

    # Save insights
    for i, insight in enumerate(insights):
        record = SessionInsight(
            session_id=session_id,
            rule_type=insight.get("rule_type", "unknown"),
            title=insight.get("title", ""),
            body=insight.get("body", ""),
            impact=_normalize_impact(insight.get("impact", "minor")),
            display_order=i,
        )
        record.evidence = insight.get("chart_data") or insight.get("evidence") or {}
        db.add(record)

    # Save recommendations
    for i, rec in enumerate(recommendations):
        record = SessionRecommendation(
            session_id=session_id,
            action=rec.get("action", ""),
            owner=rec.get("owner", ""),
            timeframe=rec.get("timeframe", ""),
            impact=_normalize_impact(rec.get("impact", "minor")),
            display_order=i,
        )
        db.add(record)

    await db.commit()
    await db.refresh(session)
    return session


async def mark_failed(db: AsyncSession, session_id: int, error: str):
    session = await db.get(AnalysisSession, session_id)
    if session:
        session.status = "failed"
        session.error_message = error
        await db.commit()


async def get_user_sessions(
    db: AsyncSession,
    user_id: int,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    offset = (page - 1) * page_size

    # Total count
    count_result = await db.execute(
        select(func.count(AnalysisSession.id))
        .where(AnalysisSession.user_id == user_id)
    )
    total = count_result.scalar()

    # Paginated results — no eager loading of insights here (list view)
    result = await db.execute(
        select(AnalysisSession)
        .where(AnalysisSession.user_id == user_id)
        .order_by(desc(AnalysisSession.created_at))
        .offset(offset)
        .limit(page_size)
    )
    sessions = result.scalars().all()

    return {
        "items": sessions,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


async def get_session_detail(
    db: AsyncSession,
    session_id: int,
    user_id: int,
) -> Optional[AnalysisSession]:
    """Fetch full session with insights + recommendations. Enforces ownership."""
    result = await db.execute(
        select(AnalysisSession)
        .where(
            AnalysisSession.id == session_id,
            AnalysisSession.user_id == user_id   # ownership check
        )
        .options(
            selectinload(AnalysisSession.insights),
            selectinload(AnalysisSession.recommendations),
        )
    )
    return result.scalar_one_or_none()


async def delete_session(
    db: AsyncSession,
    session_id: int,
    user_id: int,
) -> bool:
    session = await get_session_detail(db, session_id, user_id)
    if not session:
        return False
    await db.delete(session)
    await db.commit()
    return True
