import json
import os
import asyncio
from groq import Groq
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from config import settings
from models import AnalysisSession, ChatMessage
from services.context_builder import build_system_prompt, build_message_history

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))


async def get_session_for_chat(
    db: AsyncSession,
    session_id: int,
    user_id: int,
) -> AnalysisSession | None:
    result = await db.execute(
        select(AnalysisSession)
        .where(
            AnalysisSession.id == session_id,
            AnalysisSession.user_id == user_id,
        )
        .options(
            selectinload(AnalysisSession.insights),
            selectinload(AnalysisSession.recommendations),
            selectinload(AnalysisSession.chat_messages),
        )
    )
    return result.scalar_one_or_none()


async def save_message(
    db: AsyncSession,
    session_id: int,
    user_id: int,
    role: str,
    content: str,
) -> ChatMessage:
    msg = ChatMessage(
        session_id=session_id,
        user_id=user_id,
        role=role,
        content=content,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def stream_response(
    db: AsyncSession,
    session_id: int,
    user_id: int,
    user_message: str,
):
    """Async generator yielding SSE-formatted chunks."""
    if not user_message.strip():
        yield f"data: {json.dumps({'type': 'error', 'content': 'Empty message'})}\n\n"
        return

    if len(user_message) > 2000:
        yield f"data: {json.dumps({'type': 'error', 'content': 'Message too long (max 2000 chars)'})}\n\n"
        return

    session = await get_session_for_chat(db, session_id, user_id)
    if not session:
        yield f"data: {json.dumps({'type': 'error', 'content': 'Session not found'})}\n\n"
        return

    if session.status != "complete":
        yield f"data: {json.dumps({'type': 'error', 'content': 'Analysis not complete yet'})}\n\n"
        return

    if not os.getenv("GROQ_API_KEY", ""):
        yield f"data: {json.dumps({'type': 'error', 'content': 'GROQ_API_KEY not configured. Add it to .env'})}\n\n"
        return

    await save_message(db, session_id, user_id, "user", user_message)

    system_prompt = build_system_prompt(session)
    history = build_message_history(
        session.chat_messages[:-1],
        max_messages=settings.CHAT_MAX_HISTORY_MESSAGES,
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    full_response = ""
    try:
        def _call_groq():
            completion = _groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
            return completion.choices[0].message.content

        answer = await asyncio.to_thread(_call_groq)
        full_response = answer
        yield f"data: {json.dumps({'type': 'text', 'content': answer})}\n\n"

        await save_message(db, session_id, user_id, "assistant", full_response)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        error_msg = f"Something went wrong: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        if full_response:
            await save_message(db, session_id, user_id, "assistant", full_response)
