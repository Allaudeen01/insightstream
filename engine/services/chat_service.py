import json
import google.generativeai as genai
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from config import settings
from models import AnalysisSession, ChatMessage
from services.context_builder import build_system_prompt, build_message_history


def get_gemini_model(system_prompt: str):
    """Initialise Gemini model. Returns None if API key not set."""
    if not settings.GEMINI_API_KEY:
        return None
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=system_prompt,
    )


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

    if not settings.GEMINI_API_KEY:
        yield f"data: {json.dumps({'type': 'error', 'content': 'Gemini API key not configured. Add GEMINI_API_KEY to .env'})}\n\n"
        return

    print(f"[CHAT DEBUG] API key present: {bool(settings.GEMINI_API_KEY)}")
    print(f"[CHAT DEBUG] Model: {settings.GEMINI_MODEL}")
    print(f"[CHAT DEBUG] Session status: {session.status if session else 'NOT FOUND'}")

    await save_message(db, session_id, user_id, "user", user_message)

    system_prompt = build_system_prompt(session)
    history = build_message_history(
        session.chat_messages[:-1],  # exclude the message just saved
        max_messages=settings.CHAT_MAX_HISTORY_MESSAGES,
    )

    model = get_gemini_model(system_prompt)
    chat = model.start_chat(history=history)

    full_response = ""
    try:
        response = await chat.send_message_async(
            user_message,
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield f"data: {json.dumps({'type': 'text', 'content': chunk.text})}\n\n"

        if full_response:
            await save_message(db, session_id, user_id, "assistant", full_response)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        error_msg = f"Something went wrong: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        if full_response:
            await save_message(db, session_id, user_id, "assistant", full_response)
