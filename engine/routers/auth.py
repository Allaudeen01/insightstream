from fastapi import APIRouter, Depends, HTTPException, Response, Cookie
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta, timezone
import hashlib

from db_async import get_db
from models import User, RefreshToken
from auth import (
    hash_password, verify_password, validate_password_strength,
    create_access_token, create_refresh_token, get_current_user
)
from schemas import UserRegister, UserLogin, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserOut, status_code=201)
async def register(payload: UserRegister, db: AsyncSession = Depends(get_db)):
    # Validate password strength
    validate_password_strength(payload.password)

    # Check email not already taken
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Email already registered")

    user = User(
        email=payload.email.lower().strip(),
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/login")
async def login(
    payload: UserLogin,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    # Look up user
    result = await db.execute(select(User).where(User.email == payload.email.lower()))
    user = result.scalar_one_or_none()

    # Same error message whether email or password is wrong (prevents enumeration)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(401, "Invalid email or password")

    if not user.is_active:
        raise HTTPException(403, "Account disabled")

    # Issue tokens
    access_token = create_access_token(user.id)
    raw_refresh, hashed_refresh = create_refresh_token()

    # Store refresh token in DB
    refresh_token_record = RefreshToken(
        user_id=user.id,
        token_hash=hashed_refresh,
        expires_at=datetime.now(timezone.utc) + timedelta(days=30)
    )
    db.add(refresh_token_record)
    await db.commit()

    # Send refresh token as httpOnly cookie (not in response body)
    response.set_cookie(
        key="refresh_token",
        value=raw_refresh,
        httponly=True,       # JS cannot read this
        secure=True,         # HTTPS only
        samesite="lax",
        max_age=30 * 24 * 3600
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/refresh")
async def refresh_access_token(
    refresh_token: str = Cookie(None),
    db: AsyncSession = Depends(get_db)
):
    if not refresh_token:
        raise HTTPException(401, "No refresh token")

    hashed = hashlib.sha256(refresh_token.encode()).hexdigest()

    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == hashed)
    )
    record = result.scalar_one_or_none()

    if not record or record.expires_at < datetime.utcnow():
        raise HTTPException(401, "Invalid or expired refresh token")

    new_access_token = create_access_token(record.user_id)
    return {"access_token": new_access_token, "token_type": "bearer"}


@router.post("/logout")
async def logout(
    response: Response,
    refresh_token: str = Cookie(None),
    db: AsyncSession = Depends(get_db)
):
    if refresh_token:
        hashed = hashlib.sha256(refresh_token.encode()).hexdigest()
        result = await db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == hashed)
        )
        record = result.scalar_one_or_none()
        if record:
            await db.delete(record)
            await db.commit()

    response.delete_cookie("refresh_token")
    return {"message": "Logged out"}


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    return current_user
