from pydantic import BaseModel, EmailStr
from typing import Optional, Any
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserRegister(UserBase):
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserOut(UserBase):
    id: int
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class InsightOut(BaseModel):
    id: int
    rule_type: str
    title: str
    body: str
    impact: str
    display_order: int
    evidence_json: Optional[str] = None

    class Config:
        from_attributes = True

class RecommendationOut(BaseModel):
    id: int
    action: str
    owner: str
    timeframe: str
    impact: str
    display_order: int

    class Config:
        from_attributes = True

class SessionSummary(BaseModel):
    id: int
    original_filename: str
    detected_domain: Optional[str]
    row_count: Optional[int]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    report_path: Optional[str]

    class Config:
        from_attributes = True

class SessionListResponse(BaseModel):
    items: list[SessionSummary]
    total: int
    page: int
    page_size: int
    total_pages: int

class SessionDetailResponse(SessionSummary):
    kpis_json: Optional[str]
    insights: list[InsightOut]
    recommendations: list[RecommendationOut]
