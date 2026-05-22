import json
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float,
    ForeignKey, DateTime, Enum as SAEnum
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    sessions = relationship("AnalysisSession", back_populates="user", cascade="all, delete-orphan")

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # File metadata
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    row_count = Column(Integer)
    column_count = Column(Integer)
    file_size_bytes = Column(Integer)

    # Domain detection
    detected_domain = Column(String(100))  # e.g. "ecommerce", "saas", "logistics"

    # Currency override (user-selected or auto-detected)
    currency = Column(String(10), nullable=True, default=None)  # "INR", "USD", "GBP", etc.

    # KPIs stored as JSON string (SQLite-safe)
    kpis_json = Column(Text)

    # LLM analysis results for unknown-domain datasets (HR, salary, etc.)
    # Stored as JSON text — contains insights, recommendations, chart_specs, domain, title.
    # Plotly figures are NOT stored here; they are re-rendered at PDF export time.
    llm_results_json = Column(Text, nullable=True)

    # Status tracking
    status = Column(
        SAEnum("pending", "processing", "complete", "failed", name="session_status"),
        default="pending"
    )
    error_message = Column(Text)

    # Report
    report_path = Column(String(500))  # local path or S3 key later

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="sessions")
    insights = relationship("SessionInsight", back_populates="session", cascade="all, delete-orphan")
    recommendations = relationship("SessionRecommendation", back_populates="session", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

    # Helper properties
    @property
    def kpis(self):
        return json.loads(self.kpis_json) if self.kpis_json else {}

    @kpis.setter
    def kpis(self, value):
        self.kpis_json = json.dumps(value)

    @property
    def llm_results(self):
        return json.loads(self.llm_results_json) if self.llm_results_json else None

    @llm_results.setter
    def llm_results(self, value):
        self.llm_results_json = json.dumps(value, default=str) if value is not None else None


class SessionInsight(Base):
    __tablename__ = "session_insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    rule_type = Column(String(100))       # e.g. "temporal_peaks", "cross_dim"
    title = Column(String(255))
    body = Column(Text)
    impact = Column(String(20))           # "minor", "important", "critical"
    evidence_json = Column(Text)          # raw numbers behind the insight
    display_order = Column(Integer, default=0)

    session = relationship("AnalysisSession", back_populates="insights")

    @property
    def evidence(self):
        return json.loads(self.evidence_json) if self.evidence_json else {}

    @evidence.setter
    def evidence(self, value):
        self.evidence_json = json.dumps(value)


class SessionRecommendation(Base):
    __tablename__ = "session_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    action = Column(Text)
    owner = Column(String(100))
    timeframe = Column(String(100))
    impact = Column(String(20))
    display_order = Column(Integer, default=0)

    session = relationship("AnalysisSession", back_populates="recommendations")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    role = Column(SAEnum("user", "assistant", name="message_role"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("AnalysisSession", back_populates="chat_messages")
