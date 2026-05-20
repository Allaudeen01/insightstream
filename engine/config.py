import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/insightstream.db"
    JWT_SECRET_KEY: str = "94c2e557b7dbd7d91e84a2750e4952be7b415e612ed428ef55fc18efd6f1a8c0"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    ENVIRONMENT: str = "development"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GROQ_API_KEY: str = ""
    CHAT_MAX_HISTORY_MESSAGES: int = 20

    class Config:
        env_file = ".env"

settings = Settings()
