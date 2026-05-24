"""
Pytest configuration for integration tests.
Adds engine/ to sys.path. Does NOT stub heavy modules — integration tests
call analyze_dataset() directly with a real DataFrame (no Groq API needed
because analyze_dataset falls back to _safe_fallback when GROQ_API_KEY is absent).
"""
import sys
import os

ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "engine")
sys.path.insert(0, ENGINE_DIR)

# Stub only the modules that cause import-time side effects
# (database, FastAPI app startup) — NOT groq or insight_engine
from unittest.mock import MagicMock
for _mod in ["database", "session_cache", "google", "google.genai"]:
    sys.modules.setdefault(_mod, MagicMock())
