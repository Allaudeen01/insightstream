"""
Pytest configuration for InsightStream engine tests.

main.py is a FastAPI application — importing it triggers app startup, DB init,
and heavy engine imports. This conftest stubs all those side-effect modules so
the pure helper functions under test can be imported cleanly.
"""
import sys
import os
from unittest.mock import MagicMock

# ── Path setup ───────────────────────────────────────────────────────────────
ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "engine")
sys.path.insert(0, ENGINE_DIR)

# ── Stub heavy / side-effect modules before main.py is imported ───────────────
_STUBS = [
    "database",
    "session_cache",
    "insight_engine",
    "report_generator",
    "google",
    "google.genai",
]
for _mod in _STUBS:
    sys.modules.setdefault(_mod, MagicMock())

# insight_engine exports several names that main.py unpacks in its import statement
_ie = sys.modules["insight_engine"]
for _name in (
    "ColumnClassifier", "MetricComputer", "BusinessRuleEngine",
    "InsightNarrator", "SmartChartRecommender", "AnomalyDetector",
    "run_insight_engine", "RecommendationEngine", "StrategicBriefBuilder",
    "validate_dataframe", "auto_clean_dataframe",
):
    setattr(_ie, _name, MagicMock())

_rg = sys.modules["report_generator"]
_rg.__file__ = "<mocked report_generator>"  # main.py prints this at import time
for _name in ("ChartGenerator", "PDFReportGenerator", "ColumnMap",
              "cleanup_temp_files", "UnifiedReportGenerator"):
    setattr(_rg, _name, MagicMock())

_sc = sys.modules["session_cache"]
setattr(_sc, "SessionCache", MagicMock())
setattr(_sc, "JobTracker", MagicMock())
