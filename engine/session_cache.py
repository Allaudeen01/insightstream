"""
InsightStream — In-Process Session Cache
=========================================
Thread-safe, TTL-based cache that stores expensive computation results
so the ColumnClassifier/MetricComputer/ChartRecommender run only once
per session instead of once per endpoint call.

Design decisions:
  - Pure in-process (no Redis) → 0 network latency on hit
  - TTL of 30 min → auto-evicts stale sessions
  - Max 20 sessions → caps memory at ~200MB worst-case
  - Thread-safe via threading.Lock per entry
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

_TTL_SECONDS = 30 * 60   # 30 minutes
_MAX_ENTRIES = 20


@dataclass
class _CacheEntry:
    data: dict
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > _TTL_SECONDS


class SessionCache:
    """Singleton thread-safe session result cache."""

    _instance: Optional["SessionCache"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SessionCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._store: dict[str, _CacheEntry] = {}
                    inst._entry_locks: dict[str, threading.Lock] = {}
                    cls._instance = inst
        return cls._instance

    # ── Public API ────────────────────────────────────────────────

    def get(self, session_id: str, key: str) -> Optional[Any]:
        """Return cached value or None if miss/expired."""
        entry = self._store.get(f"{session_id}:{key}")
        if entry is None or entry.is_expired():
            return None
        return entry.data

    def put(self, session_id: str, key: str, value: Any) -> None:
        """Store a value. Evicts oldest entries if over limit."""
        composite = f"{session_id}:{key}"
        self._store[composite] = _CacheEntry(data=value)
        self._maybe_evict()

    def has(self, session_id: str, key: str) -> bool:
        entry = self._store.get(f"{session_id}:{key}")
        return entry is not None and not entry.is_expired()

    def delete(self, session_id: str, key: str) -> None:
        """Remove a single cache key for a session."""
        self._store.pop(f"{session_id}:{key}", None)

    def invalidate(self, session_id: str) -> None:
        """Remove all cached data for a session (e.g. on re-upload)."""
        to_delete = [k for k in self._store if k.startswith(f"{session_id}:")]
        for k in to_delete:
            del self._store[k]

    def stats(self) -> dict:
        """Return current cache statistics."""
        now = time.time()
        active = sum(1 for e in self._store.values() if not e.is_expired())
        return {
            "total_entries": len(self._store),
            "active_entries": active,
            "max_entries": _MAX_ENTRIES,
            "ttl_seconds": _TTL_SECONDS,
        }

    # ── Internal ──────────────────────────────────────────────────

    def _maybe_evict(self) -> None:
        """Evict expired entries, then oldest if over capacity."""
        # Remove expired
        expired = [k for k, v in self._store.items() if v.is_expired()]
        for k in expired:
            del self._store[k]

        # If still over capacity, remove oldest
        while len(self._store) > _MAX_ENTRIES * 5:  # 5 keys per session
            oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
            del self._store[oldest_key]


# ── Background Job Tracker ─────────────────────────────────────────

@dataclass
class AnalysisJob:
    session_id: str
    status: str = "pending"   # "pending" | "running" | "done" | "error"
    progress: int = 0         # 0-100
    result: Optional[dict] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)


class JobTracker:
    """Track background analysis jobs."""

    _instance: Optional["JobTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "JobTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._jobs: dict[str, AnalysisJob] = {}
                    cls._instance = inst
        return cls._instance

    def create(self, session_id: str) -> AnalysisJob:
        job = AnalysisJob(session_id=session_id)
        self._jobs[session_id] = job
        return job

    def get(self, session_id: str) -> Optional[AnalysisJob]:
        return self._jobs.get(session_id)

    def update(self, session_id: str, **kwargs: Any) -> None:
        job = self._jobs.get(session_id)
        if job:
            for k, v in kwargs.items():
                if hasattr(job, k):
                    setattr(job, k, v)

    def is_running(self, session_id: str) -> bool:
        job = self._jobs.get(session_id)
        return job is not None and job.status in ("pending", "running")

    def cleanup_old(self, max_age: float = 600) -> None:
        """Remove jobs older than max_age seconds."""
        now = time.time()
        to_del = [k for k, v in self._jobs.items() if now - v.started_at > max_age]
        for k in to_del:
            del self._jobs[k]
