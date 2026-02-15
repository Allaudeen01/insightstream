"""
SQLite persistence layer for InsightStream projects.
Replaces ephemeral temp-dir sessions with permanent project storage.
"""
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Database lives alongside the engine
DB_DIR = Path(os.environ.get("INSIGHTSTREAM_DATA_DIR", Path(__file__).parent / "data"))
DB_PATH = DB_DIR / "insightstream.db"


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with row factory."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Initialize database tables. Safe to call multiple times."""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                row_count INTEGER DEFAULT 0,
                column_count INTEGER DEFAULT 0,
                data_path TEXT NOT NULL,
                executive_summary TEXT,
                recommendations TEXT
            );

            CREATE TABLE IF NOT EXISTS project_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                action TEXT NOT NULL,
                params TEXT,
                result_summary TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dashboards (
                project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
                layout TEXT NOT NULL,
                pinned_chart_ids TEXT NOT NULL,
                text_blocks TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_history_project
                ON project_history(project_id, created_at DESC);
        """)
        conn.commit()
        print(f"Database initialized at {DB_PATH}")
    finally:
        conn.close()


# ─── Project CRUD ───────────────────────────────────────────────

def create_project(
    project_id: str,
    name: str,
    filename: str,
    data_path: str,
    row_count: int = 0,
    column_count: int = 0,
) -> dict:
    """Create a new project record."""
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO projects (id, name, filename, created_at, updated_at,
               row_count, column_count, data_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, name, filename, now, now, row_count, column_count, data_path),
        )
        conn.commit()
        return {
            "id": project_id,
            "name": name,
            "filename": filename,
            "created_at": now,
            "updated_at": now,
            "row_count": row_count,
            "column_count": column_count,
        }
    finally:
        conn.close()


def list_projects() -> list[dict]:
    """List all projects, most recent first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, name, filename, created_at, updated_at,
                      row_count, column_count
               FROM projects ORDER BY updated_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_project(project_id: str) -> Optional[dict]:
    """Get a single project by ID, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_project(project_id: str, **kwargs) -> bool:
    """Update project fields. Pass only the fields to change."""
    allowed = {"name", "executive_summary", "recommendations", "row_count", "column_count"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False

    updates["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [project_id]

    conn = get_connection()
    try:
        cursor = conn.execute(
            f"UPDATE projects SET {set_clause} WHERE id = ?", values
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_project(project_id: str) -> bool:
    """Delete a project and its history. Returns True if deleted."""
    conn = get_connection()
    try:
        # Also clean up the parquet file
        row = conn.execute(
            "SELECT data_path FROM projects WHERE id = ?", (project_id,)
        ).fetchone()

        cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()

        # Remove data files
        if row and cursor.rowcount > 0:
            data_path = Path(row["data_path"])
            if data_path.exists():
                import shutil
                # data_path is the session dir containing data.parquet + metadata.json
                if data_path.is_dir():
                    shutil.rmtree(data_path, ignore_errors=True)
                else:
                    data_path.unlink(missing_ok=True)

        return cursor.rowcount > 0
    finally:
        conn.close()


# ─── History ────────────────────────────────────────────────────

def add_history(
    project_id: str,
    action: str,
    params: Optional[dict] = None,
    result_summary: Optional[str] = None,
) -> int:
    """Record a history event for a project."""
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO project_history (project_id, action, params, result_summary, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (project_id, action, json.dumps(params) if params else None, result_summary, now),
        )
        # Also touch the project's updated_at
        conn.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_history(project_id: str, limit: int = 50) -> list[dict]:
    """Get history events for a project, most recent first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, action, params, result_summary, created_at
               FROM project_history
               WHERE project_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (project_id, limit),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("params"):
                try:
                    d["params"] = json.loads(d["params"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(d)
        return result
    finally:
        conn.close()


# ─── Dashboard ──────────────────────────────────────────────────

def save_dashboard(
    project_id: str,
    layout: list,
    pinned_chart_ids: list[str],
    text_blocks: Optional[list[dict]] = None,
) -> bool:
    """Save or update dashboard layout for a project."""
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO dashboards (project_id, layout, pinned_chart_ids, text_blocks, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(project_id) DO UPDATE SET
               layout = excluded.layout,
               pinned_chart_ids = excluded.pinned_chart_ids,
               text_blocks = excluded.text_blocks,
               updated_at = excluded.updated_at""",
            (
                project_id,
                json.dumps(layout),
                json.dumps(pinned_chart_ids),
                json.dumps(text_blocks) if text_blocks else None,
                now,
            ),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def load_dashboard(project_id: str) -> Optional[dict]:
    """Load dashboard layout for a project, or None if not saved."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM dashboards WHERE project_id = ?", (project_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["layout"] = json.loads(d["layout"])
        d["pinned_chart_ids"] = json.loads(d["pinned_chart_ids"])
        if d.get("text_blocks"):
            d["text_blocks"] = json.loads(d["text_blocks"])
        else:
            d["text_blocks"] = []
        return d
    finally:
        conn.close()
