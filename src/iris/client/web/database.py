"""SQLite database for persistent analysis session storage."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

DB_PATH = Path(__file__).parent / "static" / "analysis.db"


def init_db() -> None:
    """Initialize SQLite database with schema."""
    # Ensure parent directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.executescript("""
            -- Analysis sessions table
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'idle',
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                config TEXT NOT NULL DEFAULT '{}',
                video_file TEXT,
                annotation_file TEXT
            );

            -- Inference results table
            CREATE TABLE IF NOT EXISTS inference_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                video_time_ms INTEGER NOT NULL,
                inference_start_ms REAL NOT NULL,
                inference_end_ms REAL NOT NULL,
                inference_duration_ms REAL NOT NULL,
                frame_start INTEGER NOT NULL,
                frame_end INTEGER NOT NULL,
                result TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            -- Session logs table
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            -- Indexes for efficient querying
            CREATE INDEX IF NOT EXISTS idx_results_session
                ON inference_results(session_id);
            CREATE INDEX IF NOT EXISTS idx_results_video_time
                ON inference_results(video_time_ms);
            CREATE INDEX IF NOT EXISTS idx_logs_session
                ON logs(session_id);
            CREATE INDEX IF NOT EXISTS idx_logs_timestamp
                ON logs(timestamp);
        """)


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Get database connection with automatic cleanup.

    Yields:
        sqlite3.Connection with Row factory enabled.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert sqlite3.Row to dictionary."""
    if row is None:
        return None
    return dict(row)


def rows_to_list(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Convert list of sqlite3.Row to list of dictionaries."""
    return [dict(row) for row in rows]


# Convenience function to serialize/deserialize JSON fields
def serialize_json(data: dict | list | None) -> str:
    """Serialize data to JSON string for storage."""
    return json.dumps(data) if data else "{}"


def deserialize_json(data: str | None) -> dict | list:
    """Deserialize JSON string from storage."""
    if not data:
        return {}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}
