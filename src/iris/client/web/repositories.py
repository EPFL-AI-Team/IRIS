"""Repository classes for database operations."""

import time
from typing import Any

from iris.client.web.database import (
    deserialize_json,
    get_db,
    row_to_dict,
    rows_to_list,
    serialize_json,
)


class SessionRepository:
    """CRUD operations for analysis sessions."""

    def create(
        self,
        session_id: str,
        config: dict[str, Any],
        mode: str = "live",
        video_file: str | None = None,
        annotation_file: str | None = None,
    ) -> dict[str, Any]:
        """Create a new analysis session.

        Args:
            session_id: Unique session identifier.
            config: Session configuration dict.
            mode: Session mode ('live' or 'analysis').
            video_file: Optional video filename.
            annotation_file: Optional JSONL annotation filename.

        Returns:
            Created session as dict.
        """
        now = time.time()
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, status, mode, created_at, config, video_file, annotation_file)
                VALUES (?, 'idle', ?, ?, ?, ?, ?)
                """,
                (session_id, mode, now, serialize_json(config), video_file, annotation_file),
            )
        return self.get(session_id)

    def get(self, session_id: str) -> dict[str, Any] | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session dict or None if not found.
        """
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row:
                result = row_to_dict(row)
                result["config"] = deserialize_json(result.get("config"))
                return result
            return None

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent sessions.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session dicts ordered by created_at desc.
        """
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            sessions = rows_to_list(rows)
            for s in sessions:
                s["config"] = deserialize_json(s.get("config"))
            return sessions

    def update_status(
        self,
        session_id: str,
        status: str,
        started_at: float | None = None,
        completed_at: float | None = None,
    ) -> None:
        """Update session status.

        Args:
            session_id: Session identifier.
            status: New status (idle, running, paused, completed, error).
            started_at: Optional start timestamp.
            completed_at: Optional completion timestamp.
        """
        with get_db() as conn:
            if started_at is not None and completed_at is not None:
                conn.execute(
                    "UPDATE sessions SET status = ?, started_at = ?, completed_at = ? WHERE id = ?",
                    (status, started_at, completed_at, session_id),
                )
            elif started_at is not None:
                conn.execute(
                    "UPDATE sessions SET status = ?, started_at = ? WHERE id = ?",
                    (status, started_at, session_id),
                )
            elif completed_at is not None:
                conn.execute(
                    "UPDATE sessions SET status = ?, completed_at = ? WHERE id = ?",
                    (status, completed_at, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET status = ? WHERE id = ?", (status, session_id)
                )

    def update_config(self, session_id: str, config: dict[str, Any]) -> None:
        """Update session config.

        Args:
            session_id: Session identifier.
            config: Configuration dictionary to store.
        """
        with get_db() as conn:
            conn.execute(
                "UPDATE sessions SET config = ? WHERE id = ?",
                (serialize_json(config), session_id),
            )

    def delete(self, session_id: str) -> bool:
        """Delete a session and all related data.

        Args:
            session_id: Session identifier.

        Returns:
            True if deleted, False if not found.
        """
        with get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            return cursor.rowcount > 0


class ResultsRepository:
    """CRUD operations for inference results."""

    def store(
        self,
        session_id: str,
        job_id: str,
        video_time_ms: int,
        inference_start_ms: float,
        inference_end_ms: float,
        frame_start: int,
        frame_end: int,
        result: dict[str, Any],
    ) -> int:
        """Store an inference result.

        Args:
            session_id: Parent session ID.
            job_id: Job identifier.
            video_time_ms: Position in video when frames were captured.
            inference_start_ms: When inference started.
            inference_end_ms: When inference completed.
            frame_start: Start frame index.
            frame_end: End frame index.
            result: Inference result dict.

        Returns:
            ID of inserted row.
        """
        inference_duration_ms = inference_end_ms - inference_start_ms
        now = time.time()

        with get_db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO inference_results
                (session_id, job_id, video_time_ms, inference_start_ms, inference_end_ms,
                 inference_duration_ms, frame_start, frame_end, result, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    job_id,
                    video_time_ms,
                    inference_start_ms,
                    inference_end_ms,
                    inference_duration_ms,
                    frame_start,
                    frame_end,
                    serialize_json(result),
                    now,
                ),
            )
            return cursor.lastrowid

    def get_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """Get all results for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of result dicts ordered by video_time_ms.
        """
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT * FROM inference_results
                WHERE session_id = ?
                ORDER BY video_time_ms ASC
                """,
                (session_id,),
            ).fetchall()
            results = rows_to_list(rows)
            for r in results:
                r["result"] = deserialize_json(r.get("result"))
            return results

    def get_for_playback(
        self, session_id: str, start_ms: int, end_ms: int
    ) -> list[dict[str, Any]]:
        """Get results within a time range for playback.

        Args:
            session_id: Session identifier.
            start_ms: Start time in milliseconds.
            end_ms: End time in milliseconds.

        Returns:
            List of results where video_time_ms is within range.
        """
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT * FROM inference_results
                WHERE session_id = ? AND video_time_ms >= ? AND video_time_ms <= ?
                ORDER BY video_time_ms ASC
                """,
                (session_id, start_ms, end_ms),
            ).fetchall()
            results = rows_to_list(rows)
            for r in results:
                r["result"] = deserialize_json(r.get("result"))
            return results

    def count_by_session(self, session_id: str) -> int:
        """Count results for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Number of results.
        """
        with get_db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM inference_results WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["count"] if row else 0

    def clear_session(self, session_id: str) -> int:
        """Delete all results for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Number of deleted rows.
        """
        with get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM inference_results WHERE session_id = ?", (session_id,)
            )
            return cursor.rowcount


class LogsRepository:
    """CRUD operations for session logs."""

    def append(self, session_id: str, level: str, message: str) -> int:
        """Append a log entry.

        Args:
            session_id: Parent session ID.
            level: Log level (INFO, WARNING, ERROR, etc.).
            message: Log message.

        Returns:
            ID of inserted row.
        """
        now = time.time()
        with get_db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO logs (session_id, level, message, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, level, message, now),
            )
            return cursor.lastrowid

    def get_by_session(
        self, session_id: str, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Get logs for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of logs to return.

        Returns:
            List of log dicts ordered by timestamp desc.
        """
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT * FROM logs
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return rows_to_list(rows)

    def count_by_session(self, session_id: str) -> int:
        """Count logs for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Number of logs.
        """
        with get_db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM logs WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["count"] if row else 0

    def clear_session(self, session_id: str) -> int:
        """Delete all logs for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Number of deleted rows.
        """
        with get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM logs WHERE session_id = ?", (session_id,)
            )
            return cursor.rowcount


class ReportsRepository:
    """CRUD operations for generated reports."""

    def store(
        self,
        session_id: str,
        provider: str,
        content: str,
        generation_duration_sec: float | None = None,
    ) -> int:
        """Store a generated report.

        Args:
            session_id: Parent session ID.
            provider: LLM provider used (e.g., 'gemini').
            content: Full markdown report content.
            generation_duration_sec: Time taken to generate.

        Returns:
            ID of inserted row.
        """
        now = time.time()
        with get_db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reports (session_id, provider, content, created_at, generation_duration_sec)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, provider, content, now, generation_duration_sec),
            )
            return cursor.lastrowid

    def get_latest_by_session(self, session_id: str) -> dict[str, Any] | None:
        """Get the most recent report for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Report dict or None if not found.
        """
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT * FROM reports
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            return row_to_dict(row)

    def get_all_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """Get all reports for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of report dicts ordered by created_at desc.
        """
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT * FROM reports
                WHERE session_id = ?
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
            return rows_to_list(rows)

    def delete_by_session(self, session_id: str) -> int:
        """Delete all reports for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Number of deleted rows.
        """
        with get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM reports WHERE session_id = ?", (session_id,)
            )
            return cursor.rowcount


# Singleton instances
session_repo = SessionRepository()
results_repo = ResultsRepository()
logs_repo = LogsRepository()
reports_repo = ReportsRepository()
