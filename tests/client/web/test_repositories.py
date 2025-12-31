"""Tests for repository classes."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSessionRepository:
    """Tests for SessionRepository."""

    def test_create_session(
        self, initialized_db: Path, sample_session_data: dict
    ) -> None:
        """Test creating a new session."""
        from iris.client.web.repositories import SessionRepository

        repo = SessionRepository()
        session = repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
            video_file=sample_session_data["video_file"],
            annotation_file=sample_session_data["annotation_file"],
        )

        assert session is not None
        assert session["id"] == sample_session_data["id"]
        assert session["status"] == "idle"
        assert session["config"] == sample_session_data["config"]
        assert session["video_file"] == sample_session_data["video_file"]

    def test_get_session(
        self, initialized_db: Path, sample_session_data: dict
    ) -> None:
        """Test getting a session by ID."""
        from iris.client.web.repositories import SessionRepository

        repo = SessionRepository()
        repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        session = repo.get(sample_session_data["id"])
        assert session is not None
        assert session["id"] == sample_session_data["id"]

    def test_get_nonexistent_session(self, initialized_db: Path) -> None:
        """Test getting a nonexistent session returns None."""
        from iris.client.web.repositories import SessionRepository

        repo = SessionRepository()
        session = repo.get("nonexistent-id")
        assert session is None

    def test_list_sessions(self, initialized_db: Path) -> None:
        """Test listing sessions."""
        from iris.client.web.repositories import SessionRepository

        repo = SessionRepository()
        # Create multiple sessions
        for i in range(3):
            repo.create(session_id=f"session-{i}", config={"index": i})

        sessions = repo.list_sessions()
        assert len(sessions) == 3
        # Should be ordered by created_at DESC
        assert sessions[0]["id"] == "session-2"

    def test_list_sessions_with_limit(self, initialized_db: Path) -> None:
        """Test listing sessions with limit."""
        from iris.client.web.repositories import SessionRepository

        repo = SessionRepository()
        for i in range(5):
            repo.create(session_id=f"session-{i}", config={})

        sessions = repo.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_update_status(
        self, initialized_db: Path, sample_session_data: dict
    ) -> None:
        """Test updating session status."""
        from iris.client.web.repositories import SessionRepository

        repo = SessionRepository()
        repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo.update_status(sample_session_data["id"], "running", started_at=100.0)
        session = repo.get(sample_session_data["id"])

        assert session["status"] == "running"
        assert session["started_at"] == 100.0

    def test_update_status_completed(
        self, initialized_db: Path, sample_session_data: dict
    ) -> None:
        """Test updating status to completed with timestamps."""
        from iris.client.web.repositories import SessionRepository

        repo = SessionRepository()
        repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo.update_status(
            sample_session_data["id"],
            "completed",
            started_at=100.0,
            completed_at=200.0,
        )
        session = repo.get(sample_session_data["id"])

        assert session["status"] == "completed"
        assert session["started_at"] == 100.0
        assert session["completed_at"] == 200.0

    def test_delete_session(
        self, initialized_db: Path, sample_session_data: dict
    ) -> None:
        """Test deleting a session."""
        from iris.client.web.repositories import SessionRepository

        repo = SessionRepository()
        repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        result = repo.delete(sample_session_data["id"])
        assert result is True

        session = repo.get(sample_session_data["id"])
        assert session is None

    def test_delete_nonexistent_session(self, initialized_db: Path) -> None:
        """Test deleting nonexistent session returns False."""
        from iris.client.web.repositories import SessionRepository

        repo = SessionRepository()
        result = repo.delete("nonexistent")
        assert result is False


class TestResultsRepository:
    """Tests for ResultsRepository."""

    def test_store_result(
        self,
        initialized_db: Path,
        sample_session_data: dict,
        sample_result_data: dict,
    ) -> None:
        """Test storing an inference result."""
        from iris.client.web.repositories import ResultsRepository, SessionRepository

        # Create session first (foreign key)
        session_repo = SessionRepository()
        session_repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo = ResultsRepository()
        row_id = repo.store(
            session_id=sample_result_data["session_id"],
            job_id=sample_result_data["job_id"],
            video_time_ms=sample_result_data["video_time_ms"],
            inference_start_ms=sample_result_data["inference_start_ms"],
            inference_end_ms=sample_result_data["inference_end_ms"],
            frame_start=sample_result_data["frame_start"],
            frame_end=sample_result_data["frame_end"],
            result=sample_result_data["result"],
        )

        assert row_id is not None
        assert row_id > 0

    def test_get_by_session(
        self,
        initialized_db: Path,
        sample_session_data: dict,
        sample_result_data: dict,
    ) -> None:
        """Test getting results by session."""
        from iris.client.web.repositories import ResultsRepository, SessionRepository

        session_repo = SessionRepository()
        session_repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo = ResultsRepository()
        # Store multiple results
        for i in range(3):
            repo.store(
                session_id=sample_session_data["id"],
                job_id=f"job-{i}",
                video_time_ms=i * 1000,
                inference_start_ms=float(i * 1000),
                inference_end_ms=float(i * 1000 + 150),
                frame_start=i * 5,
                frame_end=i * 5 + 8,
                result={"index": i},
            )

        results = repo.get_by_session(sample_session_data["id"])
        assert len(results) == 3
        # Should be ordered by video_time_ms ASC
        assert results[0]["video_time_ms"] == 0
        assert results[2]["video_time_ms"] == 2000

    def test_get_for_playback(
        self,
        initialized_db: Path,
        sample_session_data: dict,
    ) -> None:
        """Test getting results within time range for playback."""
        from iris.client.web.repositories import ResultsRepository, SessionRepository

        session_repo = SessionRepository()
        session_repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo = ResultsRepository()
        # Store results at different times
        for i in range(5):
            repo.store(
                session_id=sample_session_data["id"],
                job_id=f"job-{i}",
                video_time_ms=i * 1000,  # 0, 1000, 2000, 3000, 4000
                inference_start_ms=float(i * 1000),
                inference_end_ms=float(i * 1000 + 150),
                frame_start=i * 5,
                frame_end=i * 5 + 8,
                result={},
            )

        # Get results between 1000 and 3000 ms
        results = repo.get_for_playback(sample_session_data["id"], 1000, 3000)
        assert len(results) == 3
        assert results[0]["video_time_ms"] == 1000
        assert results[2]["video_time_ms"] == 3000

    def test_count_by_session(
        self,
        initialized_db: Path,
        sample_session_data: dict,
    ) -> None:
        """Test counting results by session."""
        from iris.client.web.repositories import ResultsRepository, SessionRepository

        session_repo = SessionRepository()
        session_repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo = ResultsRepository()
        for i in range(7):
            repo.store(
                session_id=sample_session_data["id"],
                job_id=f"job-{i}",
                video_time_ms=i * 100,
                inference_start_ms=0.0,
                inference_end_ms=100.0,
                frame_start=0,
                frame_end=5,
                result={},
            )

        count = repo.count_by_session(sample_session_data["id"])
        assert count == 7

    def test_clear_session(
        self,
        initialized_db: Path,
        sample_session_data: dict,
    ) -> None:
        """Test clearing all results for a session."""
        from iris.client.web.repositories import ResultsRepository, SessionRepository

        session_repo = SessionRepository()
        session_repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo = ResultsRepository()
        for i in range(5):
            repo.store(
                session_id=sample_session_data["id"],
                job_id=f"job-{i}",
                video_time_ms=i * 100,
                inference_start_ms=0.0,
                inference_end_ms=100.0,
                frame_start=0,
                frame_end=5,
                result={},
            )

        deleted = repo.clear_session(sample_session_data["id"])
        assert deleted == 5

        count = repo.count_by_session(sample_session_data["id"])
        assert count == 0


class TestLogsRepository:
    """Tests for LogsRepository."""

    def test_append_log(
        self,
        initialized_db: Path,
        sample_session_data: dict,
    ) -> None:
        """Test appending a log entry."""
        from iris.client.web.repositories import LogsRepository, SessionRepository

        session_repo = SessionRepository()
        session_repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo = LogsRepository()
        row_id = repo.append(
            session_id=sample_session_data["id"],
            level="INFO",
            message="Test log message",
        )

        assert row_id is not None
        assert row_id > 0

    def test_get_by_session(
        self,
        initialized_db: Path,
        sample_session_data: dict,
    ) -> None:
        """Test getting logs by session."""
        from iris.client.web.repositories import LogsRepository, SessionRepository

        session_repo = SessionRepository()
        session_repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo = LogsRepository()
        repo.append(sample_session_data["id"], "INFO", "Message 1")
        repo.append(sample_session_data["id"], "WARNING", "Message 2")
        repo.append(sample_session_data["id"], "ERROR", "Message 3")

        logs = repo.get_by_session(sample_session_data["id"])
        assert len(logs) == 3
        # Should be ordered by timestamp DESC (most recent first)
        assert logs[0]["message"] == "Message 3"

    def test_get_by_session_with_limit(
        self,
        initialized_db: Path,
        sample_session_data: dict,
    ) -> None:
        """Test getting logs with limit."""
        from iris.client.web.repositories import LogsRepository, SessionRepository

        session_repo = SessionRepository()
        session_repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo = LogsRepository()
        for i in range(10):
            repo.append(sample_session_data["id"], "INFO", f"Message {i}")

        logs = repo.get_by_session(sample_session_data["id"], limit=5)
        assert len(logs) == 5

    def test_count_by_session(
        self,
        initialized_db: Path,
        sample_session_data: dict,
    ) -> None:
        """Test counting logs by session."""
        from iris.client.web.repositories import LogsRepository, SessionRepository

        session_repo = SessionRepository()
        session_repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo = LogsRepository()
        for i in range(8):
            repo.append(sample_session_data["id"], "INFO", f"Message {i}")

        count = repo.count_by_session(sample_session_data["id"])
        assert count == 8

    def test_clear_session(
        self,
        initialized_db: Path,
        sample_session_data: dict,
    ) -> None:
        """Test clearing logs for a session."""
        from iris.client.web.repositories import LogsRepository, SessionRepository

        session_repo = SessionRepository()
        session_repo.create(
            session_id=sample_session_data["id"],
            config=sample_session_data["config"],
        )

        repo = LogsRepository()
        for i in range(6):
            repo.append(sample_session_data["id"], "INFO", f"Message {i}")

        deleted = repo.clear_session(sample_session_data["id"])
        assert deleted == 6

        count = repo.count_by_session(sample_session_data["id"])
        assert count == 0
