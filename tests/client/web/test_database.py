"""Tests for database module."""

import sqlite3
from pathlib import Path

import pytest


class TestInitDb:
    """Tests for database initialization."""

    def test_init_db_creates_tables(self, initialized_db: Path) -> None:
        """Test that init_db creates all required tables."""
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "sessions" in tables
        assert "inference_results" in tables
        assert "logs" in tables

    def test_init_db_creates_indexes(self, initialized_db: Path) -> None:
        """Test that init_db creates required indexes."""
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "idx_results_session" in indexes
        assert "idx_results_video_time" in indexes
        assert "idx_logs_session" in indexes
        assert "idx_logs_timestamp" in indexes

    def test_init_db_idempotent(self, initialized_db: Path) -> None:
        """Test that init_db can be called multiple times safely."""
        from iris.client.web.database import init_db

        # Should not raise
        init_db()
        init_db()

        # Tables should still exist
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "sessions" in tables


class TestGetDb:
    """Tests for database connection context manager."""

    def test_get_db_returns_connection(self, initialized_db: Path) -> None:
        """Test that get_db returns a valid connection."""
        from iris.client.web.database import get_db

        with get_db() as conn:
            assert conn is not None
            # Test query
            cursor = conn.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1

    def test_get_db_enables_foreign_keys(self, initialized_db: Path) -> None:
        """Test that foreign keys are enabled."""
        from iris.client.web.database import get_db

        with get_db() as conn:
            cursor = conn.execute("PRAGMA foreign_keys")
            assert cursor.fetchone()[0] == 1

    def test_get_db_commits_on_success(self, initialized_db: Path) -> None:
        """Test that successful operations are committed."""
        from iris.client.web.database import get_db

        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, status, created_at, config)
                VALUES ('test', 'idle', 0.0, '{}')
                """
            )

        # Should persist
        with get_db() as conn:
            cursor = conn.execute("SELECT id FROM sessions WHERE id = 'test'")
            assert cursor.fetchone() is not None

    def test_get_db_rollbacks_on_error(self, initialized_db: Path) -> None:
        """Test that failed operations are rolled back."""
        from iris.client.web.database import get_db

        try:
            with get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (id, status, created_at, config)
                    VALUES ('rollback-test', 'idle', 0.0, '{}')
                    """
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Should not persist
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT id FROM sessions WHERE id = 'rollback-test'"
            )
            assert cursor.fetchone() is None


class TestJsonSerialization:
    """Tests for JSON serialization helpers."""

    def test_serialize_json_dict(self) -> None:
        """Test serializing a dictionary."""
        from iris.client.web.database import serialize_json

        result = serialize_json({"key": "value", "num": 42})
        assert result == '{"key": "value", "num": 42}'

    def test_serialize_json_list(self) -> None:
        """Test serializing a list."""
        from iris.client.web.database import serialize_json

        result = serialize_json([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_serialize_json_none(self) -> None:
        """Test serializing None returns empty dict."""
        from iris.client.web.database import serialize_json

        result = serialize_json(None)
        assert result == "{}"

    def test_deserialize_json_valid(self) -> None:
        """Test deserializing valid JSON."""
        from iris.client.web.database import deserialize_json

        result = deserialize_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_deserialize_json_none(self) -> None:
        """Test deserializing None returns empty dict."""
        from iris.client.web.database import deserialize_json

        result = deserialize_json(None)
        assert result == {}

    def test_deserialize_json_invalid(self) -> None:
        """Test deserializing invalid JSON returns empty dict."""
        from iris.client.web.database import deserialize_json

        result = deserialize_json("not valid json{")
        assert result == {}


class TestRowConversion:
    """Tests for row conversion helpers."""

    def test_row_to_dict(self, initialized_db: Path) -> None:
        """Test converting sqlite3.Row to dict."""
        from iris.client.web.database import get_db, row_to_dict

        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, status, created_at, config)
                VALUES ('row-test', 'idle', 123.456, '{}')
                """
            )
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = 'row-test'"
            ).fetchone()
            result = row_to_dict(row)

        assert result is not None
        assert result["id"] == "row-test"
        assert result["status"] == "idle"
        assert result["created_at"] == 123.456

    def test_row_to_dict_none(self) -> None:
        """Test converting None row returns None."""
        from iris.client.web.database import row_to_dict

        result = row_to_dict(None)
        assert result is None

    def test_rows_to_list(self, initialized_db: Path) -> None:
        """Test converting list of rows to list of dicts."""
        from iris.client.web.database import get_db, rows_to_list

        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, status, created_at, config)
                VALUES ('list-1', 'idle', 1.0, '{}'),
                       ('list-2', 'running', 2.0, '{}')
                """
            )
            rows = conn.execute(
                "SELECT * FROM sessions WHERE id LIKE 'list-%' ORDER BY id"
            ).fetchall()
            result = rows_to_list(rows)

        assert len(result) == 2
        assert result[0]["id"] == "list-1"
        assert result[1]["id"] == "list-2"
