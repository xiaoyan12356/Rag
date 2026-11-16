"""Unit tests for the file integrity checker.

The checker is what keeps a second ingestion run from doing the work twice,
so the tests pin the hash, the bookkeeping and the behaviour under a
concurrent write.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import pytest

from libs.loader.file_integrity import (
    DEFAULT_DB_PATH,
    STATUS_FAILED,
    STATUS_SUCCESS,
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
)


@pytest.fixture
def checker(tmp_path: Path) -> SQLiteIntegrityChecker:
    """Return a checker that keeps its history in a temporary database."""
    return SQLiteIntegrityChecker(tmp_path / "history.db")


class TestComputeSha256:
    def test_the_same_file_gives_the_same_hash(self, tmp_path: Path):
        target = tmp_path / "document.txt"
        target.write_text("hello world", encoding="utf-8")

        checker = SQLiteIntegrityChecker(tmp_path / "history.db")

        assert checker.compute_sha256(target) == checker.compute_sha256(target)

    def test_the_hash_matches_the_one_hashlib_computes(self, tmp_path: Path):
        target = tmp_path / "document.txt"
        target.write_bytes(b"hello world")

        checker = SQLiteIntegrityChecker(tmp_path / "history.db")
        expected = hashlib.sha256(b"hello world").hexdigest()

        assert checker.compute_sha256(target) == expected

    def test_the_hash_is_a_sha256_hex_digest(self, tmp_path: Path):
        target = tmp_path / "document.txt"
        target.write_text("content", encoding="utf-8")

        checker = SQLiteIntegrityChecker(tmp_path / "history.db")

        digest = checker.compute_sha256(target)

        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")

    def test_different_content_gives_a_different_hash(self, tmp_path: Path):
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("one", encoding="utf-8")
        second.write_text("two", encoding="utf-8")

        checker = SQLiteIntegrityChecker(tmp_path / "history.db")

        assert checker.compute_sha256(first) != checker.compute_sha256(second)

    def test_the_same_content_under_two_names_gives_the_same_hash(
        self, tmp_path: Path
    ):
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("same content", encoding="utf-8")
        second.write_text("same content", encoding="utf-8")

        checker = SQLiteIntegrityChecker(tmp_path / "history.db")

        assert checker.compute_sha256(first) == checker.compute_sha256(second)

    def test_an_empty_file_has_the_hash_of_the_empty_bytes(self, tmp_path: Path):
        target = tmp_path / "empty.txt"
        target.write_bytes(b"")

        checker = SQLiteIntegrityChecker(tmp_path / "history.db")

        assert checker.compute_sha256(target) == hashlib.sha256(b"").hexdigest()

    def test_a_large_file_is_read_in_pieces(self, tmp_path: Path):
        target = tmp_path / "large.bin"
        payload = b"a" * (1024 * 1024)
        target.write_bytes(payload)

        checker = SQLiteIntegrityChecker(tmp_path / "history.db")

        assert checker.compute_sha256(target) == hashlib.sha256(payload).hexdigest()

    def test_a_missing_file_is_reported(self, tmp_path: Path):
        checker = SQLiteIntegrityChecker(tmp_path / "history.db")

        with pytest.raises(FileNotFoundError):
            checker.compute_sha256(tmp_path / "missing.txt")

    def test_a_directory_is_reported(self, tmp_path: Path):
        checker = SQLiteIntegrityChecker(tmp_path / "history.db")

        with pytest.raises(OSError, match="not a file"):
            checker.compute_sha256(tmp_path)

    def test_a_string_path_is_accepted(self, tmp_path: Path):
        target = tmp_path / "document.txt"
        target.write_text("content", encoding="utf-8")

        checker = SQLiteIntegrityChecker(tmp_path / "history.db")

        assert checker.compute_sha256(str(target)) == checker.compute_sha256(target)


class TestShouldSkip:
    def test_an_unknown_file_is_not_skipped(self, checker: SQLiteIntegrityChecker):
        assert checker.should_skip("hash-that-was-never-seen") is False

    def test_a_file_marked_as_successful_is_skipped(
        self, checker: SQLiteIntegrityChecker
    ):
        checker.mark_success("hash_1", "data/a.txt")

        assert checker.should_skip("hash_1") is True

    def test_a_file_that_failed_is_not_skipped(self, checker: SQLiteIntegrityChecker):
        checker.mark_failed("hash_1", "boom")

        assert checker.should_skip("hash_1") is False

    def test_a_file_that_succeeds_after_a_failure_is_skipped(
        self, checker: SQLiteIntegrityChecker
    ):
        checker.mark_failed("hash_1", "boom")
        checker.mark_success("hash_1", "data/a.txt")

        assert checker.should_skip("hash_1") is True

    def test_two_files_are_tracked_separately(self, checker: SQLiteIntegrityChecker):
        checker.mark_success("hash_1", "data/a.txt")

        assert checker.should_skip("hash_1") is True
        assert checker.should_skip("hash_2") is False


class TestMarkSuccess:
    def test_the_record_keeps_the_file_details(self, checker: SQLiteIntegrityChecker):
        checker.mark_success(
            "hash_1",
            "data/a.txt",
            file_size=2048,
            chunk_count=7,
        )

        with sqlite3.connect(checker.db_path) as connection:
            row = connection.execute(
                "SELECT file_path, file_size, status, chunk_count, error_msg "
                "FROM ingestion_history WHERE file_hash = ?",
                ("hash_1",),
            ).fetchone()

        assert row == ("data/a.txt", 2048, STATUS_SUCCESS, 7, None)

    def test_the_file_size_and_the_chunk_count_are_optional(
        self, checker: SQLiteIntegrityChecker
    ):
        checker.mark_success("hash_1", "data/a.txt")

        with sqlite3.connect(checker.db_path) as connection:
            row = connection.execute(
                "SELECT file_size, chunk_count FROM ingestion_history "
                "WHERE file_hash = ?",
                ("hash_1",),
            ).fetchone()

        assert row == (None, None)

    def test_recording_the_same_file_twice_leaves_one_row(
        self, checker: SQLiteIntegrityChecker
    ):
        checker.mark_success("hash_1", "data/a.txt")
        checker.mark_success("hash_1", "data/a.txt")

        with sqlite3.connect(checker.db_path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM ingestion_history WHERE file_hash = ?",
                ("hash_1",),
            ).fetchone()[0]

        assert count == 1

    def test_recording_the_same_file_again_updates_the_record(
        self, checker: SQLiteIntegrityChecker
    ):
        checker.mark_success("hash_1", "data/a.txt", file_size=10)
        checker.mark_success("hash_1", "data/b.txt", file_size=20, chunk_count=3)

        with sqlite3.connect(checker.db_path) as connection:
            row = connection.execute(
                "SELECT file_path, file_size, chunk_count FROM ingestion_history "
                "WHERE file_hash = ?",
                ("hash_1",),
            ).fetchone()

        assert row == ("data/b.txt", 20, 3)

    def test_the_time_of_the_ingestion_is_recorded(
        self, checker: SQLiteIntegrityChecker
    ):
        checker.mark_success("hash_1", "data/a.txt")

        with sqlite3.connect(checker.db_path) as connection:
            processed_at = connection.execute(
                "SELECT processed_at FROM ingestion_history WHERE file_hash = ?",
                ("hash_1",),
            ).fetchone()[0]

        assert processed_at is not None

    def test_a_successful_record_clears_an_earlier_error(
        self, checker: SQLiteIntegrityChecker
    ):
        checker.mark_failed("hash_1", "boom", "data/a.txt")
        checker.mark_success("hash_1", "data/a.txt")

        with sqlite3.connect(checker.db_path) as connection:
            error_msg = connection.execute(
                "SELECT error_msg FROM ingestion_history WHERE file_hash = ?",
                ("hash_1",),
            ).fetchone()[0]

        assert error_msg is None


class TestMarkFailed:
    def test_the_error_is_kept_for_the_record(self, checker: SQLiteIntegrityChecker):
        checker.mark_failed("hash_1", "boom")

        with sqlite3.connect(checker.db_path) as connection:
            row = connection.execute(
                "SELECT status, error_msg FROM ingestion_history WHERE file_hash = ?",
                ("hash_1",),
            ).fetchone()

        assert row == (STATUS_FAILED, "boom")

    def test_the_file_path_is_optional(self, checker: SQLiteIntegrityChecker):
        checker.mark_failed("hash_1", "boom", "data/a.txt")

        with sqlite3.connect(checker.db_path) as connection:
            file_path = connection.execute(
                "SELECT file_path FROM ingestion_history WHERE file_hash = ?",
                ("hash_1",),
            ).fetchone()[0]

        assert file_path == "data/a.txt"

    def test_a_failure_replaces_an_earlier_success(
        self, checker: SQLiteIntegrityChecker
    ):
        checker.mark_success("hash_1", "data/a.txt")
        checker.mark_failed("hash_1", "boom")

        assert checker.should_skip("hash_1") is False


class TestDatabase:
    def test_the_database_file_is_created(self, tmp_path: Path):
        db_path = tmp_path / "nested" / "history.db"

        SQLiteIntegrityChecker(db_path)

        assert db_path.exists()

    def test_the_table_is_created(self, checker: SQLiteIntegrityChecker):
        with sqlite3.connect(checker.db_path) as connection:
            name = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'ingestion_history'"
            ).fetchone()

        assert name is not None

    def test_the_indexes_are_created(self, checker: SQLiteIntegrityChecker):
        with sqlite3.connect(checker.db_path) as connection:
            names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                )
            }

        assert "idx_status" in names
        assert "idx_processed_at" in names

    def test_the_database_runs_in_wal_mode(self, checker: SQLiteIntegrityChecker):
        with sqlite3.connect(checker.db_path) as connection:
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

        assert mode.lower() == "wal"

    def test_an_unknown_status_is_rejected(self, checker: SQLiteIntegrityChecker):
        with sqlite3.connect(checker.db_path) as connection:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO ingestion_history "
                    "(file_hash, file_path, status) VALUES (?, ?, ?)",
                    ("hash_1", "data/a.txt", "unknown"),
                )

    def test_the_history_survives_a_restart(self, tmp_path: Path):
        db_path = tmp_path / "history.db"

        SQLiteIntegrityChecker(db_path).mark_success("hash_1", "data/a.txt")

        assert SQLiteIntegrityChecker(db_path).should_skip("hash_1") is True

    def test_the_default_location_is_under_the_data_directory(self):
        assert DEFAULT_DB_PATH == "data/db/ingestion_history.db"

    def test_a_relative_path_is_resolved_against_the_repository(self):
        checker = SQLiteIntegrityChecker(DEFAULT_DB_PATH)

        assert checker.db_path.is_absolute()
        assert checker.db_path.name == "ingestion_history.db"
        assert checker.db_path.parent.name == "db"


class TestConcurrentWrites:
    def test_several_threads_can_record_at_the_same_time(self, tmp_path: Path):
        checker = SQLiteIntegrityChecker(tmp_path / "history.db")
        hashes = [f"hash_{index}" for index in range(10)]
        barrier = threading.Barrier(len(hashes))

        def record(file_hash: str) -> None:
            barrier.wait()
            checker.mark_success(file_hash, f"data/{file_hash}.txt")

        threads = [
            threading.Thread(target=record, args=(file_hash,)) for file_hash in hashes
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        with sqlite3.connect(checker.db_path) as connection:
            rows = connection.execute(
                "SELECT file_hash FROM ingestion_history"
            ).fetchall()

        assert {row[0] for row in rows} == set(hashes)


class TestAbstractBase:
    def test_the_base_class_cannot_be_used_on_its_own(self):
        with pytest.raises(TypeError):
            FileIntegrityChecker()

    def test_a_subclass_has_to_implement_every_method(self):
        class Incomplete(FileIntegrityChecker):
            def should_skip(self, file_hash: str) -> bool:
                return False

        with pytest.raises(TypeError):
            Incomplete()

    def test_a_complete_subclass_can_be_used(self, tmp_path: Path):
        class InMemory(FileIntegrityChecker):
            def __init__(self) -> None:
                self.seen: set[str] = set()

            def should_skip(self, file_hash: str) -> bool:
                return file_hash in self.seen

            def mark_success(self, file_hash, file_path, **kwargs) -> None:
                self.seen.add(file_hash)

            def mark_failed(self, file_hash, error_msg, file_path="") -> None:
                pass

        checker = InMemory()
        checker.mark_success("hash_1", "data/a.txt")

        assert checker.should_skip("hash_1") is True
