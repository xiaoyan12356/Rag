"""File integrity checking for incremental ingestion.

A file that has already been ingested does not need to be ingested again, and
the cheapest way to recognise it is the hash of its content. The checker keeps
the hashes it has seen, so the next run can skip the files that did not
change and the answer survives a restart.
"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from core.settings import resolve_path

DEFAULT_DB_PATH = "data/db/ingestion_history.db"
READ_CHUNK_SIZE = 65536
LOCK_TIMEOUT_SECONDS = 30.0

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_PROCESSING = "processing"


def compute_file_sha256(file_path: str | Path) -> str:
    """Return the SHA256 hash of the file.

    The file is read in pieces, so hashing a large document does not need
    the whole document in memory.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise OSError(f"Path is not a file: {file_path}")

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(READ_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OSError(f"Failed to read file {file_path}: {exc}") from exc
    return digest.hexdigest()


class FileIntegrityChecker(ABC):
    """Remember which files were already ingested.

    A file is recognised by the hash of its content rather than by its name,
    so the same document under a different path is still recognised.
    """

    def compute_sha256(self, file_path: str | Path) -> str:
        """Return the SHA256 hash of the file."""
        return compute_file_sha256(file_path)

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Return whether the file with this hash was ingested before."""

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        *,
        file_size: int | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """Record that the file was ingested."""

    @abstractmethod
    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        file_path: str = "",
    ) -> None:
        """Record that the file could not be ingested."""


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """Keep the ingestion history in a SQLite database.

    The database runs in WAL mode, so a reader does not hold up the writer
    and two ingestion runs can write at the same time.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = resolve_path(db_path)
        self._ensure_database()

    def should_skip(self, file_hash: str) -> bool:
        """Return whether the file was ingested before.

        Only a successful run counts, so a file that failed is tried again.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM ingestion_history "
                "WHERE file_hash = ? AND status = ?",
                (file_hash, STATUS_SUCCESS),
            ).fetchone()
        return row is not None

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        *,
        file_size: int | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """Record that the file was ingested.

        Recording the same file again replaces the earlier record, so a file
        that is ingested twice still leaves one row behind.
        """
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_history
                    (file_hash, file_path, file_size, status, error_msg, chunk_count)
                VALUES (?, ?, ?, ?, NULL, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_size = excluded.file_size,
                    status = excluded.status,
                    processed_at = CURRENT_TIMESTAMP,
                    error_msg = NULL,
                    chunk_count = excluded.chunk_count
                """,
                (file_hash, file_path, file_size, STATUS_SUCCESS, chunk_count),
            )

    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        file_path: str = "",
    ) -> None:
        """Record that the file could not be ingested.

        The failure is kept for the record but does not make the file skippable,
        so the next run tries it again.
        """
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_history
                    (file_hash, file_path, file_size, status, error_msg, chunk_count)
                VALUES (?, ?, NULL, ?, ?, NULL)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = excluded.status,
                    processed_at = CURRENT_TIMESTAMP,
                    error_msg = excluded.error_msg
                """,
                (file_hash, file_path, STATUS_FAILED, error_msg),
            )

    def _ensure_database(self) -> None:
        """Create the database file and its schema if they are not there yet."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL
                        CHECK(status IN ('success', 'failed', 'processing')),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    chunk_count INTEGER
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_at "
                "ON ingestion_history(processed_at)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit the work and close it again."""
        connection = sqlite3.connect(self.db_path, timeout=LOCK_TIMEOUT_SECONDS)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
