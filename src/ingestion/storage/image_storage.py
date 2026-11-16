"""Image storage.

The loader writes the images it pulls out of a document to disk. This module
keeps the note of where each one went, so a later question about an image
can be answered by its id alone.

The images live under `data/images/{collection}/` and the note of them lives
in a SQLite database. The database runs in WAL mode, so a lookup does not
hold up a write and two runs can work at the same time.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.settings import resolve_path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/db/image_index.db"
DEFAULT_IMAGES_ROOT = "data/images"
DEFAULT_COLLECTION = "default"
LOCK_TIMEOUT_SECONDS = 30.0


class ImageStorage:
    """Keep the images of a collection on disk and in a database.

    The database remembers which file belongs to which image id, so an image
    that was pulled out of a document can be found again later.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        images_root: str | Path = DEFAULT_IMAGES_ROOT,
    ) -> None:
        """Set up the storage and make sure the database and folders exist."""
        self.db_path = resolve_path(db_path)
        self.images_root = resolve_path(images_root)
        self._ensure_database()

    def save_image(
        self,
        image_id: str,
        image_data: bytes | str | Path,
        collection: str | None = None,
        doc_hash: str | None = None,
        page_num: int | None = None,
        extension: str = "png",
    ) -> str:
        """Write an image to disk and note where it went.

        The image can be given as bytes or as the path of a file that is
        copied over. Saving the same image again replaces the file and the
        note, so a document that is ingested twice leaves one image behind.

        Returns the path of the image.
        """
        if not image_id or not image_id.strip():
            raise ValueError("image_id cannot be empty")

        target = self.images_root / (collection or DEFAULT_COLLECTION) / (
            f"{image_id}.{extension.lstrip('.')}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(image_data, bytes):
                target.write_bytes(image_data)
            else:
                source = Path(image_data)
                if not source.exists():
                    raise FileNotFoundError(f"The image to copy is not there: {source}")
                shutil.copy2(source, target)
        except Exception as exc:
            raise OSError(f"Could not save the image {image_id}: {exc}") from exc

        stored_path = str(target.resolve())
        self._write_note(image_id, stored_path, collection, doc_hash, page_num)
        return stored_path

    def register_image(
        self,
        image_id: str,
        file_path: str | Path,
        collection: str | None = None,
        doc_hash: str | None = None,
        page_num: int | None = None,
    ) -> str:
        """Note an image that is already on disk.

        The file is left where it is; only the note in the database is added.
        This is what the pipeline uses, because the loader has already put
        the image in its place.

        Returns the path of the image.
        """
        if not image_id or not image_id.strip():
            raise ValueError("image_id cannot be empty")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"The image is not there: {file_path}")

        stored_path = str(path.resolve())
        self._write_note(image_id, stored_path, collection, doc_hash, page_num)
        return stored_path

    def get_image_path(self, image_id: str) -> str | None:
        """Return where an image is, or None when it is not known."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?", (image_id,)
            ).fetchone()
        return str(row[0]) if row else None

    def image_exists(self, image_id: str) -> bool:
        """Return whether an image is known."""
        return self.get_image_path(image_id) is not None

    def list_images(
        self,
        collection: str | None = None,
        doc_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the images of a collection, or of a document.

        The images come back oldest first.
        """
        query = "SELECT * FROM image_index WHERE 1 = 1"
        parameters: list[str] = []
        if collection is not None:
            query += " AND collection = ?"
            parameters.append(collection)
        if doc_hash is not None:
            query += " AND doc_hash = ?"
            parameters.append(doc_hash)
        query += " ORDER BY created_at ASC, image_id ASC"

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def delete_image(self, image_id: str, remove_file: bool = True) -> bool:
        """Forget an image, and optionally take the file away as well.

        Returns whether the image was known.
        """
        file_path = self.get_image_path(image_id)
        if file_path is None:
            return False

        with self._connect() as connection:
            connection.execute("DELETE FROM image_index WHERE image_id = ?", (image_id,))

        if remove_file:
            Path(file_path).unlink(missing_ok=True)
        return True

    def get_collection_stats(self, collection: str) -> dict[str, Any]:
        """Return how many images a collection holds and how much room they take."""
        images = self.list_images(collection=collection)

        total_size = 0
        for image in images:
            try:
                path = Path(image["file_path"])
                if path.exists():
                    total_size += path.stat().st_size
            except OSError:
                continue

        return {"total_images": len(images), "total_size_bytes": total_size}

    def _write_note(
        self,
        image_id: str,
        file_path: str,
        collection: str | None,
        doc_hash: str | None,
        page_num: int | None,
    ) -> None:
        """Write the note of an image, replacing an earlier one."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO image_index
                    (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    collection = excluded.collection,
                    doc_hash = excluded.doc_hash,
                    page_num = excluded.page_num
                """,
                (
                    image_id,
                    file_path,
                    collection,
                    doc_hash,
                    page_num,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _ensure_database(self) -> None:
        """Create the database file, its tables and its folders."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_root.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)"
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
