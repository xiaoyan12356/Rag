"""Loader abstraction.

A loader turns a file of one format into the `Document` the rest of the
pipeline works with: text that is already normalised, and metadata that says
where the document came from. The loader does not split the document - that
is the splitter's job - so the two can be changed independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.types import Document


class LoaderError(RuntimeError):
    """Raised when a document cannot be parsed."""


class BaseLoader(ABC):
    """Contract that every document loader follows.

    A loader reads one file and returns one document. The document's metadata
    always carries a `source_path`, so a chunk can always be traced back to
    the file it came from.
    """

    @abstractmethod
    def load(self, file_path: str | Path) -> Document:
        """Return the document at `file_path`.

        Raises:
            FileNotFoundError: the file does not exist.
            OSError: the path is not a file.
            LoaderError: the file exists but could not be parsed.
        """

    @staticmethod
    def _validate_file(file_path: str | Path) -> Path:
        """Return the path of an existing file, or say why it cannot be read."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise OSError(f"Path is not a file: {file_path}")
        return path.resolve()
