"""Citations.

A citation says where a passage came from: which file, which page, which
chunk and how well it matched the question. The list of citations is what
lets a client show the reader why an answer was given.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.types import RetrievalResult

UNKNOWN_SOURCE = "unknown"


@dataclass
class Citation:
    """Where one retrieved passage came from."""

    index: int
    source: str
    chunk_id: str
    score: float
    page: int | None = None
    text: str = ""
    source_path: str = ""
    source_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the citation as a plain dictionary."""
        return asdict(self)


class CitationGenerator:
    """Turn retrieval results into the citations that point at them."""

    def generate(self, retrieval_results: list[RetrievalResult]) -> list[Citation]:
        """Return one citation per result, numbered from one.

        Args:
            retrieval_results: The passages the search returned, best first.

        Returns:
            The citations, in the order of the results.
        """
        citations: list[Citation] = []
        for number, result in enumerate(retrieval_results, start=1):
            source_path = str(result.metadata.get("source_path") or "")
            source = _source_of(result, source_path)
            citations.append(
                Citation(
                    index=number,
                    source=source,
                    chunk_id=result.chunk_id,
                    score=round(float(result.score), 6),
                    page=_page_of(result),
                    text=result.text,
                    source_path=source_path,
                    source_file=source,
                )
            )
        return citations

    @staticmethod
    def format_marker(index: int) -> str:
        """Return the marker a citation is referred to by in the text."""
        return f"[{index}]"


def _source_of(result: RetrievalResult, source_path: str) -> str:
    """Return the name of the file the passage came from."""
    if source_path:
        return Path(source_path).name
    title = result.metadata.get("title")
    return str(title) if title else UNKNOWN_SOURCE


def _page_of(result: RetrievalResult) -> int | None:
    """Return the page the passage is on, when the document says."""
    page = result.metadata.get("page")
    if isinstance(page, int):
        return page

    images = result.metadata.get("images") or []
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict) and isinstance(image.get("page"), int):
                return int(image["page"])
    return None
