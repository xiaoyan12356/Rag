"""Recursive character splitting.

LangChain's recursive splitter does the character level work; this module adds
the Markdown awareness the ingestion pipeline needs. A fenced code block is
never broken in half, and a heading is never left behind at the end of a chunk
without the section it introduces.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.settings import IngestionSettings
from libs.splitter.base_splitter import BaseSplitter, SplitterError

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_HEADING = re.compile(r"^#{1,6}\s")


class RecursiveSplitter(BaseSplitter):
    """Split text into chunks that respect Markdown structure."""

    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        "; ",
        ", ",
        " ",
        "",
    ]

    def __init__(
        self,
        config: IngestionSettings,
        *,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(config)

        chunk_size = config.chunk_size
        chunk_overlap = config.chunk_overlap
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise SplitterError(f"ingestion.chunk_size must be a positive integer: {chunk_size}")
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise SplitterError(
                f"ingestion.chunk_overlap must be a non-negative integer: {chunk_overlap}"
            )
        if chunk_overlap >= chunk_size:
            raise SplitterError(
                f"ingestion.chunk_overlap ({chunk_overlap}) must be smaller than "
                f"ingestion.chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(self, text: str, trace: TraceContext | None = None) -> list[str]:
        """Split `text` into chunks, in reading order."""
        if not isinstance(text, str):
            raise SplitterError(f"text must be a string, got {type(text).__name__}")
        if not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks: list[str] = []
        for segment, is_code in _split_code_blocks(text):
            if is_code:
                chunks.append(segment.strip())
                continue
            chunks.extend(chunk for chunk in self._splitter.split_text(segment) if chunk.strip())

        return _merge_orphaned_headings(chunks)


def _split_code_blocks(text: str) -> list[tuple[str, bool]]:
    """Break the text into prose segments and atomic fenced code blocks."""
    segments: list[tuple[str, bool]] = []
    cursor = 0
    for match in _CODE_BLOCK.finditer(text):
        if match.start() > cursor:
            segments.append((text[cursor : match.start()], False))
        segments.append((match.group(0), True))
        cursor = match.end()
    if cursor < len(text):
        segments.append((text[cursor:], False))
    return segments


def _merge_orphaned_headings(chunks: list[str]) -> list[str]:
    """Keep a heading together with the section it introduces."""
    merged: list[str] = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        while index + 1 < len(chunks) and _ends_with_heading(chunk):
            index += 1
            chunk = f"{chunk}\n\n{chunks[index]}"
        merged.append(chunk)
        index += 1
    return merged


def _ends_with_heading(chunk: str) -> bool:
    """Whether the last line of `chunk` is a Markdown heading."""
    lines = [line for line in chunk.splitlines() if line.strip()]
    return bool(lines) and bool(_HEADING.match(lines[-1].strip()))
