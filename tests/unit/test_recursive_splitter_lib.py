"""Tests for the recursive splitter.

The splitter is what protects the Markdown structure of an ingested document,
so the tests focus on headings, fenced code blocks and the chunk bounds.
"""

from __future__ import annotations

import pytest

from core.settings import IngestionSettings, Settings
from libs.splitter.base_splitter import SplitterError
from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory

MARKDOWN = """# Project Guide

The guide explains how to run the service.

## Installation

Install the dependencies first.

```bash
pip install -r requirements.txt
python -m myproject
```

## Usage

Call the API to start working.
"""


def make_settings(chunk_size: int = 1000, chunk_overlap: int = 200, **overrides) -> Settings:
    ingestion = IngestionSettings(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter=overrides.pop("splitter", "recursive"),
    )
    return Settings(ingestion=ingestion)


def make_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveSplitter:
    return RecursiveSplitter(
        IngestionSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )


class TestFactoryRouting:
    def test_the_factory_creates_the_recursive_splitter(self):
        splitter = SplitterFactory.create(make_settings())

        assert isinstance(splitter, RecursiveSplitter)

    def test_the_strategy_name_is_matched_case_insensitively(self):
        splitter = SplitterFactory.create(make_settings(splitter="Recursive"))

        assert isinstance(splitter, RecursiveSplitter)


class TestSplitting:
    def test_a_short_document_stays_in_one_chunk(self):
        splitter = make_splitter()

        chunks = splitter.split_text(MARKDOWN)

        assert len(chunks) == 1
        assert "# Project Guide" in chunks[0]

    def test_a_long_document_is_split(self):
        splitter = make_splitter(chunk_size=120, chunk_overlap=20)

        chunks = splitter.split_text(MARKDOWN)

        assert len(chunks) > 1

    def test_every_chunk_is_within_the_size_limit(self):
        splitter = make_splitter(chunk_size=120, chunk_overlap=20)

        chunks = splitter.split_text(MARKDOWN)

        assert all(len(chunk) <= 120 for chunk in chunks)

    def test_an_empty_document_produces_no_chunks(self):
        assert make_splitter().split_text("") == []

    def test_a_blank_document_produces_no_chunks(self):
        assert make_splitter().split_text("   \n\n  ") == []

    def test_a_non_string_is_rejected(self):
        with pytest.raises(SplitterError, match="must be a string"):
            make_splitter().split_text(123)


class TestMarkdownStructure:
    def test_a_code_block_is_never_broken(self):
        splitter = make_splitter(chunk_size=120, chunk_overlap=20)

        chunks = splitter.split_text(MARKDOWN)

        code_chunks = [chunk for chunk in chunks if "```" in chunk]
        assert len(code_chunks) == 1
        assert code_chunks[0].count("```") == 2
        assert "pip install -r requirements.txt" in code_chunks[0]

    def test_a_heading_is_not_left_without_its_section(self):
        splitter = make_splitter(chunk_size=120, chunk_overlap=20)

        chunks = splitter.split_text(MARKDOWN)

        for chunk in chunks[:-1]:
            last_line = [line for line in chunk.splitlines() if line.strip()][-1]
            assert not last_line.strip().startswith("#"), chunk

    def test_the_document_content_is_preserved(self):
        splitter = make_splitter(chunk_size=120, chunk_overlap=20)

        joined = "\n".join(splitter.split_text(MARKDOWN))

        for fragment in (
            "# Project Guide",
            "The guide explains how to run the service.",
            "## Installation",
            "pip install -r requirements.txt",
            "## Usage",
            "Call the API to start working.",
        ):
            assert fragment in joined

    def test_a_code_block_larger_than_the_chunk_size_is_kept_whole(self):
        code = "```\n" + "\n".join(f"line {index}" for index in range(50)) + "\n```"

        chunks = make_splitter(chunk_size=100, chunk_overlap=10).split_text(code)

        assert len(chunks) == 1
        assert chunks[0].count("```") == 2


class TestConfiguration:
    def test_a_non_positive_chunk_size_is_rejected(self):
        with pytest.raises(SplitterError, match="chunk_size"):
            RecursiveSplitter(IngestionSettings(chunk_size=0, chunk_overlap=0))

    def test_a_negative_overlap_is_rejected(self):
        with pytest.raises(SplitterError, match="chunk_overlap"):
            RecursiveSplitter(IngestionSettings(chunk_size=100, chunk_overlap=-1))

    def test_an_overlap_larger_than_the_chunk_size_is_rejected(self):
        with pytest.raises(SplitterError, match="chunk_overlap"):
            RecursiveSplitter(IngestionSettings(chunk_size=100, chunk_overlap=100))

    def test_the_configured_size_and_overlap_are_used(self):
        splitter = make_splitter(chunk_size=200, chunk_overlap=50)

        assert splitter.chunk_size == 200
        assert splitter.chunk_overlap == 50
